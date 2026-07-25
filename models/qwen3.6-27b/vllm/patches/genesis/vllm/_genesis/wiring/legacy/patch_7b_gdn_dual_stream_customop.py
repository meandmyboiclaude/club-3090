# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 7b — GDN dual-stream via `torch.library.custom_op`.

Replaces the back-to-back serial calls
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)
in `GatedDeltaNet.forward_cuda` (non-LoRA branch, `else:` of
`if hasattr(self, "in_proj_qkv"):`) with a single call to our
custom op:
    mixed_qkvz, ba = torch.ops.genesis.dual_linear_parallel(...)

Why P7b (custom op) instead of P7 (raw streams)
------------------------------------------------
P7's `DualStreamDispatcher.maybe_parallel(...)` uses `torch.cuda.Stream`
directly in the forward path. Dynamo can't symbolically trace that
API, so `torch.compile(fullgraph=True)` fails. On vLLM's default
`aot_compile_fullgraph` path (mode=3), P7 therefore requires
`--enforce-eager` — throwing away the ~40% prefill speedup of
compile.

P7b wraps the dual-GEMM as a single **custom op**. Dynamo treats it
as an opaque node: the body (with CUDA streams) is NEVER traced;
only the fake meta is consulted for shape inference. Result: works
inside `fullgraph=True` AND keeps the parallel-streams win (+5-8%
decode on Qwen3-Next hybrid, measured in P7 eager mode).

Coexistence with P7
-------------------
P7 (text-patch to DualStreamDispatcher) remains in-tree for
`--enforce-eager` users. P7b adds a NEW text-patch that replaces the
same 2 lines with the custom op call. Only ONE of (P7, P7b) should
be active at a time — they text-patch the SAME anchor. We enforce
this at apply time: P7b, if enabled, also marks P7 as conflicting
via the text-patch marker check (p7b's anchor includes P7's original
raw-stream marker → fails if P7 already patched).

Opt-in gate
-----------
`GENESIS_ENABLE_P7B=1`. OFF by default until VM 100 validates
numerical equivalence to serial (tight diff on GSM8K + quality
harness) AND compile-cache hash collision check (fresh cache
required).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.5 implementation (opt-in)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import (
    resolve_vllm_file, vllm_install_root, is_nvidia_cuda, is_sm_at_least,
)
from vllm._genesis.kernels.gdn_dual_stream_customop import is_p7b_enabled
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p7b_gdn_dual_stream_customop")

GENESIS_P7B_MARKER = (
    "Genesis P7b GDN dual-stream custom_op import-time-cached v7.68"
)

# Drift markers: if upstream adds its own dual-stream via custom_op OR
# via multi_stream framework, we self-retire.
UPSTREAM_DRIFT_MARKERS = [
    "dual_linear_parallel",
    "multi_stream_linear",
    "genesis::gdn_dual_gemm",
    "_GENESIS_P7B_DUAL_LINEAR_OP",  # v7.68 marker
]


# ─── v7.68 — sister fix to PN25 v7.68 (same bug class) ─────────────
#
# v7.5/v7.66 P7b had the same bug class as v7.66 PN25: registration
# happens INSIDE the call site (`from ...customop import
# dual_linear_parallel`), and the call site is reached during dynamo
# tracing of `forward_cuda` under cudagraph capture. Worker spawn +
# fresh interpreter + `Library` construction inside trace = same
# Dynamo crash class:
#
#   torch._dynamo.exc.Unsupported: ... instantiate_user_defined_class_object
#
# Preventive fix (P7b is opt-in / default OFF, so noonghunna didn't hit
# this — but anyone who flips `GENESIS_ENABLE_P7B_DUAL_STREAM_CUSTOM_OP=1`
# on a TP=1 spawn config would). Same import-time pattern as PN25 v7.68:
#   1. Sub-patch 1: insert at module-import time of `gdn_linear_attn.py`
#      a try/except that calls `_register_op_once()` and caches result
#      as `_GENESIS_P7B_DUAL_LINEAR_OP` module global.
#   2. Sub-patch 2: replace the in_proj call body to read only the
#      cached global (no import inside forward_cuda).
#
# Pattern credit: noonghunna's PN25 v3 fix on club-3090
# (commit a62ad78, 2026-05-01). Applied preventively to P7b.

# Sub-patch 1: insert import-time registration at module top.
# Anchor: the import of GatedDeltaNet's surrounding helpers.
_P7B_IMPORT_ANCHOR = (
    "from vllm.distributed import get_tensor_model_parallel_world_size\n"
    "\n"
)

_P7B_IMPORT_REPLACEMENT = (
    "from vllm.distributed import get_tensor_model_parallel_world_size\n"
    "\n"
    "# [Genesis P7b v7.68] Register/cache the dual-stream custom op at\n"
    "# module import time, BEFORE any cudagraph/torch.compile context.\n"
    "# The patched in_proj call site below reads only this cached global.\n"
    "# Same pattern as PN25 v7.68 (silu_and_mul) — fixes the dynamo\n"
    "# 'instantiate_user_defined_class_object' crash class on TP=1 spawn.\n"
    "try:\n"
    "    from vllm._genesis.kernels.gdn_dual_stream_customop import (\n"
    "        dual_linear_parallel as _genesis_p7b_dual_linear_op,\n"
    "    )\n"
    "    _GENESIS_P7B_DUAL_LINEAR_OP = _genesis_p7b_dual_linear_op\n"
    "except Exception:\n"
    "    _GENESIS_P7B_DUAL_LINEAR_OP = None\n"
    "\n"
)

# Sub-patch 2: replace the call site to read the cached global.
# Anchor: the back-to-back in_proj calls in the non-LoRA branch.
# Matches the ORIGINAL upstream text (NOT the P7-patched variant —
# if P7 is already applied, P7b will not find its anchor and will
# skip with a clear reason).
_OLD_INPROJ = (
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)"
)

_NEW_INPROJ = (
    "            # [Genesis P7b v7.68] Dual GEMM through opaque op cached at\n"
    "            # module import time (no registration inside forward — fixes\n"
    "            # spawn-worker dynamo trace bug class). Falls back to serial\n"
    "            # in_proj if op cache is None (CPU build / op disabled).\n"
    "            if _GENESIS_P7B_DUAL_LINEAR_OP is not None:\n"
    "                mixed_qkvz, ba = _GENESIS_P7B_DUAL_LINEAR_OP(\n"
    "                    hidden_states,\n"
    "                    self.in_proj_qkvz.weight,\n"
    "                    getattr(self.in_proj_qkvz, 'bias', None),\n"
    "                    self.in_proj_ba.weight,\n"
    "                    getattr(self.in_proj_ba, 'bias', None),\n"
    "                )\n"
    "            else:\n"
    "                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "                ba, _ = self.in_proj_ba(hidden_states)"
)


# ── V2 shapes (2026-07-26 re-anchor) ───────────────────────────────────────
# Same two drifts as P7, plus a third of its own. P7b is not in the ledger's
# "target path gone" list only because it is keyed differently; it was sitting
# in exactly the same hole.
#
#  1. The file was renamed to mamba/gdn/qwen_gdn_linear_attn.py. Handled by
#     the _PATH_ALIASES entry in guards.py (both lanes).
#  2. The `if hasattr(self, "in_proj_qkv"): ... else:` branch is gone, so the
#     12-space call-site anchor is count==0. The 8-space pair is count==2
#     (forward_cuda AND forward_cpu), so the num_tokens + banner prefix is
#     load-bearing for uniqueness — patching the bare pair would put the
#     custom op on the CPU forward path.
#  3. The import anchor drifted independently: the single-line
#     `from vllm.distributed import get_tensor_model_parallel_world_size`
#     is now a parenthesised block importing `divide`. count==0 for the old
#     literal. Sub-patch 1 is `required=True` and BOTH sub-patches must land
#     (a half-apply NameErrors on the global at the call site), so this alone
#     was enough to make P7b unappliable.
_P7B_IMPORT_ANCHOR_V2 = (
    "from vllm.distributed import (\n"
    "    divide,\n"
    ")\n"
)

_P7B_IMPORT_REPLACEMENT_V2 = (
    "from vllm.distributed import (\n"
    "    divide,\n"
    ")\n"
    "\n"
    "# [Genesis P7b v7.68] Register/cache the dual-stream custom op at\n"
    "# module import time, BEFORE any cudagraph/torch.compile context.\n"
    "# The patched in_proj call site below reads only this cached global.\n"
    "try:\n"
    "    from vllm._genesis.kernels.gdn_dual_stream_customop import (\n"
    "        dual_linear_parallel as _genesis_p7b_dual_linear_op,\n"
    "    )\n"
    "    _GENESIS_P7B_DUAL_LINEAR_OP = _genesis_p7b_dual_linear_op\n"
    "except Exception:\n"
    "    _GENESIS_P7B_DUAL_LINEAR_OP = None\n"
)

_OLD_INPROJ_V2 = (
    "        num_tokens = hidden_states.size(0)\n"
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        ba, _ = self.in_proj_ba(hidden_states)\n"
)

_NEW_INPROJ_V2 = (
    "        num_tokens = hidden_states.size(0)\n"
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        # [Genesis P7b v7.68] Dual GEMM through opaque op cached at\n"
    "        # module import time (no registration inside forward — fixes\n"
    "        # spawn-worker dynamo trace bug class). Falls back to serial\n"
    "        # in_proj if op cache is None (CPU build / op disabled).\n"
    "        if _GENESIS_P7B_DUAL_LINEAR_OP is not None:\n"
    "            mixed_qkvz, ba = _GENESIS_P7B_DUAL_LINEAR_OP(\n"
    "                hidden_states,\n"
    "                self.in_proj_qkvz.weight,\n"
    "                getattr(self.in_proj_qkvz, 'bias', None),\n"
    "                self.in_proj_ba.weight,\n"
    "                getattr(self.in_proj_ba, 'bias', None),\n"
    "            )\n"
    "        else:\n"
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
)

# Ordered newest-first, per sub-patch. Shapes are only ever ADDED, so the
# older pinned images keep resolving.
_P7B_VARIANTS = {
    "p7b_import_time_register": (
        (_P7B_IMPORT_ANCHOR_V2, _P7B_IMPORT_REPLACEMENT_V2),
        (_P7B_IMPORT_ANCHOR, _P7B_IMPORT_REPLACEMENT),
    ),
    "p7b_inproj_via_custom_op": (
        (_OLD_INPROJ_V2, _NEW_INPROJ_V2),
        (_OLD_INPROJ, _NEW_INPROJ),
    ),
}


def pick_anchors(text: str) -> dict[str, tuple[str, str]] | None:
    """Resolve every sub-patch to the variant that occurs EXACTLY ONCE.

    All-or-nothing on purpose: P7b's two sub-patches are both required (the
    call site reads a global the import block defines), so resolving one and
    letting the other soft-skip is the PN346B failure — an announced apply
    with half the patch missing. Returns None if ANY sub-patch has no
    uniquely-present variant.
    """
    out: dict[str, tuple[str, str]] = {}
    for name, variants in _P7B_VARIANTS.items():
        hit = next(((a, r) for a, r in variants if text.count(a) == 1), None)
        if hit is None:
            return None
        out[name] = hit
    return out


def anchor_report(text: str) -> dict[str, list[int]]:
    """Per-sub-patch counts of every known shape — for offline verifiers."""
    return {name: [text.count(a) for a, _ in variants]
            for name, variants in _P7B_VARIANTS.items()}


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn_linear_attn.py"
    )
    if target is None:
        return None
    from pathlib import Path
    try:
        text = Path(str(target)).read_text(encoding="utf-8")
    except OSError:
        return None
    picked = pick_anchors(text)
    if picked is None:
        log.warning(
            "[P7b] not every sub-patch has a uniquely-present anchor in %s — "
            "counts %s. Both sub-patches are required; refusing to apply half "
            "of them. Re-derive before the next boot.",
            target, anchor_report(text),
        )
        return None
    _P7B_IMPORT_A, _P7B_IMPORT_R = picked["p7b_import_time_register"]
    _INPROJ_A, _INPROJ_R = picked["p7b_inproj_via_custom_op"]
    return TextPatcher(
        patch_name="P7b GDN dual-stream via torch.library.custom_op",
        target_file=target,
        marker=GENESIS_P7B_MARKER,
        sub_patches=[
            # v7.68: import-time registration first so the global is
            # defined BEFORE the in_proj call site references it. Both
            # required — half-applied state would NameError on the
            # global at the patched call site.
            TextPatch(
                name="p7b_import_time_register",
                anchor=_P7B_IMPORT_A,
                replacement=_P7B_IMPORT_R,
                required=True,
            ),
            TextPatch(
                name="p7b_inproj_via_custom_op",
                anchor=_INPROJ_A,
                replacement=_INPROJ_R,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def should_apply() -> bool:
    """Gate: env-opt-in + NVIDIA CUDA SM≥8.0 (streams need real GPU)."""
    if not is_p7b_enabled():
        return False
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def apply() -> tuple[str, str]:
    """Apply P7b text-patch. Never raises."""
    if not is_p7b_enabled():
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_P7B=1 to enable GDN dual-stream "
            "via torch.library.custom_op (graph-safe alternative to P7; "
            "validates numeric equiv + compile-cache freshness before prod)"
        )

    if not is_nvidia_cuda():
        return "skipped", (
            "P7b targets NVIDIA CUDA (streams) — platform skip"
        )

    if not is_sm_at_least(8, 0):
        return "skipped", "SM < 8.0 — CUDA streams parallelism weak"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found in vllm install"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "text-patch applied — in_proj_qkvz + in_proj_ba now run "
            "through torch.ops.genesis.dual_linear_parallel (opaque to "
            "dynamo → compile-fullgraph safe)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already patched this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown skip"
        # Differentiate "anchor not found because P7 already replaced it"
        # from "anchor not found for some other reason". Both result in
        # skip but the former is a USER issue (both P7 and P7b active).
        if failure and "anchor" in (failure.reason or "").lower():
            return "skipped", (
                f"anchor not found — likely P7 (raw-stream dispatcher) "
                f"already patched the same 2 lines. Disable P7 "
                f"(unset GENESIS_ENABLE_P7=1 if set) or revert first. "
                f"Details: {reason}"
            )
        return "skipped", reason
    return "failed", failure.reason if failure else "unknown failure"


def is_applied() -> bool:
    """True iff the target file contains the P7b marker."""
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn_linear_attn.py"
    )
    if target is None or not target.exists():
        return False
    try:
        return GENESIS_P7B_MARKER in target.read_text()
    except Exception:
        return False


def revert() -> bool:
    """P7b is a text-patch — no clean runtime revert. Operators must
    restart the container with `compose down && up -d` to restore
    upstream behavior (the container R/W layer holds the patched file).
    """
    return False
