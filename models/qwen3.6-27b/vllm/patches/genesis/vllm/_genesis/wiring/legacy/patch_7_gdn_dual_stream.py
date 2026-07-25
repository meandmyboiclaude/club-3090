# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 7 — GatedDeltaNet dual-stream in_proj parallelism.

Problem
-------
`GatedDeltaNet.forward_cuda` (gdn_linear_attn.py:519) serially calls
`self.in_proj_qkvz(hidden_states)` then `self.in_proj_ba(hidden_states)`
at lines 544-545. These two GEMMs are independent and can execute in
parallel on an auxiliary CUDA stream, recovering ~5% decode throughput on
Qwen3-Next / Qwen3.6 hybrid models.

Fix
---
Replace the serial call pair with a `DualStreamDispatcher.maybe_parallel`
invocation. On NVIDIA CUDA SM≥8.0, the dispatcher issues the two GEMMs on
separate CUDA streams synchronised by events. On all other platforms it
falls back to a sequential call (identical behaviour to upstream).

Platform compatibility: graceful degradation
  - NVIDIA CUDA SM≥8.0 → true parallel execution
  - AMD ROCm            → best-effort HIP stream (may serialize)
  - Intel XPU / CPU     → sequential fallback

Wiring strategy: TEXT-PATCH on gdn_linear_attn.py. Class-method rebind is
not viable because the two GEMMs are buried inside a conditional branch
(`if hasattr(self, "in_proj_qkv"):` vs `else:`) — we only want to patch
the `else` branch (non-LoRA Qwen3.5 / Qwen3-Next path where both GEMMs
are called back-to-back).

Upstream drift detection: if `DualStreamDispatcher` or `aux_stream` appears
upstream, the patch is skipped as obsolete.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p7_gdn_dual_stream")

GENESIS_P7_MARKER = "Genesis P7 GDN dual-stream in_proj v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "DualStreamDispatcher",
    "gdn_aux_stream",
    "in_proj_dual_stream",
]


# Anchor: the back-to-back in_proj calls in the non-LoRA branch of
# GatedDeltaNet.forward_cuda. The first line has leading 12-space indent
# (inside `else:` of `if hasattr(self, "in_proj_qkv")`).
_OLD_INPROJ = (
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)"
)

_NEW_INPROJ = (
    "            # [Genesis P7] Dispatch in_proj_qkvz and in_proj_ba in parallel on\n"
    "            # an auxiliary CUDA stream (SM≥8.0); sequential fallback elsewhere.\n"
    "            from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher\n"
    "            (mixed_qkvz, _), (ba, _) = DualStreamDispatcher.maybe_parallel(\n"
    "                lambda: self.in_proj_qkvz(hidden_states),\n"
    "                lambda: self.in_proj_ba(hidden_states),\n"
    "            )"
)


# ── V2 shape (2026-07-26 re-anchor) ────────────────────────────────────────
# Two independent drifts, and BOTH have to be fixed or the patch resolves to
# nothing:
#
#  1. The file was renamed. `model_executor/layers/mamba/gdn_linear_attn.py`
#     became `.../mamba/gdn/qwen_gdn_linear_attn.py` when upstream split GDN
#     per model family (siblings kimi_/olmo_). dispatcher.py's P60 row had
#     already recorded that rename; nothing propagated it here, so
#     resolve_vllm_file() returned None and apply() reported
#     "gdn_linear_attn.py not found" on every boot. Fixed in guards.py's
#     _PATH_ALIASES, both lanes, so the OLD literal keeps resolving.
#  2. The `if hasattr(self, "in_proj_qkv"): ... else:` branch is gone, so the
#     12-space V1 anchor is count==0 in the pin. The calls now sit flat at
#     8-space indent in QwenGatedDeltaNetAttention.forward_cuda.
#
# The bare two-line call pair is count==2 post-boot — forward_cuda AND
# forward_cpu — so it is NOT usable as an anchor on its own. The
# `num_tokens` + banner prefix is what makes it unique (forward_xpu has the
# same banner but binds `projected_states_qkvz` and puts a blank line after
# `num_tokens`). Verified count==1 against the post-boot bytes, not the
# pristine image: P28, PN11, PN50 and PN350 all rewrite this same file.
#
# V1 is RETAINED and the pair is CONTENT-SNIFFED (P89 6edf8386 / P101
# 92647851 / PN346B 79c655c2 precedent) so nothing is rewritten in place.
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
    "        # [Genesis P7] Dispatch in_proj_qkvz and in_proj_ba in parallel on\n"
    "        # an auxiliary CUDA stream (SM≥8.0); sequential fallback elsewhere.\n"
    "        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher\n"
    "        (mixed_qkvz, _), (ba, _) = DualStreamDispatcher.maybe_parallel(\n"
    "            lambda: self.in_proj_qkvz(hidden_states),\n"
    "            lambda: self.in_proj_ba(hidden_states),\n"
    "        )\n"
)

# Ordered newest-first. Every historical shape stays here; a shape is only
# ever ADDED.
_INPROJ_VARIANTS = (
    (_OLD_INPROJ_V2, _NEW_INPROJ_V2),
    (_OLD_INPROJ, _NEW_INPROJ),
)


def pick_anchor(text: str) -> tuple[str, str] | None:
    """Return the (anchor, replacement) whose anchor occurs EXACTLY ONCE.

    Counted, not merely present. A shape that matches twice is a drift the
    patch must refuse rather than guess at: the bare in_proj call pair is
    count==2 today (forward_cuda and forward_cpu), and silently patching the
    first hit would put a CUDA-stream dispatcher on the CPU path. Returns None
    when no variant is uniquely present, which the caller turns into a LOUD
    skip — never a silent no-op.
    """
    for anchor, replacement in _INPROJ_VARIANTS:
        if text.count(anchor) == 1:
            return anchor, replacement
    return None


def anchor_report(text: str) -> list[tuple[str, int]]:
    """(variant-label, count) for every known shape — for offline verifiers."""
    labels = ("v2_forward_cuda_flat", "v1_else_branch")
    return [(lbl, text.count(a)) for lbl, (a, _) in zip(labels, _INPROJ_VARIANTS)]


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    try:
        from pathlib import Path
        picked = pick_anchor(Path(str(target)).read_text(encoding="utf-8"))
    except OSError:
        picked = None
    if picked is None:
        # No known shape is uniquely present. Say so instead of handing the
        # patcher a V1 anchor that cannot match — an "anchor not found" skip
        # reads like ordinary drift and buries the fact that we looked at
        # every variant and none of them fit.
        log.warning(
            "[P7] no in_proj variant is uniquely present in %s — counts %s. "
            "Re-derive the anchor before the next boot.",
            target, anchor_report(Path(str(target)).read_text(
                encoding="utf-8", errors="replace")) if target else "unreadable",
        )
        return None
    anchor, replacement = picked
    return TextPatcher(
        patch_name="P7 GDN dual-stream in_proj",
        target_file=target,
        marker=GENESIS_P7_MARKER,
        sub_patches=[
            TextPatch(
                name="p7_inproj_dual_stream",
                anchor=anchor,
                replacement=replacement,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P7 wiring — currently DEFERRED under torch.compile fullgraph.

    Architectural constraint
    ------------------------
    vLLM's AOT-compile path uses `aot_compile_fullgraph` which REJECTS
    graph breaks. `DualStreamDispatcher.maybe_parallel` uses
    `torch.cuda.stream(aux_stream)` and `cuda.Event` objects — inherently
    side-effectful CUDA stream orchestration that is NOT representable
    as a SymPy-traceable op. Specifically fails with:

        RuntimeError: Worker failed with error 'cannot extract sympy
        expressions from <torch.cuda.Stream device=cuda:0 ...>',
        please check the stack trace above for the root cause

    Since prod runs with compile enabled (`enforce_eager=False`), P7's
    dual-stream benefit would be erased by the graph break anyway. The
    correct forward path is:

      1. Re-implement `maybe_parallel` as a single `torch.library.define`
         custom op that wraps the stream orchestration on the C++ side.
      2. Register the op in `splitting_ops` so vLLM's piecewise graph
         cuts around it.

    Until that custom op is authored, P7 skips so the integration stays
    compile-compatible and correct. This is a TEMPORARY deferral, NOT a
    regression — upstream's serial `in_proj` calls run unchanged.

    Re-enable via env: `GENESIS_ENABLE_P7=1` for eager-mode users who
    pass `--enforce-eager` at vllm-serve time (rare in prod).
    """
    import os
    enabled = os.environ.get("GENESIS_ENABLE_P7", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not enabled:
        return (
            "skipped",
            "deferred — incompatible with torch.compile fullgraph "
            "(CUDA streams not SymPy-graphable); "
            "custom op implementation required. "
            "Re-enable with GENESIS_ENABLE_P7=1 + --enforce-eager.",
        )

    from vllm._genesis.guards import is_cpu_only, is_intel_xpu
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        if resolve_vllm_file(
                "model_executor/layers/mamba/gdn_linear_attn.py") is None:
            return "skipped", (
                "gdn_linear_attn.py not found (checked the "
                "mamba/gdn/qwen_gdn_linear_attn.py rename alias too)"
            )
        return "skipped", (
            "required_anchor_missing: no in_proj variant is uniquely present "
            "— see the [P7] warning above for the per-variant counts"
        )

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        cpu_note = " (CPU/XPU fall back to sequential execution)" if (
            is_cpu_only() or is_intel_xpu()
        ) else ""
        return "applied", f"dual-stream in_proj wired{cpu_note} (opt-in)"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
