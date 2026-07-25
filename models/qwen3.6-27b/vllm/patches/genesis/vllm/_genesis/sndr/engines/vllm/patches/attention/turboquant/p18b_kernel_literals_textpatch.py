# SPDX-License-Identifier: Apache-2.0
"""Wiring for P18B_TEXT — TurboQuant decode stage1 kernel-literal tune.

Genesis-original. The original P18b (kernels_legacy/tq_decode_tune.py +
dispatch hook in sndr/apply/_per_patch_dispatch.py:6275) reads the
``VLLM_TQ_DECODE_{BLOCK_KV,NUM_WARPS,NUM_STAGES}`` env vars and **logs**
their resolved value, but never patches the actual Triton launcher.

Kernels-audit agent (2026-06-08) flagged this as dead code: 35B + 27B
production has been running with the upstream H100 defaults
(``num_warps=4, num_stages=2`` on the GQA branch, ``num_warps=1,
num_stages=1`` on the MHA branch) on every boot, regardless of env
overrides — under-utilising Ampere SM 8.6 (RTX A5000 / 3090) shared-
memory budgets.

This patch is the missing text-patch half. It rewrites the two launch-
parameter blocks of ``vllm/v1/attention/ops/triton_turboquant_decode.py``
in place at boot from ``VLLM_TQ_DECODE_NUM_WARPS`` /
``VLLM_TQ_DECODE_NUM_STAGES``. The SM-8.6-validated tune is
``num_warps=8, num_stages=3``, but that is a RECOMMENDED OVERRIDE, not
the shipped default: with neither env var set, each branch keeps ITS OWN
upstream literal (``4, 2`` GQA / ``1, 1`` MHA), so the rewrite is inert.
Set the env to actually realise the SM-8.6 tune.

  PER-BRANCH DEFAULTS FIX (2026-07-25). This module previously fed BOTH
  branches the single ``resolve_decode_tune()`` triple, whose no-env
  default is ``(4, 1, 1)`` — the *MHA* literal. With no env set that
  rewrote the GQA branch ``4, 2 -> 1, 1``, i.e. it silently DE-TUNED
  PN119's grouped tensor-core launch to one warp and no pipelining,
  the exact opposite of this patch's purpose and contrary to the
  "inert without env" contract stated above. Harmless on the current
  prod compose only because ``turboquant_3bit_nc`` (3-bit values) never
  reaches the grouped branch; it would have hit any k8v4 arm. Each
  branch now falls back to its own upstream literal and the env
  overrides, when set, still apply to both.

Expected impact (HIGH confidence on the fix actually applying, MEDIUM
on the TPS number): +3-8 % on 35B-A3B-FP8 + TQ k8v4 + MTP K=3. Bench
A/B before promoting from experimental.

Safety:
  - Exact text-anchor match, soft-skip on drift.
  - Per-branch (GQA / MHA) sub-patches, both optional — partial-apply
    is allowed (some pins ship only one branch).
  - Operator override ``GENESIS_DISABLE_P18B_TEXT=1`` keeps upstream
    literals; ``VLLM_TQ_DECODE_NUM_WARPS`` / ``VLLM_TQ_DECODE_NUM_STAGES``
    flow through ``resolve_decode_tune()`` and tune the replacement.
  - Self-suppresses on non-NVIDIA / pre-Ampere via
    ``tq_decode_tune.should_apply()``.
  - Idempotent marker — re-apply is a no-op.

================================================================
2026-07-25 — WHEEL-v2 RE-ANCHOR (content-sniffed, dual-pin safe)
================================================================

P18B_TEXT reported ``no_applicable_sub_patches`` on the wheel-v2 boot
(``dev1474cherrymax-1757-20260725``). Two independent causes, both from
the same KVQ/nuqv squash that broke PN119 and P101:

1. **The chain broke.** Both anchors below live at ``if``-body depth
   (12-space kwargs, ``    else:``) because they key on the launch shape
   PN119 EMITS, not on pristine upstream — PN119 boot-dispatches first and
   splits the single stage-1 launch into a grouped/scalar if/else. On
   wheel v2 PN119's raw diff stopped applying, so the 12-space shape never
   appeared and both P18B anchors necessarily missed.

2. **The MHA anchor drifted on its own.** ``cherry-max`` inserted three
   KVQ constexprs between ``FP8_E4B15=fp8_e4b15,`` and ``num_warps=1,``:

       VALUE_NUQ=value_nuq_flag,
       N_VAL_CENTROIDS=n_val_centroids,
       N_OUTLIERS=n_outliers,

   so even with PN119 restored, the pristine-shaped MHA anchor cannot
   match on wheel v2.

Fix shape (the P89 6edf8386 / P101 92647851 precedent): the KVQ block is
CONTENT-SNIFFED and spliced into BOTH the anchor and the replacement by
:func:`_mha_sub_patch`. The tree is DUAL-PIN — ``dev1060cherry-20260713``
and wheel v1 must keep booting because prod rollback depends on them — so
nothing is re-anchored in place; both shapes ship and the target decides.
Splicing the replacement matters as much as the anchor: this patch
REWRITES the launch, and a replacement built from the pristine literal
would have silently dropped the three KVQ constexprs, disabling the nuqv
value codebook and the KVQ-3 outlier scatter-gather on the live scalar
decode path (the prod compose runs ``turboquant_3bit_nc``, i.e. 3-bit
values, so the scalar branch IS the live one).

``resolve_vllm_file()`` returns a **str**, not a Path — reads go through
``Path(...)`` — and the sniff is wrapped in ``except Exception`` so an
unreadable target degrades to the plain anchors (an ordinary anchor miss)
rather than taking a boot down.

The GQA anchor needs no sniff: PN119's grouped kernel takes no KVQ
arguments (it implements the 4-bit UNIFORM value path only), so
``FP8_E4B15=fp8_e4b15,`` is still followed directly by ``num_warps=4,``
in the emitted grouped launch on every pin.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.engines.vllm.kernels_legacy.tq_decode_tune import (
    get_num_stages_override,
    get_num_warps_override,
    resolve_decode_tune,
    should_apply as tq_should_apply,
)
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.p18b_kernel_literals_textpatch")

GENESIS_P18B_TEXT_MARKER = (
    "Genesis P18b TEXT TurboQuant decode stage1 kernel-literal tune "
    "(SM 8.6 num_warps/num_stages override)"
)

# Upstream launch literals, PER BRANCH. Used when the corresponding
# VLLM_TQ_DECODE_* env var is unset, so the no-env rewrite is byte-inert
# on both branches (see the PER-BRANCH DEFAULTS FIX note in the docstring).
_UPSTREAM_GQA_WARPS, _UPSTREAM_GQA_STAGES = 4, 2
_UPSTREAM_MHA_WARPS, _UPSTREAM_MHA_STAGES = 1, 1


def _branch_tune(default_warps: int, default_stages: int) -> tuple[int, int]:
    """Resolve (num_warps, num_stages) for one branch.

    Env override wins for both; otherwise the branch keeps its own
    upstream literal.
    """
    warps = get_num_warps_override()
    stages = get_num_stages_override()
    return (
        warps if warps is not None else default_warps,
        stages if stages is not None else default_stages,
    )


# GQA path — PN119's grouped launch. The four-line literal block is
# unique in the file. Pin-invariant: the grouped kernel takes no KVQ
# arguments, so FP8_E4B15 is still adjacent to num_warps on every pin.
P18B_GQA_OLD = (
    "            FP8_E4B15=fp8_e4b15,\n"
    "            num_warps=4,\n"
    "            num_stages=2,\n"
    "        )\n"
    "    else:\n"
)

# MHA path — the same launcher, scalar branch of PN119's dispatch. Used
# for kv_group_size==1 AND for every preset the grouped kernel cannot
# serve (3-bit values — which is what the prod compose runs). The
# trailing comment ("# Stage 2:") anchors the replacement uniquely.
#
# CONTENT-SNIFFED (2026-07-25, dual-pin). ``cherry-max`` threads three KVQ
# constexprs between FP8_E4B15 and the launch tune. Byte-verified against
# source extracted from all three pinned images (each + PN14, which is the
# only sibling that edits this file first):
#   plain shape  — dev1060cherry-20260713 == 1, wheel v1 == 1, v2 == 0
#   KVQ shape    — wheel v2 == 1, the other two == 0
_P18B_MHA_HEAD = "            FP8_E4B15=fp8_e4b15,\n"
_P18B_MHA_KVQ = (
    "            VALUE_NUQ=value_nuq_flag,\n"
    "            N_VAL_CENTROIDS=n_val_centroids,\n"
    "            N_OUTLIERS=n_outliers,\n"
)
_P18B_MHA_TAIL = (
    "            num_warps=1,\n"
    "            num_stages=1,\n"
    "        )\n"
    "\n"
    "    # Stage 2:"
)
P18B_MHA_OLD = _P18B_MHA_HEAD + _P18B_MHA_TAIL
P18B_MHA_OLD_KVQ = _P18B_MHA_HEAD + _P18B_MHA_KVQ + _P18B_MHA_TAIL


def _build_replacement(
    num_warps: int, num_stages: int, branch: str, kvq: str = "",
) -> str:
    """Render the new launch-param block with our resolved tune.

    ``branch`` is ``"GQA"`` or ``"MHA"`` — only used in the comment.
    ``kvq`` is the target's KVQ constexpr block (empty on pins that do not
    carry one). It is passed through VERBATIM so the rewrite can never
    drop an argument the target had.
    """
    note = (
        f"            # [Genesis P18b TEXT, 2026-06-08] {branch} launcher\n"
        f"            # tuned for Ampere SM 8.6 (RTX A5000 / 3090).\n"
        f"            # Upstream defaults were H100-shaped (1-4 warps,\n"
        f"            # 1-2 stages) — under-utilised the 100 KB shared\n"
        f"            # / 64 KB L1 budget per SM on consumer Ampere.\n"
        f"            # Override via VLLM_TQ_DECODE_NUM_WARPS /\n"
        f"            # VLLM_TQ_DECODE_NUM_STAGES (tq_decode_tune.py).\n"
    )
    return (
        "            FP8_E4B15=fp8_e4b15,\n"
        + kvq
        + note
        + f"            num_warps={num_warps},\n"
        + f"            num_stages={num_stages},\n"
        + "        )\n"
    )


def _sniff_target(target: str) -> str | None:
    """Read the target for the content sniff. Never raises.

    ``resolve_vllm_file`` hands back a **str**, so the read goes through
    ``Path(...)``. A failure degrades to the plain anchors — i.e. an
    ordinary anchor miss and a soft skip — never a boot-killing exception.
    """
    try:
        return Path(target).read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[P18b TEXT] target sniff failed (%s: %s) — falling back to the "
            "plain (non-KVQ) MHA anchor",
            type(e).__name__, e,
        )
        return None


def _mha_sub_patch(num_warps: int, num_stages: int, kvq: bool) -> TextPatch:
    """Build the MHA sub-patch for a KVQ / non-KVQ target.

    The KVQ constexpr block is spliced into the anchor AND the replacement
    together, or into neither — the P101 lesson: fixing only the anchor
    makes the rewritten launch silently drop VALUE_NUQ / N_VAL_CENTROIDS /
    N_OUTLIERS, and the nuqv + outlier presets stop taking effect on the
    scalar decode path with nothing failing loudly.
    """
    block = _P18B_MHA_KVQ if kvq else ""
    return TextPatch(
        name="p18b_text_mha_launch_tune" + ("_kvq" if kvq else ""),
        anchor=_P18B_MHA_HEAD + block + _P18B_MHA_TAIL,
        replacement=(
            _build_replacement(num_warps, num_stages, "MHA", kvq=block)
            + "\n    # Stage 2:"
        ),
        required=False,
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if target is None:
        return None

    gqa_warps, gqa_stages = _branch_tune(_UPSTREAM_GQA_WARPS, _UPSTREAM_GQA_STAGES)
    mha_warps, mha_stages = _branch_tune(_UPSTREAM_MHA_WARPS, _UPSTREAM_MHA_STAGES)

    gqa_new = _build_replacement(gqa_warps, gqa_stages, "GQA") + "    else:\n"

    # Content sniff: prefer the shape the target actually carries. Both
    # shapes ship (required-at-least-one semantics, the P85 convention) so
    # a failed sniff still finds the right one — it just tries the other
    # first. They are mutually exclusive by construction.
    content = _sniff_target(target)
    kvq_first = bool(content is not None and _P18B_MHA_KVQ in content)
    mha_order = (True, False) if kvq_first else (False, True)

    return TextPatcher(
        patch_name=(
            "P18b TEXT v1/attention/ops/triton_turboquant_decode.py — "
            "kernel-literal tune (num_warps/num_stages SM 8.6)"
        ),
        target_file=str(target),
        marker=GENESIS_P18B_TEXT_MARKER,
        sub_patches=[
            TextPatch(
                name="p18b_text_gqa_launch_tune",
                anchor=P18B_GQA_OLD,
                replacement=gqa_new,
                required=False,
            ),
            *[_mha_sub_patch(mha_warps, mha_stages, kvq) for kvq in mha_order],
        ],
        upstream_drift_markers=["[Genesis P18b TEXT"],
    )


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_P18B_TEXT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply P18b TEXT — text-patch the TQ decode launch literals."""
    if _env_disabled():
        return "skipped", (
            "P18b TEXT disabled via GENESIS_DISABLE_P18B_TEXT=1 — leaving "
            "upstream H100-default kernel launch params"
        )

    if not tq_should_apply():
        return "skipped", (
            "P18b TEXT: TurboQuant not applicable on this device "
            "(non-CUDA or pre-Ampere) — kernel literals not patched"
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            "P18b TEXT: triton_turboquant_decode.py not found in vllm "
            "install — pin may predate TurboQuant or have a different layout"
        )

    bkv, _rw, _rs = resolve_decode_tune()
    gqa_warps, gqa_stages = _branch_tune(_UPSTREAM_GQA_WARPS, _UPSTREAM_GQA_STAGES)
    mha_warps, mha_stages = _branch_tune(_UPSTREAM_MHA_WARPS, _UPSTREAM_MHA_STAGES)
    tune = (
        f"GQA num_warps={gqa_warps} num_stages={gqa_stages} / "
        f"MHA num_warps={mha_warps} num_stages={mha_stages}"
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # never raise out of an apply hook
        log.warning(
            "[P18b TEXT] apply() raised %s — leaving upstream kernel literals",
            e,
        )
        return "skipped", f"P18b TEXT raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", (
            f"P18b TEXT: {reason}{detail}. Resolved tune was "
            f"BLOCK_KV={bkv} {tune}; "
            f"kernel literals NOT overridden — upstream H100 defaults remain."
        )

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"P18b TEXT: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", (
            f"P18b TEXT idempotent: marker already present "
            f"({tune} previously installed)."
        )

    applied = ", ".join(patcher.applied_sub_patches) or "(unknown)"
    return "applied", (
        f"P18b TEXT applied: TQ decode stage1 launch literals set to "
        f"{tune} via sub-patches [{applied}]. Closes the dead-code finding "
        f"(upstream H100 defaults 4/2 + 1/1 were silently in use despite env "
        f"overrides; tq_decode_tune was logging-only). Env-unset = byte-inert."
    )


def is_applied() -> bool:
    """Best-effort check by reading the target file for our marker.

    ``resolve_vllm_file`` returns a **str**, not a Path — the previous
    ``target.read_text(...)`` here raised AttributeError on every call and
    was swallowed as False by the (OSError, UnicodeDecodeError) handler...
    which it is NOT a subclass of, so it actually propagated. Wrapped in
    ``Path(...)`` and a blanket ``except Exception``.
    """
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if target is None:
        return False
    content = _sniff_target(target)
    return content is not None and GENESIS_P18B_TEXT_MARKER in content
