# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN54 (plan v3 P0.7) — GDN contiguous-call deduplication.

Removes redundant `.contiguous()` calls in `gdn_linear_attn.py` that are
already guaranteed contiguous by upstream operator semantics OR re-enforced
by FLA's `@input_guard` decorator on every entry point.

Direction inspired by MLX-LM PR #1077 root-cause analysis (slice-kept-alive
parent retention causes shared-buffer leak in multi-turn). vllm equivalent:
each redundant `.contiguous()` allocates a fresh block; if the "old" block
is held by a still-running async kernel, allocator marks it as fragmented.
Per-turn allocator delta on Cliff 2b (Issue #19): observed +1400 MiB/turn,
~600 MiB attributable to unnecessary copies.

Sub-patches (SAFE-set, high confidence):

**Sub-A — ssm_state advanced-index gather (HIGH IMPACT, prefill hot path)**
Anchor at `gdn_linear_attn.py:984` —
`initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()`
Advanced indexing already produces a FRESH allocation (PyTorch contract);
`.contiguous()` is a no-op copy. Saves one full ssm_state-shape allocation
per prefill batch. Safety: downstream `chunk_gated_delta_rule` is decorated
with FLA `@input_guard` which forces contiguous on every input.

**Sub-B — LoRA branch b/a.contiguous() after chunk (LOW IMPACT, defensive)**
Anchor at `gdn_linear_attn.py:551-552` (LoRA branch, `hasattr(self, 'in_proj_qkv')`).
`chunk(2, dim=-1)` along the last dim returns contiguous halves; the explicit
`.contiguous()` is no-op. Genesis PROD does not use LoRA paths so impact ~0,
but cleanup keeps Genesis-vs-upstream diff smaller.

Deferred (need deeper FLA ABI verification, separate sprint if needed):
- FlashInfer branch (lines 95-100) — 5× squeeze.contiguous(), removal needs
  `chunk_gated_delta_rule_fi` ABI confirmation.
- gqa_interleaved branch — Qwen3-Next path, not Genesis PROD.
- PN50 already replaces Qwen3.5/3.6 contiguous branch (lines 562-575) with
  fused Triton kernel — no separate dedup needed there.

Models affected
---------------
- 27B Lorbus INT4 (TQ k8v4, NGRAM, FP8 short, FP8 long, DFlash) — sub-A fires
  on prefill path. Sub-B no-op (no LoRA).
- 35B (PROD, DFlash) — does NOT have GDN; patch fires neither sub.

Default OFF until live A/B on Cliff 2b multi-turn reproducer (Genesis Issue
#19) shows per-turn allocator delta drops below ~900 MiB.

Author: Sandermage (Sander) Barzov Aleksandr.
Inspiration: adurham (MLX-LM #1077) — slice-kept-alive class of bug.

RETIRED 2026-07-25 — superseded upstream (Sub-A) + premise now false (Sub-B)
===========================================================================

This wiring reported ``skipped: gdn_linear_attn.py not found`` on the
wheel-v2 boot while PN11 text-patched *that exact module* 98 lines later.
The resolver was stale, not the file: ``gdn_linear_attn.py`` was moved to
``model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`` and the flat
path is count==0 on ALL THREE pinned images (dev1060cherry-20260713,
wheel v1 dev1474cherry-1711, wheel v2 dev1474cherrymax-1757) — so PN54
has been "not found" on every pin, not newly broken.

The resolver is fixed below (both paths, tried in order) so the log tells
the truth. But the patch itself is now a no-op, on evidence:

  * **Sub-A** (``pn54_ssm_state``, ``required=True``, the HIGH-IMPACT
    half) is **delivered natively by upstream**. All three images already
    read ``initial_state = ssm_state[prefill_state_indices]`` with **no**
    ``.contiguous()`` — the exact edit PN54 makes, done upstream (the
    index tensor was also renamed from ``non_spec_state_indices_tensor``,
    which is count==0 in the relocated file on every pin). Same job,
    equally well → drop.

  * **Sub-B** (``pn54_lora_ba``, ``required=False``) must NOT be
    re-anchored: its premise is now FALSE. The site moved from the LoRA
    branch to the Qwen3.5 ``else`` branch and now reads
    ``b, a = self.split_ba(ba)``. ``split_ba`` chunks *and then*, when
    ``disable_tp_for_ba_proj and tp_size > 1`` (Marlin replicates
    ``ba_proj`` — our AutoRound-INT4 TP=2 shape), slices
    ``b = b[:, ba_start:ba_start + ba_chunk]``. That slice is a
    NON-contiguous view, so the following ``b.contiguous()`` /
    ``a.contiguous()`` are load-bearing on exactly the config we run.
    Removing them would be a regression, not a dedup.

Lane-2/sndr retired its own copy on 2026-06-11 and disables the id; this
house-lane copy is the last live one. ``apply()`` now returns an explicit
``superseded`` skip (content-sniffed, so it degrades to the historical
behaviour on an image that still carries the pre-move shape) instead of a
misleading "not found". The anchors are retained: they are the evidence
of what was checked, and other patches' conflict docs reference them.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn54_gdn_contiguous_dedup")

GENESIS_PN54_MARKER = "Genesis PN54 GDN contiguous dedup v7.66 (P0.7 Cliff 2b)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-A — ssm_state[idx].contiguous() removal (HIGH IMPACT) ──────────────
SSM_STATE_OLD = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]"
)
SSM_STATE_NEW = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            # [Genesis PN54 P0.7] advanced index already produces fresh\n"
    "            # allocation — .contiguous() was a no-op copy. Downstream\n"
    "            # chunk_gated_delta_rule is wrapped by FLA @input_guard which\n"
    "            # restores contiguous-on-input invariant.\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor]  # type: ignore[index]"
)


# ─── Sub-B — LoRA branch b/a.contiguous() (LOW IMPACT, defensive) ───────────
LORA_BA_OLD = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = ba.chunk(2, dim=-1)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()"
)
LORA_BA_NEW = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = ba.chunk(2, dim=-1)\n"
    "            # [Genesis PN54 P0.7] chunk(dim=-1) returns contiguous halves;\n"
    "            # explicit .contiguous() was no-op copy. FLA @input_guard\n"
    "            # downstream restores invariant if any consumer needs it."
)


# The module moved; try the current home first, then the historical flat path
# so an older image (dual-pin: prod rollback rides dev1060cherry / wheel v1)
# still resolves. Kept LOCAL to PN54 on purpose — the alias table inside
# resolve_vllm_file() is shared, and widening it there would silently wake
# every other patch still pointing at the flat path (P7b, P28, …).
PN54_TARGET_CANDIDATES = (
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "model_executor/layers/mamba/gdn_linear_attn.py",
)

# Upstream's own removal of Sub-A's copy: the advanced-index gather with no
# trailing .contiguous(). Present on all three pinned images.
PN54_SUB_A_UPSTREAM_NATIVE = "initial_state = ssm_state[prefill_state_indices]"


def _resolve_target() -> str | None:
    for rel in PN54_TARGET_CANDIDATES:
        target = resolve_vllm_file(rel)
        if target is not None:
            return str(target)
    return None


def _upstream_superseded(target: str) -> bool:
    """True when upstream already ships Sub-A's edit.

    Content sniff, never a rewrite: resolve_vllm_file() returns a str, so the
    read goes through Path(str(...)), and any failure degrades to False —
    i.e. the historical anchor path — rather than killing the boot.
    """
    from pathlib import Path

    try:
        text = Path(str(target)).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    return PN54_SUB_A_UPSTREAM_NATIVE in text and SSM_STATE_OLD not in text


def _make_patcher() -> TextPatcher | None:
    target = _resolve_target()
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN54 GDN contiguous dedup (P0.7 Cliff 2b)",
        target_file=str(target),
        marker=GENESIS_PN54_MARKER,
        sub_patches=[
            TextPatch(name="pn54_ssm_state", anchor=SSM_STATE_OLD,
                      replacement=SSM_STATE_NEW, required=True),
            TextPatch(name="pn54_lora_ba", anchor=LORA_BA_OLD,
                      replacement=LORA_BA_NEW, required=False),
        ],
        upstream_drift_markers=[
            # Watch for upstream removing these calls or restructuring branch
            "ssm_state[non_spec_state_indices_tensor]",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN54")
    log_decision("PN54", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = _resolve_target()
    if target is None:
        return "skipped", (
            "qwen_gdn_linear_attn.py not found (tried "
            + " and ".join(PN54_TARGET_CANDIDATES)
            + ")"
        )

    if _upstream_superseded(target):
        return "skipped", (
            "PN54 SUPERSEDED — Sub-A is native upstream ("
            "`initial_state = ssm_state[prefill_state_indices]`, no "
            ".contiguous()) and Sub-B must not be re-anchored: the site now "
            "reads `b, a = self.split_ba(ba)`, and split_ba TP-slices b/a "
            "when disable_tp_for_ba_proj and tp_size > 1, so the trailing "
            ".contiguous() is load-bearing on our Marlin TP=2 shape. "
            "See the module docstring for the per-pin evidence."
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN54 applied: redundant .contiguous() calls removed in GDN "
            "(ssm_state advanced-index + LoRA chunk halves); reduces per-turn "
            "allocator fragmentation on Cliff 2b multi-turn",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — anchor drifted or upstream removed call"
    return "failed", failure.reason if failure else "unknown failure"
