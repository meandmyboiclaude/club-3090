# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN50 — SGLang #21019 GDN projection fusion backport.

Replaces the unfused split/reshape/cat/.contiguous() chain in the
Qwen3.5/3.6 contiguous projection branch of `gdn_linear_attn.py`
with the Genesis-ported Triton kernel `pn50_gdn_fused_proj`.

Affects only the `gqa_interleaved_layout=False` branch (Qwen3.5/3.6
contiguous-loaded weights). Qwen3-Next (interleaved layout) and the LoRA
path (`hasattr(in_proj_qkv)`) are unaffected.

Anchor stability
----------------
Anchor is the entire 9-line `else:` block of the Qwen3.5 branch in
`mamba/gdn_linear_attn.py`. Verified against pristine upstream + live
container — both match (see test_pn50_*.py).

Models affected (per Genesis 7-config matrix):
  * 27B Lorbus INT4 (TQ k8v4, FP8 short, FP8 long, NGRAM, DFlash) — APPLIES
  * 35B FP8 (PROD, DFlash) — DOES NOT APPLY (Qwen3MoE has no GDN layers)

Default OFF until live A/B prod-validates +TPS gain on at least one
27B config without numerical regression.

Author: Sandermage (Sander) Barzov Aleksandr backport.
Original Triton kernel: Yuan Luo (@yuan-luo), SGLang PR #21019.
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

log = logging.getLogger("genesis.wiring.pn50_gdn_fused_proj")

GENESIS_PN50_MARKER = "Genesis PN50 GDN fused proj v7.66 (SGLang#21019 backport)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# VERDICT 2026-07-25 — the "anchor not found — soft skip" line PN50 emits for
# `pn50_gdn_fused_proj` is CORRECT, not a degraded apply. The two sub-patches
# below are MUTUALLY EXCLUSIVE by construction (see the required=False note in
# _make_patcher): ANCHOR_OLD is the pre-vllm#41126 nested `ba.chunk(2, dim=-1)`
# shape, ANCHOR_OLD_DEV1060 the post-split de-nested `self.split_ba(ba)` shape.
# Counted against mamba/gdn/qwen_gdn_linear_attn.py extracted from all three
# pinned images (dev1060cherry-20260713, wheel v1 dev1474cherry-1711, wheel v2
# dev1474cherrymax-1757): ANCHOR_OLD count==0 and ANCHOR_OLD_DEV1060 count==1
# on every one. The dev1060 variant IS the whole patch on every pin we can
# boot; ANCHOR_OLD is retained purely for dual-pin safety against an older
# image and must NOT be "re-derived" — deriving a second anchor that also
# matched the current shape would double-apply the fusion.
#
# The separate lane-2 line ("both ENABLE and DISABLE env flags set ... DISABLE
# wins") is likewise by design: PN50 is a SHARED id, so patches/sndr_lane.py
# policy step 1 injects GENESIS_DISABLE_<bare> for lane-2 to keep THIS
# house-lane wiring authoritative. The log's "disabled by operator" wording is
# what makes it look like a defect; no compose sets that DISABLE var.

# Pristine upstream anchor — Qwen3.5/3.6 contiguous-projection branch
ANCHOR_OLD = (
    "            else:\n"
    "                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "                z_size = self.value_dim // self.tp_size\n"
    "                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "                z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "                b, a = ba.chunk(2, dim=-1)\n"
    "                b = b.contiguous()\n"
    "                a = a.contiguous()"
)

ANCHOR_NEW = (
    "            else:\n"
    "                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "                # [Genesis PN50 SGLang#21019] fused Triton kernel for\n"
    "                # split/reshape/cat/.contiguous(); replaces 5-6 launches +\n"
    "                # 2 explicit copies. Wrapper falls through to original\n"
    "                # PyTorch chain on any constraint violation (non-contig,\n"
    "                # non-pow2 head_dim, kernel failure, etc.) — strict no-regression.\n"
    "                from vllm._genesis.kernels.pn50_gdn_fused_proj import (\n"
    "                    fused_qkvzba_split_reshape_cat_contiguous as _pn50_fused,\n"
    "                )\n"
    "                _pn50_num_heads_qk = (self.key_dim // self.head_k_dim) // self.tp_size\n"
    "                _pn50_num_heads_v = (self.value_dim // self.head_v_dim) // self.tp_size\n"
    "                mixed_qkv, z, b, a = _pn50_fused(\n"
    "                    mixed_qkvz, ba,\n"
    "                    num_heads_qk=_pn50_num_heads_qk,\n"
    "                    num_heads_v=_pn50_num_heads_v,\n"
    "                    head_qk=self.head_k_dim,\n"
    "                    head_v=self.head_v_dim,\n"
    "                )"
)

# dev1060 anchor (added 2026-07-13) — 0.23.1rc1.dev1060+g9e57de719
# (club-dev1060-cherry; source builds report 0.1.dev1+g3da1671fc.d20260713).
# Two drifts vs ANCHOR_OLD: (a) forward() de-nested one level (8/12-space
# indentation, not 12/16), (b) `b, a = ba.chunk(2, dim=-1)` became
# `b, a = self.split_ba(ba)` via the vllm#41126 mamba/gdn/ split. The file
# is now mamba/gdn/qwen_gdn_linear_attn.py. forward_cpu() carries the same
# comment line but keeps chunk(2, ...) and lacks the trailing .contiguous()
# pair, so this 9-line block stays unique to forward() (verified count==1
# against the dev1060 checkout on 2026-07-13).
#
# torch.compile note: on >=0.23 the graph splits at the
# vllm::qwen_gdn_attention_core custom op, so this projection block sits
# INSIDE a piecewise-compiled subgraph. The fused wrapper's host-side
# fallback branches + try/except around the Triton launch may graph-break
# under fullgraph dynamo. Original patch ships no eager-only gate (it is
# default-OFF, opt-in) — unchanged here; A/B under the live compile config
# before trusting it on dev1060.
ANCHOR_OLD_DEV1060 = (
    "        else:\n"
    "            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "            z_size = self.value_dim // self.tp_size\n"
    "            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = self.split_ba(ba)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()"
)

ANCHOR_NEW_DEV1060 = (
    "        else:\n"
    "            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "            # [Genesis PN50 SGLang#21019] fused Triton kernel for\n"
    "            # split/reshape/cat/.contiguous(); replaces 5-6 launches +\n"
    "            # 2 explicit copies. Wrapper falls through to original\n"
    "            # PyTorch chain on any constraint violation (non-contig,\n"
    "            # non-pow2 head_dim, kernel failure, etc.) — strict no-regression.\n"
    "            from vllm._genesis.kernels.pn50_gdn_fused_proj import (\n"
    "                fused_qkvzba_split_reshape_cat_contiguous as _pn50_fused,\n"
    "            )\n"
    "            _pn50_num_heads_qk = (self.key_dim // self.head_k_dim) // self.tp_size\n"
    "            _pn50_num_heads_v = (self.value_dim // self.head_v_dim) // self.tp_size\n"
    "            mixed_qkv, z, b, a = _pn50_fused(\n"
    "                mixed_qkvz, ba,\n"
    "                num_heads_qk=_pn50_num_heads_qk,\n"
    "                num_heads_v=_pn50_num_heads_v,\n"
    "                head_qk=self.head_k_dim,\n"
    "                head_v=self.head_v_dim,\n"
    "            )"
)


def _make_patcher() -> TextPatcher | None:
    # 2026-07-13 dev1060 re-anchor: vllm#41126 split the file to
    # mamba/gdn/qwen_gdn_linear_attn.py — the old path is gone on
    # 0.23.1rc1.dev1060+g9e57de719 (club-dev1060-cherry) and on source
    # builds reporting 0.1.dev1+g3da1671fc.d20260713. Same fallback
    # pattern as the sndr lane (K.1.R.R.4).
    target = (
        resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
        or resolve_vllm_file(
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        )
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN50 GDN fused proj (SGLang#21019)",
        target_file=str(target),
        marker=GENESIS_PN50_MARKER,
        sub_patches=[
            # 2026-07-13: required=False so the historical anchor's absence
            # on dev1060-shaped trees soft-skips and lets the _dev1060
            # sibling apply. Exactly one of the two matches any given tree;
            # if neither matches, TextPatcher still returns SKIPPED
            # (no_applicable_sub_patches) — same net behavior as before.
            TextPatch(
                name="pn50_gdn_fused_proj",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=False,
            ),
            TextPatch(
                name="pn50_gdn_fused_proj_dev1060",
                anchor=ANCHOR_OLD_DEV1060,
                replacement=ANCHOR_NEW_DEV1060,
                required=False,
            ),
        ],
        upstream_drift_markers=[
            # Watch for upstream merging an equivalent fusion (vllm PR pending?)
            "fused_qkvzba_split_reshape_cat_contiguous",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN50")
    log_decision("PN50", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN50 applied: GDN proj fusion active in Qwen3.5/3.6 contiguous "
            "branch; wrapper falls through to PyTorch on constraint violation",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return (
            "skipped",
            f"{msg} — likely upstream merged an equivalent fusion or "
            "anchor drifted (check gdn_linear_attn.py Qwen3.5 branch)",
        )
    return "failed", failure.reason if failure else "unknown failure"
