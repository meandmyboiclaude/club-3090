# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N17 — FA2 softmax_lse runtime clamp (Cliff 1 mechanism A).

================================================================
WHAT THIS PATCH DOES
================================================================

Replaces the `max_seqlen_k = attn_metadata.max_seq_len` assignment in
`vllm/v1/attention/backends/flash_attn.py` with a runtime clamp to the
actual max-per-sequence value (computed from `seqused_k`), but ONLY
when CUDA stream is NOT capturing a graph. During cudagraph capture,
falls back to `attn_metadata.max_seq_len` for shape stability (the
upstream behavior).

================================================================
ROOT CAUSE
================================================================

FA2's `flash_attn_varlen_func` allocates an internal `softmax_lse`
buffer of shape `[num_seqs, num_heads, max_seqlen_k]` — sized by the
`max_seqlen_k` argument the caller passes, NOT by the actual sequence
lengths in `cu_seqlens_k` / `seqused_k`. Reference: Dao-AILab/
flash-attention#1011 (open since 2024).

vLLM's `gpu_model_runner.py` sets `attn_metadata.max_seq_len =
self.max_model_len` during cudagraph capture for shape stability
(also confirmed by upstream PR vllm#40961 for SWA models). This
choice leaks into runtime decode/prefill: at `--max-model-len=205000`
even a 25K-token chunk reserves softmax_lse for 205K tokens →
unnecessary 50-100 MiB allocation.

Empirical Cliff 1 mechanism A (noonghunna 2026-04-29 cross-rig on
RTX 3090):

  205K + 0.98 + TQ3 + no-vision: FA2 softmax_lse OOM, 50 MiB / 50 MiB free
  identical-prefill on 48K + 0.92: passes cleanly

Closing this mechanism widens the safe envelope for `long-text`
no-vision configs to ~205K. (The dual mechanism B — FFN intermediate
buffer cliff at 138 MiB on `long-vision` configs — is OUT OF SCOPE
for this patch and requires upstream-FFN changes; see Genesis
[Issue #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11)
discussion thread for full analysis.)

================================================================
UPSTREAM ABSORPTION — MRV2 path (vllm#43991, merged 2026-06-02)
================================================================

⚠ As of the 2026-06-07 nightly, vLLM **Model Runner V2** now sets
`attn_metadata.max_seq_len` to the ACTUAL batch max at runtime and to
`max_model_len` only during CUDA-graph capture — the exact clamp this
patch performs, but done one level up at metadata construction:

  - `v1/worker/gpu/model_states/mamba_hybrid.py:91-95` (our GDN hybrid
    path) — `for_capture` → `max_model_len`, else
    `seq_lens_cpu_upper_bound[:num_reqs].max().item()`.
  - `v1/worker/gpu/model_states/default.py:177-180` (dense MRV2 path) —
    identical pattern.
  - `v1/worker/gpu_model_runner.py:2226-2228` (legacy V1 runner) —
    `is_capturing` → `max_model_len`, else
    `optimistic_seq_lens_cpu[:num_reqs].max()`.

  Introduced by PR #40654, extended to the V2 hybrid + MTP-draft paths
  by **PR #43991** ("[Model Runner V2] Use actual batch max_seq_len for
  attn metadata", fanghao566, merged 2026-06-02). The OLD running build
  (aa2b56ffb) still had `mamba_hybrid.py` passing
  `max_seq_len=self.max_model_len` UNCONDITIONALLY (old line 132) — so
  on the OLD hybrid path PN17 had a real effect; on the NEW hybrid path
  `attn_metadata.max_seq_len` is already the actual batch max.

  CONSEQUENCE for our hybrid GDN + MTP deployment — PARTIAL absorption:
    * MAIN forward (hybrid prefill/decode): #43991 already feeds the
      actual batch max into `attn_metadata.max_seq_len`, so PN17's
      eager-mode clamp reduces `max_seqlen_k` to a value that equals
      (and never beats) it — a harmless NO-OP here. The defensive upper
      bound (`if max_seqlen_k > attn_metadata.max_seq_len`) makes it
      exactly inert on this path.
    * MTP DRAFT forward (n=3, the path we exercise every step): #43991
      did NOT cover the draft builder. As of 2026-06-07,
      `v1/worker/gpu/spec_decode/speculator.py` sets
      `self.draft_max_seq_len = self.max_model_len` once in __init__
      (line 84) and `_build_draft_attn_metadata` passes it UNCONDITION-
      ALLY (line 199) — no runtime clamp, no capture gate. So the draft
      FA2 calls still bloat `max_seq_len` to max_model_len, and PN17's
      clamp is STILL a real saver on the draft path. (The PR #43991 body
      claims it touched `_build_draft_attn_metadata`, but the merged
      nightly draft builder still uses the unconditional value — the
      eagle/MTP coverage referenced there did not reach this builder.)

  NET: PN17 is NOT obsolete. It is redundant on the main hybrid forward
  (absorbed by #43991) but remains beneficial on the MTP draft path and
  on the legacy V1 runner edge cases. Kept in place (anchor still valid,
  default OFF). The `max_seqlen_k = seq_lens...max()` drift markers below
  now also catch the upstream clamp landing in flash_attn.py directly,
  should a future PR ever absorb the FA2-backend-level clamp too.

================================================================
SAFETY MODEL
================================================================

- Cudagraph guard: only clamps in eager mode
  (`torch.cuda.is_current_stream_capturing()` returns False). During
  capture, behavior is identical to upstream (max_model_len padding).
  This preserves cudagraph shape stability.

- Per-rank guard: `seqused_k` is a tensor; `.max()` is a single
  GPU→CPU sync that fires once per FA2 call. Cost is one int read,
  amortized across the kernel work. Probed cost on Ampere: ~3-5 us
  per call → noise relative to FA2 kernel runtime (~ms).

- Idempotent via marker; drift detection on the upstream anchor.

- Default OFF; opt-in via `GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1`.
  Recommend enabling on `long-text-no-vision.yml` configs only;
  for `long-vision.yml` the FFN cliff dominates and PN17 is no-op.

================================================================
ANCHOR / REPLACEMENT
================================================================

The anchor block is the variable assignment lines just before the
`flash_attn_varlen_func` call in the non-cascade path of
`FlashAttentionImpl.forward`:

    if not attn_metadata.use_cascade:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len   ← THIS

We replace the last line with a conditional that consults
`seqused_k.max().item()` outside cudagraph capture.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Diagnosis credit: noonghunna (cross-rig RTX 3090, Genesis Issue #11
follow-up 2026-04-29).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn17_fa2_softmax_lse_clamp")


GENESIS_PN17_MARKER = "Genesis PN17 FA2 softmax_lse runtime clamp v1"


# Drift markers: if upstream changes the anchor block (e.g. variable
# rename, refactored cascade gate), our text-patch won't apply
# silently in the wrong place.
UPSTREAM_DRIFT_MARKERS = [
    GENESIS_PN17_MARKER,
    # If upstream natively clamps softmax_lse in this file:
    "max_seqlen_k = int(seqused_k.max",
    # vllm#43991/#40654 absorbed the equivalent clamp at metadata
    # construction (mamba_hybrid.py / default.py / gpu_model_runner.py),
    # NOT in this file — so the assignments below would only appear here
    # if a future PR also clamps in the FA2 backend. If they do, the
    # patcher reports "upstream may have absorbed this fix" and self-skips.
    "max_seqlen_k = seqused_k[:num_reqs].max",
    "max_seqlen_k = attn_metadata.seq_lens[:num_reqs].max",
    # If upstream issue Dao-AILab/flash-attention#1011 lands a fix:
    "softmax_lse_clamped",
]


# Anchor: the 4-line block of attn-metadata reads just before the
# flash_attn_varlen_func call. Sized to be unique within the file.
PN17_OLD = (
    "        if not attn_metadata.use_cascade:\n"
    "            cu_seqlens_q = attn_metadata.query_start_loc\n"
    "            seqused_k = attn_metadata.seq_lens\n"
    "            max_seqlen_q = attn_metadata.max_query_len\n"
    "            max_seqlen_k = attn_metadata.max_seq_len\n"
)


PN17_NEW = (
    "        if not attn_metadata.use_cascade:\n"
    "            cu_seqlens_q = attn_metadata.query_start_loc\n"
    "            seqused_k = attn_metadata.seq_lens\n"
    "            max_seqlen_q = attn_metadata.max_query_len\n"
    "            # [Genesis PN17 FA2 softmax_lse runtime clamp v1]\n"
    "            # FA2 varlen allocates softmax_lse[num_seqs, heads, max_seqlen_k]\n"
    "            # — sized by THIS arg, not by actual seqused_k. Upstream sets\n"
    "            # attn_metadata.max_seq_len = max_model_len during cudagraph\n"
    "            # capture for shape stability; that value leaks into runtime\n"
    "            # decode/prefill, causing 50-100 MiB over-allocation at long\n"
    "            # context (Cliff 1 mechanism A; ref Genesis Issue #11). Eager-\n"
    "            # mode runtime: clamp to actual chunk max from seqused_k.\n"
    "            # Capture mode: keep max_model_len for shape stability.\n"
    "            import torch as _genesis_pn17_torch\n"
    "            try:\n"
    "                _genesis_pn17_capturing = (\n"
    "                    _genesis_pn17_torch.cuda.is_available()\n"
    "                    and _genesis_pn17_torch.cuda.is_current_stream_capturing()\n"
    "                )\n"
    "            except Exception:\n"
    "                _genesis_pn17_capturing = False\n"
    "            if _genesis_pn17_capturing:\n"
    "                max_seqlen_k = attn_metadata.max_seq_len\n"
    "            else:\n"
    "                try:\n"
    "                    max_seqlen_k = int(seqused_k.max().item())\n"
    "                    # Defensive lower bound: should not exceed upstream's\n"
    "                    # max_model_len cap regardless of metadata corruption.\n"
    "                    if max_seqlen_k > attn_metadata.max_seq_len:\n"
    "                        max_seqlen_k = attn_metadata.max_seq_len\n"
    "                except Exception:\n"
    "                    max_seqlen_k = attn_metadata.max_seq_len\n"
)


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/flash_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN17 FA2 softmax_lse runtime clamp",
        target_file=target,
        marker=GENESIS_PN17_MARKER,
        sub_patches=[
            TextPatch(
                name="pn17_clamp",
                anchor=PN17_OLD,
                replacement=PN17_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN17. Default OFF. Opt-in via env flag.

    See module docstring for safety model + per-config recommendation:
    enable on long-text-no-vision configs (closes FA2 softmax_lse
    cliff). No-op on long-vision configs (FFN buffer dominates;
    out-of-scope upstream-FFN problem per Issue #11 dual-mechanism
    analysis).
    """
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP not set; default OFF. "
            "Enable on long-text-no-vision configs to close Cliff 1 "
            "mechanism A (FA2 softmax_lse over-allocation at long ctx). "
            "Diagnosis credit: noonghunna, Genesis Issue #11."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _patcher()
    if p is None:
        return "skipped", "v1/attention/backends/flash_attn.py not found"

    result, failure = p.apply()
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(result, failure, applied_message='PN17 applied: FA2 softmax_lse buffer now clamped to actual seqused_k at runtime, freeing 50-100 MiB on long-ctx (Cliff 1 mechanism A fix per noonghunna Genesis Issue #11).', patch_name='PN17 FA2 softmax_lse runtime clamp')


def is_applied() -> bool:
    """Reporter for verify_live_rebinds in apply_all.py."""
    if vllm_install_root() is None:
        return False
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN17_MARKER in f.read()
    except Exception:
        return False
