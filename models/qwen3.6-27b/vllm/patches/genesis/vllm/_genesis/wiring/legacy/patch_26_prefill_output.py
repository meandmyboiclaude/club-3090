# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 26 — TurboQuant `_prefill_attention` output prealloc.

Problem
-------
`TurboQuantAttentionImpl._prefill_attention` at `turboquant_attn.py:566`
does `output = torch.zeros(N, Hq, D, ...)` on every prefill call. That's
~32 MiB of zero-fill per call on Qwen3.6-35B-A3B (max_num_batched_tokens
= 4096, Hq = 32, D = 128) × 10 attention layers per step — a measurable
~1-2 % decode TGS hit on long-context, AND the allocation is invisible to
vLLM's memory profiler (same root cause class as #40420).

A second lazy allocation sits on line 575: `_cu_2 = torch.zeros(2, ...)`
is a fresh int32[2] tensor per prefill call — host→device transfer that
we'd rather do once.

Fix
---
Replace both with shared pool helpers:
  - Line 566 → `TurboQuantBufferManager.acquire_prefill_output(N, Hq, D, ...)`
  - Line 575 → `TurboQuantBufferManager.acquire_cu_2(device)`

Both helpers are pointer-stable on re-call (CUDA-graph safe) and fall
back to a fresh allocation if platform-incompatible or budget-exceeded
(correctness-over-speed safety net).

Platform guard: shared with P22 (NVIDIA CUDA + SM ≥ 8.0). Non-NVIDIA
reaches the fallback inside `acquire_*` and behaves identically to the
baseline — zero behavioural regression.

Upstream drift detection: if `acquire_prefill_output` appears in the file,
assume we (or upstream) already applied an equivalent and skip.

Partial absorption (2026-07-26 ledger pass)
-------------------------------------------
Exactly ONE of the two hunks was absorbed. On the boot pin
dev1474cherrymax-1757-20260725 (and its pristine image), counted in
`v1/attention/backends/turboquant_attn.py`:

    output = torch.zeros(N, Hq, D, ...)          count == 1   (NOT absorbed)
    _cu_2 = torch.zeros(2, ...)  [P26 form]      count == 0   (absorbed)
    if not hasattr(self, "_cu_2")                count == 1   (the absorption)

`if not hasattr(self, "_cu_2")` was listed as a whole-patch drift marker, so
the presence of the absorbed HALF retired the patch entirely and the boot
logged "patch obsolete, skip" — while the per-prefill 32 MiB zero-fill that
is P26's actual point stayed in the file. That is a drift marker scoped to
the wrong unit: it is evidence about sub-patch `p26_cu_2_alloc`, not about
`p26_output_alloc`.

It is therefore no longer a patch-level drift marker. `p26_cu_2_alloc` is
now `required=False`, so it soft-skips on its own absorbed anchor while the
output pool still applies. Only real evidence about the OUTPUT pool
(`acquire_prefill_output`, `_tq_prefill_output_slice`) retires the patch.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p26_prefill_output")

GENESIS_P26_MARKER = "Genesis P26 TQ prefill output prealloc v7.0"

# Patch-level drift markers ONLY. Each must be evidence that the PREFILL
# OUTPUT POOL — the hunk P26 exists for — is already in the file. A marker
# that only proves the cu_2 half landed belongs on that sub-patch, not here
# (see "Partial absorption" in the module docstring).
UPSTREAM_DRIFT_MARKERS = [
    "acquire_prefill_output",
    "_tq_prefill_output_slice",
]

# Sub-patch-scoped evidence: upstream nightly 7923b48047be wraps
# `_cu_2 = torch.zeros(...)` in `if not hasattr(self, "_cu_2"):`, which IS
# P26's cu_2 optimization natively. Kept as a named constant so the fact is
# not lost, and asserted against below — but it retires one sub-patch, not
# the patch.
CU2_ABSORBED_MARKER = 'if not hasattr(self, "_cu_2")'


# Anchor 1: line 566 of turboquant_attn.py — the fresh output tensor.
_OLD_OUTPUT = (
    "        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)"
)

_NEW_OUTPUT = (
    "        # [Genesis P26] Shared, profiler-visible prefill output pool.\n"
    "        # First call reserves max_num_batched_tokens × Hq × D (picked up\n"
    "        # by profile_run warmup); subsequent calls return a zeroed slice.\n"
    "        from vllm._genesis.kernels.dequant_buffer import (\n"
    "            TurboQuantBufferManager as _GenesisTQBuf,\n"
    "        )\n"
    "        output = _GenesisTQBuf.acquire_prefill_output(\n"
    "            num_tokens=N,\n"
    "            num_q_heads=Hq,\n"
    "            head_size=D,\n"
    "            device=query.device,\n"
    "            dtype=query.dtype,\n"
    "            max_batched_tokens=getattr(self, '_max_num_batched_tokens', None),\n"
    "        )"
)


# Anchor 2: line 575 — cu_seqlens scratch.
_OLD_CU2 = (
    "        # Pre-allocate cu_seqlens for single-request flash_attn calls\n"
    "        # to avoid per-request host→device tensor creation.\n"
    "        _cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)"
)

_NEW_CU2 = (
    "        # Pre-allocate cu_seqlens for single-request flash_attn calls\n"
    "        # to avoid per-request host→device tensor creation.\n"
    "        # [Genesis P26/P32] Reuse the shared cu_2 scratch (pointer-stable,\n"
    "        # CUDA-graph safe); caller writes _cu_2[1] = q_len in-place.\n"
    "        _cu_2 = _GenesisTQBuf.acquire_cu_2(query.device)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P26 TQ prefill output prealloc",
        target_file=target,
        marker=GENESIS_P26_MARKER,
        sub_patches=[
            TextPatch(
                name="p26_output_alloc",
                anchor=_OLD_OUTPUT,
                replacement=_NEW_OUTPUT,
                required=True,
            ),
            # NOT required: upstream absorbed this hunk (CU2_ABSORBED_MARKER)
            # while leaving the output pool alone. A soft skip here lets the
            # output hunk land; `required=True` used to abort the whole patch.
            # It can only ever apply when p26_output_alloc applied too — that
            # one is required and its replacement is where `_GenesisTQBuf` is
            # imported into the function scope this hunk reads it from.
            TextPatch(
                name="p26_cu_2_alloc",
                anchor=_OLD_CU2,
                replacement=_NEW_CU2,
                required=False,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P26 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        # Report the hunks that were actually WRITTEN, read back off the file.
        # "prefill output + cu_2 rewired" was printed unconditionally and is
        # wrong on every pin that already carries the upstream cu_2 guard.
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                after = f.read()
        except OSError:
            after = ""
        cu2_landed = "acquire_cu_2(query.device)" in after
        detail = "prefill output pool rewired through the shared TQ buffer"
        if cu2_landed:
            detail += "; cu_2 scratch rewired too"
        else:
            detail += (
                f"; cu_2 hunk soft-skipped — upstream already caches it "
                f"({CU2_ABSORBED_MARKER!r} present), absorbed half"
            )
        return "applied", detail
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
