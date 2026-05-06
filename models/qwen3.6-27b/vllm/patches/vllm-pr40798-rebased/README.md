# vLLM PR #40798 overlay — TurboQuant max-workspace pre-allocation

Vendored 2026-05-11 to unblock `turboquant_3bit_nc` KV + MTP on Qwen 3.6 27B
(`dual/int8-tq3.yml`).

## Source

- Upstream PR: <https://github.com/vllm-project/vllm/pull/40798>
- Title: `[TurboQuant] Share decode scratch workspace across layers`
- Head SHA: `cbe823e9b5e32b3590705f9ee86ab57db887754d`
- State at vendor time: OPEN, MERGEABLE.
- 3 source files: `turboquant_attn.py` (−1), `triton_turboquant_decode.py`
  (+15/−8), `gpu_model_runner.py` (+57/−1).

## What it fixes

Without this PR, the MTP drafter's `_decode_attention` and the long-context
`_continuation_prefill` paths request workspace allocations AFTER the
cudagraph capture has locked it (per vllm#39226). On vLLM nightly `1acd67a7`
that surfaces as:

```
AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:747:_continuation_prefill' requires 8.06 MB,
current size is 0.76 MB. Workspace growth is not allowed after locking.
```

Bug tracked at vllm-issue#41565 / #41726 / #40420.

This PR moves TurboQuant decode scratch allocation into the v1 workspace
manager so scratch tensors are shared across layers, AND reserves the
maximum TurboQuant decode workspace before CUDA graph capture locks the
workspace. Both the decode and continuation-prefill paths get the right
workspace size up front, so the strict lock invariant from vllm#39226 holds.

## Why not Genesis

The Sandermage Genesis package's PN34 solves the same problem (and many
others) via a separate draft-model workspace pool. We've intentionally
kept this compose Genesis-free to validate the upstream fix in isolation.

## Why not PR #42215

PR #42215 (Warm up decode kernels) attacks the same bug from a different
angle: it warms up the decode kernel during `kernel_warmup()` so the
workspace high-water mark is captured before lock. We tried that overlay
first on 2026-05-11 and confirmed it fixes the decode path (line 879)
but NOT the continuation-prefill path (line 747). PR #40798 reserves the
max workspace across all TQ paths, which covers both.

## Drop trigger

```
gh api repos/vllm-project/vllm/pulls/40798 --jq '.state, .merged_at'
```

reports `MERGED`. Then bump the nightly pin past the merge commit.

## Verified on

- vLLM nightly: `1acd67a7`
- Compose: `dual/int8-tq3.yml` (Qwen 3.6 27B AutoRound INT4, TQ3 KV, MTP n=3, 262K, 2 streams)
- 2026-05-11
