# vLLM PR #40108 overlay — TurboQuant + sliding-window/YOCO

Vendored 2026-05-11 to evaluate `turboquant_3bit_nc` KV on Gemma 4 31B
(`dual/int8-tq3.yml`).

**OUTCOME 2026-05-11:** Confirmed Gemma 4 + TQ3 is hardware-blocked on
Ampere SM 8.6 (RTX 3090) — `flash_attn_varlen_func` in TQ backend's
prefill path rejects head_dim=512 (Gemma 4 global layers). FA2 on Ampere
caps at head_dim=256; FA3 (which supports 512) requires Hopper. See the
full walkthrough in `dual/int8-tq3.yml` header.

This overlay + the local edits we made (boundary-skip patch, cudagraph
capture guards on .tolist() sites) WORK correctly — they're not why TQ3
is blocked. They're retained because they're the right code if/when the
hardware/upstream constraint disappears.

## Source

- Upstream PR: <https://github.com/vllm-project/vllm/pull/40108>
- Head SHA: `68afe6249a36786ea666b78ae009ad775d86965c`
- Branch: `feature/turboquant-yoco-sliding-window`
- State at vendor time: OPEN, last updated 2026-05-07.
- Adds: PyTorch sliding-window mask + Triton windowed decode kernel +
  unified KV cache page-size logic for TQ backend on YOCO/sliding-window
  models (Gemma 4 E4B / 31B).

## Why we need this

Without #40108, the TurboQuant backend rejects `turboquant_3bit_nc`
kv_cache_dtype on Gemma 4 with:

```
ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not valid
            for this configuration. Reason: ['kv_cache_dtype not supported']
```

This blocks the TQ3 head-to-head against `qwen3.6-27b/vllm/compose/dual/int8-tq3.yml`.

## Conflict with PR #40391 overlay

PR #40108 and PR #40391 (per-token-head fp8 KV for Gemma 4) both touch:

- `vllm/v1/core/kv_cache_utils.py`
- `vllm/model_executor/layers/attention/attention.py`

`int8-tq3.yml` mounts the #40108 overlay (this dir) and drops the
#40391 overlay, because TQ3 doesn't use per-token-head KV.
The INT8 PTH path (`int8.yml`) continues to mount the #40391 overlay.

## Drop this overlay when

```
gh api repos/vllm-project/vllm/pulls/40108 --jq '.state, .merged_at'
```

reports `MERGED`. After merge, just bump the nightly pin.
