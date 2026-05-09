# `fixes/` — local sidecar patches not (yet) in upstream Genesis

This directory holds patches we authored for Qwen3.6-27B-AutoRound-INT4 + vLLM
that **upstream Genesis does not currently provide**. Kept separate from
`models/qwen3.6-27b/vllm/patches/` (which tracks Sandermage's Genesis tree)
so that Genesis pulls don't smash them.

History note: the c6e6163 release deleted 6 prior local sidecars because
Genesis natives superseded them (PN30, PN25, PN34, PN35, P78). The 3
remaining sidecars below have **no Genesis equivalent** as of v7.72.5.

## Contents

### `patch_drafter_skip.py`

**Problem.** `vllm/model_executor/models/qwen3_5_mtp.py` unconditionally
allocates a `ParallelLMHead` (≈2.37 GiB transient peak in float16) for the
MTP drafter. Seconds later `eagle.py:_maybe_share_lm_head` replaces it with
the target model's `lm_head` via weight-sharing, so the alloc was never
needed. The transient peak OOMs on 24 GB cards at `gpu-memory-utilization
>= 0.95`.

**Fix.** Text-patch swaps the `ParallelLMHead(...)` call site with a
`PPMissingLayer()` placeholder. `_maybe_share_lm_head` then assigns the
shared head onto the placeholder slot — no transient peak, identical
runtime semantics.

**Where it applies.** Any compose variant that uses MTP spec-decode
(`--speculative-config '{"method":"mtp",...}'`). Idempotent — no-op if the
target source string is already replaced.

### `patch_vision_tower_skip.py`

**Problem.** `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditional
Generation` instantiate `Qwen3_VisionTransformer` (≈0.86 GiB) at construction
time **even when `--language-model-only` is set**. The runtime gate skips
calling the ViT, but the alloc has already happened.

**Fix.** Text-patch wraps the `Qwen3_VisionTransformer(...)` call site with
a `language_model_only=True` check; when True, replaces with
`PPMissingLayer()`. Reclaims the 0.86 GiB for KV cache.

**Where it applies.** Any compose variant that passes `--language-model-only`
(text-only variants — `tools-text`, `long-text`, `bounded-thinking`,
`dual-turbo`, default `docker-compose.yml`). Should NOT be applied to
vision variants (`long-vision.yml`); guarded by the `language_model_only=True`
runtime check, so applying it there is safe but pointless.

### `cliff2b/`

**Problem.** Cliff 2b OOM on single-card 24 GB rigs at long context
(>50 K tokens). Genesis P103 chunks the GDN forward pass to mitigate but
still allocates a 2.74 GiB transient `h_new_empty(B, NT, H, V, K)` buffer
where `NT = ceil(seq_len / chunk_size)` — grows with prompt length. P103
trades latency for VRAM headroom; the transient still spikes.

**Fix.** Replace the unfused chunked GDN forward (`chunk_gated_delta_rule_
fwd_h.h.new_empty(...)`) with a fused Triton kernel that computes the
output in place, eliminating the transient buffer entirely. Verified
~305 t/s decode on 4090 + INT4 (memory: project_cliff2b_deployed.md).

**Files:**
- `apply_cliff2b.py` — runs after Genesis `apply_all`, before `vllm serve`.
  Idempotent; safe to re-run.
- `chunk_fused.py` — the fused Triton kernel module.

**Where it applies.** All Qwen3.6-27B variants (model is hybrid GDN — every
forward pass touches the GDN code path). No-op if `apply_all` already
patched the call site.

## Wiring

Composes that should mount + apply our fixes add a single volume mount for
the entire dir and three `python3` invocations after Genesis `apply_all`:

```yaml
volumes:
  - ../../../../fixes:/fixes:ro

entrypoint:
  - /bin/bash
  - -c
  - |
    set -e
    pip install xxhash pandas scipy -q
    python3 -m vllm._genesis.patches.apply_all
    python3 /fixes/patch_drafter_skip.py
    python3 /fixes/patch_vision_tower_skip.py
    python3 /fixes/cliff2b/apply_cliff2b.py
    exec vllm serve "$@"
```

All three patches are idempotent and self-gate on whether their target
condition is present, so wiring them into all Genesis-using composes is
safe even when a particular variant doesn't need a given fix.
