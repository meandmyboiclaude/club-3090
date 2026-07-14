# tq-lane — TurboQuant kernel-engineering diagnostics

Authored 2026-07-13 (static analysis + authoring only; nothing here has run on
GPU yet — the live server on :8020 had a profiling session in flight).

## Contents

| File | What |
|---|---|
| `tq_parity_harness.py` | TQ 3-bit store/load bit-parity + continuation-path harness (run later, GPU) |
| `README.md` | this file |

Related (in `fixes/`): `patch_pn95_43357_tq_prefill_workspace_sizing.py` —
continuation-prefill workspace fix draft for upstream #43357 (verified
apply/idempotent/ast-clean in the image; **not wired into any compose**).

## Harness: what it covers

Geometry hardcoded from the served model's `config.json`
(`XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-MTP-BF16`): 24 q heads / **4 KV heads**
(GQA 6) / **head_dim 256**. The model is a GDN hybrid (`full_attention` every
4th layer, 16 of 64) — only full-attention layers have KV, so the harness
models one full-attention layer, which is exactly the TQ backend's view.

Presets: `turboquant_3bit_nc` (live) + `turboquant_k3v4_nc`. FP8-key presets
skipped (different storage path, not served).

Upstream ships **no** python/torch reference for TQ (module = config +
centroids + two Triton kernels), so the harness carries a reference
quantizer/dequantizer implemented op-for-op from the packing spec in
`triton_turboquant_store.py` / `triton_turboquant_decode.py`
(slot = `[3-bit MSE key idx | fp16 norm | packed values | fp16 scale | fp16 zero]`).

Checks (each prints `PASS/FAIL` + max abs/rel error):

1. **store-bytes parity** — Triton store vs reference quantizer, exact packed
   bytes (padding bytes excluded).
2. **roundtrip bounds** — store → Triton dequant vs original: values must obey
   the exact per-element bound `|err| ≤ scale/2 (+fp16 slack)`; keys must sit
   inside the 3-bit Lloyd-Max distortion envelope (rel L2 < 0.35, cos > 0.90;
   CPU selfcheck measures ~0.235 max).
3. **dequant parity** — Triton `_tq_full_dequant_kv` vs reference dequant on
   identical cache bytes (fp16-rounding tolerance).
4. **determinism** — two independent store/dequant/decode runs bitwise equal.
5. **continuation q≤128** — TQ decode-kernel continuation path vs fp32 reference.
6. **continuation q>128** — `_continuation_prefill` (dequant+flash) vs
   reference, **plus an informational vllm#43357 lock repro** (reports
   `REPRODUCED` on stock, `no-crash` once PN95 is applied).
7. **mixed batch (PN86/#46461)** — one long first-chunk prefill owning both
   batch maxima + one continuation request. **Expected FAIL on the stock
   image** (fast path drops the continuation's cached prefix); passes with
   PN86 applied.

## How to run it later (GPU, no server conflict)

Runs in its own container ⇒ own CUDA context. Small shapes; peak
torch-allocated VRAM is printed at the end and stays well under 1 GB (plus
~0.5 GB CUDA context). Do not start it while a profiling capture is actively
running on the same card; otherwise coexistence with the live server is fine.

```bash
# stock image (expect: check 7 FAIL — #46461 — and check 6 repro REPRODUCED)
podman run --rm --gpus all \
  -v /home/user/club-3090/diagnostics/tq-lane:/tq \
  --entrypoint python3 \
  docker.io/vllm/vllm-openai:nightly-9e57de7197f234f9d9187715d96e07e007048c0f \
  /tq/tq_parity_harness.py

# patched-as-deployed (what :8020 actually runs): apply the sidecars first
podman run --rm --gpus all \
  -v /home/user/club-3090/diagnostics/tq-lane:/tq \
  -v /home/user/club-3090/fixes:/fixes:ro \
  --entrypoint bash \
  docker.io/vllm/vllm-openai:nightly-9e57de7197f234f9d9187715d96e07e007048c0f \
  -c 'python3 /fixes/patch_pn79_tq_decode_scratch_cudagraph_safe.py &&
      python3 /fixes/patch_pn86_46461_tq_prefill_continuation_guard.py &&
      python3 /fixes/patch_pn95_43357_tq_prefill_workspace_sizing.py &&
      python3 /tq/tq_parity_harness.py'
```

(If the rig exposes GPUs via CDI use `--device nvidia.com/gpu=all` instead of
`--gpus all`.) Pytest also works: `--entrypoint python3 ... -m pytest /tq/tq_parity_harness.py -v`
— but the mount must then be writable or `PYTHONDONTWRITEBYTECODE=1` set.

CPU-only reference sanity pass (no GPU, already run 2026-07-13, both presets
PASS): append `--selfcheck`.

## #43357 workspace-sizing findings (dev1060 pin)

- Consumer: `turboquant_attn.py::_continuation_prefill` (stock L787–795)
  requests `2 × (1, Hk, round_up(cached_len, block), D) fp16` from the
  lockable `WorkspaceManager`. Demand scales with the **cached prefix
  length** — not the chunk size; the issue reporter's
  `2×chunk×heads×dim` formula under-reserves.
- Reservation: `TurboQuantMetadataBuilder._reserve_workspace` (stock
  L208–243). Decode scratch from `max_num_seqs` (PN79 territory) + a
  continuation reservation sized from `max_model_len − 1` (~300 MB at our
  75K/4KV/D256), gated on `enable_chunked_prefill and
  max_num_batched_tokens > 128`. The gate leaves #43357 reproducible for
  prefix-caching continuations with chunked prefill off; the reservation
  itself silently eats post-profiling VRAM headroom.
- PN95 moves the dequant buffers to a dedicated grow-only pool (PN79
  pattern), drops the ~300 MB reservation, and enforces a documented static
  bound `round_up(max_model_len − 1, block)` fail-loud. Prefill is never
  cudagraph-captured (TQ cg support = `UNIFORM_BATCH`; the path needs
  q_len > 128), so pool growth is replay-safe and PN79's invariant is
  untouched.
