# vLLM Torch-Profiler Analysis — Qwen3.6-27B on RTX 4090 (TQ lane)

**Date:** 2026-07-13 · **Server:** vLLM :8020, `XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-MTP-BF16`, gptq_marlin W4 g128, MTP n=3, TQ3 KV (3-bit V), piecewise cudagraphs, max_model_len 75000.
**Model dims (config.json):** 64 layers = 16 full-attention + 48 GDN linear-attention; hidden 5120; 24 q-heads / 4 kv-heads × head_dim 256 (**GQA group = 6**); GDN: 48 v-heads / 16 k-heads × 128; vocab **248,320**; `tie_word_embeddings=false`; GPTQ `lm_head: false` → **lm_head is unquantized BF16 (2.543 GB)**.

**Traces** (`models/qwen3.6-27b/vllm/diagnostics/torch_profiles/`):

| Capture | File | Events | Kernel span | Content (observed, not nominal) |
|---|---|---|---|---|
| A | `rank0...645726788777` (62.6 MB) | 3,376,481 | 3.86 s | 85 decode steps, steady bucket `generation_3(12)` = **3 seqs × q_len 4**, short ctx (nominal "bs=1" — production traffic overlapped) |
| B | `rank0...789778774433` (160.8 MB) | 8,580,695 | 13.17 s | 238 steps: 2 single-chunk prefills (~4,128- and ~3,650-token prompts) + decode at buckets 4/8/12/16 tokens (**up to 4 seqs simultaneously**, never 5) |
| C (orig) | `rank0...970205360277` (671 KB) | 34,784 | 0.23 s | **FAILED capture** — one 132 ms decode step (single `execute_context_0(0)_generation_2(8)` annotation, 1,556 kernels, 40.7 ms CUDA, zero prefill kernels). `profiler_out_0.txt` is this capture's summary (overwritten), not A's. |
| C2 (re-capture) | `rank0.1783959251465237922` (98.5 MB) | 5,310,715 | 22.53 s | **32,207-token prefill + 300-token decode**, captured 2026-07-13 18:13 after verified `num_requests_running=0`; one ~2.6K-prompt production request joined at t≈2.9 s (unavoidable on the live box), so decode ran at bucket 8 (2 seqs) — the long-ctx seq dominates every number of interest |

Method: per-trace stats regenerated from the chrome traces via streaming `ijson` (yajl2_c) parse; self-CUDA = sum of `cat:"kernel"` durations by name; self-CPU computed by per-thread interval nesting over `cat:"cpu_op"`. CPU times carry torch-profiler `with_stack` inflation — treat relatively.

Annotation decode (verified against kernel grids): `execute_context_N(M)_generation_K(L)` = M prefill tokens + L decode tokens in the step; decode buckets advance in steps of 4 = **q_len K+1 = 4 per sequence** (MTP n=3 verify), confirmed independently by `_tq_decode_stage1` grid.x = padded decode-token count.

---

## 1. Per-scenario kernel tables

### Capture A — decode, 3 seqs, short ctx (85 steps, median step 37.0 ms, GPU 91.4% busy)

Total kernel CUDA 3,529 ms. Top 15 by self-CUDA:

| # | Kernel | Self-CUDA | % | Calls | us/call |
|---|---|---|---|---|---|
| 1 | `marlin::Marlin<...>` (W4 GEMM, decode) | 1362.0 ms | 38.6% | 20,992 | 64.9 |
| 2 | `cutlass_80_wmma_..._128x1_tn_align8` (**BF16 lm_head**, grid (8,1940,1)) | 913.4 ms | 25.9% | 340 | **2686.5** |
| 3 | `_tq_decode_stage1` (TQ3 attn decode) | 570.8 ms | 16.2% | 1,632 | **349.7** |
| 4 | `cutlass_80_wmma_..._128x2_tn_align8` (BF16 MTP-draft GEMMs + GDN ba-proj) | 251.5 ms | 7.1% | 5,355 | 47.0 |
| 5 | `fused_sigmoid_gating_delta_rule_update_kernel` (GDN decode core) | 115.9 ms | 3.3% | 4,080 | 28.4 |
| 6 | `_topk_topp_kernel` (sampler) | 63.8 ms | 1.8% | 81 | 787.0 |
| 7 | `marlin::Marlin<...>` (2nd instantiation, 1 bool template arg differs; 9 calls/step) | 47.1 ms | 1.3% | 768 | 61.4 |
| 8 | `at::native::elementwise_kernel` (copies) | 22.0 ms | 0.6% | 12,240 | 1.8 |
| 9 | `cunn_SoftMaxForward<4,...>` | 13.5 ms | 0.4% | 174 | 77.5 |
| 10 | `cutlass_80_tensorop_s1688gemm_64x64` (small FP32 GEMMs) | 13.0 ms | 0.4% | 3,247 | 4.0 |
| 11 | `_causal_conv1d_update_kernel` (GDN) | 12.7 ms | 0.4% | 4,080 | 3.1 |
| 12 | `_fwd_kernel_stage2` (TQ split reduce) | 9.8 ms | 0.3% | 1,632 | 6.0 |
| 13 | `unrolled_elementwise_kernel` (copies) | 7.9 ms | 0.2% | 3,502 | 2.3 |
| 14 | `triton_red_fused_..._rms_norm_marlin_gemm_3` | 7.8 ms | 0.2% | 4,080 | 1.9 |
| 15 | `sample_recovered_tokens_kernel` (spec-decode) | 6.9 ms | 0.2% | 85 | 81.2 |

Per-step budget (37.0 ms wall): marlin ≈16.6 ms (256 calls), lm_head BF16 10.75 ms (4 calls), `_tq_decode_stage1` 6.7 ms (19 calls = 16 full-attn layers + 3 MTP-draft passes), MTP BF16 GEMMs ≈3.0 ms, GDN delta-rule 1.4 ms, sampler 0.75 ms.

Top self-CPU ops:

| # | Op | Self-CPU | Calls | us/call |
|---|---|---|---|---|
| 1 | `vllm::qwen_gdn_attention_core` | 1119.4 ms | 4,080 | 274.4 |
| 2 | `vllm::unified_attention_with_output` | 337.6 ms | 1,615 | 209.0 |
| 3 | `aten::copy_` | 255.3 ms | 35,524 | 7.2 |
| 4 | `vllm::unified_kv_cache_update` | 197.6 ms | 1,615 | 122.4 |
| 5 | `aten::mm` | 73.7 ms | 3,587 | 20.5 |
| 6 | `aten::empty` | 70.0 ms | 25,907 | 2.7 |
| 7 | `aten::slice` | 57.3 ms | 61,281 | 0.9 |
| 8 | `PythonDispatchMode` | 50.2 ms | 46 | 1092.1 |
| 9 | `aten::empty_strided` | 45.4 ms | 12,189 | 3.7 |
| 10 | `aten::cat` | 36.2 ms | 4,174 | 8.7 |

Runtime API: `cudaEventSynchronize` 243 ms/510, `cudaLaunchKernel` 137 ms/43,176, `cudaGraphLaunch` 76 ms/6,035 (**71 piecewise-graph segments per step**), `cuLaunchKernelEx` 64 ms/14,525 (Triton eager, mostly GDN).

### Capture B — 2 small prefills + decode at 2–4 seqs (238 steps, median decode step 38.1 ms, GPU 97.2% busy)

Total kernel CUDA 12,805 ms. Top 15 by self-CUDA:

| # | Kernel | Self-CUDA | % | Calls | us/call |
|---|---|---|---|---|---|
| 1 | `marlin::Marlin` (decode) | 3595.9 ms | 28.1% | 55,040 | 65.3 |
| 2 | `cutlass_..._128x1_tn` (**BF16 lm_head**) | 2541.9 ms | 19.9% | 993 | 2559.8 |
| 3 | `marlin::Marlin` (**prefill**; M≈4K split into 4 launches/GEMM) | 2262.1 ms | 17.7% | 2,048 | 1104.5 |
| 4 | `_tq_decode_stage1` | 1894.4 ms | 14.8% | 4,556 | **415.8** |
| 5 | `cutlass_..._128x2_tn` (MTP BF16 + ba-proj) | 700.2 ms | 5.5% | 14,734 | 47.5 |
| 6 | `fused_sigmoid_gating_delta_rule_update_kernel` | 369.3 ms | 2.9% | 11,328 | 32.6 |
| 7 | `marlin::Marlin` (3rd pool) | 307.4 ms | 2.4% | 5,120 | 60.0 |
| 8 | `flash::flash_fwd_kernel<...256,64,64,4>` (prefill full-attn) | 90.0 ms | 0.7% | 119 | 756.4 |
| 9 | `_topk_topp_kernel` | 89.8 ms | 0.7% | 217 | 413.7 |
| 10 | `elementwise_kernel` (copies) | 62.0 ms | 0.5% | 34,035 | 1.8 |
| 11 | `triton_poi_fused_marlin_gemm_mul_silu_slice_4` | 52.9 ms | 0.4% | 11,424 | 4.6 |
| 12 | `cunn_SoftMaxForward` | 38.3 ms | 0.3% | 493 | 77.7 |
| 13 | `cutlass_80_tensorop_s1688gemm` | 36.3 ms | 0.3% | 9,044 | 4.0 |
| 14 | `_causal_conv1d_update_kernel` | 36.0 ms | 0.3% | 11,328 | 3.2 |
| 15 | `triton_red_..._rms_norm_marlin_gemm_3` | 33.0 ms | 0.3% | 11,424 | 2.9 |

GDN prefill kernels (Triton, eager): `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 30.4 ms/192, `chunk_fwd_kernel_o` 25.8 ms/192, `_causal_conv1d_fwd_kernel` 13.2 ms/192, `chunk_scaled_dot_kkt_fwd` 6.8 ms/192. TQ store `_tq_fused_store_mse` 10.3 ms/4,522 (2.3 us/call — **TQ store is negligible**).

Top self-CPU ops:

| # | Op | Self-CPU | Calls | us/call |
|---|---|---|---|---|
| 1 | `vllm::qwen_gdn_attention_core` | 3102.1 ms | 11,424 | 271.5 |
| 2 | `aten::_local_scalar_dense` (**.item() GPU syncs**) | 2491.1 ms | 7,412 | 336.1 |
| 3 | `vllm::unified_attention_with_output` | 935.5 ms | 4,522 | 206.9 |
| 4 | `aten::copy_` | 708.2 ms | 99,323 | 7.1 |
| 5 | `vllm::unified_kv_cache_update` | 546.1 ms | 4,522 | 120.8 |
| 6 | `aten::mm` | 210.8 ms | 10,153 | 20.8 |
| 7 | `aten::empty` | 203.2 ms | 75,543 | 2.7 |
| 8 | `aten::slice` | 148.2 ms | 166,454 | 0.9 |
| 9 | `aten::cat` | 100.4 ms | 11,590 | 8.7 |
| 10 | `aten::add` | 98.7 ms | 14,158 | 7.0 |

Runtime API: `cudaStreamSynchronize` **2489 ms / 119 calls (~21 ms each)**, concentrated in the two prefill steps (matches `_local_scalar_dense` 2×1.22 s inside the two 1.33 s prefill annotations); `cudaEventSynchronize` 1050 ms/1,420; `cudaGraphLaunch` 221 ms/16,764.

### Capture C2 — 32,207-token prefill + 300-token decode at 32K ctx (22.53 s, GPU 98.8% busy)

Total kernel CUDA 22,253 ms. Top 15 by self-CUDA:

| # | Kernel | Self-CUDA | % | Calls | us/call |
|---|---|---|---|---|---|
| 1 | `marlin::Marlin` (**prefill**) | 11284.7 ms | 50.7% | 10,112 | 1116.0 |
| 2 | `_tq_decode_stage1` | 3811.4 ms | 17.1% | 2,189 | **1741.2** avg (2028.9 median inside 32K-ctx decode steps) |
| 3 | `marlin::Marlin` (decode) | 1681.6 ms | 7.6% | 27,392 | 61.4 |
| 4 | `flash::flash_fwd_kernel` (prefill attn) | 1627.9 ms | 7.3% | 187 | 8705.4 (3.5 ms → 18.3 ms/call as ctx grows 4K→32K) |
| 5 | `cutlass_..._128x1_tn` (**BF16 lm_head**) | 1236.1 ms | 5.6% | 509 | 2428.5 |
| 6 | `cutlass_..._128x2_tn` (MTP BF16 + ba-proj) | 333.8 ms | 1.5% | 6,793 | 49.1 |
| 7 | `triton_poi_fused_marlin_gemm_mul_silu_slice_4` | 197.5 ms | 0.9% | 5,616 | 35.2 |
| 8 | `ampere_fp16_s1688gemm_fp16_128x64...` (cuBLAS BF16 — MTP prefill GEMMs) | 151.0 ms | 0.7% | 37 | 4082.2 |
| 9 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (GDN prefill) | 143.1 ms | 0.6% | 720 | 198.8 |
| 10 | `chunk_fwd_kernel_o` (GDN prefill) | 127.3 ms | 0.6% | 720 | 176.8 |
| 11 | `fused_sigmoid_gating_delta_rule_update_kernel` | 116.2 ms | 0.5% | 5,520 | 21.0 |
| 12 | `recompute_w_u_fwd_kernel` (GDN prefill) | 96.7 ms | 0.4% | 720 | 134.3 |
| 13 | `_fused_post_conv_kernel` (GDN prefill) | 83.3 ms | 0.4% | 720 | 115.7 |
| 14 | `_causal_conv1d_fwd_kernel` (GDN prefill) | 80.9 ms | 0.4% | 720 | 112.4 |
| 15 | `merge_16x16_to_64x64_inverse_kernel` (GDN prefill) | 79.0 ms | 0.4% | 720 | 109.7 |

TQ store/dequant remain negligible: `_tq_full_dequant_kv` 12.0 ms/170, `_tq_fused_store_mse` 9.0 ms/2,223.

Top self-CPU: `aten::_local_scalar_dense` **13,572 ms / 4,995 calls (2.72 ms/call)** — dwarfs everything; `vllm::qwen_gdn_attention_core` 1,720 ms/5,616; `ChunkGatedDeltaRuleFunction` 349 ms/720. Runtime: `cudaStreamSynchronize` **13,570 ms / 187 calls (72.6 ms each)**, `cudaEventSynchronize` 3,095 ms/695.

Timeline: prefill t=0→15.4 s as ~8 chunk steps of 4,128/4,124 tokens, each **1.41 → 1.67 s wall (growing ~45 ms/chunk with context)**; decode t=15.4→22.5 s, 101 steps at bucket 8, **median 64.5 ms/step** (GPU busy 63.7 ms, idle 0.89 ms). 300 tokens over ~115 decode steps ⇒ **2.61 tok/step, MTP draft acceptance ≈54%**, ≈40 tok/s at 32K ctx.

### How rankings shift with batch size and context

- **Batch (A→B, 3→4 seqs): nearly free.** marlin 64.9→65.3 us/call, lm_head 2686→2683 us, delta-rule 28.4→32.6 us, TQ 378→452 us (that delta is ctx, not batch — see 2a). Step wall 37.0→38.5 ms. Every decode kernel is weight/KV-bandwidth- or launch-bound; concurrency ≤4 seqs costs ~4%.
- **Context (A→C2, ~2K→32K): TQ decode explodes, everything else holds.** `_tq_decode_stage1` 350→2,029 us/call (5.8×), moving from 16% of the step to **~60% of a 64.5 ms step**; lm_head and marlin decode are unchanged in absolute terms, so their share halves. The step-time increase 37→64.5 ms is accounted for almost entirely by TQ (+31.8 ms measured TQ delta vs +27.5 ms wall delta).
- **Prefill: marlin GEMMs own it at every length** (~70% of prefill GPU at 32K; 17.7% of all of B from two ~4K prefills), with flash attention rising to #2 at long ctx (quadratic: 3.5→18.3 ms/call) and GDN chunked-prefill + TQ store staying <6% combined.

---

## 2. Specific questions

### (a) `_tq_decode_stage1`: flat launch cost or bandwidth?

**Flat in batch; strongly context-scaled; never bandwidth-bound — a latency/occupancy-bound kernel with a ~330 us floor.** Per-call medians by decode-token bucket:

| Scenario / ctx | Tokens (bucket) | n calls | median us | p95 |
|---|---|---|---|---|
| A, short ctx | 8 | 85 | 329.9 | 358 |
| A, short ctx | 12 | 1,290 | 377.9 | 397 |
| B, ~4–8K ctx | 4 | 17 | 360.9 | 456 |
| B, ~4–8K ctx | 8 | 279 | 452.3 | 461 |
| B, ~4–8K ctx | 12 | 1,657 | 452.1 | 477 |
| B, ~4–8K ctx | 16 | 1,851 | 450.4 | 485 |
| **C2, 32K ctx** | 8 | 1,726 | **2028.9** | 2071 |

- **Batch scaling: none.** 4→16 tokens at fixed ctx changes the median <1% (B rows). The ~350 us at low bs is not amortized by batching — and it is not a one-time launch cost either: it is per-call kernel inefficiency.
- **Context scaling: ~55 us per 1K tokens** on top of the floor (378 us @ ~2K → 2,029 us @ 32K). At 19 calls/step (16 full-attn layers + 3 MTP-draft passes) that is **6.7 ms/step short-ctx → 38.5 ms/step @ 32K = 60% of the step**.
- **Bandwidth check @ 32K:** bytes needed per call = 32K tokens × 4 kv-heads × 256 dim × (1 B fp8 K + 3/8 B TQ3 V) ≈ 45 MB ⇒ ~45 us at peak BW. The per-q-head grid re-reads each KV page 6× (GQA group 6) ⇒ ~270 MB ⇒ ~270 us at peak. Measured 2,029 us ⇒ **~13% of achievable bandwidth even counting the 6× redundancy**.
- Root cause (live kernel = `vllm/v1/attention/ops/triton_turboquant_decode.py` in the container, mirrored at `patches/vllm-pr40798-rebased/...`, launch lines 563–598): `grid=(tokens, Hq=24, NUM_KV_SPLITS=32)` with **`num_warps=1, num_stages=1, BLOCK_KV=4`** — trace confirms `block=(32,1,1)`. One warp per block, serial 4-token tiles, zero pipelining: pure unhidden memory latency, plus 24 q-head blocks re-fetching the same KV 6×.
- The fix half-exists: `_genesis/kernels/tq_grouped_decode.py` (`BLOCK_H=16, BLOCK_KV=16, num_warps=4, num_stages=2`) with `GENESIS_ENABLE_P40=1` **set in the live container** — but `should_use_grouped_kernel()` (lines 325–345) requires `key_fp8 and value_quant_bits == 4`; this deployment is TQ3 (`value_quant_bits=3`), so **P40 silently falls back to the scalar kernel on 100% of calls**. Zero `_tq_grouped_decode_stage1` launches in any of the four traces.

### (b) Which GEMMs are the big cutlass f16 kernels?

The model card is `GPTQ-Pro-MTP-BF16`: main model W4 marlin; **MTP module and lm_head BF16**. Trace confirmation:

1. **lm_head** — `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x1_tn_align8`, grid `(8, 1940, 1)`: 1940 × 128 = **248,320 = vocab_size** exactly. **4.0 calls per decode step** (A: 340/85 = 1 verify + 3 MTP draft passes), 2,686 us/call flat from bs=1 to 4 and from 2K to 32K ctx (C2 median 2,672).
   **Effective bandwidth: 248,320 × 5,120 × 2 B = 2.543 GB / 2.686 ms = 947 GB/s = 94% of the 4090's 1,008 GB/s peak.** Hard bandwidth-bound on weight reads; only byte-shrinking helps. Cost: **10.7 ms per decode step** (29% of a short-ctx step, 17% @ 32K).
2. **MTP draft-block GEMMs** — `..._128x2_tn_align8`: grid `(8,40,1)` N=5,120 (o_proj / eh_proj / down_proj, 9 calls/step = 3 per draft pass), `(8,272,1)` N=34,816 = 2×17,408 (fused gate_up, 3/step), `(8,112,1)` N=14,336 (draft qkv block, 3/step). ~47 us/call, ~2.9 ms/step. At prefill M=4,128 these route to cuBLAS `ampere_fp16_s1688gemm` kernels (151 ms total in C2).
3. **GDN ba-projection** — same 128x2 kernel at grid `(8,1,16)` (N≤128, split-K=16, `cublasLt::splitKreduce` companion), 48 calls/step: the tiny BF16 projection excluded from GPTQ.
4. Embeddings never register (index_select-class, negligible).

Marlin reference points: decode pool reads ~12 GB W4 weights over 256 calls / 16.6 ms ⇒ **~720–760 GB/s (72–75% of peak)** — close enough to roofline that only byte-reduction pays. Marlin *prefill* (M=4,128, 4-way M-split ⇒ ~1,264 launches/chunk-forward) ≈ 198 TFLOP / 1.13 s ≈ **~175 TF/s effective** — compute-bound near the card's FP16/FP32-acc dense rate; near its practical ceiling too.

Discrepancy worth chasing: `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1` is set, yet all MTP draft GEMMs run BF16 — the flag isn't biting on this checkpoint (~3 ms/step at stake).

### (c) GDN eager-gap cost

GPU idle inside decode-step windows (annotation wall − union of kernel intervals):

| Scenario | Bucket | Steps | Median wall | GPU busy | Idle | Idle % | GDN CPU-op in window |
|---|---|---|---|---|---|---|---|
| A | 8 | 4 | 37.54 ms | 36.17 ms | 1.37 ms | 3.7% | 17.7 ms |
| A | 12 | 81 | 37.00 ms | 36.18 ms | 0.83 ms | 2.2% | 17.2 ms |
| B | 8 | 18 | 36.82 ms | 35.96 ms | 0.85 ms | 2.3% | 16.7 ms |
| B | 12 | 100 | 37.87 ms | 37.04 ms | 0.83 ms | 2.2% | 17.1 ms |
| B | 16 | 117 | 38.45 ms | 37.59 ms | 0.86 ms | 2.2% | 16.8 ms |
| C2 | 8 (32K ctx) | 101 | 64.55 ms | 63.67 ms | 0.89 ms | 1.4% | 16.9 ms |

**At current step times the GDN eager gap is nearly fully hidden: ~0.8–0.9 ms/step GPU idle at every batch size and context** (plus ~5.8 ms/step between annotations in A, itself ~93% GPU-busy). The 48 eager `vllm::qwen_gdn_attention_core` calls cost ~272 us self-CPU each ⇒ **~17 ms/step of host work** (with_stack-inflated; real is lower but same order), running concurrently with 36–64 ms of GPU work. The original capture C shows the failure mode when launch-ahead collapses (cold step: 132 ms wall, 80.8% idle, GDN 1.4 ms CPU/call).
**Implication:** GDN CPU orchestration is not today's bottleneck — but at short ctx it caps the winnings from targets 1–2: fixing lm_head+TQ takes GPU busy to ~22 ms/step while the serialized eager CPU chain needs ~17 ms, so the step floors at the CPU chain instead of 22 ms unless GDN launches are also fused/graph-included. (GDN sits in `splitting_ops`; each step is 71 piecewise-graph segments with eager GDN between them — 14.5K `cuLaunchKernelEx` per trace in A.)

### (d) Prefill: what dominates (capture C2, 32,207 tokens)

Prefill = 15.4 s wall, ~8 chunk steps of ≤4,128 tokens, **1.41 → 1.67 s per chunk** (growing ~45 ms/chunk as flash attention lengthens) ⇒ **~2.1K tok/s prefill at 32K, TTFT ≈ 15.4 s**. Per-4,128-token chunk:

| Component | Per chunk | Share | Evidence |
|---|---|---|---|
| **marlin W4 GEMMs** | ~1.13 s | **~70%** | 10,112 calls × 1,116 us = 11.28 s = 50.7% of trace; ~1,264 launches/chunk (4-way M-split × 256 GEMMs + drafts) |
| flash_fwd (16 full-attn layers + MTP) | 60 → 310 ms | 4→19% | 3.5 ms/call @ 4K ctx → 18.3 ms @ 32K (quadratic); 1.63 s total |
| GDN chunked prefill (6 Triton kernels × 48 layers) | ~75 ms | ~5% | `chunk_gated_delta_rule_fwd_h` 143 ms + `chunk_fwd_o` 127 + `recompute_w_u` 97 + `post_conv` 83 + `conv1d_fwd` 81 + `merge_inverse` 79 ≈ 610 ms total |
| MTP BF16 prefill GEMMs (cuBLAS) | ~19 ms | ~1% | `ampere_fp16_s1688gemm` 151+39 ms / 37+11 calls |
| TQ store/quant | <3 ms | <0.2% | `_tq_full_dequant_kv` 12 ms + `_tq_fused_store_mse` 9 ms total |

**Chunked-prefill GDN and TQ store are non-issues; prefill is a marlin compute problem** (~175 TF/s effective, near the FP32-acc tensor-core ceiling — launch-config tuning won't move it; only cheaper FLOPs would).
The host side is ugly but currently masked: **13.6 s of `.item()` syncs (4,995 calls × 2.72 ms; `cudaStreamSynchronize` 72.6 ms/call × 187) inside 15.4 s of prefill** — GPU stays 98.8% busy, but the sync chain serializes the host thread, and the co-running decode seq got exactly **one 4-token step per ~1.5 s chunk** (mixed steps `execute_context_1(4124)_generation_1(4)`): any co-resident decode request sees ~1.5 s inter-token stalls during a long prefill.

### (e) MTP verify shape

- Decode runs **q_len = 4 = K+1 per sequence** — proven twice: decode buckets advance in 4-token steps per extra seq (`generation_1(4)…generation_4(16)`), and `_tq_decode_stage1` grid.x = padded decode-token count. Draft passes run q_len=1/seq (grid.x 2–4 observed as separate small launches inside the same step annotation).
- **Kernels are shape-specialized, not tile-padded**: marlin always uses its persistent grid `(128,1,1)×256thr` (weight-stationary — M=4..16 rides along free; decode per-call invariant 64.9→65.3 us from 12→16 tokens); lm_head wmma uses 16-row tiles — M≤16 fits one tile stripe and the kernel is weight-bandwidth-bound, so padded M is free; TQ grid.x tracks tokens exactly. The only padding is the uniform-decode cudagraph bucket itself (multiples of 4 tokens — zero waste at these sizes).
- MTP's real overhead is not shape padding but **the 4× BF16 lm_head reads (10.7 ms/step) + the BF16 draft block (~2.9 ms/step)**; measured draft acceptance ≈54% (2.61 tok/step incl. bonus) at 32K ctx.

---

## 3. Top-3 attack targets

Ranked by production impact (agent workloads live at 20–70K ctx, where TQ decode is 60% of every step).

### Target 1 — TQ3 decode stage1: GQA grouping + warp/pipeline retune
**6.7 ms/step short-ctx → 38.5 ms/step @ 32K (60% of step)**

- **File/kernel:** `_tq_decode_stage1`, live at `vllm/v1/attention/ops/triton_turboquant_decode.py` in the container (mirror: `patches/vllm-pr40798-rebased/v1/attention/ops/triton_turboquant_decode.py`, launch config lines 563–598). Grouped kernel: `patches/genesis/vllm/_genesis/kernels/tq_grouped_decode.py`; gate `should_use_grouped_kernel()` lines 325–345; wiring `wiring/legacy/patch_40_tq_grouped_decode.py`.
- **Current behavior:** 19 calls/step; `num_warps=1, num_stages=1, BLOCK_KV=4`, fixed 32 splits, one block per q-head ⇒ 6× redundant KV reads; ~13% of achievable bandwidth at 32K (2,029 us measured vs ~270 us roofline incl. redundancy, ~45 us without). P40 grouped kernel env-enabled but **dead on TQ3** (`value_quant_bits==4` gate + `tl.static_assert(VQB == 4)`).
- **Proposed change:**
  1. *Day-1 text-patch:* raise the scalar launch to `num_warps=4, num_stages=2` (the PN26 sparse-v sweep already crowned BLOCK_KV=4 / num_warps=4 on SM86; env-knob pattern `GENESIS_PN26_SPARSE_V_BLOCK_KV` exists). Grid unchanged ⇒ cudagraph-safe (recapture only).
  2. *Main fix:* add a VQB=3 dequant path to `_tq_grouped_decode_stage1` (relax the static_assert, port the scalar kernel's 3-bit unpack) and widen the P40 predicate to `value_quant_bits in (3, 4)`. Grid `(tok,24,32)` → `(tok, 4·⌈6/BLOCK_H⌉, 32)`: KV read 6×→1×, 4 warps, 2-stage pipeline.
  3. Autotune keys over (ctx-bucket, tokens) for BLOCK_KV/num_warps with per-bucket caching — NUM_KV_SPLITS must stay 32 (grid constant for cudagraph; see comment `vllm-pr40914-k1-only/.../turboquant_attn.py:299`).
- **Expected gain (from measured numbers):** conservatively 2,029 → ~500 us/call @ 32K (4×; the roofline says 10× is available) ⇒ step 64.5 → ~35 ms ⇒ **~1.8× decode throughput at 32K ctx** (40 → ~74 tok/s); short-ctx step −4.5 ms (37 → 32.5 ms). The day-1 warp bump alone should take the 350 us floor to ~150–200 us, worth ~20–30% at long ctx.
- **Risk:** 3-bit dequant correctness in the grouped kernel — bit-compare vs scalar path across (ctx, tokens) grid (reuse the `test_pn14_tq_decode_oob_clamp.py` harness) + logit A/B on live prompts. The old A5000 t-test (p=0.284) that shelved P40 tested only the k8v4 path, short-ctx, single-stream — irrelevant to the measured 32K bottleneck.
- **Ships as:** genesis text-patch (kernel + predicate already in tree; wire VQB=3 through patch_40). No custom build.

### Target 2 — Quantize the BF16 lm_head (10.7 ms/step: 29% short-ctx, 17% @ 32K)

- **File/kernel:** `cutlass_80_wmma_..._128x1_tn` via `aten::mm` on `ParallelLMHead.weight` BF16 [248,320 × 5,120]; checkpoint has `quantization_config.lm_head=false`.
- **Current behavior:** 4 × 2.543 GB weight reads per decode step (1 verify + 3 draft lm_head calls), 2.686 ms each at 947 GB/s — pure bandwidth, invariant to batch and ctx.
- **Proposed change (staged):** (1) *W8 online repack* at load (marlin W8, or SM89-native FP8 cutlass) via genesis text-patch wrapping the lm_head weight loader → 1.27 GB ⇒ ~1.35 ms/call ⇒ **−5.4 ms/step**; (2) *W4 marlin* → 0.64 GB ⇒ ~0.7 ms/call ⇒ **−8 ms/step**, gated on quality.
- **Expected gain:** short-ctx step 37 → 31.6 ms (W8) / ~29 ms (W4) ⇒ **+17–27% decode throughput**; @ 32K (after Target 1) another ~15%. Compounds with Target 1: short-ctx step → ~27 ms, 32K step → ~30 ms (~87 tok/s at measured acceptance).
- **Risk:** logit fidelity hits both output quality and MTP acceptance (54% measured; a 5-pt drop costs ~7% throughput — watch `vllm:spec_decode_num_accepted_tokens` before/after + perplexity; 248K-vocab logits are the most quant-sensitive tensor in the model). W8 is near-lossless; W4 needs the check.
- **Ships as:** genesis text-patch (new PN, same online-quant machinery as PN8). While there: **fix PN8 itself** — `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1` is set but every MTP GEMM still runs BF16 cutlass (~2 ms/step more on the table).

### Target 3 — GDN eager launch-chain + prefill `.item()` storm (the post-1+2 wall)

- **File/kernel:** `vllm::qwen_gdn_attention_core` host path (48 calls/step, in `splitting_ops`, outside all 71 graph segments; ~10 small launches/layer: `fused_sigmoid_gating_delta_rule_update` 21–33 us + `_causal_conv1d_update` 3 us + glue); chunked-prefill host loop issuing ~500 `.item()` syncs per chunk (`aten::_local_scalar_dense` → `cudaStreamSynchronize`; 13.6 s CPU per 32K prefill).
- **Current behavior:** ~17 ms/step self-CPU hidden behind 36–64 ms GPU (idle only 0.8–0.9 ms/step) — but the cold-step evidence (capture C orig: 80.8% idle, 132 ms wall) shows the chain fully exposed when launch-ahead breaks, and after Targets 1+2 the short-ctx GPU budget (~22 ms) approaches the CPU chain (~17 ms inflated / est. 8–12 ms real), which then floors the step. Prefill: the sync chain limits any co-running decode to 1 step per ~1.5 s chunk.
- **Proposed change:** (1) fuse the GDN decode layer path into 1–2 Triton launches per layer (all operands are tiny at decode shapes) — cuts ~480 launches/step to ~100; (2) then drop GDN from `splitting_ops` for uniform-decode buckets so the whole step captures into one graph (capture-safety groundwork exists: P57 spec-decode capture-safe buffers, P78 tolist-capture guard, PN59 streaming-GDN); (3) batch the chunked-prefill `.item()` reads into one D2H copy per chunk.
- **Expected gain:** decode — nothing standalone today; **it converts Targets 1+2 from "GPU idle waiting on CPU" into the full ~27 ms short-ctx step** (~35% throughput at short ctx post-1+2) and removes the cold-step 132 ms failure mode; prefill — restores decode interleaving (co-resident inter-token stalls 1.5 s → ~70 ms) and trims host serialization from TTFT.
- **Risk:** graph-capturing stateful GDN updates is the classic trap (in-place recurrent-state writes must be capture-safe and bucket-stable) — stage behind an env flag, bit-compare states over replay batches; the kernel fusion and `.item()` batching are low-risk and land first.
- **Ships as:** fusion + sync batching = genesis text-patches; graph inclusion = `splitting_ops` config change + patch. No custom build.

**Not top-3, noted:** marlin prefill is 70% of prefill GPU but already ~175 TF/s effective (near FP32-acc ceiling) — only algorithmic changes move TTFT materially; flashinfer `_topk_topp_kernel` costs 787 us/step at low bs (2%); `cudaEventSynchronize` 3.1 s in C2 is worth a look alongside Target 3's sync work.

---

## Appendix: capture C2 provenance

Original capture C (`rank0...970205360277`) contained no prefill — a single 132 ms decode step; the 50K request evidently never executed under the profiler. Re-captured 2026-07-13 18:13 (`rank0.1783959251465237922`, 98.5 MB): verified `vllm:num_requests_running == 0` twice, `POST /start_profile`, one `/v1/completions` with a programmatically built 30,000-word prompt (**32,207 tokens** via `/tokenize`, under the 75K cap), `max_tokens=300`, `POST /stop_profile`; request completed in 22.4 s (`usage: prompt 32207, completion 300`). One ~2.6K-token production request arrived at t≈2.9 s and co-ran (decode bucket 8 instead of 4); its short-ctx contribution is negligible to the long-ctx numbers quoted. Analysis scripts (streaming ijson aggregator + per-step gap/bucket analysis) lived in the session scratchpad; every aggregate is reproducible from the `.pt.trace.json.gz` files.
