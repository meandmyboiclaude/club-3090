# TQPLUS-DIGEST — TheTom/turboquant_plus papers → vLLM kernel-lane actions

Authored 2026-07-13. Digest only — no implementation.

**Our context (what "applicable" means below):** vLLM dev1060
(`nightly-9e57de71…`, pinned), Qwen3.6-27B hybrid GDN — 64 layers, **16
full-attention (indices 3,7,…,63)**, 24 q / 4 KV heads (GQA 6), **head_dim
256** — `turboquant_3bit_nc` KV, MTP n=3, single RTX 4090 (SM89), single-shot
4–25K workload. Just shipped: GQA-grouped TQ3 decode kernel (P40 v7.73,
`patches/genesis/vllm/_genesis/kernels/tq_grouped_decode.py`, 2.87× at 32K).

**Critical caveat on transfer:** every first-party number in these papers is
**llama.cpp/Metal on Apple Silicon**, where TQ dequant is a centroid-LUT
bottleneck (14–34% of decode). vLLM's TQ is a different animal: V is plain
uniform scale/zero (no WHT, no LUT — cheap dequant), K is Lloyd-Max
centroid gather or FP8, and one fp16 norm covers the whole head_dim vector.
Several headline wins shrink dramatically or vanish on our stack; each
verdict below says which.

---

## 1. sparse-v-dequant.md — attention-gated V-dequant skip

**Claim.** In flash-attention decode, softmax weights are known before V is
touched; at 32K ≥90% of weights are < 1e-6. Skipping V dequant+accumulate for
those positions: **+22.8% decode at 32K** (M5 Max, Qwen3.5-35B MoE, turbo3),
PPL delta exactly 0.0000 across τ∈[1e-8,1e-4], formats (turbo3/q8_0/q4_0),
and corpora. Explicitly "not TurboQuant-specific."

**Evidence quality.** Strong for the safety claim: 50-chunk wikitext-103 32K
run (CI ±0.021), threshold ablation, cross-format A/B, NIAH. The +22.8% is
Metal-specific (expensive LUT dequant). The honest CUDA datapoint is buried
in the addendum: **@Madreag, 4-GPU CUDA sweep incl. SM89 — only +4.6% at 32K
at skip rates of 96.8–99.7%** (τ=5e-3). The NIAH "improvement" (7/9→9/9) did
not replicate in Madreag's control (identical ON/OFF) — treat sparse-V as
quality-*neutral*, never quality-positive.

**Applicability: actionable-now (as a port into OUR grouped kernel), with
capped expectations.** Three facts from our own tree:

- Upstream already has this as **OPEN PR vllm#41422** (per-tile skip in the
  scalar TQ decode kernel, AMD MI300X-validated only), and Genesis already
  carries a fork: `_genesis/kernels/triton_turboquant_decode_sparse_v.py`
  (PN26b). Its 2026-05-01 A/B on A5000/SM86 found the *skip itself* nearly
  never fired on short outputs; the +3–5% shipped win came from kernel
  restructuring (`tl.range` pipelining, `.cg` cache hints, BLOCK_KV/warp
  tuning), and it ships default-OFF.
- PN26b wraps the **scalar** kernel. After P40 v7.73, our 32K decode hot
  path is the **grouped** kernel — the PN26 fork no longer covers the path
  that matters. The technique must be ported *into*
  `_tq_grouped_decode_stage1`.
- The gate is mathematically safe with online softmax: `p_running =
  exp(s − m_running) ≥ p_final` since the running max only grows, so
  `max(p_tile) < τ` is a conservative skip test. PN26's docstring already
  derives the bounded-drift argument (skip V load + weighted sum; still
  decay `acc`, still accumulate `l_prev`).

Composition caveat: in the grouped kernel a V tile is shared by all 6 q-heads
of the GQA group (that sharing *is* the 2.87×). A tile may be skipped only if
`max over (BLOCK_H × BLOCK_KV)` of p < τ — heads attend to different
positions, so the effective skip rate is the *intersection* across the group
and will be meaningfully lower than the paper's per-head 90%. Also our V
bytes are only ~half the slot (TQ3 @ D=256: key 98 B, value 100 B of a 198 B
slot) and V dequant is cheap uniform math, so the skippable share of tile
cost is ≤ ~50% of bytes, not the Metal LUT-heavy profile.

**Work item.**
- File: `patches/genesis/vllm/_genesis/kernels/tq_grouped_decode.py` — add
  `SPARSE_V: tl.constexpr` tile-skip to `_tq_grouped_decode_stage1`
  (constexpr-DCE'd when off, byte-equivalent OFF-path), env-gated like PN26;
  reuse PN26's threshold/min-ctx envs and skip-stat counters.
- Threshold: start τ=1e-3–5e-3 (upstream/TRT-LLM/Madreag range; the paper's
  1e-6 is safe but leaves skip rate on the table), min_ctx 8192.
- Expected win on OUR box: **low single-digit % at 25–32K on top of the
  2.87×**; ~0 below 8K. If a quick skip-rate probe at 25K shows <30% at the
  tile level, drop the item.
- Validation: SPARSE_V=0 bit-parity vs shipped grouped kernel + scalar
  reference (extend `diagnostics/tq-lane/tq_parity_harness.py`), skip-rate
  counters, PPL/logit A/B at 8/16/25/32K, `canary_grammar_mtp.py`.

---

## 2. layer-aware-v-compression.md + asymmetric-kv-compression.md — boundary protection + K-dominance

**Claims.**
- *Asymmetric:* K precision dominates quality (softmax amplifies K error
  exponentially; V error is linear in attention weight). q8_0-K + 2–4-bit V
  is nearly free (+0.3–2%), symmetric low-bit K is catastrophic on sensitive
  families (Qwen: PPL 3,556 vs 6.7 at identical total bits). GQA makes it
  worse: with group 6, K error hits 6 q-heads per K head.
- *Layer-aware:* protecting first-2+last-2 **KV-layer-ordinal** layers with
  higher V precision recovers 24–62% of the turbo2→turbo3 quality gap. Their
  own implementation mis-targeted hybrids (raw layer index vs KV ordinal —
  upgraded 1–2 of 16 KV layers on Qwen3.5-27B); hybrid numbers are
  confounded. Gains **dilute to ~zero at 16K+ context**. Mechanistic follow-up
  (@sztlink): boundary-*K* protection does NOT help — layer-0 K has extreme
  norms (146.8 vs 20–40 mid-stack) that quantize *better*; it's V-side
  boundary protection that pays.

**Evidence quality.** Asymmetric is the strongest-evidenced paper in the set:
replicated across 5 backends (Metal/CUDA/HIP/Vulkan ×2), 7+ model families,
including an E8-lattice cross-method confirmation and Ada (SM89) data.
Layer-aware is weaker: 512-ctx/4-chunk PPL runs, hybrid results
self-acknowledged as buggy, effect vanishes at long context.

**Applicability — two separate verdicts:**

**(a) K-dominance → the dtype ladder ordering: actionable-now, zero code.**
vLLM's presets map directly onto the paper's axes (upstream's own PPL doc in
`turboquant/config.py`): k8v4 +1.17% · 4bit_nc +2.71% · k3v4_nc +10.63% ·
**3bit_nc +20.59% (what we run today — the worst-quality preset)**. The
papers predict and upstream confirms: FP8-K + 4-bit-V is the sweet spot
("compress V, spend bits on K"). On the 4090 (SM89) `key_fp8` uses native
e4nv, and k8v4 is the *original* scope of the P40 grouped kernel (upstream
measured +16–27% grouped on k8v4). Cost: k8v4 slot = 356 B vs 3bit_nc 198 B
per head-token → 1.8× KV bytes (≈22.8 KB/token vs 12.7 KB/token over 16
layers × 4 KV heads; at 120K ctx that's 2.7 GB vs 1.5 GB — check headroom,
trivial at 25K).
→ **Bake-off priority: k8v4 first, then 4bit_nc, k3v4_nc; 3bit_nc is the
memory-floor arm, not the default candidate.** Score with a reasoning task
(GSM8K-style), not PPL alone — the catastrophic failures are task-visible
(the vLLM TQ PR's 0% gsm8k incident was exactly symmetric-V-too-low).

**(b) Boundary protection: actionable-now via CLI, but protect the right
layers and expect small returns at our context.** Found in the pinned image:

- Machinery exists: `--kv-cache-dtype-skip-layers` (`config/cache.py`,
  applied in `attention.py` — listed layer indices fall back to
  `kv_cache_dtype="auto"` = bf16, get their own standard-shaped cache
  allocation/group with `skip_page_size_padded` alignment).
- Auto-boundary (`TurboQuantConfig.get_boundary_skip_layers`, n=2 first+last,
  "empirically required for k3v4_nc/3bit_nc — without it GSM8K drops ~30
  points on Qwen3-4B") **explicitly returns [] for hybrids** — i.e. our
  3bit_nc deployment runs with NO boundary protection today, and upstream
  disabled it for our architecture class on a heuristic, not on data.
- Per-KV-ordinal boundary for our model = raw indices **3, 7, 59, 63**
  (first-2 + last-2 full-attention layers) — exactly the fix the paper's
  addendum says their own code lacks. `--kv-cache-dtype-skip-layers 3,7,59,63`
  does it today. Cost: those 4 layers at bf16 (1024 B/head-token vs 198) →
  total KV ≈ **2.0×** of uniform 3bit_nc. A cheaper probe: `3,63` only
  (first+last, ≈1.5×).
- Per-layer *mixed TQ presets* (e.g. k8v4 boundary + 3bit_nc middle — the
  paper's actual LA-V7 shape, ~1.3× instead of 2×) = **needs-patch**:
  `TurboQuantConfig` is built once from the global `--kv-cache-dtype` string
  and `TurboQuantAttentionBackend.get_kv_cache_shape` bakes one slot size;
  you'd need a layer→preset map and per-group TQ configs. The KV-cache-group
  framework itself supports heterogeneous per-layer specs (it already
  partitions GDN vs attention vs skip layers), so the patch is contained to
  the TQ config/backend plumbing. vllm#38479's `TQ_boundary_layers` cites
  this same paper — check its state before writing our own.
- Tempering: the paper's boundary gains dilute at 16K+ and are V-side;
  skip-layers protects K+V at 16-bit (coarse but free). Our 4–25K single-shot
  window is exactly where boundary effects are strongest, so it's worth one
  ladder arm — but if arm (a) lands on k8v4, boundary protection likely adds
  ~nothing (K already FP8, V4 ≈ "free" per the asymmetric data). Run it as a
  3bit_nc/k3v4_nc rescue arm, not on k8v4.

**Work item.** Add two arms to the planned dtype bake-off: (i)
`turboquant_3bit_nc` + `--kv-cache-dtype-skip-layers 3,7,59,63`; (ii) same
with `3,63`. Compare against k8v4 / 4bit_nc / k3v4_nc / 3bit_nc uniform on
GSM8K-style + tool-call canary + TPS at 4/8/16/25K. No code. Decision rule:
mixed-preset patch only if a skip-layers arm beats k8v4 on
quality-per-KV-byte.

---

## 3. block-size-experiment.md — turbo3 storage block 32→128

**Claim.** llama.cpp's turbo3 stores a 2-byte norm per 32-element block, but
norm correction is computed per 128-element rotation group → 3 of 4 norms
are redundant. bs=128 gives 3.125 b/val (5.12× vs 4.57×), byte-identical PPL
(mechanically guaranteed — same math), speed flat on M5 / +3–7% decode on
bandwidth-poor M2.

**Evidence quality.** High for what it claims — the PPL identity is
arithmetic, not empirics — but it's a fix for **llama.cpp's storage layout**.

**Applicability: irrelevant — vLLM already stores at the "bs = head_dim"
granularity.** Read from the pinned image
(`v1/attention/ops/triton_turboquant_store.py` + `turboquant/config.py`):
K = packed MSE indices + **one fp16 norm per whole head_dim(256) vector**
(98 B key slot); V = packed uniform indices + **one fp16 scale + one fp16
zero per whole vector** (100 B value slot). There are no 32-element
sub-blocks anywhere; the redundancy the paper removes does not exist in
vLLM's layout. Effective bits at D=256: K 3.06, V 3.125 — already at/below
the paper's bs=128 target. Nothing to adopt; no ladder implication.

---

## 4. dflash-self-draft-investigation.md — 31 spec-decode experiments on GDN hybrids

**Claim/findings.** (1) Single-hidden-state draft heads cap at ~36%
acceptance no matter the scale; (2) DFlash-style multi-layer KV-injection
architecture works (flat 13–15%/position) but needs ~300K gradient steps +
800K target-generated tokens + ~500M params to reach useful acceptance —
**not trainable on consumer hardware** (their full z-lab-recipe replications
landed at 7.1% and 0.6%/mode-collapse); (3) GDN verify cost floor: on a
75%-GDN model, N-token verify costs 0.25+0.75N → **max theoretical speedup
1.33×** with sequential GDN verification (tree-aware GDN kernels reportedly
break this: 3.43–5.46×, external claim); (4) partial-rejection Mamba-state
corruption is THE integration killer; tape-replay (record δ/k/g per step,
replay accepted steps only) is the fix shape; (5) pre-trained z-lab drafts:
3.5-draft-on-3.6 = 67% accept → 0.69× (slower!); z-lab's own 3.6 draft (WIP)
= 75% accept → **0.92× — still a slowdown** on their stack; (6) built-in MTP
layer = "the best deal in speculative decoding" (80%+ acceptance, zero
training — spiritbuun).

**Evidence quality.** n=1 experimenter, Apple/MLX stack, but unusually
honest (28 documented failures) and the architecture math is
platform-independent. No Ada/SM89 data; the 1.33× ceiling and acceptance
economics transfer, the absolute tok/s numbers don't.

**Applicability to our z-lab drafter revival: mostly a warning — don't
spend GPU time on the vLLM path.**
- Our arch is 75% GDN (48/64 GDN layers): the paper's verify-cost ceiling
  applies. With MTP n=3 already accepted-and-running, a DFlash drafter's
  *marginal* win must clear (a) the 1.33× sequential-GDN verify ceiling
  unless vLLM grows tree/batch-aware GDN verify, and (b) draft-model forward
  cost — the paper shows 75% acceptance still netted 0.92× on a hybrid.
- Inside vLLM, EAGLE/DFlash on Qwen3-Next-family is blocked by DeltaNet
  rollback (`docs/UPSTREAM.md` → vllm#39931) — which is *exactly* the
  Mamba-state-corruption problem this paper documents; their tape-replay
  design is the shape of the upstream fix. Until vllm#39931 moves, the
  z-lab drafter has no vLLM integration point at all.
- Draft-version pairing matters: a 3.5-era drafter on a 3.6 target degraded
  to 67%/0.69×. Verify which target our downloaded z-lab drafter was trained
  against before *any* benchmarking.
- **Ada/sm_89 note the paper doesn't have but our tracker does:** DFlash is
  currently **broken on Ada** via beellama (gibberish, cameronr 4090 repro;
  club-3090 commit 96149ba2 gates 4090 off DFlash — see
  `~/shared/TODO-PLANNING.md` 07-13 mining notes). So even the "A/B it via
  beellama" fallback is closed on this box until that's fixed. Same notes:
  EAGLE3 lost to MTP decisively (−25%, accept 0.41 vs 0.74).
- **Verdict: park the DFlash revival entirely on this box.** vLLM path
  blocked (vllm#39931), beellama path arch-gated on 4090, training path
  infeasible (this paper), and MTP n=3 already holds the spec-decode slot.
  The paper's actionable residue for the vLLM lane is nil beyond "MTP was
  the right call" — at most re-check the spec-sweep n=2 vs n=3 question.

---

## 5. One-liners (skimmed)

- **block-selector-sparse-attention.md** — Quest-style learned pre-SDPA K-block
  selection, 1.16–1.73× decode at 16–48K, but WIP on MLX-Swift/Metal with
  unmerged PRs and stack-specific kernel conclusions; nothing portable to our
  Triton lane that upstream vLLM sparse-attention work doesn't already cover
  — skip.
- **longctx-1m-and-triattention.md** — 1M-token RAG/eviction-rescue *service*
  glue (MRCR 0.688 at 1M); application-layer, irrelevant to a 4–25K
  single-shot kernel lane.
- **turbo4-resurrection.md** — 7 bugs + ablation proving QJL correction is
  actively harmful; redesigned 4-bit polar KV beats q4_0 — already absorbed
  by upstream vLLM (config.py: "QJL is intentionally omitted", norm-correction
  `_nc` presets exist); indirectly supports weighting 4-bit arms in the
  ladder.
- **eden-optimal-s-revisit.md** — EDEN's optimal-S is real but second-order
  (~1% once rotation is right); matched-norm (what vLLM's vec_norm scheme
  effectively does) already captures the first-order fix — no action.

---

## Priority-ordered work queue (digest verdicts only)

| # | Item | Type | Expected win | Cost |
|---|------|------|--------------|------|
| 1 | Dtype ladder re-ordered by K-dominance: **k8v4 as lead arm** (native FP8 on SM89 + original P40 grouped scope), then 4bit_nc, k3v4_nc; 3bit_nc = memory floor | actionable-now, no code | large quality headroom (+20.6%→+1.2% PPL band per upstream), grouped-kernel speed retained | bench time only; 1.8× KV bytes vs 3bit |
| 2 | Boundary arms in the same bake-off: `--kv-cache-dtype-skip-layers 3,7,59,63` (and `3,63`) on 3bit_nc/k3v4_nc — upstream disabled auto-boundary for hybrids without data | actionable-now, no code | rescues aggressive presets if we must stay ≤3.5 b/val; ~0 on k8v4 | bench time; 1.5–2.0× KV bytes |
| 3 | Sparse-V tile-skip ported into `_tq_grouped_decode_stage1` (reuse PN26 gate math + envs; skip test conservative under online softmax) | small patch in our own Genesis kernel | low single-digit % at 25–32K, ~0 below 8K; kill if tile-level skip rate <30% at 25K | contained; parity harness exists |
| 4 | Per-layer mixed TQ presets (k8v4 boundary + 3bit middle) | needs-patch (TQ config/backend; check vllm#38479 first) | only if #2 wins on quality-per-byte | moderate |
| 5 | DFlash / z-lab drafter revival in vLLM | blocked (vllm#39931 = the paper's Mamba-corruption problem) + economics (GDN verify ceiling 1.33×, 75% accept was still 0.92× on a hybrid) | — | park; A/B via beellama if at all |
| 6 | Block-size 32→128 | irrelevant — vLLM already stores per-head_dim-vector norms/scales | — | — |

---

# Part 2 — llama-cpp-turboquant fork → vLLM port questions (scope add, 2026-07-13)

Fork cloned depth-1 to `scratchpad/llama-cpp-turboquant` (tip `4503343`,
"docs: point prebuilt table at tqp-v0.3.0").

## (A) Is vLLM's `turboquant_4bit_nc` the rehabilitated turbo4? — **YES, post-resurrection. No codebook port needed.**

Side-by-side:

| | Fork rehabilitated turbo4 (KV) | Pinned-image vLLM 4-bit K |
|---|---|---|
| File | `ggml/src/ggml-common.h` L305–329 (`TURBO4_USE_4BIT=1` default: `block_turbo4_0` = fp16 norm + 64 B nibble-packed indices, 66 B/128 vals) | `turboquant/config.py` + `centroids.py` + `triton_turboquant_store.py` (per-head_dim-256 vector: packed 4-bit indices + fp16 vec_norm) |
| Codebook | 16 **Lloyd-Max optimal** centroids for N(0,1) post-WHT (`[-2.733 … +2.733]`, PolarQuant) | 16 **Lloyd-Max optimal** centroids solved for N(0,1/d) post-rotation (`centroids.py::solve_lloyd_max` — same table, scaled by 1/√d because vLLM quantizes the *unit-normalized* vector) |
| QJL residual correction | **removed** (that removal IS the resurrection; legacy 3-bit+QJL kept behind `TURBO4_USE_4BIT=0`) | **never present** — config.py docstring: "QJL is intentionally omitted: community consensus (5+ independent groups) found it hurts attention quality" |
| Norm correction | group-level `grp_norm/recon_norm` | `_nc` presets: re-normalize centroid vector to unit norm in-kernel (`NORM_CORRECTION` in both store/decode) |

The resurrection paper's 7 bugs were llama.cpp-specific (shared set_rows
template writing turbo3 packing into turbo4 blocks; QJL matmul missing; QJL
signs never computed) — none of those code paths exist in vLLM. Upstream's
4-bit K is already the post-resurrection design: WHT-family rotation +
Lloyd-Max codebook + per-vector norm + norm-correction, no QJL.

**Residual gap (low-priority needs-patch, not a codebook port):** the fork's
rehabilitated turbo4 applies rotated PolarQuant to **V as well**; vLLM's V is
plain uniform min/max (no rotation, `_store_quantized_value`). TheTom's note
on the vllm#38479 fix ("their fix uses more bits; our approach achieves
better compression at equal quality via WHT rotation") + "value precision is
the quality bottleneck" (varjoranta) suggest a rotated Lloyd-Max V path could
lift the 3-bit-V arm's quality at equal bits. Only worth specing if the
Part-1 ladder shows the V axis is our binding quality constraint.

## (B) TQ4_1S / TQ3_1S weight formats + the dp4a kernel — honest read: **no speed case vs gptq_marlin on our box; VRAM case is real but small (~2 GB) and expensive to port.**

**What the formats are** (`ggml/src/ggml-common.h` L346–368):
- `TQ4_1S`: WHT-rotated 4-bit weights, 16 Lloyd-Max centroids, block 32, dual
  fp16 half-block scales → 20 B/32 vals = **5.0 bpw**.
- `TQ3_1S`: same recipe at 3 bits (8 centroids), 16 B/32 vals = **4.0 bpw**.

**What "240 vs 68" actually was** (`scripts/autoresearch/track-weight/
program.md`, `baseline.json`, `history.jsonl`): an autonomous kernel-tuning
loop on **RTX 5090 (sm_120)**, Qwen2.5-7B-Instruct TQ4_1S, `llama-bench -p 0
-n 128` decode-only. The 68–69 t/s baseline was **their own naive fused V8
kernel** (f32 activation, 1 float FMA per element — not a real W4 kernel).
The loop's dp4a rewrite (`ggml/src/ggml-cuda/mmvq-tq.cu`: WHT pre-rotate
activation → q8_1, fixed int8 centroid LUT in registers, dp4a inner loop,
multi-token weight-reuse for ne1≤8, runtime →q8_0 scratch + cuBLAS for
ne1>8) plateaued at **~220–226 t/s** (history best 225.9). The honest
comparator on the same rig is llama.cpp's native q4_0 mmvq at **267 t/s** —
i.e. the tuned dp4a kernel reaches ~84% of a real W4 kernel, in *their own
program's* framing ("Target: close the gap to q4_0 (267 t/s)"). Note also
TQ3_1S has **no dp4a path at all** — it runs the scalar-half kernel
(`mmvq-tq.cu` L411: dp4a is TQ4_1S-only; TQ3_1S + AMD fall back to scalar).

**Port assessment vs gptq_marlin (SM89, bs=1, our shapes):** bs=1 decode is
weight-bandwidth-bound, so bytes-moved decides:
- TQ4_1S = 5.0 bpw **> GPTQ g128 ≈ 4.15–4.25 bpw**. It moves ~20% more bytes
  than what marlin reads at ~90% BW efficiency. There is **no regime on our
  box** — bs=1 or the MTP verify batch (≤4, covered by marlin's small-batch
  path) — where a 5.0 bpw dp4a kernel at ≤84%-of-q4_0 efficiency beats
  marlin on 4.2 bpw. dp4a's 4×-MAC density is irrelevant when the roof is
  HBM, and the fork's own ablation agrees (centroid LUT ≠ bottleneck; f32
  activation bandwidth + FMA density were).
- TQ3_1S = 4.0 bpw ≈ byte-parity with GPTQ — and it only has the scalar
  kernel. Speed-parity at best, likely worse.

**The VRAM angle, with real numbers:** the artifact-level delta on our model
is **15.4 GB (TQ3 weights) vs 17.4 GB (our GPTQ-Pro artifact) → ~2 GB KV
headroom** (already logged as a low-prio TRIAL in
`~/shared/TODO-PLANNING.md` L101). Naive bpw math (27B × (4.25−4.0)/8)
explains <1 GB; the rest of the 2 GB is what GPTQ-Pro keeps at bf16
(embeddings/head/kept tensors), so the delta is artifact-specific, not a
format property. Weight-quality caveat from the fork's own paper
(`weight-compression-tq4.md`): rotated-TQ weights win on **attention
tensors** but *lose to Q4_K on ffn_down* (their shipped "Config I" is a
hybrid for this reason) — a uniform TQ3_1S 27B is not a validated quality
point.

**What a port would involve:** out-of-tree vLLM **quantization plugin**
(template: `varjoranta/turboquant-vllm` — "TurboQuant+ KV cache compression
for vLLM", the plugin-packaged precedent from the 04-14 recon sweep) —
register a quant config/method via vLLM's plugin entry-point, a
weights converter (bf16 → WHT+Lloyd-Max GGUF-free layout; our GPTQ artifact
can't be transcoded, it's already 4-bit), a Triton/CUDA fused GEMV (port of
`mmvq-tq.cu` incl. the activation pre-rotation stage), plus a cuBLAS/marlin
fallback for prefill (ne1>8 in the fork = dequant-to-q8 + cuBLAS — on vLLM
that's a dequant-to-bf16 + cutlass path we'd have to build). That's a
multi-kernel port for ~2 GB of headroom and probable speed/quality
regression. **Verdict: not worth GPU or engineering time while KV dtype
(k8v4 ladder, Part 1 #1) and context cap are cheaper levers for the same
headroom; revisit only if 120K-ctx VRAM becomes binding AND the ladder
lands on a fat KV dtype.**

## (C) turbo2 + Boundary V — the exact aggressive-arm recipe

From `src/llama-kv-cache.cpp` L310–370 (production fork) + the two papers:

- **Mode: `TURBO_LAYER_ADAPTIVE=7` (LA-V7), auto-enabled** whenever
  `type_v == turbo2 && n_layer ≥ 8` (opt-out `TURBO_LAYER_ADAPTIVE=0`).
- **K: q8_0 on ALL layers, always** (8.5 bpw; K is never boundary-tiered —
  boundary-K protection was tested and does not help, layer-0 K quantizes
  *better* due to extreme norms).
- **V: q8_0 on first-2 + last-2 layers; turbo2 (2.5 bpw, 4 Lloyd-Max
  centroids, block 128) on all middle layers.**
- Rejected tiers, for the record: mode 5 (turbo4 boundaries — equal on
  phi-4, worse on sensitive models), mode 6 (last-8 — worse than 2+2), 4+4
  width (not stable across models/contexts), f16 boundaries (no better than
  q8_0). 2+2 q8_0 is the settled recipe.
- **Known bug that carries over to us:** the fork's boundary test is still
  raw layer index (`il < 2 || il >= n_layer - 2`) — on a hybrid like ours it
  would touch 1–2 of 16 real KV layers. The KV-layer-**ordinal** fix the
  paper's addendum calls for remains unimplemented in the fork. Any vLLM
  translation must use full-attention ordinals: **first-2 + last-2 KV layers
  = raw indices 3, 7, 59, 63** on Qwen3.6-27B.
- Headline result at this operating point (MoE frontier follow-up, cited in
  layer-aware paper): 7.53× V compression, PPL within 1% of q8_0 at
  512c–32K on Qwen3.5-35B MoE.

**vLLM translation of the aggressive arm** (extends Part-1 item #4): nearest
native analogue = per-layer mixed presets patch with `value_quant_bits=2`
middle + higher-V boundary — but vLLM has **no 2-bit V path at all** today
(store/decode kernels implement VQB ∈ {3,4} only), so the true turbo2 arm
needs BOTH the per-layer preset map AND a VQB=2 uniform (or rotated-polar,
see (A) residual) kernel path. Digest verdict: keep this as the *last* rung
of the per-layer item — only reachable if #2's skip-layers arms show
boundary protection pays on our hybrid at 4–25K, and #4's mixed-preset
plumbing exists anyway. Do not build a VQB=2 kernel on spec.

---

Sources inspected: papers under `scratchpad/tqplus/docs/papers/`; fork
`scratchpad/llama-cpp-turboquant` (`ggml-common.h`, `ggml-cuda/mmvq-tq.cu`,
`src/llama-kv-cache.cpp`, `scripts/autoresearch/track-weight/*`); pinned image
`vllm/vllm-openai:nightly-9e57de71…` → `triton_turboquant_store.py`,
`triton_turboquant_decode.py`, `turboquant/config.py`, `turboquant_attn.py`,
`v1/kv_cache_interface.py`, `engine/arg_utils.py`, `config/cache.py`,
`layers/attention/attention.py`; Genesis lane →
`_genesis/kernels/tq_grouped_decode.py` (P40 v7.73),
`_genesis/kernels/triton_turboquant_decode_sparse_v.py` (PN26b, default-OFF);
`diagnostics/tq-lane/PROFILE-ANALYSIS.md`, `README.md`.
