# Quantization — a field guide for the club-3090 community

Quantization is how a 27B model that would need ~54 GB at FP16 fits in 24 GB of VRAM. This page explains the **quant families** you'll see in the wild, what actually differs between them, and which ones this stack ships — including the **IQK imatrix quants** that exist only in [ik_llama.cpp](engines/IK_LLAMA.md).

> **The one idea to take away:** at the same *bits-per-weight*, not all quants are equal. The two levers that separate good from bad are (1) **non-linear levels** that match the weight distribution and (2) **calibration** (an "importance matrix") that spends bits where they matter. The best quants use both.

See also: [GLOSSARY.md](GLOSSARY.md) · [DTYPE_MATRIX.md](DTYPE_MATRIX.md) (KV/compute dtypes) · [engines/IK_LLAMA.md](engines/IK_LLAMA.md).

---

## 1. The vocabulary

- **bpw (bits per weight):** the headline number. FP16 = 16 bpw. A "4-bit" quant is ~4-4.5 bpw once you count the per-block scale/zero-point overhead. Lower bpw = smaller file = more context room, but more quality risk.
- **Block / group:** quants don't store one scale for the whole tensor — they chunk weights into blocks (e.g. 32 weights) and store a scale per block. Smaller blocks = finer = more accurate, but more overhead.
- **imatrix (importance matrix):** a calibration pass over real text that records *which weights matter most* for the model's outputs, so the quantizer protects those and compresses the rest harder. "i-quant" / "IQ" prefixes signal imatrix use.
- **Weight quant vs KV-cache quant:** two independent knobs. One shrinks the *model*; the other shrinks the *context* (see §5). You pick both.

---

## 2. The GGUF ladder (llama.cpp + ik_llama.cpp)

GGUF is the llama.cpp-family weight format. Roughly in order of quality-per-bit (worst → best at a given bpw):

| Family | Examples | Calibrated? | Where | Notes |
|---|---|---|---|---|
| **Legacy** | `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0` | ❌ | mainline | Simple round-to-nearest, one scale/block. `Q8_0` is still a great near-lossless choice; the low-bit legacy ones are superseded. |
| **K-quants** | `Q3_K_M`, **`Q4_K_M`**, `Q5_K_M`, `Q6_K` | ❌ (data-free) | mainline | Mixed precision per tensor-type + 2-level block scales. The mainstream default. **`Q4_K_M` is what our shipped `llamacpp/mtp` runs.** Good, but data-free — no calibration. |
| **i-quants** | `IQ2_XXS` … `IQ3_M`, `IQ4_XS` | ✅ imatrix | mainline | Non-linear lattice codebooks + importance matrix. Clearly better quality-per-bit than k-quants, *especially below 4 bpw*. Slightly slower dequant than k-quants. |
| **IQK quants** ⭐ | **`IQ4_KS`**, `IQ5_KS`, `IQ4_K`, `IQ2_K` … | ✅ imatrix | **[ik_llama.cpp](engines/IK_LLAMA.md) only** | Refined grids + imatrix + **kernels co-designed for those grids**. Best quality-per-bit in the GGUF world *and* fast (the dequant path is hand-tuned). Fork-exclusive. |

**The progression that matters:** `Q4_K_M` (data-free) → `IQ4_XS` (imatrix, mainline) → `IQ4_KS` (imatrix + co-designed kernels, ik fork). Each step is better quality at similar bpw. Our shipped `llamacpp/mtp` is at the *first* rung (`Q4_K_M`); the [ik_llama track](engines/IK_LLAMA.md) is at the *last* (`IQ4_KS`).

---

## 3. What "imatrix" actually buys you

A data-free quant treats every weight as equally important and rounds uniformly. But in a trained model, a small fraction of weights carry most of the signal. An **importance matrix** is computed by running calibration text through the model and measuring how much each weight influences activations. The quantizer then:
- protects high-importance weights (more bits / closer grid points), and
- compresses low-importance weights harder.

Result: at 4 bpw, an imatrix quant loses noticeably less quality than a data-free one — and the gap *widens* as you go lower (at 2-3 bpw, imatrix is the difference between usable and broken). The cost is a one-time calibration step when *building* the quant; inference is the same speed.

> **Calibration corpus matters.** An imatrix calibrated on chat+code+tool-calling preserves those skills; one calibrated on Wikipedia may quietly drop tool-call formatting. This is exactly the kind of thing that shows up in our 8-pack quality tests (see [QUALITY_TEST.md](QUALITY_TEST.md)).

---

## 4. The vLLM / safetensors side (not GGUF)

vLLM and SGLang don't use GGUF — they load **safetensors** with these quant schemes:

| Quant | Bits | Calibrated? | Notes |
|---|---|---|---|
| **AutoRound** ⭐ | INT4 | ✅ (sign-gradient) | Intel's method; **what our shipped `vllm/dual` runs** (`qwen3.6-27b-autoround-int4`). Strong 4-bit quality. |
| **AWQ** | INT4 | ✅ (activation-aware) | Protects salient channels by inspecting activations. Widely available. |
| **GPTQ** | INT3/4/8 | ✅ (second-order) | Older, well-supported; AWQ/AutoRound usually edge it at 4-bit. |
| **FP8 (e4m3/e5m2)** | 8 | ❌ | Native on Hopper+; on Ampere it's emulated. We use **fp8 mostly for the KV cache**, not weights. |
| **bitsandbytes** | 4/8 | ❌ | Easy/on-the-fly; lower quality-per-bit than AWQ/AutoRound. |

These are conceptually the same idea as imatrix i-quants (calibrate, protect what matters) in a different ecosystem. There is **no GGUF↔safetensors interchange** — a quant is tied to its engine family.

---

## 5. KV-cache quantization (a separate knob)

Independent of the weight quant, you can quantize the **KV cache** — this is what sets your max context, not your model quality:

| KV type | Bits | Engine | Notes |
|---|---|---|---|
| `f16` | 16 | all | Lossless, biggest. Rarely needed. |
| `q8_0` | 8 | llama.cpp / ik | Near-lossless; good default when context is moderate. |
| `q4_0` | 4 | llama.cpp / ik | Halves KV vs q8_0 → enables **262K on one 3090** (ik IQ4_KS). Tiny quality cost. |
| `fp8_e5m2` | 8 | vLLM | Our `vllm/dual` default. |
| **TQ3 (TurboQuant)** | 3 | vLLM (Genesis) | 3-bit KV — beats fp8 on long-context memory; powers our `dual-turbo`. See [TQ3_MTP_GENESIS.md](TQ3_MTP_GENESIS.md) + [CLIFFS.md](CLIFFS.md). |
| `-khad` (modifier) | — | **ik only** | Hadamard transform on the K-cache → recovers accuracy lost to KV quantization, so you keep quality at q4_0/q8_0. |

---

## 6. Engine × quant support

| Quant family | vLLM | mainline llama.cpp | ik_llama.cpp | SGLang |
|---|---|---|---|---|
| K-quants (`Q4_K_M`…) | ❌ | ✅ | ✅ | ❌ |
| i-quants (`IQ4_XS`…) | ❌ | ✅ | ✅ | ❌ |
| **IQK (`IQ4_KS`…)** | ❌ | ❌ | ✅ **only** | ❌ |
| AutoRound / AWQ / GPTQ | ✅ | ❌ | ❌ | ✅ |
| FP8 weights | ✅ | ❌ | ❌ | ✅ |

---

## 7. Why doesn't every GGUF repo ship IQK?

If IQK is the best quality-per-bit, why are most community GGUFs still `Q4_K_M`?

1. **It's fork-locked.** IQK quants run *only* on ik_llama.cpp. A `Q4_K_M` runs on mainline llama.cpp, Ollama, LM Studio, LocalAI, Jan — everything. Quant authors optimize for reach.
2. **Kernel co-design.** IQK's quality comes partly from kernels written *for* its grids. Porting that to mainline isn't a small patch, and upstreaming has been slow.
3. **Inertia + tooling.** `Q4_K_M` is the well-trodden default; build pipelines, docs, and "recommended download" buttons all point at it.

So IQK is a deliberate "I'll run the fork to get the better quant" choice — which is exactly the niche the [ik_llama track](engines/IK_LLAMA.md) fills on this stack.

---

## 8. What this stack ships (and why)

| Path | Quant (weights) | KV | Rationale |
|---|---|---|---|
| `vllm/dual` | AutoRound INT4 | fp8_e5m2 | Production dual-card; deepest Qwen3-Next feature support |
| `vllm/dual-turbo` | AutoRound INT4 | **TQ3** | Max throughput + long context (3-bit KV) |
| `llamacpp/mtp` | **Q4_K_M** | q4_0 | Conservative, mainline image, cliff-immune single-card |
| `ik-llama/iq4ks-mtp` ⭐ | **IQ4_KS** (imatrix) | q4_0 + `-khad` | Advanced-quant track: best quality-per-bit + 262K single-card |

**Rule of thumb for your own rig:**
- Tightest VRAM / lowest bpw → reach for an **imatrix quant** (`IQ4_XS` mainline, or `IQ4_KS` on ik_llama), not a data-free `Q4_K_M`.
- Want maximum quality-per-bit and willing to run the fork → **ik_llama + IQK**.
- Multi-tenant / vision / tools at scale → **vLLM + AutoRound**.
- "Just works everywhere, no fork" → mainline **llama.cpp + Q4_K_M**.

---

## See also
- [engines/IK_LLAMA.md](engines/IK_LLAMA.md) — the engine that unlocks IQK
- [INFERENCE_ENGINES.md](INFERENCE_ENGINES.md) — engine comparison
- [DTYPE_MATRIX.md](DTYPE_MATRIX.md) — compute/KV dtype matrix
- [CLIFFS.md](CLIFFS.md) + [TQ3_MTP_GENESIS.md](TQ3_MTP_GENESIS.md) — KV-cache quant deep-dives
- [BENCHMARKS.md](../BENCHMARKS.md) — measured quality + TPS per quant/engine
