# ik_llama.cpp — the advanced-quant engine

**Role on this stack:** the engine you reach for when you want **newer, higher-quality-per-bit quants** than mainline llama.cpp ships — specifically the **IQK imatrix family** (`IQ4_KS`, `IQ5_KS`, …) that exists *only* in this fork. It's a llama.cpp fork (ikawrakow), so it inherits llama.cpp's cliff-immune memory model and broad hardware support, then adds a co-designed quant + kernel stack on top.

> **In one line:** llama.cpp's robustness + fork-exclusive IQK quants + fused CUDA kernels → on our rig, **~62 narr / ~69 code TPS** single-3090 (decode ~64 / ~72) with **MTP**, clean to **262K context on one card** — ~+18-20% over the shipped `llamacpp/mtp` Q4_K_M path at equal-or-better quality.

For *what the quants actually are* and how IQK compares to k-quants / i-quants / AWQ, see **[../QUANTIZATION.md](../QUANTIZATION.md)**. For the cross-engine overview see **[../INFERENCE_ENGINES.md](../INFERENCE_ENGINES.md)**.

---

## TL;DR

- **Image:** `ghcr.io/ikawrakow/ik-llama-cpp:cu13-server` (cu13 = CUDA 13.x; matches our 13.2 host driver). Official, digest-pinnable. A `cu12` tag exists for older drivers.
- **Compose:** `models/qwen3.6-27b/ik-llama/compose/single/iq4ks-mtp.yml` (text) + `iq4ks-mtp-vision.yml` (vision).
- **Model:** ubergarm `Qwen3.6-27B-MTP-IQ4_KS.gguf` (IQK imatrix quant, built-in MTP head).
- **Interface:** same `--jinja` + `--reasoning on|off` server contract as mainline llama.cpp — so the stack-wide thinking-off policy works unchanged.

```bash
MODEL_DIR=/your/models docker compose \
  -f models/qwen3.6-27b/ik-llama/compose/single/iq4ks-mtp.yml up -d
curl http://localhost:8020/v1/models
```

---

## Why pick ik_llama.cpp over mainline llama.cpp?

Both are cliff-immune (same ggml flat allocator — see [LLAMA_CPP.md](LLAMA_CPP.md) "Why llama.cpp doesn't hit the prefill cliffs"). ik_llama adds, on top:

1. **IQK imatrix quants (fork-exclusive).** `IQ4_KS` / `IQ5_KS` use refined non-linear grids + an importance matrix + **kernels co-designed for those grids**. Net: better quality-per-bit than mainline `Q4_K_M`, and *faster* than mainline i-quants because the dequant path is hand-tuned. `IQ4_KS` is ~15.1 GB vs `Q4_K_M`'s ~17 GB on Qwen3.6-27B — smaller weights leave room for **262K context** on a single 24 GB card.
2. **Fused CUDA kernels** — `--merge-qkv` (fused QKV projection), `--merge-up-gate-experts` (MoE, no-op on dense), and a fast IQK dequant path.
3. **`-khad` (Hadamard K-cache)** — a Hadamard transform on the K-cache that improves quantized-KV accuracy, so you keep more quality at `q4_0`/`q8_0` KV.
4. **MoE-on-consumer tooling** — `-ser` (smart expert reduction) + on-the-fly MLA tensors make big-MoE-over-VRAM (DeepSeek/Kimi-class) practical, an alternative to ktransformers.
5. **Qwen3.x MTP on `main`** — MTP merged on the fork's main branch (mainline still has it as open [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)), so no PR-branch building.

The cost: it's a **fork** with no tagged releases (rolling `main`, smaller community) — a second engine image to track vs the clean `ggml-org/llama.cpp:server-cuda`. That's the trade for the IQK quants. Use mainline `llamacpp/mtp` for the conservative production path; use ik_llama when you want the cutting-edge quant.

---

## Pros

- **IQK imatrix quants** — fork-exclusive, best quality-per-bit available in the GGUF world.
- **Cliff-immune** (inherits llama.cpp's ggml allocator — no Cliff 1/2 GDN OOM).
- **262K context on a single 3090** with MTP (q4_0 KV ~5 GB; verified, verify-stress 7/7 incl. 91K needle).
- **Broad hardware** — CUDA (incl. CC 7.0 Volta), ROCm, Apple Metal, Intel, CPU (same as mainline).
- **MoE-over-VRAM** path (`-ser` + MLA) for models bigger than your cards.
- Same `--jinja` / `--reasoning` server contract as mainline → froggeric chat template + thinking-off both work (validated on this fork; see "Gotchas").

## Cons

- **A fork to track** — rolling `main`, no tags, smaller community; pin by image digest.
- **GGUF-only** — no safetensors / vLLM-class continuous batching (single-stream `-np 1` is the sweet spot here).
- **Tooling lag** — some mainline llama.cpp server flags differ or arrive later (e.g. `--alias`, `--no-mmproj-offload` are in source builds but not every published image tag).
- Not the production default — that's vLLM (dual, max TPS) and mainline llama.cpp (single, conservative).

---

## Quick recipe

### 1. Pull the engine image
```bash
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu13-server   # cu12-server for CUDA 12 drivers
```

### 2. Get an IQK GGUF
```bash
hf download ubergarm/Qwen3.6-27B-GGUF Qwen3.6-27B-MTP-IQ4_KS.gguf \
  --local-dir $MODEL_DIR/qwen3.6-27b-gguf/ubergarm-mtp-iq4ks
# Always SHA256-verify multi-GB GGUFs after download.
```

### 3. Launch (via our compose)
```bash
MODEL_DIR=$MODEL_DIR docker compose \
  -f models/qwen3.6-27b/ik-llama/compose/single/iq4ks-mtp.yml up -d
```
Defaults: q4_0 KV, 131K ctx, MTP n=2, froggeric v19 template, thinking-off. Overrides:
- **Max context (262K):** `UBATCH_SIZE=512 CTX_SIZE=262144` (q4_0 KV is the default; ~21.5 GB / 24).
- **Higher KV fidelity:** `KV_TYPE=q8_0` (caps ~131-200K — q8_0 KV @262K OOMs).
- **Reasoning on:** `REASONING=on` (pair with `MTP_DRAFT_N_MAX=5 DRAFT_P_MIN=0.5` — reasoning text drafts deeper).

---

## Tuning levers (ik-specific)

| Flag | What it does | Default here |
|---|---|---|
| `-ctk` / `-ctv` | KV cache quant (`q4_0` / `q8_0` / `f16`) — biggest VRAM lever | `q4_0` |
| `-khad` / `--k-cache-hadamard` | Hadamard transform on K-cache → better quantized-KV accuracy | on |
| `--merge-qkv` (`-mqkv`) | Fused QKV projection | on |
| `--merge-up-gate-experts` (`-muge`) | Fused MoE up/gate (no-op on dense models) | — |
| `-ser N,f` | Smart expert reduction (big-MoE) | — |
| `--multi-token-prediction` + `--draft-max` / `--draft-p-min` | Built-in MTP spec-decode | n=2 / p-min 0.0 |
| `-b` / `-ub` | Batch / micro-batch — `-ub 512` unlocks higher ctx at ~2-4% TPS | 4096 / 1024 |

---

## ik_llama-specific gotchas

- **froggeric chat template works here** (unlike mainline). The mainline `llama.cpp` note that froggeric "silently suppresses `--reasoning off`" is a *mainline* issue — on ik_llama, froggeric v19 + `--reasoning off` suppresses thinking cleanly **and** renders tool-calls correctly (validated 2026-05-21). So the ik composes ship froggeric v19; mainline composes stay native.
- **The published image lags source on a few flags.** `--alias` and `--no-mmproj-offload` exist in from-source builds but not the `cu13-server` tag we pull — don't copy a from-source config verbatim. Check `llama-server --help` in the container.
- **Single-stream is the regime.** `-np 1` — this isn't a continuous-batching server. For multi-tenant, use vLLM.
- **MoE flags are no-ops on dense models** (`--merge-up-gate-experts`, `-ser` do nothing on dense Qwen3.6-27B; they matter for the MoE catalog).

---

## Measured on this stack (Qwen3.6-27B, IQ4_KS + MTP, single 3090)

| Metric | Value | vs shipped `llamacpp/mtp` (Q4_K_M) |
|---|---|---|
| Narrative TPS (wall / decode) | ~62.8 / 64.4 | +18-20% |
| Code TPS (wall / decode) | ~68.7 / 71.9 | |
| Max context (1× 3090) | **262K** (q4_0 KV) | vs 131K |
| verify-stress | 7/7 (incl. 91K Cliff 2 needle) | parity |
| Quality 8-pack | ≈ parity with the vLLM autoround path | — |
| toolcall-15 | 60% — **no regression** (froggeric recovers the formatting) | — |

> Bench: canonical prompt, 3 warmup + measured runs, q4_0 KV / 131K / MTP n=2 / thinking-off, 370 W cap. See [../../BENCHMARKS.md](../../BENCHMARKS.md).

---

## When ik_llama.cpp is the right pick

- You want the **best quality-per-bit GGUF** (IQK imatrix) on a single card.
- You want **262K context on one 3090** with MTP decode speed.
- You're running a **big MoE that doesn't fit VRAM** and want `-ser`/MLA instead of ktransformers.
- You're **experimenting with newer quants** generally — this is the stack's advanced-quant track.

## When to use something else

- **Production multi-tenant / max dual-card TPS** → vLLM (`dual` / `dual-turbo`).
- **Conservative single-card, mainline image, no fork** → llama.cpp (`llamacpp/mtp`).
- **Apple Silicon** → either llama.cpp (Metal) or ik_llama (also Metal).

## See also
- [../QUANTIZATION.md](../QUANTIZATION.md) — what IQK / imatrix / k-quants actually are
- [LLAMA_CPP.md](LLAMA_CPP.md) — the mainline sibling (shared cliff-immunity)
- [../INFERENCE_ENGINES.md](../INFERENCE_ENGINES.md) — full cross-engine comparison
- [../../BENCHMARKS.md](../../BENCHMARKS.md) — measured TPS across engines/configs
