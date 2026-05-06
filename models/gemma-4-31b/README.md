# Gemma 4 31B — on 2× RTX 3090

**Run [Gemma 4 31B](https://blog.google/technology/developers/gemma-4/) — with vision, tool calling, and MTP/DFlash spec-decode — on 2× RTX 3090s.**

> ⚠️ **Single-card boot OOMs on 24 GB Ampere** regardless of KV format. Needs ≥32 GB single-card (validated on RTX 5090 by [@apnar](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042)).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven config picks. TL;DR:

| Config | Max ctx | Narr / Code TPS | Best for |
|--------|---------|----------------|----------|
| `vllm/dual` (default) | 32K | 106 / 141 | General-purpose, vision + tools |
| `vllm/int8` | **262K** | 95 / 126 | Long-context via INT8 PTH KV (PR #40391) |
| `vllm/dflash` | 32K | 105 / 177 | Peak code TPS (z-lab DFlash n=7) |
| `vllm/awq` | 118K | varies | AWQ-4bit weights (cyankiwi quant) |

Run via:
```bash
bash scripts/launch.sh --variant vllm/gemma-mtp     # MTP default
bash scripts/launch.sh --variant vllm/gemma-int8    # long-context
bash scripts/launch.sh --variant vllm/gemma-dflash  # peak code TPS
```

---

## Models

- **Target:** [`Intel/gemma-4-31B-it-int4-AutoRound`](https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound) (~21.2 GB, vision preserved)
- **Draft (MTP):** [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) (0.5B / 927 MB BF16)
- **Draft (DFlash):** z-lab Gemma 4 DFlash n=7 block-diffusion drafter

## Key details

| Aspect | Notes |
|--------|-------|
| **Quants** | Intel AutoRound INT4 (default), cyankiwi AWQ-4bit |
| **KV** | bfloat16 (32K) or INT8 PTH via vendored PR #40391 (262K) |
| **Drafter** | MTP n=3 (Google official) or DFlash n=7 (z-lab) |
| **Vision** | ✅ Yes |
| **Tools** | ✅ `--tool-call-parser gemma4` |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- [vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745) — Gemma 4 MTP support (merged)
- [vLLM PR #40391](https://github.com/vllm-project/vllm/pull/40391) — INT8 PTH KV
- [Discussion #67](https://github.com/noonghunna/club-3090/discussions/67) — first Ampere consumer cross-rig data
