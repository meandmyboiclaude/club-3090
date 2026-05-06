# Gemma 4 26B-A4B MoE — on 2× RTX 3090

**Run [Gemma 4 26B-A4B](https://blog.google/technology/developers/gemma-4/) — a 128-expert MoE with ~4B active params, vision, and tool calling — on 2× RTX 3090s.**

> ⭐ v0.7.3 onboarding target. AWQ production path validated; Intel AutoRound INT4 blocked on Ampere (Marlin K-dim alignment).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven config picks. TL;DR:

| Config | Max ctx | Narr / Code TPS | Best for |
|--------|---------|----------------|----------|
| `vllm/gemma4-26b-a4b-tp2` | 32K | 139 / 139 | General-purpose, vision + tools |

Run via:
```bash
bash scripts/launch.sh --variant vllm/gemma4-26b-a4b-tp2
```

---

## Models

- **Target:** [`Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound`](https://huggingface.co/Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound) (~16 GB, MoE expert layers quant-mixed)
- **AWQ alternative:** cyankiwi AWQ-4bit weights (validated production path on Ampere)
- **Draft:** TBD (Google's `gemma-4-26B-A4B-it-assistant` available, compose pending)

## Key details

| Aspect | Notes |
|--------|-------|
| **Arch** | MoE — 128 experts × 8 active, ~4B active params |
| **Quants** | AWQ-4bit (production), Intel AutoRound INT4 (Ampere-blocked for now) |
| **KV** | bfloat16 |
| **Vision** | ✅ Yes (off by default in base compose for first-boot validation) |
| **Tools** | ✅ `--tool-call-parser gemma4` |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- [vLLM PR #40886](https://github.com/vllm-project/vllm/pull/40886) — compressed-tensors MoE key remapping (vendored for AWQ path)
- [Discussion #67](https://github.com/noonghunna/club-3090/discussions/67) — first Ampere consumer cross-rig data thread
