# Archived gemma-4-26b-a4b vLLM composes

Archived 2026-06-06 (#326). Superseded by the AWQ-on-stable-v0.22.0 wiring:
the live slugs are `vllm/gemma-26ba4b-single` (AWQ tp=1) + `vllm/gemma-26ba4b-dual`
(AWQ tp=2 + MTP), both on `vllm-stable`.

| Orig path | Why archived |
|---|---|
| `single/autoround-int4-mixed/bf16.yml` | AutoRound INT4-mixed is **Ampere-dead** (uint8b128 → no sm_86 W4A16 kernel). Single-card replaced by `single/awq/base.yml`. |
| `dual/autoround-int4-mixed/bf16.yml` | Same — AutoRound-mixed Ampere-dead. |
| `dual/awq/bf16.yml` | Redundant base-AWQ dual (no MTP); the shipped dual is `dual/awq/mtp.yml` (+55% from MTP). |

Revival: `git mv` back + re-add a `compose_registry.py` entry.
