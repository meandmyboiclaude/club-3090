#!/usr/bin/env bash
#
# pull.sh — v0.8.0 Pull-Gate orchestrator (PR #147, STEP P4).
#
# Derives an HF repo via [A], gates it through the LOCKED 6-stratum abort
# taxonomy (deriver-errors → --profile-like → [C0] → [C2a] → eligibility →
# [B]→[C1] → Path-A [D] dry-run), and on a curated, [D]-emittable, gate-
# passing Path A run hands the validated registry key to the existing #141
# generator for real emission. Path B (--dry-run / any non-curated slug)
# prints a §7-caveated verdict and NEVER calls [D] / downloads.
#
# Honest by construction (design §1): every non-eligible / non-pass outcome
# hard-stops with a precise structured reason; only `exact × fits-clean`
# reaches `proceed` silently (§4.1). `--force-download` is a no-op + notice
# this phase (download/telemetry deferred to the Loop phase).
#
# Usage:
#   scripts/pull.sh <hf-slug> --profile-like <COMPOSE_REGISTRY-key> [opts]
#
#   # Path A — curated pull-and-emit:
#   scripts/pull.sh Lorbus/Qwen3.6-27B-int4-AutoRound \
#       --profile-like vllm/minimal --out /tmp/qwen.yml
#
#   # Path B — universal evaluate (never emits/downloads):
#   scripts/pull.sh some-org/Some-Llama-7B --profile-like vllm/minimal --dry-run
#
#   # Failure on-ramp — submit a captured failed pull (a SEPARATE, consented
#   # verb: the ONLY step that touches the network, and only after an
#   # explicit y; reuses the shipped dedup; needs no slug/--profile-like):
#   scripts/pull.sh --submit-last            # the most-recent capture
#   scripts/pull.sh --submit <capture-dir>   # an explicit bundle dir
#
# Opts: --yes  --force-download  --experimental-arch  --trust-remote-code
#       --hf-home DIR  --out FILE (Path A)  --hardware SM (override nvidia-smi)
#
# All decision logic lives in scripts/lib/profiles/pull.py (this is a thin
# argv pass-through, matching the generate-compose.sh / diagnose-profile.sh
# pattern). Exit: 0 = download-eligible / clean verdict; 3 = needs a flag
# (confirm→proceed / advisory); 2 = honest hard-stop; 64 = usage.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT_DIR}/scripts/lib/profiles/pull.py" "$@"
