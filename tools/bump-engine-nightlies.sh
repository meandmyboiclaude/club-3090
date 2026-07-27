#!/usr/bin/env bash
#
# Bump vLLM nightly pins in EngineProfile YAML.
#
# The vLLM composes consume VLLM_NIGHTLY_SHA at launch time; the canonical pin
# lives in scripts/lib/profiles/engines/<engine-id>.yml -> install.spec.
#
# Usage:
#   bash tools/bump-engine-nightlies.sh --engine vllm-nightly-mtp --sha <sha>
#   bash tools/bump-engine-nightlies.sh --engine vllm-nightly-mtp,vllm-nightly-full --sha <sha>
#   bash tools/bump-engine-nightlies.sh --all --sha <sha>
#
# Options:
#   --repo <image-repo>   Default: vllm/vllm-openai
#   --no-tests           Skip post-bump profile tests

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_IDS=""
SHA=""
IMAGE_REPO="vllm/vllm-openai"
RUN_TESTS=1

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE_IDS="$2"; shift 2 ;;
    --engine=*) ENGINE_IDS="${1#--engine=}"; shift ;;
    --sha) SHA="$2"; shift 2 ;;
    --sha=*) SHA="${1#--sha=}"; shift ;;
    --repo) IMAGE_REPO="$2"; shift 2 ;;
    --repo=*) IMAGE_REPO="${1#--repo=}"; shift ;;
    --all) ENGINE_IDS="vllm-nightly-mtp,vllm-nightly-dflash,vllm-nightly-full"; shift ;;
    --no-tests) RUN_TESTS=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 2 ;;
  esac
done

if [[ -z "$ENGINE_IDS" || -z "$SHA" ]]; then
  echo "ERROR: --engine/--all and --sha are required." >&2
  usage 2
fi
if [[ ! "$SHA" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: --sha contains unsupported characters: $SHA" >&2
  exit 2
fi
if [[ ! "$IMAGE_REPO" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "ERROR: --repo contains unsupported characters: $IMAGE_REPO" >&2
  exit 2
fi

IFS=',' read -ra ENGINES <<< "$ENGINE_IDS"

python3 - "$ROOT_DIR" "$IMAGE_REPO" "$SHA" "${ENGINES[@]}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
image_repo = sys.argv[2]
sha = sys.argv[3]
engine_ids = sys.argv[4:]
profile_dir = root / "scripts/lib/profiles/engines"
new_spec = f"{image_repo}:nightly-{sha}"

for engine_id in engine_ids:
    path = profile_dir / f"{engine_id}.yml"
    if not path.exists():
        raise SystemExit(f"unknown engine profile: {engine_id}")
    text = path.read_text(encoding="utf-8")
    if "type: vllm" not in text:
        raise SystemExit(f"{engine_id} is not a vLLM engine profile")
    if "method: docker_image" not in text:
        raise SystemExit(f"{engine_id} is not a docker-image engine profile")
    updated, count = re.subn(
        r"(?m)^(\s*spec:\s*)(?:[A-Za-z0-9._/-]+:nightly-[A-Za-z0-9._-]+)(\s*)$",
        rf"\1{new_spec}\2",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"{engine_id}: could not find docker nightly install.spec")
    path.write_text(updated, encoding="utf-8")
    print(f"{engine_id}: spec -> {new_spec}")
PY

for engine_id in "${ENGINES[@]}"; do
  python3 "${ROOT_DIR}/scripts/lib/profiles/launch_compat.py" resolve-engine-pin --engine-id "$engine_id" --format shell
done

if [[ "$RUN_TESTS" -eq 1 ]]; then
  bash "${ROOT_DIR}/scripts/tests/test-profiles-compat.sh"
fi
