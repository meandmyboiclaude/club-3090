#!/usr/bin/env bash
# list-image-pins.sh — audit which container images each club-3090 compose pins,
# what patches each compose mounts, and which composes share a pin.
#
# Engine-agnostic — works for vLLM, llama.cpp, SGLang, Luce, xtransformers, or
# any engine whose composes use `image: <repo>:<tag>`. Output groups composes
# by `<repo>:<tag>` so you can see at a glance:
#   - which composes share a pin (= safe to bump together)
#   - which patches each compose adds (= migration cost per compose)
#   - which engines have pin-drift (multiple tags of the same repo)
#
# Usage:
#   bash scripts/maintenance/list-image-pins.sh                # default: club-3090 root auto-detected
#   REPO=/path/to/club-3090 bash scripts/maintenance/list-image-pins.sh
#   bash scripts/maintenance/list-image-pins.sh --filter vllm  # only show vllm/* repos
#
# When to run:
#   - Before bumping an engine pin (see docs/NIGHTLY_BUMP_RUNBOOK.md)
#   - After landing a new compose, to verify it joined an existing pin not a new one
#   - Periodically to surface accumulated pin sprawl across engines

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FILTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --filter) FILTER="$2"; shift 2 ;;
    --filter=*) FILTER="${1#--filter=}"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "$REPO/models" ]; then
  echo "error: $REPO does not look like a club-3090 repo (no models/ dir)" >&2
  exit 1
fi

CYAN=$'\033[0;36m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
NC=$'\033[0m'

echo ""
if [ -n "$FILTER" ]; then
  echo "${CYAN}═══ Image pins across ${REPO} (filter: $FILTER) ═══${NC}"
else
  echo "${CYAN}═══ Image pins across ${REPO} ═══${NC}"
fi
echo ""

# Build temp file: image_repo_tag<TAB>compose-relpath<TAB>patch-list
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

shopt -s nullglob
# Walk all engine compose dirs: vllm, llama-cpp, sglang, luce, xtransformers, etc.
# Pattern: models/<model>/<engine>/compose/<topology>/<file>.yml
for f in "$REPO"/models/*/*/compose/*/*.yml "$REPO"/models/*/*/compose/*.yml; do
  [ -f "$f" ] || continue
  # Extract first `image: <repo>:<tag>` line. Match repo/tag flexibly so we
  # cover stable tags (server-cuda), nightly hashes, semver, etc.
  image=$(grep -E "^[[:space:]]*image:[[:space:]]+" "$f" 2>/dev/null \
    | head -1 \
    | sed -E 's/^[[:space:]]*image:[[:space:]]+//' \
    | tr -d '"' | awk '{print $1}' || true)
  if [ -z "$image" ]; then
    continue  # No image: line, skip (might be an override-only compose)
  fi
  if [[ "$image" == *'VLLM_NIGHTLY_SHA'* ]]; then
    engine_profile=$(grep -E "^[[:space:]]*#[[:space:]]*Engine-profile:[[:space:]]*" "$f" 2>/dev/null \
      | head -1 \
      | sed -E 's/^[[:space:]]*#[[:space:]]*Engine-profile:[[:space:]]*//' \
      | awk '{print $1}' || true)
    engine_file="$REPO/scripts/lib/profiles/engines/${engine_profile}.yml"
    if [ -n "$engine_profile" ] && [ -f "$engine_file" ]; then
      resolved_spec=$(grep -E "^[[:space:]]*spec:[[:space:]]+" "$engine_file" 2>/dev/null \
        | head -1 \
        | sed -E 's/^[[:space:]]*spec:[[:space:]]+//' \
        | tr -d '"' | awk '{print $1}' || true)
      if [ -n "$resolved_spec" ]; then
        image="${resolved_spec}#${engine_profile}"
      fi
    fi
  fi
  # Apply filter if set
  if [ -n "$FILTER" ] && [[ "$image" != *"$FILTER"* ]]; then
    continue
  fi
  patches=$(grep -oE "patches/[a-zA-Z0-9_./-]+" "$f" 2>/dev/null \
    | sed -E 's|^(patches/[a-zA-Z0-9_-]+).*|\1|' \
    | sort -u | tr '\n' ',' | sed 's/,$//' || true)
  rel="${f#$REPO/}"
  printf "%s\t%s\t%s\n" "$image" "$rel" "${patches:-none}" >> "$tmp"
done

if [ ! -s "$tmp" ]; then
  echo "${YELLOW}(no compose files matched)${NC}"
  exit 0
fi

# Per-image summary, sorted by repo+tag
echo "${CYAN}── Pin distribution ──${NC}"
printf "%-72s %s\n" "Image" "Composes"
printf "%-72s %s\n" "-----" "--------"
awk -F'\t' '{print $1}' "$tmp" | sort | uniq -c | sort -rn \
  | awk '{cnt=$1; $1=""; sub(/^ /, ""); printf "%-72s %d\n", $0, cnt}'

# Highlight engines with pin DRIFT (same repo, multiple tags)
echo ""
echo "${CYAN}── Pin-drift detection ──${NC}"
drift=$(awk -F'\t' '{print $1}' "$tmp" | awk -F: '{print $1}' | sort -u | while read repo; do
  ntags=$(awk -F'\t' -v r="$repo" '$1 ~ "^"r":" {split($1, a, "#"); print a[1]}' "$tmp" | sort -u | wc -l)
  if [ "$ntags" -gt 1 ]; then
    printf "  ${YELLOW}%s${NC} has %d distinct tags pinned\n" "$repo" "$ntags"
  fi
done)
if [ -n "$drift" ]; then
  echo "$drift"
  echo "  ${YELLOW}^^${NC} consider consolidating to one tag per repo (see docs/NIGHTLY_BUMP_RUNBOOK.md)"
else
  echo "  ${GREEN}✓ No drift — every repo pins exactly one tag.${NC}"
fi

# Per-pin compose breakdown
echo ""
echo "${CYAN}── Per-pin compose + patch breakdown ──${NC}"
images=$(awk -F'\t' '{print $1}' "$tmp" | sort -u)
for img in $images; do
  echo ""
  echo "${GREEN}● $img${NC}"
  awk -F'\t' -v img="$img" '$1==img {
    printf "  %-58s  patches=%s\n", $2, $3
  }' "$tmp" | sort
done

# Patch surface ranking
echo ""
echo "${CYAN}── Patch surface (highest = most expensive to migrate when pin bumps) ──${NC}"
echo ""
awk -F'\t' '$3 != "none" {
  n = split($3, a, ",")
  print n "\t" $2 "\t" $3
}' "$tmp" | sort -rn | head -10 | awk -F'\t' '{
  printf "  %d patches  %s\n             (%s)\n", $1, $2, $3
}'

echo ""
echo "${YELLOW}Tip:${NC} See docs/NIGHTLY_BUMP_RUNBOOK.md for the bump procedure."
echo ""
