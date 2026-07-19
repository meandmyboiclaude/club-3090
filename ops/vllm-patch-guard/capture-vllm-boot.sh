#!/usr/bin/env bash
# Capture a vLLM container's FULL boot log to disk, plus a triage summary.
#
# Why this exists: the containers log to journald, which is persistent but holds
# only ~21 hours here — vLLM's per-step chatter evicts everything within a day
# (45.6M total journal). Boot logs are the only record of which patches applied,
# skipped, or DRIFTed, and that record is exactly what patch-regression triage
# needs. Losing it means a later "why isn't PNxx active?" cannot be answered.
# Verified 2026-07-19: `podman logs` had already lost the boot-time patch lines
# for a container that was still running.
#
# Usage:
#   capture-vllm-boot.sh <container> [port] [--wait]
#     --wait   poll /health first (default 600s) so the capture includes the
#              whole boot, not just the first seconds
#
# Wire into a unit with:
#   ExecStartPost=-/bin/bash -c '/home/user/shared/tools/capture-vllm-boot.sh vllm-qwen36-endgame 8020 --wait &'

set -uo pipefail

CTR="${1:?usage: capture-vllm-boot.sh <container> [port] [--wait]}"
PORT="${2:-}"
WAIT=0
for a in "$@"; do [ "$a" = "--wait" ] && WAIT=1; done

OUTDIR="${VLLM_BOOTLOG_DIR:-/home/user/shared/vllm-boot-logs}"
KEEP="${VLLM_BOOTLOG_KEEP:-30}"
TIMEOUT="${VLLM_BOOTLOG_TIMEOUT:-600}"

mkdir -p "$OUTDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$OUTDIR/${CTR}-${STAMP}.log"

if [ "$WAIT" = "1" ] && [ -n "$PORT" ]; then
  deadline=$(( $(date +%s) + TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:${PORT}/health" 2>/dev/null)
    [ "$code" = "200" ] && break
    sleep 5
  done
  sleep 5   # let the last startup lines flush
fi

# journalctl, not `podman logs` — the latter returns a truncated view under the
# journald driver and silently drops boot-time lines.
#
# Scope to THIS container instance. CONTAINER_NAME= matches every incarnation of
# the name, so a container recreated N times yields N boots concatenated and
# every count comes out N× inflated (measured 2026-07-19: 162 sub-patches read
# as 486 after three recreates). Anchor on the current StartedAt.
STARTED=$(podman inspect "$CTR" --format '{{.State.StartedAt}}' 2>/dev/null)
[ -n "${STARTED:-}" ] || STARTED=$(sudo podman inspect "$CTR" --format '{{.State.StartedAt}}' 2>/dev/null)
if [ -n "${STARTED:-}" ]; then
  # podman emits "2026-07-19 10:23:12.336855471 +0200 CEST" — GNU date rejects
  # that (nanoseconds plus BOTH a numeric offset and a zone name). Strip the
  # fraction and the trailing zone name, keep the offset.
  _clean=$(printf '%s' "$STARTED" | sed -E 's/\.[0-9]+//; s/ [A-Z]{3,5}$//')
  SINCE=$(date -d "$_clean" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
fi
if [ -n "${SINCE:-}" ]; then
  journalctl "CONTAINER_NAME=${CTR}" --since "$SINCE" --no-pager -o cat > "$OUT" 2>/dev/null
else
  journalctl "CONTAINER_NAME=${CTR}" --no-pager -o cat > "$OUT" 2>/dev/null
fi

if [ ! -s "$OUT" ]; then
  podman logs "$CTR" > "$OUT" 2>&1 || sudo podman logs "$CTR" > "$OUT" 2>&1
fi

lines=$(wc -l < "$OUT")

# Triage summary — the fields that answer "did my patch actually engage?".
SUM="${OUT%.log}.summary.txt"
{
  echo "container : $CTR"
  echo "captured  : $(date -Is)"
  echo "log lines : $lines"
  echo "log file  : $OUT"
  echo
  echo "--- genesis dispatcher ---"
  grep -oE 'Genesis Results:.*' "$OUT" | tail -1
  echo
  echo "--- counts ---"
  # Count DISTINCT patches, not lines and not registry-table rows. A patch that
  # touches 3 files logs 3 "applied N hunks" lines, and the dispatcher prints a
  # full registry table at boot — counting either inflates the total badly
  # (measured 2026-07-19: line/ID counting gave 93/122/155 for a real 88).
  # House scripts prefix their output with [<name>]; that prefix is the identity.
  _house=$(grep -E '^\[[a-z0-9_-]+\]' "$OUT" | grep -iE 'applied|wired|patched' \
           | grep -oE '^\[[a-z0-9_-]+\]' | tr -d '[]' | sort -u | wc -l)
  _gen=$(grep -oE 'Genesis Results: [0-9]+' "$OUT" | tail -1 | grep -oE '[0-9]+')
  # NOTE: "how many patches" has no single answer -- these are different units.
  # Report the layers rather than one misleading total.
  _sub=$(grep -oE 'applied [0-9]+ sub-patches?' "$OUT" | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}')
  _hunk=$(grep -oE 'applied [0-9]+ hunk\(s\)' "$OUT" | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}')
  # The dispatcher prints "Genesis Results" ONCE PER LANE (legacy genesis.apply_all
  # AND the sndr v12 registry). Quoting one undercounts by more than half — that
  # mistake cost an hour on 2026-07-19. Sum every occurrence.
  _genall=$(grep -oE 'Genesis Results: [0-9]+ applied' "$OUT" | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}')
  printf 'dispatcher lanes total : %s\n' "${_genall:-0}"
  printf 'house /fixes engaged   : %s\n' "$_house"
  printf 'TOTAL ACTIVE PATCHES   : %s\n' "$(( ${_genall:-0} + _house ))"
  printf 'sub-patch applications : %s\n' "$_sub"
  printf 'direct file hunks      : %s\n' "$_hunk"
  printf 'DRIFT skipped       : %s\n' "$(grep -ci 'DRIFT skipped' "$OUT")"
  printf 'anchor not found    : %s\n' "$(grep -ci 'anchor not found\|required anchor' "$OUT")"
  printf 'partial-apply warns : %s\n' "$(grep -ci 'partial-apply' "$OUT")"
  printf 'hard failures       : %s\n' "$(grep -ciE 'FAILED to apply|patch failed' "$OUT")"
  echo
  echo "--- memory / KV ---"
  grep -oE 'GPU KV cache size:.*|Available KV cache memory:.*|Model loading took.*' "$OUT" | sort -u
  grep -oE 'Applying persisted startup plan[^.]*\.' "$OUT" | head -1
  echo
  echo "--- DRIFT / anchor problems (triage these) ---"
  grep -iE 'DRIFT skipped|required anchor .* not found' "$OUT" \
    | sed 's/^.*\] //' | cut -c1-160 | sort -u | head -25
} > "$SUM" 2>/dev/null

# Retention: keep the newest N pairs.
ls -1t "$OUTDIR"/*.log 2>/dev/null | tail -n +$((KEEP+1)) | while read -r old; do
  rm -f "$old" "${old%.log}.summary.txt"
done

echo "captured $lines lines -> $OUT"
echo "summary -> $SUM"
