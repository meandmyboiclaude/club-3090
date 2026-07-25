#!/usr/bin/env bash
# Watch podman's event stream and record every vLLM container boot's patch set.
#
# Deliberately NOT an ExecStartPost on the vLLM units. A hook living inside the
# thing it audits shares that thing's fate: the next compose rewrite, unit
# regeneration, or "clean up the drop-ins" pass silently removes it, and the
# auditing stops exactly when a config churn makes it most necessary. Hooking
# podman's own event stream means this fires on ANY start path -- systemd,
# podman-compose, a manual `podman run`, an auto-restart after a crash -- and
# nothing in the vLLM config can switch it off by accident.
#
# Records into Postgres (vllmops) via vllm-patch-record.py, then diffs against
# the previous boot and raises an alert file if patches vanished.

set -uo pipefail

RECORD=/home/user/shared/tools/vllm-patch-record.py
CAPTURE=/home/user/shared/tools/capture-vllm-boot.sh
ABORTREC=/home/user/shared/tools/vllm-abort-record.py
ALERT=/home/user/shared/PATCH-REGRESSION-ALERT.md
MATCH="${VLLM_WATCH_MATCH:-vllm}"
SETTLE="${VLLM_WATCH_SETTLE:-150}"   # let the boot finish before reading the log
ABORT_SCAN_SEC="${VLLM_ABORT_SCAN_SEC:-600}"  # hourly abort-rate rows converge per scan

log() { echo "[patch-watcher] $*"; }

handle() {
  local ctr="$1"
  log "start detected: $ctr — settling ${SETTLE}s before capture"
  sleep "$SETTLE"

  # Archive the full boot log first; the DB row is a summary, the log is evidence.
  [ -x "$CAPTURE" ] && "$CAPTURE" "$ctr" >/dev/null 2>&1

  local out rc
  out=$(python3 "$RECORD" record "$ctr" 2>&1); rc=$?
  log "$out"
  if [ $rc -ne 0 ]; then
    log "record FAILED for $ctr (log may be untrustworthy — see message above)"
    return
  fi

  out=$(python3 "$RECORD" diff "$ctr" 2>&1); rc=$?
  log "$out"
  if [ $rc -ne 0 ]; then
    {
      echo "# vLLM PATCH REGRESSION — $(date -Is)"
      echo
      echo "container: \`$ctr\`"
      echo
      echo '```'
      echo "$out"
      echo '```'
      echo
      echo "Inspect: \`python3 $RECORD history $ctr\`"
      echo "Slow drift (lost several boots ago): \`python3 $RECORD missing $ctr\`"
      echo
      echo "If intentional, no action needed — the next boot becomes the new comparison point."
    } > "$ALERT"
    log "REGRESSION -> $ALERT"
  else
    rm -f "$ALERT"
  fi
}

# BUG-127 visibility: the abort class is HTTP-200-invisible (finish_reason=abort,
# 0 completion tokens, runner counts 0 errors), so nothing event-driven ever sees
# it. A periodic journald scan banks per-container per-hour abort/oom/request
# counts into vllmops.abort_hourly; anubis's vllm-aborts module alerts on the rate.
if [ -x "$ABORTREC" ]; then
  (
    while :; do
      out=$(python3 "$ABORTREC" scan --match "$MATCH" 2>&1) \
        && log "abort-scan: $out" \
        || log "abort-scan FAILED: $out"
      sleep "$ABORT_SCAN_SEC"
    done
  ) &
else
  log "abort-scan disabled: $ABORTREC missing or not executable"
fi

log "watching podman events for containers matching '$MATCH'"
podman events --filter event=start --format '{{.Name}}' 2>/dev/null | while read -r name; do
  case "$name" in
    *${MATCH}*) handle "$name" & ;;
  esac
done
