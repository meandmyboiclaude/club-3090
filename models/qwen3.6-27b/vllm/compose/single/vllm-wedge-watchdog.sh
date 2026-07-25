#!/bin/bash
# Wedge watchdog (BUG-072 class): health lies during the abort-wedge — a REAL
# completion is the only signal that can't lie. v2 (2026-07-18): hybrid.
#   PASSIVE first: diff vllm:request_success_total counters from /metrics.
#     - stop/length advanced since last tick  => real traffic is completing:
#       healthy, skip the probe (removes the ~15% probe share of request stats
#       and the per-2min seq-slot tax during active hours).
#     - ONLY abort advanced                   => wedge-suspect: confirm with the
#       real probe immediately (faster detection than probe-only cadence; a
#       client-disconnect abort alone can't trigger a restart — the probe stays
#       the ground truth).
#     - nothing advanced (idle) or metrics unreadable => active probe as before
#       (idle is exactly where passive is blind).
#   Probe prompt carries the [wedge-probe] tag; otelcol drops those spans so
#   Phoenix stays clean. Two consecutive bad probes => restart via the SYSTEMD
#   UNIT that owns the container (see UNIT_FOR_PORT below).
#
# [2026-07-25] Recovery goes through `systemctl restart <unit>`, NOT a raw
# `podman restart <container>`. A raw podman restart reuses the SAME container
# and so bypasses the unit's BUG-128 boot guard (ExecStopPost `podman rm -f` +
# ExecStartPost endgame8020-boot-guard.sh) — which is precisely the
# stale-degraded-boot class that guard exists to close: a mid-boot engine death
# gets silently restarted onto warm Triton/compile caches and the container
# comes up degraded while reporting healthy. Going through the unit gives the
# documented stop -> rm -f -> fresh create -> guard sequence. The unit's
# TimeoutStartSec (2700s on :8020) bounds the blocking wait; this service has
# TimeoutStartUSec=infinity so nothing kills us mid-restart, and the timer
# (OnUnitActiveSec) simply will not re-trigger while we are still running.
PORT="${1:-8020}"
STATE=/run/vllm-wedge-watchdog.state          # exists => last probe was bad
SNAP=/run/vllm-wedge-watchdog.metrics.snap    # "good_sum abort_sum" from last tick

# Port -> owning systemd unit. The container name alone is not enough: recovery
# must run the unit so its boot guard executes. An unmapped port restarts
# NOTHING (the old code hardcoded the PROD container regardless of $PORT, so a
# `vllm-wedge-watchdog.sh 8021` invocation would have bounced prod).
unit_for_port() {
  case "$1" in
    8020) echo vllm-endgame-8020.service ;;   # container vllm-qwen36-endgame
    8021) echo vllm-tcbench-8021.service ;;   # container vllm-tcbench-8021
    *)    return 1 ;;
  esac
}

probe() {
  curl -s -m 45 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6","messages":[{"role":"user","content":"[wedge-probe] Say OK."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys
try: print(json.load(sys.stdin)['choices'][0]['finish_reason'])
except Exception: print('parse-error')" 2>/dev/null
}

# metrics_sums: prints "good_sum abort_sum" summed across engine/model labels;
# non-zero exit if the counter family is absent/unreadable.
metrics_sums() {
  curl -s -m 5 "http://localhost:${PORT}/metrics" | awk '
    /^vllm:request_success_total\{/ {
      v = $NF
      if (index($0, "finished_reason=\"stop\"") || index($0, "finished_reason=\"length\"")) good += v
      else if (index($0, "finished_reason=\"abort\"")) abort += v
      seen = 1
    }
    END { if (!seen) exit 1; printf "%d %d\n", good, abort }'
}

# skip if container is mid-(re)start
curl -s -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "health down — restart policy owns this; skip"; rm -f "$STATE" "$SNAP"; exit 0; }

if cur=$(metrics_sums); then
  cur_good=${cur% *}; cur_abort=${cur#* }
  if [ -f "$SNAP" ]; then
    read -r prev_good prev_abort < "$SNAP"
    if [ "$cur_good" -lt "$prev_good" ] || [ "$cur_abort" -lt "$prev_abort" ]; then
      echo "counters went backwards (engine restarted) — resnapshot, probing"
    elif [ "$cur_good" -gt "$prev_good" ]; then
      echo "$cur_good $cur_abort" > "$SNAP"
      rm -f "$STATE"
      exit 0   # passive: real traffic completing => healthy, no probe
    elif [ "$cur_abort" -gt "$prev_abort" ]; then
      echo "passive wedge-suspect: only aborts advanced ($prev_abort->$cur_abort) — confirming with real probe"
    fi
    # else: idle (nothing advanced) => probe
  fi
  echo "$cur_good $cur_abort" > "$SNAP"
else
  echo "metrics unreadable — falling back to active probe"
fi

fr=$(probe)
if [ "$fr" = "stop" ] || [ "$fr" = "length" ]; then rm -f "$STATE"; exit 0; fi
echo "wedge-suspect probe: finish=${fr:-none}"
if [ -f "$STATE" ]; then
  rm -f "$STATE"
  if ! unit=$(unit_for_port "$PORT"); then
    echo "wedge confirmed on :${PORT} but no systemd unit is mapped for that port — NOT restarting (map it in unit_for_port)"
    exit 1
  fi
  echo "second consecutive bad probe -> systemctl restart ${unit} (wedge; runs the unit's boot guard)"
  rm -f "$SNAP"   # counters reset with the fresh container; don't diff across it
  if systemctl restart "$unit"; then
    echo "restart of ${unit} completed"
  else
    echo "restart of ${unit} FAILED (rc=$?) — see: systemctl status ${unit}"
    exit 1
  fi
else
  touch "$STATE"
fi
