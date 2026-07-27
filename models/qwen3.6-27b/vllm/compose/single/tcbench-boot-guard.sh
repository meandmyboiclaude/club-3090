#!/bin/bash
# tcbench-boot-guard.sh — wired as ExecStartPost on vllm-tcbench-8021.service.
#
# BUG-128 (2026-07-25): the first boot after a pin/patch change that alters
# kernel code paths pays cold Triton-JIT compilation during vLLM's memory
# profile run. The transient peak (~2.3 GiB observed on nightly-0ba2aa35) can
# fail the KV-memory check; the engine dies and podman's `unless-stopped`
# policy restarts the SAME container, which then passes on the now-warm
# host-mounted Triton cache — yielding a silently DEGRADED boot (idempotent
# re-patch, wrong vllmops record: the 112-patch / sub=0 class). The same
# signature was seen at the 07-13 dev1060 swap and misread as a "transient
# VRAM-release race (restart policy recovers)".
#
# This guard converts that failure mode into a self-healing clean boot:
# wait for the container to settle; if any mid-boot engine death occurred
# (RestartCount > 0), bank the log and perform ONE clean recycle
# (stop -> rm -f -> compose up = fresh container on warm caches), which is
# exactly the documented manual recovery. Never bricks boot (always exits 0);
# logs every decision to the journal. Covers the whole stale-degraded-boot
# class (BUG-126 note, 112-patch class), not just cold-JIT.
#
# BUG-172 / review 2026-07-27 M2: RestartCount>0 alone cannot tell "engine died
# from cold JIT" (recycle helps) from "a /fixes applier refused to apply"
# (recycle can NEVER help — /fixes is a read-only bind mount, so the fresh
# container re-runs the same applier against the same drifted anchor and dies
# identically, costing 2x WAIT_S and destroying the container for nothing).
# The banked log carries the discriminator: every applier prints
# "[patch_<name>] FATAL ..." before exit 1. Grep for it and skip the recycle.
set -u
C=vllm-tcbench-8021
COMPOSE_DIR=/home/user/club-3090/models/qwen3.6-27b/vllm/compose/single
COMPOSE_FILE=tcbench8021.yml
WAIT_S=${TCBENCH_BOOT_GUARD_WAIT_S:-900}
# Matches the /fixes applier convention: LOG = "[patch_<module>]" (35 appliers)
# plus patch_oom_resilience.py's "[oom_resilience]".
FATAL_RE='\[(patch_[A-Za-z0-9_]+|oom_resilience)\] FATAL'

log() { echo "[tcbench-boot-guard] $*"; }

# Poll until: container healthy (echo healthy), a mid-boot restart is seen
# (echo restarted), or the window closes (echo timeout).
wait_state() {
  local deadline=$(( $(date +%s) + $1 ))
  local r h
  while (( $(date +%s) < deadline )); do
    r=$(podman inspect "$C" --format '{{.RestartCount}}' 2>/dev/null || echo "")
    h=$(podman inspect "$C" --format '{{.State.Health.Status}}' 2>/dev/null || echo "")
    if [[ "$r" =~ ^[0-9]+$ ]] && (( r > 0 )); then
      echo restarted; return
    fi
    if [[ "$h" == healthy ]]; then
      echo healthy; return
    fi
    sleep 10
  done
  echo timeout
}

outcome=$(wait_state "$WAIT_S")
case "$outcome" in
  healthy)
    log "clean boot: healthy with RestartCount=0"
    ;;
  restarted)
    log "WARN mid-boot engine death detected (RestartCount>0, BUG-128 class)"
    log "banking container log"
    ts=$(date +%Y%m%d-%H%M%S)
    banked="/home/user/shared/tcbench-bootguard-recycle-$ts-container.log"
    podman logs "$C" > "$banked" 2>&1 \
      || log "WARN could not bank container log"

    # Cheap discriminator (M2): an applier FATAL is not the cold-JIT class.
    if grep -Eq "$FATAL_RE" "$banked" 2>/dev/null; then
      log "PATCH-FATAL a /fixes applier refused to apply — NOT recycling"
      grep -Ehm 5 "$FATAL_RE" "$banked" 2>/dev/null \
        | while IFS= read -r line; do log "PATCH-FATAL   $line"; done
      log "PATCH-FATAL a recycle cannot fix anchor drift (/fixes is a read-only"
      log "PATCH-FATAL bind mount; a fresh container re-runs the same applier)."
      log "PATCH-FATAL fix the applier under /home/user/club-3090/fixes, then"
      log "PATCH-FATAL restart the unit. Banked log: $banked"
      exit 0
    fi

    log "no applier FATAL in the log — ONE clean recycle (stop -> rm -> fresh up)"
    podman stop -t 30 "$C" >/dev/null 2>&1 || true
    podman rm -f "$C" >/dev/null 2>&1 || true
    if ! (cd "$COMPOSE_DIR" && podman-compose -f "$COMPOSE_FILE" up -d); then
      log "ERROR recycle compose-up failed — leaving unit state as-is"
      exit 0
    fi
    outcome2=$(wait_state "$WAIT_S")
    if [[ "$outcome2" == healthy ]]; then
      log "recycle OK — clean fresh boot achieved (warm caches)"
    else
      log "ERROR recycle outcome=$outcome2 — NOT recycling again; investigate"
      log "(a second failure means warm caches did not fix it: real config/memory problem)"
    fi
    ;;
  timeout)
    log "WARN container neither healthy nor restarted within ${WAIT_S}s — no action taken"
    ;;
esac
exit 0
