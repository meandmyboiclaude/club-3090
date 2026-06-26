#!/bin/bash
# boot-guard.sh — wired as ExecStartPre on vllm-qwen36.service.
# Guarantees the DEFAULT boot is the validated Qwopus / nightly-b53b1c7 build, even if
# someone left a repo on the wrong branch. Safe: only auto-restores a CLEAN tree;
# never bricks boot (always exits 0). Logs every decision to the journal.
#
# Validated anchors (2026-06-18 — Qwopus3.6-27B-v2-GPTQ-Pro-MTP-BF16 adoption on
# vLLM nightly-b53b1c7 / 0.23.1rc1.dev178; gptq_marlin + TQ3 KV + MTP n=3 +
# froggeric template; util 0.91 / 5-seq / 75K model-len; all PN71-74 fixes wired.
# Validation: Genesis 36 applied / 0 failed / 13 partial-warn (benign — Rust-migrated
# parser files + GDN/TQ path moves); health 200, reasoning correct, streaming
# <tool_call> preserved, MTP accepts):
#   club-3090 tree:  branch rebase-vllm-1033ffac / tag validated-qwopus-b53b1c7
#   genesis tree:    tag validated-qwopus-b53b1c7
#   image:           vllm/vllm-openai:nightly-b53b1c7ffe7aebdafd0876350f30e51d1226c92a
set -u
CLUB=/home/user/club-3090
G="$CLUB/models/qwen3.6-27b/vllm/patches/genesis"
COMPOSE="$CLUB/models/qwen3.6-27b/vllm/compose/single/tools-text-aibox.yml"
TAG=validated-qwopus-b53b1c7
PIN_IMG=nightly-b53b1c7ffe7aebdafd0876350f30e51d1226c92a
PIN_VER=0.23.1rc1.dev178+gb53b1c7ff

log() { echo "[vllm-boot-guard] $*"; }

# Restore a repo's working tree to the validated tag IF it has drifted and is clean.
restore() {
  local dir="$1" name="$2"
  local want; want=$(git -C "$dir" rev-parse "$TAG" 2>/dev/null) || { log "$name: tag $TAG missing — skip guard"; return; }
  local have; have=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  if [ "$have" = "$want" ]; then log "$name: OK on validated $(git -C "$dir" rev-parse --short HEAD)"; return; fi
  if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
    log "$name: WARN drifted to ${have:0:9} and tree is DIRTY — booting AS-IS (manual review needed)"; return
  fi
  log "$name: drifted to ${have:0:9} — auto-restoring to $TAG (${want:0:9})"
  git -C "$dir" checkout -q "$TAG" 2>&1 | sed 's/^/[vllm-boot-guard]   /'
}

restore "$CLUB" "club-3090"
restore "$G" "genesis"

# Verify the compose image pin survived (catches a reverted compose).
if grep -q "$PIN_IMG" "$COMPOSE"; then
  log "compose image pin OK ($PIN_IMG)"
else
  log "WARN compose image is NOT the validated pin — booting whatever is set:"
  grep -m1 'image:' "$COMPOSE" | sed 's/^/[vllm-boot-guard]   /'
fi

# Verify the genesis pin-gate will recognize this build.
if grep -q "$PIN_VER" "$G/vllm/_genesis/guards.py" 2>/dev/null; then
  log "genesis pin-gate allowlist contains $PIN_VER"
else
  log "WARN genesis guards.py missing pin $PIN_VER (pin-gate will warn, non-fatal)"
fi

log "guard complete — proceeding to boot"
exit 0
