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
# [2026-06-26 dev424 promote] crash fix = tq_buffer_pool DISABLED + PN75; util 0.91.
# [2026-06-26 CUDAGRAPHS] PIECEWISE cudagraphs ON (dropped --enforce-eager) -> ~+50% decode
#   TPS. Made stable on TQ3+MTP+GDN-hybrid by PN79 (fixed-size TQ decode scratch, backport of
#   vllm#46067 — the prior config crashed with a device-side assert at ~round 13 of the killer
#   stress because the growable WorkspaceManager scratch baked into captured decode graphs was
#   freed+realloc'd under load) + P66 (capture-size ÷ filter) + P78 (.tolist() guard). util
#   0.91->0.90 for cudagraph capture headroom (KV 152K->134K; pool only 0.09 GiB). Validated:
#   PN79 20/20 killer-stress rounds clean (RestartCount 0), functional 6/6, +50% decode.
#   Rollback (cudagraph only): re-add --enforce-eager + util 0.91 + drop the cudagraph env/args
#   in the compose (PN79 is harmless under eager), restart.
# [2026-06-26 PN76] retired PN73/PN73T (vendored legacy parsers) — now run upstream's
#   streaming parser engine + PN76 engine-level deferred tool-call commit + PN72.
#   Validated live: validate_bump.py 6/6 (test D 537 chars) + streaming tool-call gate 5/5.
#   Rollback (parser only): uncomment PN73/PN73T in the compose entrypoint, or checkout
#   tag pre-pn76-3f5a1e17 (commit 3133409).
# [2026-07-07 PROMOTE] dev799 (nightly-69715823) promoted to validated after
#   full gate: 6/6 functional, PN76 streaming 5/5, bench +6..47% vs dev424,
#   15-round BUG-028 soak clean, /rerank live (PN81). Carries PN80/82 (upstream
#   crash fixes) + PN81/83. Genesis at 8e15d1f (dev799 KNOWN_GOOD + PN8 re-anchor).
# Rollback (full build): tag validated-qwopus-3f5a1e17 (dev424) + image
#   vllm-rollback:3f5a1e17-20260707; older tag validated-qwopus-b53b1c7 (2dc5938) + vllm-rollback:b53b1c7-20260625.
# [2026-07-13 PROMOTE] dev1060 (nightly-9e57de71) promoted to validated after full
#   gate: verify-full 6/6, verify-stress 8/8, bench idle 75.6/99.0 (+1.7%/+0.6% vs
#   same-day dev799 baseline). Brings 481e481b (hybrid partial prefix-cache) +
#   51878e5b (KV layout repack). PN80 re-anchored dual-form; genesis re-anchors
#   PN8/PN12/P34/P83 on reanchor-dev1060 (d5e5604..930c3ff), guards.py pin added.
#   First-boot KV ValueError during swap = transient VRAM-release race (restart
#   policy recovers); not a pin defect.
# Rollback (full build): tag validated-qwopus-69715823 (dev799) + image
#   vllm-rollback:69715823-20260713; older: validated-qwopus-3f5a1e17 (dev424) +
#   vllm-rollback:3f5a1e17-20260707.
TAG=validated-qwopus-9e57de71
PIN_IMG=nightly-9e57de7197f234f9d9187715d96e07e007048c0f
PIN_VER=0.23.1rc1.dev1060+g9e57de719

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
