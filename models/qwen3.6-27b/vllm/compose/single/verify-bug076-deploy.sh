#!/bin/bash
# BUG-076 deploy verifier — proves prod :8020 boots with EVERY patch of the
# 2026-07-11..18 week actually applied (container recreates lose nothing).
# Run after `systemctl start vllm-endgame-8020` reports healthy.
# Exit 0 = all green; every failure line starts with FAIL.
set -u
C=vllm-qwen36-endgame
PASS=0; FAILN=0
ok()   { echo "  ok    $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $1"; FAILN=$((FAILN+1)); }
L=$(mktemp); sudo podman logs "$C" > "$L" 2>&1

echo "== 1. /fixes appliers (from compose command block) =="
for m in \
  pn86-tq-prefill-continuation-guard pn87 pn88 pn89 pn90 pn91g pn92 pn93 pn94 pn95 \
  pn96-structured-output-marker-step-fsm pn98-toolcall-text-fragment-demotion \
  pn99-trace-content pn100-auto-thinking-budget pn101-answer-rescue \
  pn103-spec-entry-reconcile pn104-mamba-align-gather-clamp \
  pn105-nan-logits-abort pn106d-nan-slot-audit; do
  base=$(echo "$m" | cut -d- -f1)
  # applied | multi-file applied | idempotent skip | legitimate terminal states:
  # self-retire (upstream merged the fix) and verified/by-design no-op.
  if grep -qE "\[$base[^]]*\] ([A-Za-z0-9_./-]+: )?(applied|already applied|upstream drift|no-op by design|verified NO-OP)" "$L"; then
    ok "$base"
  else
    fail "$base applier line missing"
  fi
done

echo "== 2. Genesis dispatcher state =="
grep -qE 'PN358 compilation/cuda_graph.py[^]]*\] applied 3 sub-patches' "$L" && ok "PN358 3 hunks applied" || fail "PN358 hunks"
sudo podman exec "$C" grep -q 'genesis_pn358_should_fallback' /usr/local/lib/python3.12/dist-packages/vllm/compilation/cuda_graph.py 2>/dev/null && ok "PN358 v2 fallback code byte-present" || fail "PN358 fallback code"
grep -qE 'SKIP +P58' "$L" && ok "P58 skipped (=0, validated stack)" || fail "P58 not skipped"
GEN_APPLIES=$(grep -oE '\[Genesis Dispatcher\] APPLY [A-Z0-9n]+' "$L" | awk '{print $NF}' | sort -u | wc -l)
[ "$GEN_APPLIES" -ge 105 ] && ok "genesis fleet size $GEN_APPLIES (expect >=105)" || fail "genesis fleet only $GEN_APPLIES applies (expect >=105)"
grep -qE 'GENESIS_ENABLE_PN102_CONTRACT=1' <(sudo podman inspect "$C" --format '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}') && ok "PN102 contract env" || fail "PN102 env"
grep -qE 'P68.*(SKIP|=0)|GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=0' <(sudo podman inspect "$C" --format '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}') && ok "P68 off" || fail "P68 state"

echo "== 3. Engine config (week's promoted settings) =="
A=$(grep -m1 'non-default args' "$L")
echo "$A" | grep -q "thinkingcap-gptq-pro-v2" && ok "TC gptq-pro-v2 model" || fail "model not TC"
echo "$A" | grep -q "'kv_cache_memory_bytes': 3489660928" && ok "KV pin 3.25GiB" || fail "KV pin"
echo "$A" | grep -q "'async_scheduling': False" && fail "async DISABLED (should be stock-on)" || ok "async scheduling stock (on)"
echo "$A" | grep -q "hermes" && ok "hermes tool parser" || fail "tool parser"
echo "$A" | grep -q "chat_template.jinja" && ok "vendored chat template" || fail "chat template"
grep -qE "Mamba cache mode is set to 'align'" "$L" && ok "mamba align mode (prefix caching on)" || fail "mamba mode"
grep -qE 'VLLM_COMPUTE_NANS_IN_LOGITS=1' <(sudo podman inspect "$C" --format '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}') && ok "NaN-detect env" || fail "NaN-detect env"
grep -qE 'GENESIS_PN358_MODE=fallback' <(sudo podman inspect "$C" --format '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}') && ok "PN358 fallback mode" || fail "PN358 mode env"

echo "== 4. Live probes =="
curl -sf -m 5 http://localhost:8020/health >/dev/null && ok "/health" || fail "/health"
R=$(curl -s -m 60 http://localhost:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}')
echo "$R" | grep -q '"finish_reason"' && ok "plain completion" || fail "plain completion: $(echo "$R" | head -c 120)"
J=$(curl -s -m 120 http://localhost:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Return valid json only. Give me {\"a\": 1}."}],"max_tokens":50,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":false}}')
echo "$J" | python3 -c "import json,sys; d=json.load(sys.stdin); json.loads(d['choices'][0]['message']['content'])" 2>/dev/null && ok "json_object completion parses" || fail "json_object completion"
curl -sf -m 30 http://localhost:8020/rerank -H 'Content-Type: application/json' -d '{"query":"q","documents":["a","b"]}' >/dev/null 2>&1 && ok "/rerank (PN81)" || fail "/rerank"
systemctl is-active --quiet vllm-wedge-watchdog.timer && ok "wedge-watchdog timer" || fail "watchdog timer"

echo "== 5. Zero-corruption counters =="
NAN=$(grep -c 'PN106D nan-event' "$L"); FSM=$(grep -c 'Failed to advance FSM' "$L"); FAT=$(grep -c 'fatal error' "$L")
[ "$NAN" = 0 ] && ok "nan-events 0" || fail "nan-events $NAN"
[ "$FSM" = 0 ] && ok "FSM failures 0" || fail "FSM failures $FSM"
[ "$FAT" = 0 ] && ok "fatals 0" || fail "fatals $FAT"

rm -f "$L"
echo "== RESULT: pass=$PASS fail=$FAILN =="
[ "$FAILN" = 0 ]
