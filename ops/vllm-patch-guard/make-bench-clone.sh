#!/usr/bin/env bash
# Regenerate the :8021 bench clone from the live :8020 prod compose.
#
# The bench clone must differ from prod in EXACTLY the mechanical bits below —
# port, container name, telemetry name, and a served-model alias. Everything
# else (patch list, env flags, memory config, engine args) has to be inherited,
# or the window measures a configuration nobody runs.
#
# This exists because the previous clone was hand-maintained and silently
# drifted: it was missing all three v4 flags and still carried a --kv-cache-memory
# pin that prod no longer has. A stale clone produces a clean-looking run that
# measures the wrong thing, which is worse than no run.
#
#   make-bench-clone.sh            # regenerate
#   make-bench-clone.sh --check    # show the prod-vs-clone delta, change nothing

set -euo pipefail

DIR=/home/user/club-3090/models/qwen3.6-27b/vllm/compose/single
PROD="$DIR/endgame8020.yml"
BENCH="$DIR/tcbench8021.yml"

[ -f "$PROD" ] || { echo "prod compose missing: $PROD"; exit 1; }

generate() {
  sed \
    -e 's/container_name: vllm-qwen36-endgame/container_name: vllm-tcbench-8021/' \
    -e 's#http://localhost:8020/health#http://localhost:8021/health#' \
    -e 's/openinference.project.name=vllm,service.name=vllm-qwen36/openinference.project.name=vllm,service.name=vllm-tcbench-8021/' \
    -e 's/^\(\s*\)- "8020"$/\1- "8021"/' \
    -e 's/\${PORT:-8020}/${PORT:-8021}/g' \
    "$PROD"
}

if [ "${1:-}" = "--check" ]; then
  if [ ! -f "$BENCH" ]; then echo "no clone yet at $BENCH"; exit 0; fi
  echo "=== differences beyond the expected port/name substitutions ==="
  diff <(generate | grep -vE '^\s*#') <(grep -vE '^\s*#' "$BENCH") \
    && echo "(clone is in sync with prod)"
  exit 0
fi

[ -f "$BENCH" ] && cp "$BENCH" "$BENCH.bak-$(date +%Y%m%d-%H%M%S)"
generate > "$BENCH"

# The served-model alias is additive, not a substitution: bench suites address
# the model as "thinkingcap" while prod serves it as "qwen3.6".
python3 - "$BENCH" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
if "- thinkingcap" not in s:
    s = re.sub(r"(\n(\s+)- qwen3\.6\n)", r"\1\2- thinkingcap\n", s, count=1)
    open(p, "w", encoding="utf-8").write(s)
    print("added 'thinkingcap' served-model alias")
else:
    print("'thinkingcap' alias already present")
PY

python3 -c "import yaml;yaml.safe_load(open('$BENCH',encoding='utf-8'));print('YAML parses OK')"

echo
echo "regenerated: $BENCH"
echo "inherited from prod (verify these are what you expect):"
grep -E 'GENESIS_PN100_TIER_BUDGETS|GENESIS_PN102_STATIC_BANNER|GENESIS_PN101_ESCALATE|kv-cache-memory' "$BENCH" \
  | sed 's/^\s*/  /' || echo "  (none found — that is a red flag if prod has them)"
