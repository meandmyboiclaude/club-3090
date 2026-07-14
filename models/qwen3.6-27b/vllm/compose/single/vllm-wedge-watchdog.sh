#!/bin/bash
# Wedge watchdog (BUG-072 class): health lies during the abort-wedge — probe a REAL
# completion; two consecutive finish=abort (or empty finish) => restart the container.
# Runs from systemd timer every 2 min. Logs to journal.
PORT="${1:-8020}"
STATE=/run/vllm-wedge-watchdog.state
probe() {
  curl -s -m 45 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys
try: print(json.load(sys.stdin)['choices'][0]['finish_reason'])
except Exception: print('parse-error')" 2>/dev/null
}
# skip if container is mid-(re)start
curl -s -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "health down — restart policy owns this; skip"; rm -f "$STATE"; exit 0; }
fr=$(probe)
if [ "$fr" = "stop" ] || [ "$fr" = "length" ]; then rm -f "$STATE"; exit 0; fi
echo "wedge-suspect probe: finish=${fr:-none}"
if [ -f "$STATE" ]; then
  echo "second consecutive bad probe -> restarting vllm-qwen36-endgame (wedge)"
  rm -f "$STATE"
  podman restart vllm-qwen36-endgame
else
  touch "$STATE"
fi
