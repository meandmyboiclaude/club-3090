#!/usr/bin/env python3
"""Grammar x MTP x TQ3 canary: structured output (json_schema via xgrammar) under
MTP spec-decode. Exercises the crash chain: grammar rejects draft -> -1 padding ->
embedding gather. 12 rounds, varied schemas; any 5xx / engine death / invalid JSON = RED."""
import json, sys, urllib.request, urllib.error

URL = "http://localhost:8020/v1/chat/completions"

SCHEMAS = [
    {"name": "person", "schema": {"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "age", "tags"], "additionalProperties": False}},
    {"name": "invoice", "schema": {"type": "object", "properties": {
        "id": {"type": "string", "pattern": "^INV-[0-9]{4}$"},
        "total": {"type": "number"},
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku", "qty"], "additionalProperties": False}}},
        "required": ["id", "total", "lines"], "additionalProperties": False}},
    {"name": "verdict", "schema": {"type": "object", "properties": {
        "decision": {"type": "string", "enum": ["approve", "reject", "escalate"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
        "required": ["decision", "confidence", "reasons"], "additionalProperties": False}},
]
PROMPTS = [
    "Invent a plausible person record for a sci-fi novel character.",
    "Create an invoice for 3 hardware items with realistic SKUs.",
    "Review this plan: 'deploy on Friday with no rollback'. Give a verdict.",
    "Generate a person record for a medieval blacksmith.",
]

fails = []
for i in range(12):
    sc = SCHEMAS[i % len(SCHEMAS)]
    body = {"model": "qwen3.6", "max_tokens": 1200,
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)] + f" (round {i})"}],
            "response_format": {"type": "json_schema", "json_schema": sc}}
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        content = d["choices"][0]["message"].get("content") or ""
        try:
            json.loads(content)
            ok = True
        except Exception:
            ok = False
            fails.append((i, sc["name"], "invalid-json", content[:100]))
        print(f"round {i:2d} {sc['name']:8s} {'OK' if ok else 'INVALID-JSON'}", flush=True)
    except urllib.error.HTTPError as e:
        fails.append((i, sc["name"], f"HTTP {e.code}", e.read()[:150].decode(errors='replace')))
        print(f"round {i:2d} {sc['name']:8s} HTTP {e.code}", flush=True)
    except Exception as e:
        fails.append((i, sc["name"], "EXC", str(e)[:150]))
        print(f"round {i:2d} {sc['name']:8s} EXC {type(e).__name__}", flush=True)

import subprocess
alive = subprocess.run(["curl", "-sf", "http://localhost:8020/health"],
                       capture_output=True).returncode == 0
print(f"\nengine alive after run: {alive}")
print("=== VERDICT:", "GREEN" if not fails and alive else f"RED — {len(fails)} failures, alive={alive}")
for f in fails: print("  ", f)
sys.exit(0 if not fails and alive else 1)
