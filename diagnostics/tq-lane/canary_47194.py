#!/usr/bin/env python3
"""#47194 canary: hybrid + prefix caching + MTP -> tool-call leakage / needle-recall failure.
Repeated shared long prefix (partial prefix-cache hits on GDN after round 1) + per-round
tool-call and needle checks. Any leak/wrong needle = RED (gate for backporting #48361)."""
import json, random, string, sys, urllib.request

URL = "http://localhost:8020/v1/chat/completions"
random.seed(4711)

def call(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

# ~12K-token shared prefix with a needle buried mid-way
para = ("Operational notes for the fleet: " +
        " ".join("".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(60)))
chunks = [f"[doc {i}] {para}" for i in range(140)]
NEEDLE = "the vault code is CRIMSON-HERON-42"
chunks.insert(70, f"[doc 70x] IMPORTANT: {NEEDLE}.")
SYSTEM = ("You are a precise assistant with tool access.\n" + "\n".join(chunks))

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"},
                                  "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city", "unit"]}}}]

CITIES = ["Oslo", "Lima", "Kyoto", "Perth", "Quito", "Turku", "Basel",
          "Leeds", "Miami", "Dakar", "Hanoi", "Sofia", "Adana", "Bergen", "Malmo"]

fails = []
for i, city in enumerate(CITIES):
    # (a) tool-call round
    r = call({"model": "qwen3.6", "max_tokens": 900,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user",
                            "content": f"Round {i}: call get_weather for {city} in celsius. Use the tool."}],
              "tools": TOOLS})
    ch = r["choices"][0]; msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    content = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    ok_tool = (len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather")
    args_ok = False
    if ok_tool:
        try:
            a = json.loads(tcs[0]["function"]["arguments"])
            args_ok = a.get("city", "").lower() == city.lower() and a.get("unit") in ("c", "celsius")
        except Exception:
            pass
    leaked = "<tool_call>" in content or "</tool_call>" in content or "<function=" in content
    if not (ok_tool and args_ok) or leaked:
        fails.append((i, city, "tool", f"tcs={len(tcs)} args_ok={args_ok} leaked={leaked} "
                      f"content={content[:120]!r}"))
    # (b) needle round (same prefix, different tail -> partial hit)
    r2 = call({"model": "qwen3.6", "max_tokens": 700,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user",
                             "content": f"Round {i}b: What exactly is the vault code per doc 70x? Answer with the code only."}]})
    out = (r2["choices"][0]["message"].get("content") or "")
    if "CRIMSON-HERON-42" not in out:
        fails.append((i, city, "needle", out[:120]))
    print(f"round {i:2d} {city:7s} tool={'OK' if ok_tool and args_ok and not leaked else 'FAIL'} "
          f"needle={'OK' if 'CRIMSON-HERON-42' in out else 'FAIL'}", flush=True)

print("\n=== VERDICT:", "GREEN — no #47194 reproduction" if not fails else f"RED — {len(fails)} failures")
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
