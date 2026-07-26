#!/usr/bin/env python3
"""BUG-069 present-tense canary — <tool_call> tag leaking into message content.

Re-check of the #47194 class on the 2026-07-26 shipping config
(thinkingcap-gptq-pro-v2 / dev1474cherrymax-1757 / turboquant_3bit_nc /
seqs 6 / util 0.91 / MTP n=3 probabilistic / prefix caching OFF /
--tool-call-parser hermes + --enable-auto-tool-choice / P68+P69 OFF).

Descendant of models/qwen3.6-27b/vllm/diagnostics/endgame-20260714/canary_47194_8021.py,
trimmed for shared-server volume and extended with the two paths the old
canary never covered:
  A  repeated ~12K shared prefix, non-streaming tool calls  (the #47194 shape)
  B  streaming tool calls                                   (the PN72/PN76/PN98 class)
  C  storm bait: many tools + "call several" prompts        (the 21+ tool_call storms)
  D  ~30K-token prefill tool calls                          (where the 07-14 residual
                                                             expressed: 6/15 at 30K,
                                                             0/15 at 14K)

RED on any of:
  * '<tool_call' / '</tool_call>' / '<function=' / '<tools>' appearing in
    message.content, message.reasoning_content or a streamed content delta
  * a degenerate tool-call storm (>= STORM_N tool_calls in one response)

Read-only against the server: plain /v1/chat/completions POSTs, no restarts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import time
import urllib.request

URL_DEFAULT = "http://localhost:8021/v1/chat/completions"
MODEL = "qwen3.6"
STORM_N = 6

# The leak signatures. Deliberately includes the *unterminated* '<tool_call'
# form: the 07-14 residual was a mid-content flush fragment, not a well
# formed tag pair.
LEAK_PAT = re.compile(r"<tool_call|</tool_call>|<function=|</function>|<tools>")


def leaks(text: str) -> list[str]:
    return sorted(set(LEAK_PAT.findall(text or "")))


TOOLS_ONE = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city", "unit"],
        },
    },
}]

TOOLS_MANY = TOOLS_ONE + [{
    "type": "function",
    "function": {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
} for name, desc in [
    ("search_docs", "Search the internal documentation"),
    ("lookup_flight", "Look up a flight by number"),
    ("convert_currency", "Convert an amount between currencies"),
    ("send_email", "Send an email to a recipient"),
    ("query_db", "Run a read-only SQL query"),
    ("get_stock", "Get a stock quote by ticker"),
    ("translate", "Translate text to English"),
]]

CITIES = ["Oslo", "Lima", "Kyoto", "Perth", "Quito", "Turku", "Basel",
          "Leeds", "Miami", "Dakar", "Hanoi", "Sofia"]


def long_prefix(seed: int = 4711, docs: int = 140) -> str:
    """~12K-token shared system prompt with a needle at doc 70 (same shape as
    canary_47194 so results stay comparable)."""
    rnd = random.Random(seed)
    para = ("Operational notes for the fleet: " + " ".join(
        "".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(60)))
    chunks = [f"[doc {i}] {para}" for i in range(docs)]
    chunks.insert(70, "[doc 70x] IMPORTANT: the vault code is CRIMSON-HERON-42.")
    return "You are a precise assistant with tool access.\n" + "\n".join(chunks)


def post(url: str, payload: dict, timeout: int = 900):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_stream(url: str, payload: dict, timeout: int = 900) -> dict:
    """Collect a streamed completion into {content, reasoning, tool_calls, raw}."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    content, reasoning, raw = [], [], []
    tool_names: dict[int, str] = {}
    tool_args: dict[int, list[str]] = {}
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            raw.append(body)
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                for key in ("reasoning_content", "reasoning"):
                    if d.get(key):
                        reasoning.append(d[key])
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tool_names[i] = fn["name"]
                    if fn.get("arguments"):
                        tool_args.setdefault(i, []).append(fn["arguments"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [{"name": tool_names.get(i, ""),
                        "arguments": "".join(tool_args.get(i, []))}
                       for i in sorted(set(tool_names) | set(tool_args))],
        "finish_reason": finish,
        "raw_events": raw,
    }


def unpack(resp: dict) -> dict:
    """Non-streaming response -> the same shape as post_stream."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tcs = msg.get("tool_calls") or []
    return {
        "content": msg.get("content") or "",
        "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or ""),
        "tool_calls": [{"name": (t.get("function") or {}).get("name", ""),
                        "arguments": (t.get("function") or {}).get("arguments", "")}
                       for t in tcs],
        "finish_reason": ch.get("finish_reason"),
        "raw_events": [],
    }


def judge(tag: str, got: dict, want_tool: str | None, want_args: dict | None,
          n_tools_offered: int = 1):
    """-> (verdict_row, failure_or_None).

    `storm` = the degenerate repeat-emission class (07-14: 21-26 tool_calls
    against a ONE-tool schema).  It must be judged relative to how many tools
    were offered, otherwise "call every tool you have" on an 8-tool schema
    scores as a storm — a false positive the first run of this canary hit.
    """
    text = (got["content"] or "") + "\n" + (got["reasoning"] or "")
    found = leaks(text)
    n_tc = len(got["tool_calls"])
    storm = n_tc >= STORM_N and n_tc > n_tools_offered

    args_ok = None
    if want_tool is not None:
        args_ok = False
        for t in got["tool_calls"]:
            if t["name"] != want_tool:
                continue
            try:
                a = json.loads(t["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            if want_args is None:
                args_ok = True
                break
            if all(str(a.get(k, "")).lower().startswith(str(v).lower())
                   for k, v in want_args.items()):
                args_ok = True
                break

    bad = []
    if found:
        bad.append(f"LEAK{found}")
    if storm:
        bad.append(f"STORM(tcs={n_tc})")
    if want_tool is not None and not args_ok:
        bad.append(f"NO_TOOL(tcs={n_tc})")

    row = (f"{tag:28s} tcs={n_tc:<2d} finish={str(got['finish_reason']):10s} "
           f"leak={found or '-'} {'FAIL: ' + ','.join(bad) if bad else 'ok'}")
    if not bad:
        return row, None
    # The recorded tool-call list is TRUNCATED for readability, and that used
    # to make a failure row unreadable: n_tool_calls=8 next to a 4-entry list
    # reads as duplicate emission, and on 2026-07-25 an hour went into a row
    # that was actually correct (TOOLS_MANY offers exactly 8 and the prompt
    # asked for each once). A REAL storm presents identically — big count,
    # short list — so neither number can be trusted alone. Three additions make
    # the two cases distinguishable from the JSON alone, with no reader
    # arithmetic and no run of the canary:
    #   * name_counts is the full multiset over EVERY call, so a storm shows
    #     as {"get_weather": 19} and a breadth case as eight 1s. It is sized by
    #     DISTINCT names, so it stays small precisely when the list does not —
    #     which is why it beats simply raising the cap (a 26-call storm capped
    #     at 12 distinct-looking names is still ambiguous).
    #   * n_tools_offered is the denominator `storm` is judged against; without
    #     it a reader cannot re-derive the verdict.
    #   * tool_calls_recorded / _truncated make the truncation explicit, so the
    #     list length can never again be mistaken for the count.
    # The cap is still raised (4 -> STORM_N * 2) so the ordered sample spans a
    # storm's threshold and keeps the arguments payloads that name_counts drops.
    name_counts: dict[str, int] = {}
    for t in got["tool_calls"]:
        name_counts[t["name"]] = name_counts.get(t["name"], 0) + 1
    sample = got["tool_calls"][:STORM_N * 2]
    return row, {"tag": tag, "why": bad, "n_tool_calls": n_tc,
                 "n_tools_offered": n_tools_offered,
                 "n_distinct_tools": len(name_counts),
                 "tool_call_name_counts": name_counts,
                 "content": (got["content"] or "")[:600],
                 "reasoning": (got["reasoning"] or "")[:600],
                 "tool_calls_recorded": len(sample),
                 "tool_calls_truncated": n_tc - len(sample),
                 "tool_calls": sample}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--rounds-a", type=int, default=10, help="long-prefix non-streaming rounds")
    ap.add_argument("--rounds-b", type=int, default=6, help="streaming rounds")
    ap.add_argument("--rounds-c", type=int, default=4, help="storm-bait rounds")
    ap.add_argument("--rounds-d", type=int, default=4, help="~30K-prefill deep rounds")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default=None, help="write failures as JSON here")
    args = ap.parse_args()

    fails, rows = [], []
    SYSTEM = long_prefix()
    t0 = time.time()

    # ---- A: repeated ~12K shared prefix, non-streaming --------------------
    for i in range(args.rounds_a):
        city = CITIES[i % len(CITIES)]
        got = unpack(post(args.url, {
            "model": MODEL, "max_tokens": 900, "temperature": args.temperature,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content":
                          f"Round {i}: call get_weather for {city} in celsius. Use the tool."}],
            "tools": TOOLS_ONE,
        }))
        row, f = judge(f"A{i:02d}-longprefix-{city}", got, "get_weather", {"city": city},
                       n_tools_offered=len(TOOLS_ONE))
        rows.append(row)
        print(row, flush=True)
        if f:
            fails.append(f)

    # ---- B: streaming tool calls -----------------------------------------
    for i in range(args.rounds_b):
        city = CITIES[(i + 3) % len(CITIES)]
        got = post_stream(args.url, {
            "model": MODEL, "max_tokens": 900, "temperature": args.temperature,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content":
                          f"Stream round {i}: call get_weather for {city} in celsius. Use the tool."}],
            "tools": TOOLS_ONE,
        })
        row, f = judge(f"B{i:02d}-stream-{city}", got, "get_weather", {"city": city},
                       n_tools_offered=len(TOOLS_ONE))
        rows.append(row)
        print(row, flush=True)
        if f:
            fails.append(f)

    # ---- C: storm bait — 8 tools, "call several", short context ----------
    bait = [
        "Look up flight AA100, convert 50 USD to EUR, and get the weather in Oslo "
        "in celsius. Use the tools, one call each.",
        "For each of Oslo, Lima and Kyoto get the weather in celsius. Use the tool.",
        "Search the docs for 'retention policy', then translate the phrase "
        "'buenos dias', then get a stock quote for NVDA. Use the tools.",
        "Call every tool you have once with a plausible argument.",
    ]
    for i in range(args.rounds_c):
        got = unpack(post(args.url, {
            "model": MODEL, "max_tokens": 1200, "temperature": args.temperature,
            "messages": [{"role": "system", "content":
                          "You are a precise assistant with tool access."},
                         {"role": "user", "content": bait[i % len(bait)]}],
            "tools": TOOLS_MANY,
        }))
        row, f = judge(f"C{i:02d}-stormbait", got, None, None,
                       n_tools_offered=len(TOOLS_MANY))
        rows.append(row)
        print(row, flush=True)
        if f:
            fails.append(f)

    # ---- D: deep context — the shape the 07-14 residual needed -----------
    # 07-14: 6/15 degenerate 21+ tool-call storms w/ leak at a ~30K prefill,
    # 0/15 at 14K.  Later re-attributed to P68 auto-force tool_choice=required
    # (>50K CHARS), which is OFF today — so this leg is the direct check that
    # the deep-context expression is gone rather than merely untested.
    DEEP = long_prefix(seed=90210, docs=360)          # ~30K tokens, ~120K chars
    for i in range(args.rounds_d):
        city = CITIES[(i + 7) % len(CITIES)]
        got = unpack(post(args.url, {
            "model": MODEL, "max_tokens": 1200, "temperature": args.temperature,
            "messages": [{"role": "system", "content": DEEP},
                         {"role": "user", "content":
                          f"Deep round {i}: call get_weather for {city} in celsius. Use the tool."}],
            "tools": TOOLS_ONE,
        }))
        row, f = judge(f"D{i:02d}-deep30k-{city}", got, "get_weather", {"city": city},
                       n_tools_offered=len(TOOLS_ONE))
        rows.append(row)
        print(row, flush=True)
        if f:
            fails.append(f)

    n = args.rounds_a + args.rounds_b + args.rounds_c + args.rounds_d
    print(f"\n=== {n} rounds in {time.time() - t0:.0f}s")
    print("=== VERDICT:", "GREEN — BUG-069 does not reproduce"
          if not fails else f"RED — {len(fails)}/{n} failures")
    for f in fails:
        print(json.dumps(f, indent=2)[:1500])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "fails": fails}, fh, indent=2)
        print(f"(detail -> {args.out})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
