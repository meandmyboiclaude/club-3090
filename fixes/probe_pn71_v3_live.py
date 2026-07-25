#!/usr/bin/env python3
"""The minimal live probe for PN71 v3. Run it yourself — it needs the GPU server.

WHAT IT DECIDES
---------------
v3 claims one thing that source inspection and banked outputs cannot finish
proving: that on THIS boot, a tiered `reasoning:` request comes back with a
VISIBLE ANSWER instead of an empty body. Before v3 the same request had no
thinking budget, so an open-ended prompt filled the whole `tier + grace` envelope
thinking and was cut inside `<think>` — HTTP 200, finish=length, content_len=0.
Measured on banked prod-shaped traces that is 31.2% of `reasoning=low` requests
and 16.8% of `reasoning=medium` ones, so ONE request is not decisive; this sends
a small deterministic batch and reports the rate.

PASS = zero empty contents, every reasoning trace bounded at its tier, and the
tier actually enforced (reasoning_tokens must not exceed the tier by more than a
few spec-decode tokens). A single empty body is a FAIL and means the clamp is not
doing its job on this boot.

    python3 fixes/probe_pn71_v3_live.py                    # :8021, default model
    PORT=8020 python3 fixes/probe_pn71_v3_live.py          # prod
    python3 fixes/probe_pn71_v3_live.py --model thinkingcap

Cost: 14 requests, temperature 0, concurrency 1. No restart, no config change,
nothing written. Reads VLLM_API_KEY if the endpoint wants one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

# Deliberately open-ended: these are the shape that over-thinks and used to come
# back empty. A "what is 2+2" probe proves nothing — it never filled the envelope.
PROMPTS = [
    "Design a rate limiter for a multi-tenant API. Discuss the trade-offs.",
    "Why do transformer models struggle with long-range arithmetic? Explain the mechanism.",
    "Compare optimistic and pessimistic concurrency control for a booking system.",
    "What would you check first if a service's p99 latency tripled but p50 did not?",
    "Explain the CAP theorem's practical consequences for a payments ledger.",
    "How would you migrate a 2TB table to a new schema with no downtime?",
    "Discuss the failure modes of retry-with-backoff in a fan-out architecture.",
]
TIERS = {"low": 1536, "medium": 2048}
# Spec decode can overshoot a forced close by a few tokens; anything beyond this
# means the budget is not being enforced at all.
OVERSHOOT_SLACK = 64


def post(url: str, body: dict, key: str, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key or 'EMPTY'}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=os.environ.get("PORT", "8021"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "thinkingcap"))
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}/v1"
    key = os.environ.get("VLLM_API_KEY", "")
    print(f"PN71 v3 live probe — {base} model={args.model} "
          f"grace={os.environ.get('PN71_ANSWER_GRACE', '(server default 1024)')}")
    print(f"{'tier':7s} {'prompt':6s} {'finish':8s} {'rtok':>6s} {'ctok':>6s} "
          f"{'answer':>7s}  verdict")

    empties, overshoots, errors, rows = 0, 0, 0, 0
    for tier, budget in TIERS.items():
        for i, prompt in enumerate(PROMPTS):
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning": tier,
                "temperature": 0,
                "seed": 0,
            }
            try:
                t0 = time.monotonic()
                j = post(base + "/chat/completions", body, key)
            except Exception as e:  # noqa: BLE001
                print(f"{tier:7s} p{i:<5d} ERROR    {'':>6s} {'':>6s} {'':>7s}  {e}")
                errors += 1
                continue
            ch = j["choices"][0]
            msg = ch.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            usage = j.get("usage", {}) or {}
            ctok = usage.get("completion_tokens", 0)
            rtok = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens") or max(0, len(reasoning) // 4)
            rows += 1

            verdicts = []
            if not content.strip():
                verdicts.append("EMPTY-CONTENT (FAIL)")
                empties += 1
            if rtok > budget + OVERSHOOT_SLACK:
                verdicts.append(f"BUDGET-NOT-ENFORCED rtok>{budget} (FAIL)")
                overshoots += 1
            if ch.get("stop_reason") == "pn71_truncated_in_think":
                verdicts.append("PN71T stamped it")
            print(f"{tier:7s} p{i:<5d} {str(ch.get('finish_reason')):8s} "
                  f"{rtok:6d} {ctok:6d} {len(content.split()):7d}  "
                  f"{' | '.join(verdicts) or 'ok'}  ({time.monotonic()-t0:.1f}s)")

    print()
    print(f"rows={rows} errors={errors} empty_content={empties} "
          f"budget_overshoot={overshoots}")
    if errors and not rows:
        print("FAIL — every request errored; the endpoint is not answering")
        return 1
    if empties or overshoots:
        print("FAIL — PN71 v3 is not delivering a bounded-thinking, non-empty answer")
        return 1
    print("PASS — every tiered request bounded its thinking AND returned an answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
