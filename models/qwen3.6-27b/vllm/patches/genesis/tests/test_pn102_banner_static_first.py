#!/usr/bin/env python3
"""Offline tests for PN102 GENESIS_PN102_BANNER_STATIC_FIRST (prefix-order fix).

The number-carrying banners (v3 sized, v8 hybrid, v8b lean-anchor) open with
per-request figures, so the banner's leading tokens differ on every request and
prefix-cache blocks containing the banner head can never be reused (hit rate
0%). With the flag ON the same information is emitted static-text-first with
one trailing "Figures:" sentence. This file proves, without a GPU:

  (a) flag ON:  the static prefix is byte-identical across two different
      request contexts (different budgets AND different planner step counts);
  (b) flag OFF: the banners are NOT prefix-identical — they diverge at the
      first per-request number (the bug being fixed);
  (c) the set of information conveyed (every number + every semantic clause)
      is unchanged between the two orderings;
  (d) flag OFF is byte-identical to the pre-change banner (no regression);
  (e) the v5 banner (already static) is untouched by the flag; the think-seed
      is untouched in both states (BUG-075: must end "Step 1:").

Run: python3 test_pn102_banner_static_first.py
"""

import os
import re
import sys
from pathlib import Path

MOD_DIR = Path(__file__).resolve().parents[1] / "vllm" / "_genesis" / "middleware"
sys.path.insert(0, str(MOD_DIR))

os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
# start from a clean banner-version env
for _v in ("GENESIS_PN102_BANNER_V8", "GENESIS_PN102_BANNER_V7",
           "GENESIS_PN102_BANNER_V6A", "GENESIS_PN102_BANNER_V6B",
           "GENESIS_PN102_BANNER_V5", "GENESIS_PN102_STATIC_BANNER",
           "GENESIS_PN102_BANNER_STATIC_FIRST", "GENESIS_PN102_V8_LEAN_ANCHOR",
           "GENESIS_PN102_V3_RANGE", "GENESIS_PN102_V3_ANS_FREEZE",
           "GENESIS_PN102_V5_ANSWER_CLAUSE", "GENESIS_PN102_ANNOUNCE_CEILING"):
    os.environ.pop(_v, None)

import answer_rescue as ar  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


class Request:
    def __init__(self, budget, steps=None):
        self.model = "qwen3.6"
        self.messages = [{"role": "user", "content": "hard question"}]
        self.stream = False
        self.thinking_token_budget = budget
        self.chat_template_kwargs = {"pn100_steps": steps} if steps else {}
        self.max_tokens = 2048
        self.tools = None
        self.response_format = None


def banner(budget, steps=None):
    r = Request(budget, steps)
    ar.maybe_add_answer_hint(r)
    return r.chat_template_kwargs.get("pn_env_banner", ""), \
        r.chat_template_kwargs.get("pn_env_seed", "")


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


NUMS = re.compile(r"\d+")

# every semantic field the v3 headroom banner carries, as detectable phrases
V3_CLAUSES = [
    "[envelope]",
    "Number your steps",
    "wrap up around Step",              # checkpoint anchor
    "once your answer is settled",      # settled -> stop license
    "keep reasoning past Step",         # deeper-than-planned license
    "the budget is generous",
    "do not conclude while your answer is still uncertain",
    "genuinely exhausted your approaches, commit to your best answer",
    "Do not let the budget cut you off",
    "final answer in the FIRST sentence",   # answer-shape clause
    "sentences total",
    "short reasoning steps",            # step-count figure context
    "thinking tokens",                  # token-allowance figure context
]


def test_v3():
    print("\nv3 sized banner (default path — the number-carrying prod banner)")

    # (b)+(d) flag OFF: current behaviour, numbers lead, prefixes diverge
    os.environ.pop("GENESIS_PN102_BANNER_STATIC_FIRST", None)
    off_a, seed_a = banner(6000, steps=8)
    off_b, seed_b = banner(10240, steps=21)
    print(f"\n  BEFORE (flag off, budget=6000 steps=8):\n    {off_a}")
    print(f"\n  BEFORE (flag off, budget=10240 steps=21):\n    {off_b}\n")
    cp_off = common_prefix_len(off_a, off_b)
    check("flag OFF: banners diverge at the first per-request number (the bug)",
          cp_off < 60, f"common prefix ran {cp_off} chars")
    expected_off_b = (
        "[envelope] Thinking budget: about 21 short reasoning steps "
        "(budget allows up to ~10240 thinking tokens). Number your steps and "
        "wrap up around Step 21 once your answer is settled; if the problem "
        "proves deeper than planned, keep reasoning past Step 21 — the budget "
        "is generous — and do not conclude while your answer is still "
        "uncertain. If you have genuinely exhausted your approaches, commit "
        "to your best answer. Do not let the budget cut you off. Unless the "
        "user asked for longer form, put your final answer in the FIRST "
        "sentence of your reply, then at most 3 sentences total."
    )
    check("flag OFF is byte-identical to the pre-change banner (no regression)",
          off_b == expected_off_b, f"got: {off_b!r}")

    # (a) flag ON: static prefix byte-identical across different contexts
    os.environ["GENESIS_PN102_BANNER_STATIC_FIRST"] = "1"
    on_a, seed_on_a = banner(6000, steps=8)
    on_b, seed_on_b = banner(10240, steps=21)
    print(f"\n  AFTER (flag on, budget=6000 steps=8):\n    {on_a}")
    print(f"\n  AFTER (flag on, budget=10240 steps=21):\n    {on_b}\n")
    static_a = on_a.rsplit(" Figures:", 1)[0]
    static_b = on_b.rsplit(" Figures:", 1)[0]
    check("flag ON: static part is byte-identical across the two contexts",
          static_a == static_b and len(static_a) > 200,
          f"prefix ran only {common_prefix_len(on_a, on_b)} chars")
    check("flag ON: banners differ ONLY in the trailing Figures sentence",
          common_prefix_len(on_a, on_b) >= len(static_a))
    # env-static constants (e.g. "3 sentences") may remain — they are boot-
    # invariant; only PER-REQUEST values must not appear before Figures.
    check("flag ON: no per-request value leaks into the static part",
          not any(t in static_b for t in
                  ("21", "10240", "Step 8", "6000")), f"static: {static_b!r}")
    check("flag ON: figures land at the very end",
          on_b.endswith("thinking tokens).") and " Figures: N = 21 " in on_b)

    # lean (no-headroom) branch: same property within the branch
    on_c, _ = banner(1930, steps=7)
    on_d, _ = banner(2600, steps=12)
    st_c = on_c.rsplit(" Figures:", 1)[0]
    st_d = on_d.rsplit(" Figures:", 1)[0]
    check("flag ON (lean branch): static part byte-identical across contexts",
          st_c == st_d and not any(t in st_c for t in ("1930", "2600", " 7 ", " 12 ")))

    # (c) information set unchanged between orderings (same request context)
    check("numbers conveyed are identical between orderings",
          set(NUMS.findall(off_b)) == set(NUMS.findall(on_b)),
          f"{sorted(set(NUMS.findall(off_b)))} vs {sorted(set(NUMS.findall(on_b)))}")
    missing_off = [c for c in V3_CLAUSES
                   if c.replace("Step", "Step 21") not in off_b
                   and c not in off_b]
    missing_on = [c for c in V3_CLAUSES
                  if c.replace("Step", "Step N") not in on_b and c not in on_b]
    check("every semantic clause present in the OFF banner", not missing_off,
          str(missing_off))
    check("every semantic clause present in the ON banner", not missing_on,
          str(missing_on))
    check("ON banner states N explicitly (anchor value not dropped)",
          "N = 21" in on_b and "about 21 short reasoning steps" in on_b
          and "~10240 thinking tokens" in on_b)

    # (e) seed untouched in both states (prompt tail; BUG-075)
    check("seed unchanged by the flag (BUG-075 tail invariant)",
          seed_on_b == seed_b and seed_on_b.rstrip().endswith("Step 1:"),
          f"{seed_b!r} vs {seed_on_b!r}")


def test_v8():
    print("\nv8 hybrid + v8b lean-anchor")
    os.environ["GENESIS_PN102_BANNER_V8"] = "1"

    os.environ.pop("GENESIS_PN102_BANNER_STATIC_FIRST", None)
    off_a, _ = banner(6000)
    off_b, _ = banner(10240)
    check("v8 flag OFF: banners diverge early (the bug)",
          common_prefix_len(off_a, off_b) < 110)

    os.environ["GENESIS_PN102_BANNER_STATIC_FIRST"] = "1"
    on_a, _ = banner(6000)
    on_b, _ = banner(10240)
    print(f"\n  v8 BEFORE (flag off, budget=10240):\n    {off_b}")
    print(f"\n  v8 AFTER  (flag on,  budget=10240):\n    {on_b}\n")
    st_a = on_a.rsplit(" Figures:", 1)[0]
    st_b = on_b.rsplit(" Figures:", 1)[0]
    check("v8 flag ON: static part byte-identical across contexts",
          st_a == st_b and len(st_a) > 200 and not NUMS.search(st_a))
    check("v8 numbers conveyed are identical between orderings",
          set(NUMS.findall(off_b)) == set(NUMS.findall(on_b)))

    # v8b lean-anchor branch
    os.environ["GENESIS_PN102_V8_LEAN_ANCHOR"] = "1"
    os.environ.pop("GENESIS_PN102_BANNER_STATIC_FIRST", None)
    off_a, _ = banner(6000, steps=8)
    off_b, _ = banner(10240, steps=21)
    check("v8b flag OFF: banners diverge early (the bug)",
          common_prefix_len(off_a, off_b) < 60)
    os.environ["GENESIS_PN102_BANNER_STATIC_FIRST"] = "1"
    on_a, seed_a = banner(6000, steps=8)
    on_b, seed_b = banner(10240, steps=21)
    st_a = on_a.rsplit(" Figures:", 1)[0]
    st_b = on_b.rsplit(" Figures:", 1)[0]
    check("v8b flag ON: static part byte-identical across contexts",
          st_a == st_b and len(st_a) > 200 and not NUMS.search(st_a))
    check("v8b numbers conveyed are identical between orderings",
          set(NUMS.findall(off_b)) == set(NUMS.findall(on_b)))
    check("v8b seed still ends mid-reasoning (BUG-075)",
          seed_b.rstrip().endswith("Step 1:"))
    os.environ.pop("GENESIS_PN102_V8_LEAN_ANCHOR", None)
    os.environ.pop("GENESIS_PN102_BANNER_V8", None)


def test_v5_untouched():
    print("\nv5 (live bench config) — flag must be a no-op")
    os.environ["GENESIS_PN102_BANNER_V5"] = "1"
    os.environ.pop("GENESIS_PN102_BANNER_STATIC_FIRST", None)
    off_a, _ = banner(6000)
    os.environ["GENESIS_PN102_BANNER_STATIC_FIRST"] = "1"
    on_a, _ = banner(6000)
    on_b, _ = banner(10240)
    check("v5 banner identical with flag on/off", off_a == on_a)
    check("v5 banner already request-invariant", on_a == on_b)
    os.environ.pop("GENESIS_PN102_BANNER_V5", None)


if __name__ == "__main__":
    test_v3()
    test_v8()
    test_v5_untouched()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("all offline checks passed")
