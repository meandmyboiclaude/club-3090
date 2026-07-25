#!/usr/bin/env python3
"""CPU house test for PN71 v3's (B) block — budget + clamped total bound.

Runs the REAL patched source: it lifts B_NEW out of the patcher, wraps it in a
stand-in ``to_sampling_params`` and executes it against a fake request object.
That way the test cannot drift from what boots — if the patch text stops doing
what this asserts, this fails.

    python3 fixes/test_pn71_clamp_logic.py       # prints PASS/FAIL, exit != 0 on FAIL
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_pn71_reasoning_alias as P  # noqa: E402

FAILS: list[str] = []


class Req:
    """Minimal stand-in for ChatCompletionRequest."""

    def __init__(self, **kw):
        self.reasoning = kw.pop("reasoning", None)
        self.reasoning_effort = kw.pop("reasoning_effort", None)
        self.thinking_token_budget = kw.pop("thinking_token_budget", None)
        self.temperature = kw.pop("temperature", None)
        self.top_p = kw.pop("top_p", None)
        self.top_k = kw.pop("top_k", None)
        self.presence_penalty = kw.pop("presence_penalty", None)
        assert not kw, kw


def _build_runner():
    """Compile B_NEW into a callable ``run(req, max_tokens) -> max_tokens``."""
    body = P.B_NEW
    # Strip the anchor's own two framing lines: the signature tail we re-supply
    # and the trailing "# Default parameters" comment.
    body = body.split(") -> SamplingParams:\n", 1)[1]
    body = body.rsplit("        # Default parameters\n", 1)[0]
    src = "def run(self, max_tokens):\n" + body + "        return max_tokens\n"
    g: dict = {}
    exec(compile(src, "pn71_B_NEW", "exec"), g)  # noqa: S102
    return g["run"]


RUN = _build_runner()


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAILS.append(name)


def main() -> int:
    os.environ.pop("PN71_ANSWER_GRACE", None)
    GRACE = 1024

    print("PN71 v3 (B) — tier -> budget + clamped total bound")

    # 1. The v1 restore: a tier sets an engine-enforced thinking budget.
    r = Req(reasoning="medium")
    mt = RUN(r, 100000)
    check("tier medium sets thinking_token_budget=2048",
          r.thinking_token_budget == 2048, f"got {r.thinking_token_budget}")
    check("tier medium bounds total to 2048+grace",
          mt == 2048 + GRACE, f"got {mt}")

    # 2. The 5192458d clamp — the whole point of the reversal. A caller passing a
    #    small max_tokens must still get a visible answer.
    r = Req(reasoning="medium")
    mt = RUN(r, 1500)
    check("small max_tokens is respected (not raised)", mt == 1500, f"got {mt}")
    check("budget clamped to max_tokens-grace so an answer fits",
          r.thinking_token_budget == max(1, 1500 - GRACE),
          f"got {r.thinking_token_budget}")
    check("clamped budget leaves >= grace tokens for the answer",
          mt - r.thinking_token_budget >= GRACE,
          f"{mt} - {r.thinking_token_budget}")

    # 3. Clamp floors at 1 rather than going negative/zero.
    r = Req(reasoning="high")
    mt = RUN(r, 200)
    check("clamp floors the budget at 1 when max_tokens < grace",
          r.thinking_token_budget == 1, f"got {r.thinking_token_budget}")

    # 4. max / -1 -> uncapped: no budget, no cap.
    for val in ("max", -1):
        r = Req(reasoning=val)
        mt = RUN(r, 100000)
        check(f"reasoning={val!r} leaves budget unset",
              r.thinking_token_budget is None, f"got {r.thinking_token_budget}")
        check(f"reasoning={val!r} leaves max_tokens untouched",
              mt == 100000, f"got {mt}")

    # 5. OFF intents produce no budget and no cap (thinking is disabled in (A)).
    for val in ("off", "none", 0):
        r = Req(reasoning=val)
        mt = RUN(r, 100000)
        check(f"reasoning={val!r} sets no budget",
              r.thinking_token_budget is None, f"got {r.thinking_token_budget}")
        check(f"reasoning={val!r} does not cap max_tokens", mt == 100000, f"got {mt}")

    # 6. An explicit caller budget is never overwritten...
    r = Req(reasoning="medium", thinking_token_budget=512)
    mt = RUN(r, 100000)
    check("explicit caller budget wins over the tier",
          r.thinking_token_budget == 512, f"got {r.thinking_token_budget}")

    # ...but is still clamped when it mathematically cannot emit an answer.
    r = Req(reasoning="medium", thinking_token_budget=8192)
    mt = RUN(r, 3000)
    check("an impossible caller budget (>= max_tokens) is lowered, not honoured",
          r.thinking_token_budget == 3000 - GRACE, f"got {r.thinking_token_budget}")
    check("clamp only ever LOWERS a budget",
          r.thinking_token_budget < 8192, f"got {r.thinking_token_budget}")

    # 7. PN100 disjointness: no reasoning/reasoning_effort => this block is inert,
    #    so a PN100-set budget is never read or rewritten.
    r = Req(thinking_token_budget=10240)
    mt = RUN(r, 100000)
    check("a PN100-shaped request (no reasoning field) keeps its budget",
          r.thinking_token_budget == 10240, f"got {r.thinking_token_budget}")
    check("a PN100-shaped request keeps its max_tokens", mt == 100000, f"got {mt}")

    # 8. reasoning_effort is honoured as a fallback, incl. the Responses-API object.
    r = Req(reasoning_effort="low")
    RUN(r, 100000)
    check("reasoning_effort=low maps to the low tier",
          r.thinking_token_budget == 1536, f"got {r.thinking_token_budget}")
    r = Req(reasoning={"effort": "high"})
    RUN(r, 100000)
    check("Responses-API {'effort': 'high'} maps to the high tier",
          r.thinking_token_budget == 4096, f"got {r.thinking_token_budget}")

    # 9. Raw int tiers.
    r = Req(reasoning=3000)
    mt = RUN(r, 100000)
    check("raw int 3000 becomes the budget", r.thinking_token_budget == 3000,
          f"got {r.thinking_token_budget}")
    check("raw int 3000 bounds total to 3000+grace", mt == 3000 + GRACE, f"got {mt}")

    # 10. The grace default is 1024 in the shipped text (the 512 halving is reverted).
    check("PN71_ANSWER_GRACE default in the patch text is 1024",
          'environ.get(\\"PN71_ANSWER_GRACE\\", \\"1024\\")' in P.B_NEW
          or 'PN71_ANSWER_GRACE", "1024"' in P.B_NEW.replace('\\"', '"'),
          "patch text still defaults to something other than 1024")

    # 11. Env override still works, and a bogus grace can't disable the answer.
    os.environ["PN71_ANSWER_GRACE"] = "2048"
    try:
        r = Req(reasoning="medium")
        mt = RUN(r, 100000)
        check("PN71_ANSWER_GRACE=2048 widens the total bound",
              mt == 2048 + 2048, f"got {mt}")
        os.environ["PN71_ANSWER_GRACE"] = "0"
        r = Req(reasoning="medium")
        mt = RUN(r, 1500)
        check("grace=0 is floored to 1 so the answer allowance never vanishes",
              r.thinking_token_budget == 1499, f"got {r.thinking_token_budget}")
    finally:
        os.environ.pop("PN71_ANSWER_GRACE", None)

    # 12. Booleans must never be read as tiers (True == 1 in Python).
    r = Req(reasoning=True)
    mt = RUN(r, 100000)
    check("reasoning=True is not treated as a 1-token budget",
          r.thinking_token_budget is None and mt == 100000,
          f"budget={r.thinking_token_budget} mt={mt}")

    print()
    if FAILS:
        print(f"FAIL — {len(FAILS)} check(s) failed: {FAILS}")
        return 1
    print("PASS — PN71 v3 clamp logic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
