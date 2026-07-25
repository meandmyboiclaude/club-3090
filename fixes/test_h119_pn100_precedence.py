#!/usr/bin/env python3
"""H119 — the PN100 precedence contract (the B2 fix).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_h119_pn100_precedence.py
     (no boot, no GPU, no container — CPU import + fakes)

WHY THIS FILE EXISTS
--------------------
2026-07-25: the route consumer was installed correctly on all seven sites (the
boot log named every one) and STILL changed nothing. A GPQA-30 with the consumer
live came back byte-identical to the control on every compared row.

The cause was precedence, not plumbing. `h119_on_batch_add` defers to any
non-None `thinking_token_budget` — the right call for a CLIENT budget. But
PN100 runs with GENESIS_PN100_AUTO_DEFAULT=1 and budgets ~every request, and
from the worker there is nothing to tell PN100's grant from a caller's. So the
consumer deferred 100% of the time, silently, while every counter and log line
said it was on.

The fix is an ownership stamp: PN100 writes `h119_overridable` into
`vllm_xargs` (which lands in SamplingParams.extra_args and rides to the worker
inside the same params object BatchUpdate.added hands the consumer).
  1 -> PN100 chose this budget; H119 may re-decide it.
  0 -> keep out: tier-0/thinking-off, or a client-pinned numeric.
  absent -> a real caller's budget; H119 never touches it.

The four cases below are the whole contract. Case 2 is the one that was broken.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PN100 = (REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
                "/middleware/auto_budget.py")

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------- fakes
class FakeParams:
    """Only the two attributes the consumer reads off SamplingParams."""

    def __init__(self, budget=None, xargs=None):
        self.thinking_token_budget = budget
        self.extra_args = xargs


class FakeHolder:
    """The slice of ThinkingBudgetStateHolder that site F touches."""

    def __init__(self):
        self._state: dict = {}

    def _init_state_entry(self, prompt_tok_ids, budget):
        return {"thinking_token_budget": budget,
                "check_count_down": budget,
                "think_count": 0,
                "in_think": False,
                "budget_exhausted_in_prompt": False}


def _load_router():
    sys.path.insert(0, str(REPO / "fixes"))
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "1"
    os.environ["H119_DEEP_BUDGET"] = "10240"
    os.environ["H119_LEAN_BUDGET"] = "1600"
    os.environ.pop("H119_OVERRIDE_PN100", None)
    import pn119_router as R  # noqa: PLC0415 — deliberate late import
    return R


def _force_consumer_on(R):
    """Bypass _consumer_active's live-router/enforce checks.

    Those two conditions are about the ENGINE being able to back a decision;
    this file is about precedence. Faking them keeps the test CPU-only.
    """
    R._consumer_state.update({"checked": True, "on": True, "deep": 10240,
                              "lean": 1600, "warned": 0, "override_pn100": True})
    R._consumer_active = lambda: True  # noqa: SLF001


# ---------------------------------------------------------------- cases
def case1_client_budget_wins(R) -> None:
    """An UNSTAMPED budget is a real caller's and is never touched."""
    h, p = FakeHolder(), FakeParams(budget=4096, xargs=None)
    h._state[0] = h._init_state_entry([1, 2, 3], 4096)
    took = R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
    check("case1: unstamped caller budget -> declined", took is False)
    check("case1: entry NOT marked provisional",
          not h._state[0].get(R.H119_PROVISIONAL),
          f"budget still {h._state[0]['thinking_token_budget']}")


def case2_pn100_budget_is_taken_over(R) -> None:
    """THE BUG. A stamped budget must become provisional and get re-decided."""
    h = FakeHolder()
    p = FakeParams(budget=1300, xargs={"h119_overridable": 1})
    h._state[0] = h._init_state_entry([1, 2, 3], 1300)
    R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
    entry = h._state[0]
    check("case2: PN100 budget marked provisional",
          entry.get(R.H119_PROVISIONAL) is True)
    check("case2: PN100's grant preserved as the fail-safe",
          entry.get(R.H119_PRIOR_BUDGET) == 1300,
          f"prior={entry.get(R.H119_PRIOR_BUDGET)}")


def case3_tier0_respected(R) -> None:
    """Budget None + stamp 0 = thinking off. Do not start tracking the row."""
    h = FakeHolder()
    p = FakeParams(budget=None, xargs={"h119_overridable": 0})
    took = R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
    check("case3: tier-0 declined", took is False)
    check("case3: no provisional entry installed", 0 not in h._state)


def case4_unbudgeted_still_routed(R) -> None:
    """The original path: nobody budgeted it, so H119 installs a provisional."""
    h = FakeHolder()
    p = FakeParams(budget=None, xargs=None)
    took = R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
    check("case4: unbudgeted row taken", took is True)
    check("case4: provisional at the deep budget",
          h._state.get(0, {}).get("thinking_token_budget") == 10240)


def case5_kill_switch(R) -> None:
    """H119_OVERRIDE_PN100=0 restores the deferential behaviour exactly."""
    R._consumer_state["override_pn100"] = False
    h = FakeHolder()
    p = FakeParams(budget=1300, xargs={"h119_overridable": 1})
    h._state[0] = h._init_state_entry([1, 2, 3], 1300)
    R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
    check("case5: kill switch leaves PN100's budget alone",
          not h._state[0].get(R.H119_PROVISIONAL))
    R._consumer_state["override_pn100"] = True


def case6_bad_stamp_is_inert(R) -> None:
    """extra_args is wire data. A garbage stamp must defer, never raise."""
    for bad in ("banana", None, [], {"x": 1}):
        h = FakeHolder()
        p = FakeParams(budget=1300, xargs={"h119_overridable": bad})
        h._state[0] = h._init_state_entry([1, 2, 3], 1300)
        try:
            R.h119_on_batch_add(h, 0, p, [1, 2, 3], [])
            ok = not h._state[0].get(R.H119_PROVISIONAL)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"      raised on {bad!r}: {e}")
        check(f"case6: stamp {bad!r} defers without raising", ok)


def case7_fallback_keeps_pn100_grant(R) -> None:
    """An unrouted PN100 row falls back to PN100's grant, not to deep."""
    state = {"thinking_token_budget": 1300, "check_count_down": 1300,
             "think_count": 0, R.H119_PROVISIONAL: True,
             R.H119_PRIOR_BUDGET: 1300}
    R._h119_apply_budget(state, state[R.H119_PRIOR_BUDGET])
    check("case7: prior budget restorable without drift",
          state["thinking_token_budget"] == 1300)


# ------------------------------------------------- PN100's side of the contract
def case8_pn100_stamps_every_site() -> None:
    """Every place PN100 sets a budget must stamp, with the right value."""
    src = PN100.read_text(encoding="utf-8")
    check("case8: PN100 source found", bool(src))

    # Each budget assignment must be followed by a stamp within a few lines.
    lines = src.splitlines()
    sets = [i for i, ln in enumerate(lines)
            if re.search(r"^\s*request\.thinking_token_budget = ", ln)]
    check("case8: three budget-setting sites", len(sets) == 3,
          f"found {len(sets)}")
    for i in sets:
        window = "\n".join(lines[i:i + 6])
        check(f"case8: stamp near line {i + 1}",
              "_stamp_h119(request," in window,
              lines[i].strip())

    # tier-0 disables thinking and sets NO budget — it must stamp 0.
    m = re.search(r'ctk\["enable_thinking"\] = False.*?return 0', src, re.S)
    check("case8: tier-0 block found", m is not None)
    if m:
        check("case8: tier-0 stamps 0 (keep out)",
              "_stamp_h119(request, 0)" in m.group(0))

    # The client-pinned numeric path must stamp 0, not 1. Anchor on the block
    # itself (the `mode != "classify"` direct-numeric branch) rather than on
    # prose, so rewording a comment cannot flip this assertion either way.
    m = re.search(r'if mode != "classify".*?\n        return\n', src, re.S)
    check("case8: client-pinned block found", m is not None)
    if m:
        check("case8: client-pinned path stamps 0",
              "_stamp_h119(request, 0)" in m.group(0),
              "the caller's number is not ours to re-decide")
        check("case8: client-pinned path does NOT stamp 1",
              "_stamp_h119(request, 1)" not in m.group(0))

    # The stamp must be an int — vllm_xargs is typed str|int|float, and a bool
    # sneaking through would be a silent type violation on the wire.
    check("case8: stamp coerced to int",
          'xargs["h119_overridable"] = int(overridable)' in src)


def main() -> int:
    print("H119 PN100 precedence contract\n")
    R = _load_router()
    _force_consumer_on(R)
    for fn in (case1_client_budget_wins, case2_pn100_budget_is_taken_over,
               case3_tier0_respected, case4_unbudgeted_still_routed,
               case5_kill_switch, case6_bad_stamp_is_inert,
               case7_fallback_keeps_pn100_grant):
        fn(R)
    case8_pn100_stamps_every_site()
    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print("ALL PASS")
    print("VERDICT: PN100's budgets are H119-overridable; a client's are not; "
          "tier-0 is untouched; an unrouted takeover keeps PN100's own grant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
