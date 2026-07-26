#!/usr/bin/env python3
"""The explicit thinking-OFF contract, end to end (USER directive 2026-07-26).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_thinking_off_bypass.py
     (no boot, no GPU, no container — CPU import + fakes)

THE RULE
--------
"prompts with reasoning / thinking off in request correctly bypasses all
thinking things while something that doesnt have that runs on thinking."

A REQUEST PARAMETER rule. Not a classifier, not a lane policy, not a
model-side decision. Two properties, both asserted here:

  explicit off  -> zero thinking tokens, zero classify calls, no budget
                   stamped, no PN102 banner, no H119 grant.
  unspecified   -> thinking runs exactly as it does today.

WHAT WAS ACTUALLY BROKEN
------------------------
PN100's frontend already honoured all five off-forms — `_decide_mode` returned
"skip", so no classify call was ever spent on them and no budget was ever
written. The 95-98% tier-0/engage figure comes from requests that specified
NOTHING (GENESIS_PN100_AUTO_DEFAULT=1 classifies those), not from requests
that asked for off. What leaked was everything "skip" did not SAY:

  * `chat_template_kwargs.thinking_budget: "off"` and `reasoning: off` left
    `enable_thinking` unset -> the Qwen3.6 template defaults it ON, so the
    model thought anyway, with no budget at all.
  * `thinking_token_budget: 0` (the house-preferred off form) reached the
    worker as a real budget -> a holder state entry, and with it the
    forced-`</think>` path and PN108's plateau cap.
  * no `h119_overridable` stamp -> the H119 consumer read the row as
    unclaimed and granted a provisional DEEP budget in sync_batch, a whole
    phase before the router's own thinking-off tap could see it. 101 rows on
    the live boot.

Cases 1-8 cover the frontend (auto_budget.py), 9-15 the worker
(pn119_router.py), 16 the two joined.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import types
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parent.parent
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm"

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


# ─── module loading ─────────────────────────────────────────────────────────
# `vllm` is a namespace package in the genesis tree (no __init__.py), so a
# synthetic parent with __path__ pointed at it imports the middleware without
# a vllm install. Done before any import so a real vllm cannot win.
def _load_genesis():
    mod = types.ModuleType("vllm")
    mod.__path__ = [str(GENESIS)]
    sys.modules["vllm"] = mod
    from vllm._genesis.middleware import answer_rescue as ar  # noqa: PLC0415
    from vllm._genesis.middleware import auto_budget as ab  # noqa: PLC0415
    return ab, ar


def _load_router():
    sys.path.insert(0, str(REPO / "fixes"))
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "1"
    os.environ["H119_DEEP_BUDGET"] = "10240"
    os.environ.pop("H119_OVERRIDE_PN100", None)
    import pn119_router as R  # noqa: PLC0415 — deliberate late import
    return R


# The live serving env this contract has to hold under: PN100 on, auto-default
# on, tier budgets as the compose ships them.
for _k, _v in {
    "GENESIS_ENABLE_PN100_AUTO_BUDGET": "1",
    "GENESIS_PN100_AUTO_DEFAULT": "1",
    "GENESIS_PN100_TIER_BUDGETS": "0,10240,10240,10240",
    "GENESIS_ENABLE_PN101_ANSWER_RESCUE": "1",
    "GENESIS_ENABLE_PN102_CONTRACT": "1",
}.items():
    os.environ[_k] = _v
for _k in ("GENESIS_THINKING_OFF_STRICT", "GENESIS_PN100_SHAPE_TIER0",
           "GENESIS_PN100_CONTINUOUS", "GENESIS_PN100_CLASSIFY_MAX_CHARS"):
    os.environ.pop(_k, None)

AB, AR = _load_genesis()
R = _load_router()

THINK_START, THINK_END = 248068, 248069


# ─── frontend fakes ─────────────────────────────────────────────────────────
def req(**kw) -> SimpleNamespace:
    """ChatCompletionRequest's surface, mutable. None = field absent."""
    base = dict(
        messages=[{"role": "user", "content": "how many primes below 100?"}],
        model="qwen", tools=None, chat_template_kwargs=None,
        response_format=None, temperature=None, max_tokens=4096,
        max_completion_tokens=None, tool_choice=None,
        thinking_token_budget=None, reasoning_effort=None, reasoning=None,
        vllm_xargs=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class Serving:
    """Counts classify calls — the whole point of the latency half."""

    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(self, request, raw_request=None):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="2|12"), logprobs=None)])


def run(request) -> Serving:
    """The two frontend hooks, in the order the patched serving.py runs them:
    PN100 at the top of _create_chat_completion, PN101a's banner below it."""
    AB._TIER_CACHE.clear()
    serving = Serving()
    asyncio.run(AB.apply_hook_async(serving, request))
    AR.maybe_add_answer_hint(request)
    return serving


def ctk_of(request) -> dict:
    return dict(getattr(request, "chat_template_kwargs", None) or {})


def stamp_of(request):
    return (getattr(request, "vllm_xargs", None) or {}).get("h119_overridable")


OFF_FORMS = {
    "chat_template_kwargs.enable_thinking=false":
        dict(chat_template_kwargs={"enable_thinking": False}),
    "chat_template_kwargs.thinking_budget='off'":
        dict(chat_template_kwargs={"thinking_budget": "off"}),
    "chat_template_kwargs.thinking_budget=0":
        dict(chat_template_kwargs={"thinking_budget": 0}),
    "thinking_token_budget=0":
        dict(thinking_token_budget=0),
    "reasoning='off'":
        dict(reasoning="off"),
    "reasoning={'effort':'none'}":
        dict(reasoning={"effort": "none"}),
    "reasoning_effort='none'":
        dict(reasoning_effort="none"),
}

ON_FORMS = {
    "unspecified": dict(),
    "thinking_token_budget=2048": dict(thinking_token_budget=2048),
    "reasoning='high'": dict(reasoning="high"),
    "reasoning_effort='low'": dict(reasoning_effort="low"),
    "chat_template_kwargs.enable_thinking=true":
        dict(chat_template_kwargs={"enable_thinking": True}),
    "chat_template_kwargs.thinking_budget=4096":
        dict(chat_template_kwargs={"thinking_budget": 4096}),
}


# ═══ 1-4. explicit off: zero classify, zero budget, no banner, keep-out ═════
print("\n[1] explicit off -> NO classify call (the 0.3-0.6s / 2.7s-worst spend)")
for name, kw in OFF_FORMS.items():
    s = run(req(**kw))
    check(name, s.calls == 0, f"classify calls={s.calls}")

print("\n[2] explicit off -> NO thinking budget stamped")
for name, kw in OFF_FORMS.items():
    r = req(**kw)
    run(r)
    b = getattr(r, "thinking_token_budget", None)
    check(name, b is None, f"thinking_token_budget={b!r}")

print("\n[3] explicit off -> enable_thinking=False reaches the chat template")
for name, kw in OFF_FORMS.items():
    r = req(**kw)
    run(r)
    et = ctk_of(r).get("enable_thinking")
    check(name, et is False, f"enable_thinking={et!r}")

print("\n[4] explicit off -> no PN102 banner, and H119 told to keep out")
for name, kw in OFF_FORMS.items():
    r = req(**kw)
    run(r)
    c = ctk_of(r)
    ok = not c.get("pn_env_banner") and stamp_of(r) == 0 and "pn100_steps" not in c
    check(name, ok, f"banner={bool(c.get('pn_env_banner'))} "
                    f"h119_overridable={stamp_of(r)!r}")

# ═══ 5. unspecified / explicit-on: thinking runs as today ══════════════════
print("\n[5] no off in the request -> thinking runs (unchanged behaviour)")
r = req()
s = run(r)
c = ctk_of(r)
check("unspecified: classify spent", s.calls == 1, f"calls={s.calls}")
check("unspecified: budget granted", r.thinking_token_budget == 10240,
      f"budget={r.thinking_token_budget!r}")
check("unspecified: enable_thinking True", c.get("enable_thinking") is True)
check("unspecified: PN102 banner attached", bool(c.get("pn_env_banner")))
check("unspecified: H119 may override", stamp_of(r) == 1,
      f"stamp={stamp_of(r)!r}")

print("\n[6] non-off values are never read as off (bool/int traps)")
for name, kw in ON_FORMS.items():
    r = req(**kw)
    run(r)
    et = ctk_of(r).get("enable_thinking")
    # The only thing asserted here: nothing FORCED thinking off.
    check(name, et is not False or kw.get("chat_template_kwargs", {})
          .get("enable_thinking") is False,
          f"enable_thinking={et!r}")
check("_is_off_value(True) is False", AB._is_off_value(True) is False)
check("_is_off_value(False) is True", AB._is_off_value(False) is True)
check("_is_off_value(1) is False", AB._is_off_value(1) is False)
check("_is_off_value('high') is False", AB._is_off_value("high") is False)
check("_is_off_value('OFF') is True", AB._is_off_value("OFF") is True)

print("\n[7] PN100's own classify call is never bypassed (no recursion)")
internal = req(chat_template_kwargs={"enable_thinking": False,
                                     "pn100_internal": True})
s = run(internal)
check("classify self-call: not stamped", stamp_of(internal) is None,
      f"stamp={stamp_of(internal)!r}")
check("classify self-call: no recursion", s.calls == 0, f"calls={s.calls}")

print("\n[8] GENESIS_THINKING_OFF_STRICT=0 restores pre-fix silence")
os.environ["GENESIS_THINKING_OFF_STRICT"] = "0"
r = req(thinking_token_budget=0)
s = run(r)
check("strict off: no stamp written", stamp_of(r) is None)
check("strict off: zero budget left alone", r.thinking_token_budget == 0)
check("strict off: still no classify", s.calls == 0)
os.environ.pop("GENESIS_THINKING_OFF_STRICT")


# ─── worker fakes ───────────────────────────────────────────────────────────
class FakeParams:
    """The two attributes h119_on_batch_add reads off SamplingParams."""

    def __init__(self, budget=None, xargs=None):
        self.thinking_token_budget = budget
        self.extra_args = xargs


class FakeHolder:
    """The slice of ThinkingBudgetStateHolder that site F touches."""

    def __init__(self) -> None:
        self._state: dict = {}

    def _init_state_entry(self, prompt_tok_ids, budget):
        return {"thinking_token_budget": budget, "check_count_down": budget,
                "think_count": 0, "continue_thinking": False}


class FakeRouter:
    """Enough of PN119Router for the consumer: mode + the marker scan."""

    mode = "enforce"
    _think_start_ids = [THINK_START]
    _think_end_ids = [THINK_END]
    _tail_window = 64
    _last_subseq = staticmethod(R.PN119Router._last_subseq)


R.ROUTER = FakeRouter()
R.reset_consumer_cache()

# A thinking-OFF render pre-closes the region: BOTH markers present, `</think>`
# last. Thinking-ON leaves `<think>` open (optionally with a PN102 seed after
# it). The third shape is a prior turn's closed region followed by this turn's
# open one — thinking-ON, and the last-marker-wins scan has to say so.
IDS_OFF = [10, 11, 12, THINK_START, THINK_END]
IDS_ON = [10, 11, 12, THINK_START]
IDS_ON_SEEDED = [10, 11, THINK_START, 77, 78, 79]
IDS_ON_PRIOR_TURN = [THINK_START, THINK_END, 20, 21, THINK_START, 77]
IDS_RAW = [10, 11, 12, 13]  # /v1/completions: no markers either way


def add(ids, params) -> tuple[bool, dict]:
    """One sync_batch add, faithful to site F: upstream builds the entry for a
    budgeted row BEFORE the shim is called, and calls the shim first on the
    unbudgeted path."""
    h = FakeHolder()
    if params.thinking_token_budget is not None:
        h._state[0] = h._init_state_entry(ids, params.thinking_token_budget)
    installed = R.h119_on_batch_add(h, 0, params, ids, [])
    return installed, h._state


print("\n[9] unstamped + thinking-OFF render -> H119 declines (the 101 rows)")
inst, st = add(IDS_OFF, FakeParams())
check("no provisional entry installed", inst is False and st == {}, f"state={st}")

print("\n[10] unstamped + thinking-ON render -> unchanged (provisional deep)")
for label, ids in (("open <think>", IDS_ON),
                   ("seeded <think>", IDS_ON_SEEDED),
                   ("prior turn closed, this turn open", IDS_ON_PRIOR_TURN)):
    inst, st = add(ids, FakeParams())
    check(label, inst is True and st.get(0, {}).get("thinking_token_budget") == 10240,
          f"installed={inst} state={st.get(0, {}).get('thinking_token_budget')}")

print("\n[11] no markers in the prompt -> never CLAIMS off (fail-safe)")
inst, st = add(IDS_RAW, FakeParams())
check("raw prompt still routed", inst is True, f"installed={inst}")

print("\n[12] PN100 stamp=0 still keeps H119 out (the cheap first line)")
inst, st = add(IDS_OFF, FakeParams(xargs={"h119_overridable": 0}))
check("stamped tier-0 declined", inst is False and st == {})
inst, st = add(IDS_ON, FakeParams(xargs={"h119_overridable": 0}))
check("stamped tier-0 declined on a thinking-ON render too",
      inst is False and st == {})

print("\n[13] an explicit CALLER budget is still never overridden")
inst, st = add(IDS_ON, FakeParams(budget=2048))
check("unstamped caller budget untouched",
      inst is False and st[0]["thinking_token_budget"] == 2048
      and R.H119_PROVISIONAL not in st[0], f"state={st}")

print("\n[14] markers unavailable -> the guard cannot fire at all")
saved_start = FakeRouter._think_start_ids
FakeRouter._think_start_ids = []
inst, _ = add(IDS_OFF, FakeParams())
check("empty marker pattern does not disable the consumer", inst is True)
FakeRouter._think_start_ids = saved_start

print("\n[15] H119_RESPECT_THINKING_OFF=0 restores pre-fix behaviour")
os.environ["H119_RESPECT_THINKING_OFF"] = "0"
inst, st = add(IDS_OFF, FakeParams())
check("flag off: provisional deep granted again", inst is True)
os.environ.pop("H119_RESPECT_THINKING_OFF")

print("\n[16] end to end: frontend stamp -> worker decline")
for name, kw in OFF_FORMS.items():
    r = req(**kw)
    run(r)
    params = FakeParams(budget=r.thinking_token_budget,
                        xargs=getattr(r, "vllm_xargs", None))
    # enable_thinking=False means the template pre-closes: IDS_OFF is what the
    # worker sees for every one of these.
    inst, st = add(IDS_OFF, params)
    check(name, inst is False and st == {},
          f"installed={inst} budget={params.thinking_token_budget!r} "
          f"xargs={params.extra_args}")

print("\n[17] and the unspecified request still reaches the router")
r = req()
run(r)
params = FakeParams(budget=r.thinking_token_budget,
                    xargs=getattr(r, "vllm_xargs", None))
inst, st = add(IDS_ON, params)
check("PN100-budgeted row marked provisional for H119 as before",
      inst is False and st[0].get(R.H119_PROVISIONAL) is True
      and R.STATS.get("h119_pn100_override", 0) > 0,
      f"installed={inst} override_count={R.STATS.get('h119_pn100_override')} "
      f"state={st}")

print("\n[18] the decline is counted where the health surface can see it")
check("h119_prompt_thinking_off counter bumped",
      R.STATS.get("h119_prompt_thinking_off", 0) >= 1,
      f"count={R.STATS.get('h119_prompt_thinking_off')}")
check("PN100 explicit_off counter bumped",
      AB.get_stats().get("explicit_off", 0) >= len(OFF_FORMS),
      f"count={AB.get_stats().get('explicit_off')}")

print()
if _fails:
    print(f"FAILED ({len(_fails)}): " + ", ".join(_fails))
    sys.exit(1)
print("all thinking-off bypass checks passed")
