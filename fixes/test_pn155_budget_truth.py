#!/usr/bin/env python3
"""PN155 / BUG-155 budget-truth guard — offline tests for answer_rescue.py Leg 5.

No GPU, no vLLM, no container, no network. `answer_rescue` imports only stdlib
at module scope (its `vllm.logger` import is already in a try/except), so this
collects and runs on a bare host interpreter:

    /usr/bin/python3 fixes/test_pn155_budget_truth.py     # standalone
    python -m pytest fixes/test_pn155_budget_truth.py -q  # where pytest exists

Every `test_*` below is a plain function with no fixtures, so both entry points
execute exactly the same assertions.

What is covered
  * `_pn155_is_empty` — the grammar's cheapest legal completion, and the three
    shapes that must NOT count (non-empty container, unparseable text, prose).
  * `_pn155_budget` — PN100's grant wins; the H119 flat-substitution case is
    recovered from the published route ONLY when the consumer flag is on and the
    router is enforcing.
  * `_pn155_spend` — `reasoning_tokens` preferred, then reasoning TEXT, then the
    BUG-158 fallback (`completion_tokens` minus the answer). The fallback is the
    live path on :8021 today: `reasoning_tokens` is 0 on every response there.
  * `_pn155_forced` — the exact holder signal when some future patch publishes
    it, from any of the three plausible seats; None when nobody has.
  * `_pn155_stamp` — details block preferred, usage as fallback, loud (not
    silent) on a frozen model.
  * the leg end to end: dark by default, inert on unguided traffic, fires only
    on (structured AND empty AND at/near cap), the three modes, the retry's
    recursion guard and its fall-through, and fail-open on any exception.
  * BUG-167 — the retry's ACCEPTANCE test. A truncated grammar-constrained
    retry (non-blank, unparseable, so `_pn155_is_empty` says False) must never
    be served, and must never be served under the FIRST pass's
    `finish_reason="stop"`: the parse gate rejects it back to the visible flag
    path, and a payload we DO serve carries the retry's own finish_reason.
  * placement: PN155 runs with the PN101 master flag OFF — the whole point,
    since every PN101 leg is gated behind `not _skip_common()` and that gate
    excludes exactly the structured requests BUG-155 lives in.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

MOD_DIR = (Path(__file__).resolve().parents[1]
           / "models" / "qwen3.6-27b" / "vllm" / "patches" / "genesis"
           / "vllm" / "_genesis" / "middleware")
sys.path.insert(0, str(MOD_DIR))

import answer_rescue as ar  # noqa: E402

_OWNED_ENV = (
    "GENESIS_ENABLE_PN155_BUDGET_TRUTH",
    "GENESIS_PN155_MODE",
    "GENESIS_PN155_STAMP_BUDGET",
    "GENESIS_PN155_MARGIN",
    "GENESIS_PN155_RETRY_MULT",
    "GENESIS_PN155_RETRY_CEIL",
    "GENESIS_PN155_TIMEOUT_S",
    "GENESIS_ENABLE_H119_ROUTE_BUDGET",
    "H119_DEEP_BUDGET",
    "H119_LEAN_BUDGET",
    "GENESIS_ENABLE_PN101_ANSWER_RESCUE",
    "GENESIS_ENABLE_PN123_CLOSEGATE",
    "GENESIS_ENABLE_PN118_CLOSEGATE",
    "GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT",
    "GENESIS_PN101_ESCALATE",
)


def reset_env() -> None:
    for key in _OWNED_ENV:
        os.environ.pop(key, None)
    for key in list(ar._STATS):
        ar._STATS[key] = 0
    os.environ["GENESIS_ENABLE_PN155_BUDGET_TRUTH"] = "1"
    os.environ["GENESIS_PN155_MODE"] = "observe"


# ─── fakes ───────────────────────────────────────────────────────────────────
# Plain objects, because that is what the real ones behave like: UsageInfo /
# CompletionTokenUsageInfo / ChatCompletionResponseChoice all derive from
# OpenAIBaseModel, whose model_config is ConfigDict(extra="allow") — an
# attribute set on an instance lands in __pydantic_extra__ and IS serialized
# (checked against the live pin: a stamped thinking_token_budget survives
# model_dump()).


class Details:
    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens
        self.accepted_prediction_tokens = None
        self.rejected_prediction_tokens = None


class Usage:
    def __init__(self, completion_tokens=0, details=None, prompt_tokens=59):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.completion_tokens_details = details


class FrozenUsage:
    """A usage block that refuses new attributes (pydantic extra='forbid')."""

    __slots__ = ("completion_tokens", "completion_tokens_details")

    def __init__(self, completion_tokens=0):
        self.completion_tokens = completion_tokens
        self.completion_tokens_details = None


class Message:
    def __init__(self, content=None, reasoning=None):
        self.content = content
        self.reasoning = reasoning
        self.reasoning_content = None
        self.tool_calls = None


class Choice:
    def __init__(self, message, finish_reason="stop"):
        self.index = 0
        self.message = message
        self.finish_reason = finish_reason
        self.stop_reason = None
        self.logprobs = None


_AUTO = object()   # "give me the default usage block"; None means NO usage


class Result:
    def __init__(self, content='{"facts": []}', usage=_AUTO,
                 finish_reason="stop", reasoning=None, kvt=None):
        self.choices = [Choice(Message(content, reasoning), finish_reason)]
        self.usage = Usage(completion_tokens=3907,
                           details=Details(0)) if usage is _AUTO else usage
        self.kv_transfer_params = kvt
        self.id = "chatcmpl-test"

    @property
    def choice(self):
        return self.choices[0]

    @property
    def details(self):
        return self.usage.completion_tokens_details


SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "facts", "strict": True, "schema": {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["facts"], "additionalProperties": False}},
}


class Request:
    """Pydantic-shaped enough for the module: model_fields + attributes."""

    model_fields = {
        "model": 1, "messages": 1, "temperature": 1, "stream": 1, "n": 1,
        "thinking_token_budget": 1, "chat_template_kwargs": 1,
        "max_completion_tokens": 1, "max_tokens": 1, "response_format": 1,
        "tools": 1, "top_p": 1, "top_k": 1, "seed": 1,
    }
    _DEFAULTS = {
        "model": "qwen3.6", "messages": None, "temperature": 0.6,
        "stream": False, "n": 1, "thinking_token_budget": 3900,
        "chat_template_kwargs": None, "max_completion_tokens": None,
        "max_tokens": 8192, "response_format": None, "tools": None,
        "top_p": 0.95, "top_k": 20, "seed": None,
    }

    def __init__(self, **kwargs):
        for key, val in self._DEFAULTS.items():
            setattr(self, key, val)
        for key, val in kwargs.items():
            setattr(self, key, val)
        if self.messages is None:
            self.messages = [{"role": "user", "content": "extract the facts"}]
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {}


def guided(**kwargs) -> Request:
    kwargs.setdefault("response_format", SCHEMA)
    return Request(**kwargs)


class Serving:
    """Records synthetic retries and answers them from a scripted queue.

    `finishes` scripts the RETRY's own finish_reason (BUG-167): the retry that
    hit its cap mid-schema comes back `length`, and the served choice has to say
    so instead of inheriting the first pass's `stop`.
    """

    def __init__(self, replies=(), raises=None, finishes=()):
        self.calls = []
        self.replies = list(replies)
        self.finishes = list(finishes)
        self.raises = raises

    async def create_chat_completion(self, request, raw_request=None):
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises
        content = self.replies.pop(0) if self.replies else '{"facts": []}'
        finish = self.finishes.pop(0) if self.finishes else "stop"
        return Result(content=content, finish_reason=finish,
                      usage=Usage(completion_tokens=900, details=Details(0)))


def run(coro):
    return asyncio.run(coro)


def guard(request, result, serving=None):
    run(ar._maybe_pn155_budget_truth(serving or Serving(), request, result))
    return result


# ─── _pn155_is_empty ─────────────────────────────────────────────────────────


def test_is_empty_containers():
    reset_env()
    for text in ('{"facts": []}', "[]", "{}", '  {"facts":[]}  ',
                 '{"facts": [], "notes": []}', '{"a": {}}'):
        assert ar._pn155_is_empty(text), text


def test_is_empty_rejects_content():
    reset_env()
    for text in ('{"facts": ["the build broke"]}', "[1]", '{"n": 0}',
                 '{"facts": [], "n": 1}', '{"ok": false}'):
        assert not ar._pn155_is_empty(text), text


def test_is_empty_rejects_non_json():
    # Unparseable output is a DIFFERENT failure and already visible to the
    # caller as broken JSON — claiming it here would mask it.
    reset_env()
    for text in ("", "   ", "I could not find any facts.", '{"facts": [',
                 None, 42):
        assert not ar._pn155_is_empty(text), repr(text)


# ─── flags ───────────────────────────────────────────────────────────────────


def test_dark_by_default():
    reset_env()
    os.environ.pop("GENESIS_ENABLE_PN155_BUDGET_TRUTH")
    assert ar._pn155_master_on() is False
    for val in ("1", "true", "YES", "on"):
        os.environ["GENESIS_ENABLE_PN155_BUDGET_TRUTH"] = val
        assert ar._pn155_master_on() is True
    os.environ["GENESIS_ENABLE_PN155_BUDGET_TRUTH"] = "0"
    assert ar._pn155_master_on() is False


def test_mode_defaults_to_observe():
    reset_env()
    os.environ.pop("GENESIS_PN155_MODE")
    assert ar._pn155_mode() == "observe"
    for val, want in (("flag", "flag"), ("RETRY", "retry"), ("observe", "observe"),
                      ("enforce", "observe"), ("", "observe")):
        os.environ["GENESIS_PN155_MODE"] = val
        assert ar._pn155_mode() == want, val


# ─── _pn155_budget ───────────────────────────────────────────────────────────


def test_budget_prefers_the_request_grant():
    reset_env()
    assert ar._pn155_budget(guided(thinking_token_budget=2100), Result()) == 2100


def test_budget_none_without_a_grant():
    reset_env()
    for val in (None, 0, -1, "3900"):
        assert ar._pn155_budget(guided(thinking_token_budget=val), Result()) is None


def test_budget_h119_flat_substitution():
    # With no prior to modulate, _h119_route_budget substitutes the flat
    # constant — the one case where the request-side value is WRONG, not merely
    # approximate. Recovered from the route the API bridge publishes.
    reset_env()
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "1"
    req = guided(thinking_token_budget=None)
    deep = Result(kvt={"h119": {"route": "deep", "mode": "enforce"}})
    lean = Result(kvt={"h119": {"route": "lean", "mode": "enforce"}})
    assert ar._pn155_budget(req, deep) == 10240
    assert ar._pn155_budget(req, lean) == 1600
    os.environ["H119_DEEP_BUDGET"] = "8192"
    assert ar._pn155_budget(req, deep) == 8192


def test_budget_h119_ignored_when_not_enforcing():
    reset_env()
    req = guided(thinking_token_budget=None)
    shadow = Result(kvt={"h119": {"route": "deep", "mode": "shadow"}})
    enforce = Result(kvt={"h119": {"route": "deep", "mode": "enforce"}})
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "1"
    assert ar._pn155_budget(req, shadow) is None
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "0"
    assert ar._pn155_budget(req, enforce) is None


def test_budget_h119_ignores_a_junk_payload():
    reset_env()
    os.environ["GENESIS_ENABLE_H119_ROUTE_BUDGET"] = "1"
    req = guided(thinking_token_budget=None)
    for kvt in (None, {}, {"h119": None}, {"h119": {"mode": "enforce"}},
                {"h119": {"route": "sideways", "mode": "enforce"}}, "nonsense"):
        assert ar._pn155_budget(req, Result(kvt=kvt)) is None, kvt


# ─── _pn155_spend ────────────────────────────────────────────────────────────


def test_spend_prefers_usage_reasoning_tokens():
    reset_env()
    res = Result(usage=Usage(completion_tokens=3907, details=Details(3899)))
    assert ar._pn155_spend(res, res.choice.message, '{"facts": []}') == (3899, "usage")


def test_spend_falls_back_to_reasoning_text():
    reset_env()
    res = Result(usage=Usage(completion_tokens=3907, details=Details(0)),
                 reasoning="x" * 4000)
    spend, src = ar._pn155_spend(res, res.choice.message, '{"facts": []}')
    assert (spend, src) == (1000, "reasoning_text")


def test_spend_bug158_fallback():
    # The LIVE path on :8021 today: reasoning_tokens 0, reasoning_content null,
    # and the whole think block is inside completion_tokens.
    reset_env()
    res = Result(usage=Usage(completion_tokens=3907, details=Details(0)))
    spend, src = ar._pn155_spend(res, res.choice.message, '{"facts": []}')
    assert src == "derived"
    assert spend == 3907 - len('{"facts": []}') // 4   # 3907 - 3


def test_spend_unavailable_without_any_signal():
    reset_env()
    res = Result(usage=Usage(completion_tokens=0, details=Details(0)))
    assert ar._pn155_spend(res, res.choice.message, "")[1] == "unavailable"
    res.usage = None
    assert ar._pn155_spend(res, res.choice.message, "")[1] == "unavailable"


# ─── _pn155_forced ───────────────────────────────────────────────────────────


def test_forced_none_when_nobody_publishes_it():
    reset_env()
    assert ar._pn155_forced(Result()) is None
    assert ar._pn155_forced(Result(usage=None)) is None


def test_forced_read_from_every_plausible_seat():
    reset_env()
    res = Result()
    res.details.budget_forced = True
    assert ar._pn155_forced(res) is True

    res = Result()
    res.details.censor_forced = False
    assert ar._pn155_forced(res) is False

    res = Result(usage=Usage(completion_tokens=10, details=None))
    res.usage.budget_forced = True
    assert ar._pn155_forced(res) is True

    res = Result(kvt={"h119": {"route": "deep", "censor_forced": True}})
    assert ar._pn155_forced(res) is True


# ─── _pn155_stamp ────────────────────────────────────────────────────────────


def test_stamp_prefers_the_details_block():
    reset_env()
    res = Result()
    assert ar._pn155_stamp(res, thinking_token_budget=3900) is True
    assert res.details.thinking_token_budget == 3900
    assert not hasattr(res.usage, "thinking_token_budget")


def test_stamp_falls_back_to_usage():
    reset_env()
    res = Result(usage=Usage(completion_tokens=10, details=None))
    assert ar._pn155_stamp(res, thinking_token_budget=2100) is True
    assert res.usage.thinking_token_budget == 2100


def test_stamp_is_false_without_usage():
    reset_env()
    assert ar._pn155_stamp(Result(usage=None), thinking_token_budget=1) is False


def test_stamp_survives_a_frozen_model():
    reset_env()
    res = Result(usage=FrozenUsage(completion_tokens=10))
    assert ar._pn155_stamp(res, thinking_token_budget=1) is False


# ─── the leg: observability half (3a) ────────────────────────────────────────


def test_budget_is_stamped_on_unguided_traffic_too():
    # Pure addition — it runs for every budgeted request. Today the grant is
    # reported NOWHERE, which is why the harness has to infer it from the grid.
    reset_env()
    res = guard(Request(thinking_token_budget=2100), Result(content="prose"))
    assert res.details.thinking_token_budget == 2100
    assert ar._STATS["pn155_stamped"] == 1
    assert ar._STATS["pn155_fired"] == 0


def test_stamp_can_be_turned_off_on_its_own():
    reset_env()
    os.environ["GENESIS_PN155_STAMP_BUDGET"] = "0"
    res = guard(guided(), Result())
    assert not hasattr(res.details, "thinking_token_budget")
    assert ar._STATS["pn155_fired"] == 1        # detection is independent


def test_no_budget_means_no_stamp_and_no_fire():
    reset_env()
    res = guard(guided(thinking_token_budget=None), Result())
    assert not hasattr(res.details, "thinking_token_budget")
    assert ar._STATS["pn155_seen"] == 0
    assert ar._STATS["pn155_fired"] == 0


# ─── the leg: detection half (3b) ────────────────────────────────────────────


def test_fires_on_the_prod016_shape():
    # prod-016 (aibox-20260726-guided-prodv3): 0 facts after rtok=3899 against
    # a 3900 grant, atok=8, finish=stop.
    reset_env()
    res = guard(guided(thinking_token_budget=3900),
                Result(usage=Usage(completion_tokens=3907, details=Details(0))))
    assert ar._STATS["pn155_fired"] == 1
    assert res.details.budget_empty is True
    assert res.details.thinking_token_budget == 3900
    assert res.choice.finish_reason == "stop"   # observe changes nothing


def test_never_fires_on_the_prod038_shape():
    # prod-038's chunk really is empty (`<local-command-stdout>Bye!`), rtok 158
    # against the same grant. Both arms and the 111-row champion agree on it.
    reset_env()
    res = guard(guided(thinking_token_budget=3900),
                Result(usage=Usage(completion_tokens=166, details=Details(0))))
    assert ar._STATS["pn155_fired"] == 0
    assert not hasattr(res.details, "budget_empty")


def test_never_fires_without_structured_output():
    # The leg must be unreachable on the unguided control (0 fires on the same
    # 40 items) — an unguided empty answer is PN101's escalate lane, not this.
    reset_env()
    res = guard(Request(thinking_token_budget=3900),
                Result(usage=Usage(completion_tokens=3907, details=Details(0))))
    assert ar._STATS["pn155_fired"] == 0
    assert not hasattr(res.details, "budget_empty")


def test_never_fires_on_a_cap_pinned_row_that_carried_data():
    # 25/40 guided rows pin at a ceiling; only ONE of them emptied out. The
    # other 24 must be untouched.
    reset_env()
    res = guard(guided(thinking_token_budget=2100),
                Result(content='{"facts": ["the build broke at 3pm"]}',
                       usage=Usage(completion_tokens=2110, details=Details(0))))
    assert ar._STATS["pn155_fired"] == 0
    assert res.choice.finish_reason == "stop"


def test_margin_is_configurable_and_bounds_the_fire():
    reset_env()
    def fire_at(spend, margin):
        reset_env()
        os.environ["GENESIS_PN155_MARGIN"] = str(margin)
        guard(guided(thinking_token_budget=2100),
              Result(usage=Usage(completion_tokens=spend + 3,
                                 details=Details(0))))
        return ar._STATS["pn155_fired"]
    assert fire_at(2087, 16) == 1        # gap 13 — the measured BUG-139 mode
    assert fire_at(2095, 16) == 1        # gap 5 — the other measured mode
    assert fire_at(2000, 16) == 0        # gap 100 — natural stop
    assert fire_at(2000, 200) == 1       # a wider margin does reach it


def test_holder_signal_overrides_the_threshold_both_ways():
    reset_env()
    # forced=False on a cap-pinned row: believe the holder, do not fire.
    res = Result(usage=Usage(completion_tokens=3907, details=Details(0)))
    res.details.budget_forced = False
    guard(guided(thinking_token_budget=3900), res)
    assert ar._STATS["pn155_fired"] == 0

    # forced=True far from the cap: the exact signal beats the arithmetic. This
    # is the false-negative the spec's known-limits section names.
    reset_env()
    res = Result(usage=Usage(completion_tokens=200, details=Details(0)))
    res.details.budget_forced = True
    guard(guided(thinking_token_budget=3900), res)
    assert ar._STATS["pn155_fired"] == 1
    assert res.details.budget_forced is True


def test_unknown_forced_is_not_stamped_as_false():
    # Absent means "nobody published the holder bit", which is not the same
    # claim as "the holder did not force it".
    reset_env()
    res = guard(guided(thinking_token_budget=3900), Result())
    assert res.details.budget_empty is True
    assert not hasattr(res.details, "budget_forced")


def test_no_spend_signal_refuses_to_guess():
    reset_env()
    res = guard(guided(thinking_token_budget=3900),
                Result(usage=Usage(completion_tokens=0, details=Details(0))))
    assert ar._STATS["pn155_fired"] == 0


# ─── the leg: modes ──────────────────────────────────────────────────────────


def test_flag_mode_tells_the_truth_and_keeps_the_original():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "flag"
    res = guard(guided(thinking_token_budget=3900), Result())
    assert res.choice.finish_reason == "length"
    assert res.choice.genesis_finish_reason_original == "stop"
    assert res.choice.stop_reason is None        # upstream field untouched
    assert res.details.budget_empty is True
    assert ar._STATS["pn155_flagged"] == 1


def test_flag_mode_leaves_the_payload_alone():
    # The empty array is what the model produced; the guard changes how the
    # response is LABELLED, never its content.
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "flag"
    res = guard(guided(thinking_token_budget=3900), Result())
    assert res.choice.message.content == '{"facts": []}'


def test_retry_serves_the_recovered_payload():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=['{"facts": ["the build broke", "ab12 reverted"]}'])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert ar._STATS["pn155_retries"] == 1
    assert ar._STATS["pn155_retry_rescued"] == 1
    assert json.loads(res.choice.message.content)["facts"]
    assert res.choice.finish_reason == "stop"    # a real answer arrived
    assert res.details.budget_empty is False
    assert ar._STATS["pn155_flagged"] == 0


def test_retry_carries_the_grammar_and_a_bigger_budget():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=['{"facts": ["x"]}'])
    guard(guided(thinking_token_budget=3900), Result(), serving)
    synthetic = serving.calls[0]
    assert synthetic.response_format == SCHEMA
    assert synthetic.thinking_token_budget == 7800
    assert synthetic.stream is False
    assert synthetic.top_p == 0.95 and synthetic.top_k == 20
    assert synthetic.max_completion_tokens == 7800 + 512
    assert synthetic.chat_template_kwargs[ar._PN155_MARKER_KEY] is True
    assert synthetic.chat_template_kwargs[ar._MARKER_KEY] is True


def test_retry_is_bounded_by_the_ceiling():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    os.environ["GENESIS_PN155_RETRY_CEIL"] = "4096"
    serving = Serving(replies=['{"facts": ["x"]}'])
    guard(guided(thinking_token_budget=3900), Result(), serving)
    assert serving.calls[0].thinking_token_budget == 4096

    # already at the ceiling -> no retry at all, straight to flag
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    os.environ["GENESIS_PN155_RETRY_CEIL"] = "3900"
    serving = Serving(replies=['{"facts": ["x"]}'])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert serving.calls == []
    assert res.choice.finish_reason == "length"


def test_retry_falls_through_to_flag_when_it_empties_again():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=['{"facts": []}'])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert ar._STATS["pn155_retries"] == 1
    assert ar._STATS["pn155_retry_rescued"] == 0
    assert res.choice.finish_reason == "length"
    assert res.details.budget_empty is True


# ─── BUG-167: the retry's acceptance test ────────────────────────────────────
# `_pn155_is_empty` returns False for anything unparseable, so a TRUNCATED
# grammar-constrained retry — a legal prefix of the schema, non-blank, invalid
# JSON — used to satisfy both accept conditions and be served under the FIRST
# pass's finish_reason="stop". Strictly worse than the well-formed empty payload
# PN155 exists to flag, and undetectable by any client.


TRUNCATED = '{"facts": [{"claim": "the build brok'


def test_a_truncated_retry_is_never_served_as_stop():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=[TRUNCATED], finishes=["length"])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    # the retry ran, and was REJECTED
    assert ar._STATS["pn155_retries"] == 1
    assert ar._STATS["pn155_retry_unparseable"] == 1
    assert ar._STATS["pn155_retry_rescued"] == 0
    # the ORIGINAL payload is what is served — the truncated one never lands
    assert res.choice.message.content == '{"facts": []}'
    # ... under the visible PN155 flag semantics
    assert res.choice.finish_reason == "length"
    assert res.choice.genesis_finish_reason_original == "stop"
    assert res.details.budget_empty is True
    assert ar._STATS["pn155_flagged"] == 1


def test_a_truncated_retry_is_rejected_even_when_it_says_stop():
    # The parse gate does not depend on the retry's finish_reason: a grammar
    # backend that reports `stop` on a cap-ended sequence must not slip past.
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=[TRUNCATED], finishes=["stop"])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert ar._STATS["pn155_retry_unparseable"] == 1
    assert res.choice.message.content == '{"facts": []}'
    assert res.choice.finish_reason == "length"


def test_a_parse_ok_retry_propagates_its_own_finish_reason():
    # Valid JSON with real content, but the retry itself ended at its cap. The
    # payload IS served (it parses and it is not empty) and the served choice
    # carries the RETRY's label, not the first pass's.
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=['{"facts": ["the build broke"]}'],
                      finishes=["length"])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert ar._STATS["pn155_retry_rescued"] == 1
    assert json.loads(res.choice.message.content)["facts"] == ["the build broke"]
    assert res.choice.finish_reason == "length"
    # the original is preserved, never destroyed — same contract as the flag path
    assert res.choice.genesis_finish_reason_original == "stop"
    assert res.details.budget_empty is False
    assert ar._STATS["pn155_flagged"] == 0


def test_a_clean_retry_leaves_the_finish_reason_alone():
    # finish_reason == the original: nothing to propagate, and no
    # genesis_finish_reason_original field invented for a no-op.
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    serving = Serving(replies=['{"facts": ["x"]}'], finishes=["stop"])
    res = guard(guided(thinking_token_budget=3900), Result(), serving)
    assert res.choice.finish_reason == "stop"
    assert not hasattr(res.choice, ar._PN155_ORIG_FR_FIELD)
    assert ar._STATS["pn155_retry_unparseable"] == 0


def test_the_parse_gate_only_applies_to_structured_requests():
    # `_pn155_retry` is reachable only from the structured branch today, but the
    # gate is written as a condition, not an assumption: prose from an unguided
    # caller must not be rejected as "unparseable".
    reset_env()
    serving = Serving(replies=["a plain prose answer"])
    msg = Message('{"facts": []}')
    res = Result()
    ok = run(ar._pn155_retry(serving, Request(thinking_token_budget=3900), res,
                             msg, res.choice, 3900))
    assert ok is True
    assert msg.content == "a plain prose answer"
    assert ar._STATS["pn155_retry_unparseable"] == 0


def test_a_raising_retry_still_flags():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    res = guard(guided(thinking_token_budget=3900), Result(),
                Serving(raises=RuntimeError("engine busy")))
    assert ar._STATS["pn155_errors"] == 1
    assert res.choice.finish_reason == "length"


def test_the_retry_never_recurses():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "retry"
    req = guided(thinking_token_budget=3900)
    req.chat_template_kwargs = {ar._PN155_MARKER_KEY: True}
    serving = Serving(replies=['{"facts": ["x"]}'])
    res = guard(req, Result(), serving)
    assert serving.calls == []
    assert ar._STATS["pn155_seen"] == 0
    assert res.choice.finish_reason == "stop"


def test_an_unknown_mode_degrades_to_observe():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "enforce"
    res = guard(guided(thinking_token_budget=3900), Result())
    assert ar._STATS["pn155_fired"] == 1
    assert res.choice.finish_reason == "stop"


# ─── placement and fail-open ─────────────────────────────────────────────────


def test_master_flag_off_is_byte_identical():
    reset_env()
    os.environ["GENESIS_ENABLE_PN155_BUDGET_TRUTH"] = "0"
    os.environ["GENESIS_PN155_MODE"] = "flag"
    res = Result()
    run(ar.maybe_rescue_answer(Serving(), guided(thinking_token_budget=3900), res))
    assert res.choice.finish_reason == "stop"
    assert not hasattr(res.details, "thinking_token_budget")
    assert ar._STATS["pn155_seen"] == 0


def test_runs_with_the_pn101_master_off():
    # The whole reason this is a separate leg: every PN101 leg is gated behind
    # `not _skip_common()`, and _skip_common() returns True for structured
    # requests — so PN155 must not depend on PN101's master flag.
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "flag"
    os.environ.pop("GENESIS_ENABLE_PN101_ANSWER_RESCUE", None)
    res = Result()
    run(ar.maybe_rescue_answer(Serving(), guided(thinking_token_budget=3900), res))
    assert res.choice.finish_reason == "length"
    assert ar._STATS["pn155_flagged"] == 1


def test_skipped_on_streaming():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "flag"
    res = Result()
    run(ar.maybe_rescue_answer(Serving(),
                               guided(thinking_token_budget=3900, stream=True), res))
    assert ar._STATS["pn155_seen"] == 0
    assert res.choice.finish_reason == "stop"


def test_fail_open_on_a_broken_response():
    reset_env()
    os.environ["GENESIS_PN155_MODE"] = "flag"

    class Exploding:
        choices = []
        kv_transfer_params = None

        @property
        def usage(self):
            raise RuntimeError("usage exploded")

    res = Exploding()
    run(ar.maybe_rescue_answer(Serving(), guided(thinking_token_budget=3900), res))
    assert ar._STATS["pn155_errors"] == 1
    assert ar._STATS["pn155_flagged"] == 0


def test_still_registered_with_the_patch_id_linter():
    # patch_id_lint.HOUSE_IDS_OUTSIDE_FIXES[PN155] asserts this exact string
    # lives in answer_rescue.py; deleting it turns the lint gate red.
    src = (MOD_DIR / "answer_rescue.py").read_text(encoding="utf-8")
    assert "GENESIS_ENABLE_PN155_BUDGET_TRUTH" in src


# ─── standalone runner ───────────────────────────────────────────────────────


def main() -> int:
    failures = []
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — this IS the reporter
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  PASS  {name}")
    reset_env()
    print()
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)}: {failures}")
        return 1
    print(f"ALL {len(tests)} PN155 / BUG-155 TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
