# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN100 shape-tier0 rules (GENESIS_PN100_SHAPE_TIER0).

[2026-07-25] The tier-0/engage defect (showdown verdict #1): under
thinking_budget:"auto" the classify step engages thinking on ~95% of rows
prod runs thinking-OFF. Fix = request-SHAPE rules (_shape_tier0) that land
structured-output shapes in tier 0 without a classify call, DEFAULT-DARK.

Covered:
  - flag OFF (default): _decide_mode identical to before, request untouched
  - flag ON: json_schema / json_object / guided-decoding / temp-0+small-cap /
    forced-named-tool shapes -> "shape0"; open-ended reasoning -> "classify"
  - explicit enable_thinking=True wins (no shape claim)
  - apply_hook_async on shape0: enable_thinking=False set, NO classify call
  - deliberately-unclaimed shapes stay on the classify path
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm._genesis.middleware import auto_budget as ab

FLAG = "GENESIS_PN100_SHAPE_TIER0"


def _req(**kw) -> SimpleNamespace:
    """Mimic ChatCompletionRequest with mutable attrs; None = field absent."""
    base = dict(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        chat_template_kwargs=None,
        response_format=None,
        temperature=None,
        max_tokens=None,
        max_completion_tokens=None,
        tool_choice=None,
        thinking_token_budget=None,
        reasoning_effort=None,
        reasoning=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _auto(**kw) -> SimpleNamespace:
    """A request explicitly routed thinking_budget:'auto' (the pilot shape)."""
    ctk = dict(kw.pop("chat_template_kwargs", None) or {})
    ctk["thinking_budget"] = "auto"
    return _req(chat_template_kwargs=ctk, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (FLAG, "GENESIS_PN100_SHAPE_SMALL_MAX_TOKENS",
                "GENESIS_PN100_SHAPE_MAX_TEMP", "GENESIS_PN100_AUTO_DEFAULT",
                "GENESIS_ENABLE_PN100_AUTO_BUDGET",
                "GENESIS_PN100_TIER_BUDGETS",
                "GENESIS_PN100_CLASSIFY_MAX_CHARS"):
        monkeypatch.delenv(var, raising=False)
    yield


# ─── flag OFF: behavior byte-identical to before ────────────────────────


class TestFlagOff:
    def test_shape_tier0_never_fires(self):
        r = _req(response_format={"type": "json_schema"}, temperature=0.0,
                 max_tokens=64)
        assert ab._shape_tier0(r) is False

    def test_auto_control_still_classifies(self):
        mode, allow = ab._decide_mode(_auto(
            response_format={"type": "json_schema"}))
        assert mode == "classify"
        assert allow is True

    def test_request_untouched(self):
        r = _auto(response_format={"type": "json_schema"})
        ab._decide_mode(r)
        # control key popped (pre-existing behavior), nothing else set
        assert r.thinking_token_budget is None
        assert "enable_thinking" not in (r.chat_template_kwargs or {})


# ─── flag ON: claimed shapes ────────────────────────────────────────────


class TestClaimedShapes:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")

    def test_fixture_a_temp0_tiny_cap_json_schema(self):
        # (a) from the task: temp-0 + tiny max_tokens + json schema -> tier 0
        r = _auto(temperature=0.0, max_tokens=200,
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": "x", "schema": {}}})
        assert ab._decide_mode(r) == ("shape0", True)

    def test_json_schema_alone(self):
        r = _auto(response_format={"type": "json_schema"})
        assert ab._decide_mode(r) == ("shape0", True)

    def test_json_object_alone(self):
        r = _auto(response_format={"type": "json_object"})
        assert ab._decide_mode(r) == ("shape0", True)

    def test_response_format_as_object_attr(self):
        r = _auto(response_format=SimpleNamespace(type="json_schema"))
        assert ab._decide_mode(r) == ("shape0", True)

    @pytest.mark.parametrize("attr", ["guided_json", "guided_regex",
                                      "guided_choice", "guided_grammar",
                                      "structured_outputs"])
    def test_guided_decoding_params(self, attr):
        r = _auto(**{attr: {"any": "thing"}})
        assert ab._decide_mode(r) == ("shape0", True)

    def test_temp0_small_cap_no_schema(self):
        r = _auto(temperature=0.0, max_tokens=300)
        assert ab._decide_mode(r) == ("shape0", True)

    def test_forced_named_tool_small_cap(self):
        r = _auto(max_tokens=256,
                  tool_choice={"type": "function",
                               "function": {"name": "lookup"}})
        assert ab._decide_mode(r) == ("shape0", True)

    def test_auto_default_lane_and_length_prefilter_precedence(self, monkeypatch):
        # No explicit control at all — server-default auto lane; the shape
        # rule must beat the length prefilter (long structured prompt).
        monkeypatch.setenv("GENESIS_PN100_AUTO_DEFAULT", "1")
        monkeypatch.setenv("GENESIS_PN100_CLASSIFY_MAX_CHARS", "100")
        r = _req(messages=[{"role": "user", "content": "x" * 5000}],
                 response_format={"type": "json_schema"})
        assert ab._decide_mode(r) == ("shape0", True)


# ─── flag ON: NOT-claimed shapes (conservatism) ─────────────────────────


class TestNotClaimed:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")

    def test_fixture_b_open_ended_reasoning_unchanged(self):
        # (b) from the task: a normal open-ended reasoning request
        r = _auto(messages=[{"role": "user", "content":
                             "Prove there are infinitely many primes 4k+3."}],
                  temperature=0.6, max_tokens=12288)
        assert ab._decide_mode(r) == ("classify", True)

    def test_explicit_enable_thinking_true_wins(self):
        r = _auto(response_format={"type": "json_schema"},
                  chat_template_kwargs={"enable_thinking": True})
        mode, allow = ab._decide_mode(r)
        assert mode == "classify"          # shape rule skipped entirely
        assert allow is False              # classify clamps tier 0 -> 1

    def test_text_response_format_not_claimed(self):
        r = _auto(response_format={"type": "text"}, temperature=0.6,
                  max_tokens=12288)
        assert ab._decide_mode(r) == ("classify", True)

    def test_temp0_large_cap_not_claimed(self):
        # temp-0 reasoning evals live here — must NOT go dark
        r = _auto(temperature=0.0, max_tokens=12288)
        assert ab._decide_mode(r) == ("classify", True)

    def test_low_but_nonzero_temp_not_claimed_by_default(self):
        r = _auto(temperature=0.1, max_tokens=300)
        assert ab._decide_mode(r) == ("classify", True)

    def test_tool_choice_required_not_claimed(self):
        r = _auto(max_tokens=256, tool_choice="required",
                  tools=[{"type": "function", "function": {"name": "a"}}])
        assert ab._decide_mode(r) == ("classify", True)

    def test_tool_choice_auto_not_claimed(self):
        r = _auto(max_tokens=256, tool_choice="auto",
                  tools=[{"type": "function", "function": {"name": "a"}}])
        assert ab._decide_mode(r) == ("classify", True)

    def test_forced_tool_large_cap_not_claimed(self):
        r = _auto(max_tokens=8192,
                  tool_choice={"type": "function",
                               "function": {"name": "lookup"}})
        assert ab._decide_mode(r) == ("classify", True)

    def test_explicit_budget_still_skips(self):
        r = _req(thinking_token_budget=2048,
                 response_format={"type": "json_schema"})
        assert ab._decide_mode(r) == ("skip", True)


# ─── end-to-end hook: shape0 sets thinking off, never calls classify ────


class _CountingServing:
    """Counts classify calls; each one fails (no engine here)."""

    def __init__(self):
        self.calls = 0

    async def create_chat_completion(self, *a, **kw):
        self.calls += 1
        raise RuntimeError("no engine in unit tests")


class TestApplyHook:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN100_AUTO_BUDGET", "1")
        monkeypatch.setenv(FLAG, "1")

    def test_shape0_disables_thinking_no_classify(self):
        r = _auto(temperature=0.0, max_tokens=200,
                  response_format={"type": "json_schema"})
        s = _CountingServing()
        before = ab.get_stats()["shape_tier0"]
        asyncio.run(ab.apply_hook_async(s, r))
        assert s.calls == 0                          # zero classify spend
        assert r.chat_template_kwargs.get("enable_thinking") is False
        assert r.thinking_token_budget is None       # tier 0 = OFF, no budget
        assert ab.get_stats()["shape_tier0"] == before + 1

    def test_nonzero_tier0_budget_respected(self, monkeypatch):
        # operator with budgets[0]>0 gets a small budget instead of OFF
        monkeypatch.setenv("GENESIS_PN100_TIER_BUDGETS", "256,1024,4096,10240")
        r = _auto(response_format={"type": "json_object"})
        s = _CountingServing()
        asyncio.run(ab.apply_hook_async(s, r))
        assert s.calls == 0
        assert r.chat_template_kwargs.get("enable_thinking") is True
        assert r.thinking_token_budget == 256

    def test_flag_off_hook_takes_classify_path(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        r = _auto(response_format={"type": "json_schema"})
        s = _CountingServing()
        before = ab.get_stats()["shape_tier0"]
        asyncio.run(ab.apply_hook_async(s, r))
        # flag off -> classify WAS attempted (pre-existing path), and its
        # engine failure fell back to the default tier — not shape0
        assert s.calls >= 1
        assert ab.get_stats()["shape_tier0"] == before
        assert r.chat_template_kwargs.get("enable_thinking") is True
