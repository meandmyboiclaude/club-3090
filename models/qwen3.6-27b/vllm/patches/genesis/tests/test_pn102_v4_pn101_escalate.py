#!/usr/bin/env python3
"""Offline tests for PN102 v4 (static banner) + PN101 escalation.

The engine round-trip (does an unclosed <think> partial survive template render
and vLLM's continue_final_message containment check?) cannot be tested without a
GPU. Everything else in the leg can: trigger conditions, field reads, the merge,
usage summing, finish_reason correction, and the fail-safe guarantee. Those are
pure Python, and a bug in any of them would waste a bench window.

Run: python3 test_pn102_v4_pn101_escalate.py
"""

import asyncio
import os
import sys
from pathlib import Path

MOD_DIR = Path(__file__).resolve().parents[1] / "vllm" / "_genesis" / "middleware"
sys.path.insert(0, str(MOD_DIR))

os.environ["GENESIS_ENABLE_PN101_ANSWER_RESCUE"] = "1"
os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"

import answer_rescue as ar  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# ─── fakes ───────────────────────────────────────────────────────────────────


class Usage:
    def __init__(self, prompt=100, completion=50):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class Message:
    def __init__(self, content=None, reasoning=None):
        self.content = content
        self.reasoning = reasoning
        self.reasoning_content = None
        self.tool_calls = None


class Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class Result:
    def __init__(self, message, finish_reason, usage=None):
        self.choices = [Choice(message, finish_reason)]
        self.usage = usage or Usage()


class Request:
    """Mimics the pydantic request model: constructible from arbitrary fields."""

    model_fields = {
        "model": 1, "messages": 1, "temperature": 1, "stream": 1,
        "thinking_token_budget": 1, "chat_template_kwargs": 1,
        "max_tokens": 1, "continue_final_message": 1, "add_generation_prompt": 1,
    }

    _DEFAULTS = {
        "model": "qwen3.6",
        "messages": None,
        "temperature": 0.0,
        "stream": False,
        "thinking_token_budget": 4096,
        "chat_template_kwargs": None,
        "max_tokens": 2048,
        "tools": None,
        "response_format": None,
        "continue_final_message": None,
        "add_generation_prompt": None,
    }

    def __init__(self, budget=None, stream=None, ctk=None, tools=None,
                 response_format=None, max_tokens=None, **kwargs):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
        # convenience aliases used by the tests
        if budget is not None:
            self.thinking_token_budget = budget
        if stream is not None:
            self.stream = stream
        if ctk is not None:
            self.chat_template_kwargs = ctk
        if max_tokens is not None:
            self.max_tokens = max_tokens
        self.tools = tools
        self.response_format = response_format
        # real-model construction path: req_cls(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.messages is None:
            self.messages = [{"role": "user", "content": "hard question"}]
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {}


class Serving:
    """Stands in for the OpenAIServingChat instance."""

    def __init__(self, reply_content=None, reply_reasoning=None,
                 finish="stop", raise_exc=None, usage=None):
        self.reply_content = reply_content
        self.reply_reasoning = reply_reasoning
        self.finish = finish
        self.raise_exc = raise_exc
        self.usage = usage
        self.calls: list = []

    async def create_chat_completion(self, request, raw_request=None):
        self.calls.append(request)
        if self.raise_exc:
            raise self.raise_exc
        msg = Message(content=self.reply_content, reasoning=self.reply_reasoning)
        return Result(msg, self.finish, self.usage or Usage(0, 900))


# ─── PN102 banner selection ──────────────────────────────────────────────────


def test_banner():
    print("\nPN102 banner")

    os.environ.pop("GENESIS_PN102_STATIC_BANNER", None)
    r = Request(budget=4096, ctk={"pn100_steps": 12})
    ar.maybe_add_answer_hint(r)
    v3 = r.chat_template_kwargs.get("pn_env_banner", "")
    check("v3 is the default path", "about 12 short reasoning steps" in v3, f"got: {v3[:70]}")
    check("v3 seed ends mid-reasoning (BUG-075)",
          r.chat_template_kwargs["pn_env_seed"].rstrip().endswith("Step 1:"))

    os.environ["GENESIS_PN102_STATIC_BANNER"] = "1"
    r = Request(budget=10240, ctk={"pn100_steps": 12})
    ar.maybe_add_answer_hint(r)
    v4 = r.chat_template_kwargs.get("pn_env_banner", "")
    check("v4 mentions the checkpoint", "Around Step 10" in v4, f"got: {v4[:70]}")
    check("v4 states no budget/step-count", "10240" not in v4 and "12 short" not in v4)
    check("v4 licenses stopping early", "stop reasoning and give" in v4)
    check("v4 frames continuing as normal", "keep going" in v4)
    check("v4 drops answer-shape steering", "sentence" not in v4.lower())
    check("v4 seed ends mid-reasoning (BUG-075)",
          r.chat_template_kwargs["pn_env_seed"] == "Step 1:")
    check("v4 discards the planner estimate", "pn100_steps" not in r.chat_template_kwargs)

    # v4 must be identical regardless of budget — that is the whole point.
    r2 = Request(budget=1024)
    ar.maybe_add_answer_hint(r2)
    check("v4 is budget-invariant",
          r2.chat_template_kwargs["pn_env_banner"] == v4)

    # gating
    r3 = Request(budget=0)
    ar.maybe_add_answer_hint(r3)
    check("no banner without an assigned budget", "pn_env_banner" not in r3.chat_template_kwargs)

    r4 = Request(budget=4096, tools=[{"type": "function"}])
    ar.maybe_add_answer_hint(r4)
    check("no banner on tool requests", "pn_env_banner" not in r4.chat_template_kwargs)

    r5 = Request(budget=4096, response_format={"type": "json_object"})
    ar.maybe_add_answer_hint(r5)
    check("no banner on structured requests", "pn_env_banner" not in r5.chat_template_kwargs)

    r6 = Request(budget=4096, ctk={"pn_env_banner": "already"})
    ar.maybe_add_answer_hint(r6)
    check("idempotent", r6.chat_template_kwargs["pn_env_banner"] == "already")


# ─── PN101 escalation ────────────────────────────────────────────────────────


def test_escalation():
    print("\nPN101 escalation triggers")
    os.environ["GENESIS_PN101_ESCALATE"] = "1"

    def run(result, serving, request=None):
        req = request or Request()
        return asyncio.run(ar._maybe_escalate(serving, req, result)), serving

    # starvation: burned the budget still reasoning
    res = Result(Message(content="", reasoning="Step 1: thinking..."), "length")
    srv = Serving(reply_content="42", reply_reasoning=" Step 9: done.")
    fired, srv = run(res, srv)
    check("fires on length + empty content", fired is True)
    check("issues exactly one continuation", len(srv.calls) == 1)
    if srv.calls:
        partial = srv.calls[0].messages[-1]["content"]
        check("partial opens an unclosed think block",
              partial.startswith("<think>\n") and "</think>" not in partial,
              f"got: {partial[:40]!r}")
        check("partial carries prior reasoning", "Step 1: thinking" in partial)
        check("continuation is marked internal",
              srv.calls[0].chat_template_kwargs.get("pn101_internal") is True)
        check("continuation gets the escalation budget",
              srv.calls[0].thinking_token_budget == 10240)
    msg = res.choices[0].message
    check("reasoning is concatenated, not replaced",
          msg.reasoning == "Step 1: thinking... Step 9: done.", f"got: {msg.reasoning!r}")
    check("answer lands in content", msg.content == "42")
    check("finish_reason corrected to stop", res.choices[0].finish_reason == "stop")
    check("usage summed across passes", res.usage.completion_tokens == 950)

    # R24 row 9: closed the think block and emitted nothing
    res = Result(Message(content="", reasoning="Step 1: hmm"), "stop")
    fired, _ = run(res, Serving(reply_content="7", reply_reasoning=" more"))
    check("fires on stop + empty content (R24 gap)", fired is True)

    # non-triggers
    res = Result(Message(content="The answer is 42.", reasoning="Step 1:"), "length")
    fired, srv = run(res, Serving(reply_content="x"))
    check("does not fire when content exists", fired is False)
    check("no continuation issued", len(srv.calls) == 0)

    res = Result(Message(content="", reasoning=""), "length")
    fired, _ = run(res, Serving(reply_content="x"))
    check("does not fire with no reasoning to continue", fired is False)

    os.environ["GENESIS_PN101_ESCALATE"] = "0"
    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    fired, srv = run(res, Serving(reply_content="x"))
    check("respects the flag being off", fired is False and len(srv.calls) == 0)
    os.environ["GENESIS_PN101_ESCALATE"] = "1"

    print("\nPN101 escalation fail-safe")

    # the risk flagged as unverifiable offline: containment rejects the partial
    res = Result(Message(content="", reasoning="Step 1: x"), "length", Usage(10, 20))
    fired, _ = run(res, Serving(raise_exc=ValueError(
        "The request cannot be continued: rendered prompt does not end with content")))
    check("containment rejection returns False", fired is False)
    m = res.choices[0].message
    check("original response untouched on exception",
          m.content == "" and m.reasoning == "Step 1: x"
          and res.choices[0].finish_reason == "length")
    check("usage untouched on exception", res.usage.completion_tokens == 20)

    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    fired, _ = run(res, Serving(reply_content="", reply_reasoning=""))
    check("empty continuation returns False", fired is False)

    # continuation that itself runs out: reasoning extended, no answer yet ->
    # falls through to the repair leg rather than escalating again
    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    fired, _ = run(res, Serving(reply_content="", reply_reasoning=" Step 2: y", finish="length"))
    check("continuation without an answer reports False", fired is False)
    check("but its reasoning is still kept",
          res.choices[0].message.reasoning == "Step 1: x Step 2: y")

    print("\nPN101 escalation gating in maybe_rescue_answer")

    async def rescue(req, result, serving):
        return await ar.maybe_rescue_answer(serving, req, result)

    req = Request(stream=True)
    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    srv = Serving(reply_content="42")
    asyncio.run(rescue(req, res, srv))
    check("streaming requests are never escalated", len(srv.calls) == 0)

    req = Request(response_format={"type": "json_object"})
    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    srv = Serving(reply_content="42")
    asyncio.run(rescue(req, res, srv))
    check("structured requests are never escalated", len(srv.calls) == 0)

    req = Request(budget=0)
    res = Result(Message(content="", reasoning="Step 1: x"), "length")
    srv = Serving(reply_content="42")
    asyncio.run(rescue(req, res, srv))
    check("unbudgeted requests are never escalated", len(srv.calls) == 0)


def test_stats():
    print("\nstats counters")
    s = ar.get_stats()
    for k in ("escalations_attempted", "escalations_succeeded", "escalation_errors"):
        check(f"{k} exposed", k in s)
    check("attempts were counted", s["escalations_attempted"] > 0)
    check("successes were counted", s["escalations_succeeded"] > 0)
    check("errors were counted", s["escalation_errors"] > 0)


if __name__ == "__main__":
    test_banner()
    test_escalation()
    test_stats()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("all offline checks passed")
