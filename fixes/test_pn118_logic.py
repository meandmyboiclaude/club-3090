#!/usr/bin/env python3
"""Offline tests for PN118 — premature-close gate (answer_rescue.py Leg 3).

The engine round-trip (does an unclosed <think> partial survive template render
+ vLLM's continue_final_message containment check?) needs a GPU. Everything
else is pure Python and a bug in any of it would waste a bench window:
  - the trigger truth-table (early+weak fires; early+confident no; late no;
    cap-bound no; disabled no; second fire blocked)
  - fail-open on every raised path (margin read, continuation call)
  - the continuation budget arithmetic (leftover clamped to [512, 6144])
  - the first-person cue text present in the continuation request

Run: python3 test_pn118_logic.py
"""

import asyncio
import os
import sys
from pathlib import Path

MOD_DIR = (Path(__file__).resolve().parents[1]
           / "models" / "qwen3.6-27b" / "vllm" / "patches" / "genesis"
           / "vllm" / "_genesis" / "middleware")
sys.path.insert(0, str(MOD_DIR))

import answer_rescue as ar  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def reset_env() -> None:
    for k in list(os.environ):
        if k.startswith("GENESIS_PN118") or k == "GENESIS_ENABLE_PN118_CLOSEGATE":
            del os.environ[k]
    # PN101 master OFF so we isolate PN118 behaviour in maybe_rescue_answer.
    os.environ.pop("GENESIS_ENABLE_PN101_ANSWER_RESCUE", None)
    for k in list(ar._STATS):
        ar._STATS[k] = 0


# ─── fakes ───────────────────────────────────────────────────────────────────


class CTD:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class Usage:
    def __init__(self, prompt=100, completion=50, reasoning=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.completion_tokens_details = CTD(reasoning) if reasoning is not None else None


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
    def __init__(self, message, finish_reason, usage=None, rid=None):
        self.choices = [Choice(message, finish_reason)]
        self.usage = usage or Usage()
        self.id = rid  # ChatCompletionResponse.id == engine req_id (join key)


class TopLP:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class TokLP:
    def __init__(self, top_logprobs):
        self.top_logprobs = top_logprobs


class LogProbs:
    def __init__(self, content):
        self.content = content


class LPChoice:
    """A choice carrying logprobs for the 1-token echo self-call."""
    def __init__(self, letter_lps):
        self.message = Message(content="", reasoning=None)
        self.finish_reason = "stop"
        self.logprobs = LogProbs([TokLP([TopLP(t, lp) for t, lp in letter_lps])])


class LPResult:
    def __init__(self, letter_lps):
        self.choices = [LPChoice(letter_lps)]
        self.usage = Usage(0, 1)


class Request:
    model_fields = {
        "model": 1, "messages": 1, "temperature": 1, "stream": 1,
        "thinking_token_budget": 1, "chat_template_kwargs": 1, "max_tokens": 1,
        "continue_final_message": 1, "add_generation_prompt": 1,
        "logprobs": 1, "top_logprobs": 1, "top_p": 1,
    }
    _DEFAULTS = {
        "model": "qwen3.6", "messages": None, "temperature": 0.0, "stream": False,
        "thinking_token_budget": 8000, "chat_template_kwargs": None,
        "max_tokens": 2048, "tools": None, "response_format": None,
        "continue_final_message": None, "add_generation_prompt": None,
    }

    def __init__(self, budget=None, **kwargs):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
        if budget is not None:
            self.thinking_token_budget = budget
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.messages is None:
            self.messages = [{"role": "user", "content": "hard MCQ"}]
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {}


class Serving:
    """First self-call = margin echo (returns LPResult), second = continuation
    (returns a Result). Either can be forced to raise."""
    def __init__(self, margin_letters=None, cont_content=None, cont_reasoning=None,
                 raise_on_margin=None, raise_on_cont=None):
        self.margin_letters = margin_letters
        self.cont_content = cont_content
        self.cont_reasoning = cont_reasoning
        self.raise_on_margin = raise_on_margin
        self.raise_on_cont = raise_on_cont
        self.calls: list = []

    async def create_chat_completion(self, request, raw_request=None):
        self.calls.append(request)
        is_margin = getattr(request, "logprobs", None) is True
        if is_margin:
            if self.raise_on_margin:
                raise self.raise_on_margin
            return LPResult(self.margin_letters or [])
        if self.raise_on_cont:
            raise self.raise_on_cont
        return Result(Message(content=self.cont_content, reasoning=self.cont_reasoning),
                      "stop", Usage(0, 500))


def make_result(content="The answer is B.", reasoning="Step 1: ...", spent=1000,
                finish="stop", rid=None):
    return Result(Message(content=content, reasoning=reasoning),
                  finish, Usage(0, 900, reasoning=spent), rid=rid)


# weak = near-tie letters (margin ~0.05); strong = dominant (margin ~0.85)
WEAK = [("A", -1.7), ("B", -1.8), ("C", -2.5)]
STRONG = [("B", -0.1), ("A", -2.4), ("C", -3.0)]


def run(serving, request, result):
    return asyncio.run(ar.maybe_rescue_answer(serving, request, result))


# ─── margin read unit ────────────────────────────────────────────────────────


def test_margin_read():
    print("\nletter-margin posterior read")
    import math
    weak = ar._letter_posterior_margin(LPChoice(WEAK))
    strong = ar._letter_posterior_margin(LPChoice(STRONG))
    check("weak margin < 0.5", weak is not None and weak < 0.5, f"got {weak}")
    check("strong margin >= 0.5", strong is not None and strong >= 0.5, f"got {strong}")
    check("no-letter mass → None",
          ar._letter_posterior_margin(LPChoice([("(", -0.1), ("the", -2.0)])) is None)
    expect = math.exp(-1.7) - math.exp(-1.8)
    check("margin = top1-top2", weak is not None and abs(weak - expect) < 1e-9,
          f"{weak} vs {expect}")


# ─── trigger truth-table (enforce mode) ──────────────────────────────────────


def test_disabled():
    print("\ntruth-table: master disabled → no self-calls, no fire")
    reset_env()
    s = Serving(margin_letters=WEAK, cont_content="Actually the answer is C.")
    run(s, Request(budget=8000), make_result(spent=1000))
    check("disabled: zero self-calls", len(s.calls) == 0, f"{len(s.calls)}")
    check("disabled: no fire", ar._STATS["pn118_fires"] == 0)


def test_early_weak_fires():
    print("\ntruth-table: early close + weak answer → FIRES (enforce)")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="Actually the answer is C.")
    res = make_result(content="The answer is B.", spent=1000)  # 1000/8000 = 12.5%
    run(s, Request(budget=8000), res)
    check("fired once", ar._STATS["pn118_fires"] == 1, str(ar._STATS))
    check("two self-calls (margin + continuation)", len(s.calls) == 2, f"{len(s.calls)}")
    check("answer spliced from continuation",
          res.choices[0].message.content == "Actually the answer is C.")


def test_early_confident_no_fire():
    print("\ntruth-table: early close + confident answer → NO fire")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=STRONG, cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000)
    run(s, Request(budget=8000), res)
    check("no fire on confident", ar._STATS["pn118_fires"] == 0)
    check("margin read happened, continuation did NOT", len(s.calls) == 1, f"{len(s.calls)}")
    check("original answer preserved", res.choices[0].message.content == "The answer is B.")
    check("skip counted", ar._STATS["pn118_skips"] == 1)


def test_late_no_fire():
    print("\ntruth-table: late close (spent > FRAC*budget) → NO fire, no self-call")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="changed")
    res = make_result(content="The answer is B.", spent=6000)  # 6000/8000 = 75% > 0.6
    run(s, Request(budget=8000), res)
    check("no fire when late", ar._STATS["pn118_fires"] == 0)
    check("no self-call (gated before margin read)", len(s.calls) == 0, f"{len(s.calls)}")


def test_capbound_no_fire():
    print("\ntruth-table: cap-bound close (no leftover) → NO fire")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    os.environ["GENESIS_PN118_FRAC"] = "0.99"  # would pass frac gate; grace must catch it
    s = Serving(margin_letters=WEAK, cont_content="changed")
    res = make_result(content="The answer is B.", spent=7900)  # 8000-7900=100 < grace 256
    run(s, Request(budget=8000), res)
    check("no fire when cap-bound", ar._STATS["pn118_fires"] == 0)
    check("no self-call (grace gate before margin)", len(s.calls) == 0, f"{len(s.calls)}")


def test_unbounded_no_fire():
    print("\ntruth-table: no thinking budget → NO fire")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="changed")
    run(s, Request(budget=0), make_result(spent=1000))
    check("no fire when unbounded", ar._STATS["pn118_fires"] == 0)
    check("no self-call", len(s.calls) == 0)


def test_second_fire_blocked():
    print("\ntruth-table: one fire per request — re-invocation blocked")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="Actually the answer is C.")
    res = make_result(spent=1000)
    req = Request(budget=8000)
    run(s, req, res)
    calls_after_first = len(s.calls)
    run(s, req, res)  # same result object → _pn118_seen guard
    check("fired only once", ar._STATS["pn118_fires"] == 1)
    check("no extra self-calls on 2nd pass", len(s.calls) == calls_after_first)


def test_tool_and_structured_skip():
    print("\ntruth-table: tools / structured output → NO fire")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="changed")
    run(s, Request(budget=8000, tools=[{"type": "function"}]), make_result(spent=1000))
    check("tools → no self-call", len(s.calls) == 0)


# ─── shadow mode ─────────────────────────────────────────────────────────────


def test_shadow_mode():
    print("\nshadow mode: computes margin, logs would-fire, changes NOTHING")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "shadow"
    s = Serving(margin_letters=WEAK, cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000)
    run(s, Request(budget=8000), res)
    check("shadow: would-fire recorded", ar._STATS["pn118_shadow_would_fire"] == 1)
    check("shadow: no real fire", ar._STATS["pn118_fires"] == 0)
    check("shadow: margin read ran (1 self-call), NO continuation",
          len(s.calls) == 1, f"{len(s.calls)}")
    check("shadow: answer untouched", res.choices[0].message.content == "The answer is B.")


def test_shadow_default():
    print("\nshadow is the default mode (enabling master alone does not enforce)")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"  # no MODE set
    s = Serving(margin_letters=WEAK, cont_content="changed")
    res = make_result(spent=1000)
    run(s, Request(budget=8000), res)
    check("default mode → no enforce fire", ar._STATS["pn118_fires"] == 0)
    check("default mode → would-fire logged", ar._STATS["pn118_shadow_would_fire"] == 1)


# ─── fail-open on every raised path ──────────────────────────────────────────


def test_failopen_margin_raise():
    print("\nfail-open: margin echo call raises → original kept")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(raise_on_margin=RuntimeError("boom"))
    res = make_result(content="The answer is B.", spent=1000)
    run(s, Request(budget=8000), res)
    check("no fire", ar._STATS["pn118_fires"] == 0)
    check("error counted", ar._STATS["pn118_errors"] >= 1)
    check("answer untouched", res.choices[0].message.content == "The answer is B.")


def test_failopen_cont_raise():
    print("\nfail-open: continuation call raises → original kept")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, raise_on_cont=RuntimeError("boom"))
    res = make_result(content="The answer is B.", spent=1000)
    run(s, Request(budget=8000), res)
    check("no fire", ar._STATS["pn118_fires"] == 0)
    check("error counted", ar._STATS["pn118_errors"] >= 1)
    check("answer untouched", res.choices[0].message.content == "The answer is B.")


def test_failopen_empty_continuation():
    print("\nfail-open: continuation returns empty → original kept")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=WEAK, cont_content="", cont_reasoning="")
    res = make_result(content="The answer is B.", spent=1000)
    run(s, Request(budget=8000), res)
    check("no fire on empty continuation", ar._STATS["pn118_fires"] == 0)
    check("answer untouched", res.choices[0].message.content == "The answer is B.")


def test_no_letter_margin_skip():
    print("\nopen-ended answer (no letter mass) → skip, no continuation")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s = Serving(margin_letters=[("The", -0.1), ("answer", -2.0)], cont_content="changed")
    res = make_result(content="It is a long prose answer.", spent=1000)
    run(s, Request(budget=8000), res)
    check("no fire when margin None", ar._STATS["pn118_fires"] == 0)
    check("margin read ran but no continuation", len(s.calls) == 1)


# ─── continuation budget arithmetic + cue text ───────────────────────────────


def test_continuation_budget_and_cue():
    print("\ncontinuation: leftover budget clamped [512,6144] + cue present")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"

    # leftover = 8000-1000 = 7000 → clamp to 6144
    s = Serving(margin_letters=WEAK, cont_content="Actually C.")
    run(s, Request(budget=8000), make_result(spent=1000))
    cont = s.calls[-1]
    # C1 fix: total = spent + clamped room (engine re-charges the prefill)
    check("total = spent + room(clamped 6144)", cont.thinking_token_budget == 1000 + 6144,
          str(cont.thinking_token_budget))
    txt = cont.messages[-1]["content"]
    check("cue text present in continuation", ar._PN118_DEFAULT_CUE in txt)
    check("resumes inside think (open <think>, no close)",
          txt.startswith("<think>") and "</think>" not in txt)
    check("original think content carried", "Step 1:" in txt)

    # leftover small → clamp UP to 512.  spent=4700, budget=5000, frac 0.99, grace 50
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    os.environ["GENESIS_PN118_FRAC"] = "0.99"
    os.environ["GENESIS_PN118_GRACE"] = "50"
    s2 = Serving(margin_letters=WEAK, cont_content="Actually C.")
    run(s2, Request(budget=5000), make_result(spent=4700))
    check("total = spent + room(min 512)",
          s2.calls[-1].thinking_token_budget == 4700 + 512,
          str(s2.calls[-1].thinking_token_budget))

    # mid leftover unclamped: 8000-3000 = 5000
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    s3 = Serving(margin_letters=WEAK, cont_content="Actually C.")
    run(s3, Request(budget=8000), make_result(spent=3000))
    check("total = spent + mid room (3000+5000)",
          s3.calls[-1].thinking_token_budget == 3000 + 5000,
          str(s3.calls[-1].thinking_token_budget))


def test_custom_cue_env():
    print("\ncustom cue via GENESIS_PN118_CUE")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"
    os.environ["GENESIS_PN118_CUE"] = "Hold on, let me reconsider the other options."
    s = Serving(margin_letters=WEAK, cont_content="Actually C.")
    run(s, Request(budget=8000), make_result(spent=1000))
    check("custom cue used",
          "Hold on, let me reconsider the other options." in s.calls[-1].messages[-1]["content"])


def test_reasoning_tokens_fallback():
    print("\nspend read: usage.reasoning_tokens preferred, char/4 fallback")
    reset_env()
    # no reasoning_tokens on usage → fall back to len(reasoning)//4
    res = Result(Message(content="B", reasoning="x" * 400), "stop", Usage(0, 900))
    check("char/4 fallback", ar._reasoning_tokens(res, "x" * 400) == 100)
    res2 = Result(Message(content="B", reasoning="x" * 400), "stop",
                  Usage(0, 900, reasoning=2222))
    check("usage.reasoning_tokens preferred",
          ar._reasoning_tokens(res2, "x" * 400) == 2222)


# ─── PN118 c_mean gate (engine→serving bridge) ───────────────────────────────
# GENESIS_PN118_GATE = margin (default) | cmean | both. cmean reads pn112's
# exported rolling confidence from the /tmp bridge file instead of the letter-
# margin echo self-call. Calibration: wrong items c_mean ~9.16 vs 11.37 right;
# threshold GENESIS_PN118_CMEAN default 10.0 → fire (rescue) iff c_last < 10.0.

import json as _json  # noqa: E402
import time as _time  # noqa: E402
import tempfile as _tempfile  # noqa: E402


def _write_conf(entries: dict) -> str:
    """entries: {req_id: (c_last, n, age_seconds)}. age → ts = monotonic()-age."""
    fd, path = _tempfile.mkstemp(suffix="_pn112_conf.json")
    os.close(fd)
    now = _time.monotonic()
    obj = {rid: {"c_last": c, "n": n, "ts": now - age}
           for rid, (c, n, age) in entries.items()}
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(obj, f)
    ar._PN112_CONF_PATH = path
    return path


def _cmean_env(gate="cmean", mode="enforce"):
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = mode
    os.environ["GENESIS_PN118_GATE"] = gate


def test_cmean_below_fires_no_echo():
    print("\ncmean gate: low c_last + n>=MINN + fresh → FIRES, NO echo self-call")
    reset_env()
    _cmean_env()
    _write_conf({"chatcmpl-1": (9.0, 128, 1.0)})  # 9.0 < 10.0, n ok, fresh
    s = Serving(margin_letters=STRONG, cont_content="Actually the answer is C.")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-1")
    run(s, Request(budget=8000), res)
    check("fired via cmean", ar._STATS["pn118_fires"] == 1, str(ar._STATS))
    check("exactly ONE self-call (continuation only, no echo)", len(s.calls) == 1,
          f"{len(s.calls)}")
    check("answer spliced", res.choices[0].message.content == "Actually the answer is C.")


def test_cmean_above_no_fire():
    print("\ncmean gate: high c_last (settled) → NO fire")
    reset_env()
    _cmean_env()
    _write_conf({"chatcmpl-2": (11.4, 128, 1.0)})  # 11.4 >= 10.0
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-2")
    run(s, Request(budget=8000), res)
    check("no fire when confident", ar._STATS["pn118_fires"] == 0)
    check("no self-call at all", len(s.calls) == 0, f"{len(s.calls)}")
    check("skip counted", ar._STATS["pn118_skips"] == 1)


def test_cmean_missing_entry_no_fire():
    print("\ncmean gate: no conf entry for this req → NO fire (conservative)")
    reset_env()
    _cmean_env()
    _write_conf({"chatcmpl-other": (9.0, 128, 1.0)})
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-missing")
    run(s, Request(budget=8000), res)
    check("no fire on missing entry", ar._STATS["pn118_fires"] == 0)
    check("no self-call", len(s.calls) == 0)


def test_cmean_stale_no_fire():
    print("\ncmean gate: stale entry (ts older than TTL) → NO fire")
    reset_env()
    _cmean_env()
    os.environ["GENESIS_PN118_CMEAN_TTL_S"] = "600"
    _write_conf({"chatcmpl-3": (9.0, 128, 1200.0)})  # 1200s old > 600 TTL
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-3")
    run(s, Request(budget=8000), res)
    check("no fire on stale entry", ar._STATS["pn118_fires"] == 0)
    check("no self-call", len(s.calls) == 0)


def test_cmean_minn_no_fire():
    print("\ncmean gate: n < MINN (too few samples) → NO fire")
    reset_env()
    _cmean_env()
    os.environ["GENESIS_PN118_CMEAN_MINN"] = "64"
    _write_conf({"chatcmpl-4": (9.0, 40, 1.0)})  # 40 < 64
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-4")
    run(s, Request(budget=8000), res)
    check("no fire when under MINN", ar._STATS["pn118_fires"] == 0)
    check("no self-call", len(s.calls) == 0)


def test_cmean_id_normalization():
    print("\ncmean gate: engine key '-0' parallel-sample suffix joins result.id")
    reset_env()
    _cmean_env()
    # engine stored the per-sequence id; serving result.id lacks the suffix
    _write_conf({"chatcmpl-abc-0": (9.0, 128, 1.0)})
    s = Serving(cont_content="Actually C.")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-abc")
    run(s, Request(budget=8000), res)
    check("fired via normalized-id join", ar._STATS["pn118_fires"] == 1, str(ar._STATS))


def test_cmean_missing_file_failopen():
    print("\ncmean gate: conf file absent → fail-open NO fire")
    reset_env()
    _cmean_env()
    ar._PN112_CONF_PATH = "/tmp/does_not_exist_pn112_conf_xyz.json"
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-5")
    run(s, Request(budget=8000), res)
    check("no fire when file missing", ar._STATS["pn118_fires"] == 0)
    check("no self-call", len(s.calls) == 0)


def test_cmean_shadow_logs_no_fire():
    print("\ncmean gate shadow: would-fire recorded, NOTHING changed, no echo")
    reset_env()
    _cmean_env(mode="shadow")
    _write_conf({"chatcmpl-6": (9.0, 128, 1.0)})
    s = Serving(cont_content="changed")
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-6")
    run(s, Request(budget=8000), res)
    check("shadow would-fire recorded", ar._STATS["pn118_shadow_would_fire"] == 1)
    check("no real fire", ar._STATS["pn118_fires"] == 0)
    check("no self-call in cmean shadow", len(s.calls) == 0, f"{len(s.calls)}")
    check("answer untouched", res.choices[0].message.content == "The answer is B.")


def test_both_gate_and_semantics():
    print("\nboth gate: cmean AND margin must both pass")
    # cmean pass + margin pass → fire (echo call runs, then continuation)
    reset_env()
    _cmean_env(gate="both")
    _write_conf({"chatcmpl-7": (9.0, 128, 1.0)})  # cmean passes
    s = Serving(margin_letters=WEAK, cont_content="Actually C.")  # margin passes
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-7")
    run(s, Request(budget=8000), res)
    check("both-pass fires", ar._STATS["pn118_fires"] == 1, str(ar._STATS))
    check("both: echo + continuation = 2 self-calls", len(s.calls) == 2, f"{len(s.calls)}")

    # cmean pass + margin FAIL (confident) → no fire
    reset_env()
    _cmean_env(gate="both")
    _write_conf({"chatcmpl-8": (9.0, 128, 1.0)})
    s2 = Serving(margin_letters=STRONG, cont_content="changed")  # margin fails
    res2 = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-8")
    run(s2, Request(budget=8000), res2)
    check("both: cmean-pass margin-fail → no fire", ar._STATS["pn118_fires"] == 0)
    check("both: echo ran, no continuation", len(s2.calls) == 1, f"{len(s2.calls)}")

    # cmean FAIL → margin echo never even runs
    reset_env()
    _cmean_env(gate="both")
    _write_conf({"chatcmpl-9": (11.4, 128, 1.0)})  # cmean fails
    s3 = Serving(margin_letters=WEAK, cont_content="changed")
    res3 = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-9")
    run(s3, Request(budget=8000), res3)
    check("both: cmean-fail short-circuits before echo", len(s3.calls) == 0,
          f"{len(s3.calls)}")
    check("both: no fire", ar._STATS["pn118_fires"] == 0)


def test_margin_mode_ignores_conf_file():
    print("\ndefault margin gate: conf file never read (echo path unchanged)")
    reset_env()
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_PN118_MODE"] = "enforce"  # no GATE set → margin default
    _write_conf({"chatcmpl-10": (11.4, 128, 1.0)})  # confident — would block cmean
    s = Serving(margin_letters=WEAK, cont_content="Actually C.")  # but margin is weak
    res = make_result(content="The answer is B.", spent=1000, rid="chatcmpl-10")
    run(s, Request(budget=8000), res)
    check("margin gate fires on weak margin regardless of conf file",
          ar._STATS["pn118_fires"] == 1, str(ar._STATS))
    check("margin gate did the echo (2 calls)", len(s.calls) == 2, f"{len(s.calls)}")


def main():
    for t in (test_margin_read, test_disabled, test_early_weak_fires,
              test_early_confident_no_fire, test_late_no_fire, test_capbound_no_fire,
              test_unbounded_no_fire, test_second_fire_blocked,
              test_tool_and_structured_skip, test_shadow_mode, test_shadow_default,
              test_failopen_margin_raise, test_failopen_cont_raise,
              test_failopen_empty_continuation, test_no_letter_margin_skip,
              test_continuation_budget_and_cue, test_custom_cue_env,
              test_reasoning_tokens_fallback,
              test_cmean_below_fires_no_echo, test_cmean_above_no_fire,
              test_cmean_missing_entry_no_fire, test_cmean_stale_no_fire,
              test_cmean_minn_no_fire, test_cmean_id_normalization,
              test_cmean_missing_file_failopen, test_cmean_shadow_logs_no_fire,
              test_both_gate_and_semantics, test_margin_mode_ignores_conf_file):
        t()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL PN118 TESTS PASSED")


if __name__ == "__main__":
    main()
