#!/usr/bin/env python3
"""Offline tests for BUG-157 / BUG-156 / BUG-144 in answer_rescue.py.

No GPU, no vLLM, no network — same shape as test_pn118_logic.py: the engine
round-trip is the only part that needs a card, and everything below it is pure
Python whose failure would cost a bench window.

  BUG-157 (Leg 1b, server-side deep/lean BANNER autosplit)
    - the wrapper installs once on the serving CLASS and is idempotent
    - master flag OFF  -> zero probes, request byte-identical
    - route "deep"     -> pn102_auto_v5 set -> the v5 banner is what renders
    - route "lean"     -> untouched -> the normal env chain renders (v3)
    - route unavailable / bridge missing / probe raises -> fail-open, served
    - the probe itself: max_tokens=1, stream off, markers on, same messages
    - candidate filters (internal, thinking-off, tools, structured, n>1,
      1-token, pre-rendered banner, caller already decided)
    - the promotion OBEYS the skip gates (this is what keeps BUG-156 fixed for
      structured requests) while the PN123 rerun's force_v5 still bypasses them

  BUG-156 (Leg 4, output-side banner echo net)
    - a leading "[envelope]" is stripped only when WE injected that banner
    - a reply that is only numbered steps is never emptied
    - the numbered-step shape is detect-only until opted in

  BUG-144 (PN118 -> PN123 renumber)
    - the new flag arms the close gate, the legacy flag still does, and the
      per-knob env resolution prefers the canonical name

Run: python3 test_bug157_autosplit.py
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

_OWNED_ENV = (
    "GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT",
    "GENESIS_PN102_AUTOSPLIT_V5_ROUTE",
    "GENESIS_PN102_AUTOSPLIT_TIMEOUT_S",
    "GENESIS_ENABLE_PN102_CONTRACT",
    "GENESIS_PN102_STRIP_ECHO",
    "GENESIS_PN102_STRIP_STEP_ECHO",
    "GENESIS_ENABLE_PN101_ANSWER_RESCUE",
    "GENESIS_ENABLE_PN123_CLOSEGATE",
    "GENESIS_ENABLE_PN118_CLOSEGATE",
)


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def reset_env() -> None:
    for k in list(os.environ):
        if (k.startswith("GENESIS_PN102") or k.startswith("GENESIS_PN118")
                or k.startswith("GENESIS_PN123") or k in _OWNED_ENV):
            del os.environ[k]
    for k in list(ar._STATS):
        ar._STATS[k] = 0
    ar._H119_BRIDGE = None


# ─── fakes ───────────────────────────────────────────────────────────────────


class Message:
    def __init__(self, content=None, reasoning=None):
        self.content = content
        self.reasoning = reasoning
        self.reasoning_content = None
        self.tool_calls = None


class Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class Result:
    def __init__(self, content="the answer", reasoning="Step 1: ...", tag="real"):
        self.choices = [Choice(Message(content, reasoning))]
        self.usage = None
        self.id = "chatcmpl-" + tag
        self.tag = tag
        self.kv_transfer_params = None


class Request:
    """Pydantic-shaped enough for the module: model_fields + model_copy."""

    model_fields = {
        "model": 1, "messages": 1, "temperature": 1, "stream": 1,
        "stream_options": 1, "n": 1, "thinking_token_budget": 1,
        "chat_template_kwargs": 1, "max_tokens": 1, "logprobs": 1,
        "top_logprobs": 1, "prompt_logprobs": 1, "echo": 1,
        "kv_transfer_params": 1, "response_format": 1, "tools": 1,
    }
    _DEFAULTS = {
        "model": "qwen3.6", "messages": None, "temperature": 0.6,
        "stream": False, "stream_options": None, "n": 1,
        "thinking_token_budget": 4096, "chat_template_kwargs": None,
        "max_tokens": 2048, "logprobs": None, "top_logprobs": None,
        "prompt_logprobs": None, "echo": None, "kv_transfer_params": None,
        "response_format": None, "tools": None,
    }

    def __init__(self, **kwargs):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.messages is None:
            self.messages = [{"role": "user", "content": "a hard question"}]
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {}

    def model_copy(self, update=None):
        clone = Request()
        for k in self._DEFAULTS:
            setattr(clone, k, getattr(self, k))
        clone.messages = list(self.messages)
        for k, v in (update or {}).items():
            setattr(clone, k, v)
        return clone


class Bridge:
    """Stand-in for vllm._genesis_h119_api."""

    def __init__(self, route="deep", on=True, payload_missing=False):
        self.route = route
        self.on = on
        self.payload_missing = payload_missing

    def enabled(self):
        return self.on

    def payload(self, obj):
        if self.payload_missing or self.route is None:
            return None
        return {"v": "h119.route/1", "route": self.route, "source": "score",
                "score": 0.71, "req_id": "engine-1"}


def make_serving(raise_on_probe=None, render_hint=True):
    """A fresh serving CLASS per test — the wrapper installs on the class."""

    class Serving:
        def __init__(self):
            self.calls = []

        async def create_chat_completion(self, request, raw_request=None):
            self.calls.append(request)
            ctk = getattr(request, "chat_template_kwargs", None) or {}
            is_probe = bool(ctk.get(ar._AUTOSPLIT_MARKER))
            if is_probe and raise_on_probe:
                raise raise_on_probe
            # emulate the PN101a hint site living inside this method
            if render_hint:
                ar.maybe_add_answer_hint(request)
            return Result(tag="probe" if is_probe else "real")

    return Serving


def call(serving, request, raw_request=None):
    return asyncio.run(serving.create_chat_completion(request, raw_request))


def banner_of(request):
    return (getattr(request, "chat_template_kwargs", None) or {}).get(
        "pn_env_banner", "")


V5_TELL = "The moment your answer is settled"
V3_TELL = "Thinking budget:"


# ─── BUG-157: install ────────────────────────────────────────────────────────


def test_install_idempotent():
    print("\nBUG-157 install: wraps the class once, idempotent, fail-open")
    reset_env()
    cls = make_serving()
    s = cls()
    check("first install lands", ar.install_route_autosplit(s) is True)
    wrapped = cls.create_chat_completion
    check("second install is a no-op",
          ar.install_route_autosplit(s) is False)
    check("method not re-wrapped", cls.create_chat_completion is wrapped)
    check("install never raises on junk",
          ar.install_route_autosplit(object()) is False)


# ─── BUG-157: the split itself ───────────────────────────────────────────────


def test_flag_off_is_inert():
    print("\nBUG-157 master flag OFF → no probe, banner unchanged")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    ar._H119_BRIDGE = Bridge("deep")
    cls = make_serving()
    s = cls()
    ar.install_route_autosplit(s)
    r = Request()
    call(s, r)
    check("no probe issued", len(s.calls) == 1, f"{len(s.calls)} calls")
    check("no probe counted", ar._STATS["autosplit_probes"] == 0)
    check("auto key never set",
          "pn102_auto_v5" not in (r.chat_template_kwargs or {}))
    check("default chain rendered v3", V3_TELL in banner_of(r), banner_of(r)[:60])


def test_deep_route_promotes_to_v5():
    print("\nBUG-157 route=deep → v5 banner, server-side, no client involvement")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    os.environ["GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT"] = "1"
    ar._H119_BRIDGE = Bridge("deep")
    cls = make_serving()
    s = cls()
    ar.install_route_autosplit(s)
    r = Request()
    out = call(s, r)
    check("probe + real request", len(s.calls) == 2, f"{len(s.calls)} calls")
    check("probe counted", ar._STATS["autosplit_probes"] == 1)
    check("deep counted", ar._STATS["autosplit_deep"] == 1)
    check("v5 banner rendered", V5_TELL in banner_of(r), banner_of(r)[:60])
    check("not the v3 banner", V3_TELL not in banner_of(r))
    check("the REAL response is returned", out.tag == "real")
    check("promotion key consumed by the hint",
          "pn102_auto_v5" not in (r.chat_template_kwargs or {}))


def test_probe_shape():
    print("\nBUG-157 probe: 1 token, non-streaming, marked, same messages")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    os.environ["GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT"] = "1"
    ar._H119_BRIDGE = Bridge("deep")
    cls = make_serving()
    s = cls()
    ar.install_route_autosplit(s)
    r = Request(max_tokens=4096)
    call(s, r)
    probe = s.calls[0]
    check("probe is first", probe is not s.calls[1])
    check("max_tokens=1", probe.max_tokens == 1, str(probe.max_tokens))
    check("stream off", probe.stream is False)
    check("n=1", probe.n == 1)
    check("marked internal",
          probe.chat_template_kwargs.get(ar._MARKER_KEY) is True)
    check("marked as our probe",
          probe.chat_template_kwargs.get(ar._AUTOSPLIT_MARKER) is True)
    check("same messages as the real request", probe.messages == r.messages)
    check("probe got NO banner (scored un-bannered, like the client protocol)",
          not probe.chat_template_kwargs.get("pn_env_banner"))
    check("real request cap untouched", r.max_tokens == 4096)


def test_lean_route_falls_through():
    print("\nBUG-157 route=lean → default env chain (v3 in prod)")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    os.environ["GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT"] = "1"
    ar._H119_BRIDGE = Bridge("lean")
    cls = make_serving()
    s = cls()
    ar.install_route_autosplit(s)
    r = Request()
    call(s, r)
    check("lean counted", ar._STATS["autosplit_lean"] == 1)
    check("deep not counted", ar._STATS["autosplit_deep"] == 0)
    check("v3 banner rendered", V3_TELL in banner_of(r), banner_of(r)[:60])


def test_route_inverted_by_env():
    print("\nBUG-157 GENESIS_PN102_AUTOSPLIT_V5_ROUTE selects which route wins")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    os.environ["GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT"] = "1"
    os.environ["GENESIS_PN102_AUTOSPLIT_V5_ROUTE"] = "lean"
    ar._H119_BRIDGE = Bridge("lean")
    cls = make_serving()
    s = cls()
    ar.install_route_autosplit(s)
    r = Request()
    call(s, r)
    check("lean promoted when configured", V5_TELL in banner_of(r))


def test_unavailable_paths_fail_open():
    print("\nBUG-157 fail-open: bridge missing / disabled / no payload / raises")
    for label, bridge, raiser in (
        ("bridge absent", False, None),
        ("route API disabled", Bridge("deep", on=False), None),
        ("no route published", Bridge("deep", payload_missing=True), None),
        ("probe raises", Bridge("deep"), RuntimeError("engine busy")),
    ):
        reset_env()
        os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
        os.environ["GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT"] = "1"
        ar._H119_BRIDGE = bridge
        cls = make_serving(raise_on_probe=raiser)
        s = cls()
        ar.install_route_autosplit(s)
        r = Request()
        out = call(s, r)
        check(f"{label}: request still served", out.tag == "real")
        check(f"{label}: fell back to the default banner",
              V3_TELL in banner_of(r), banner_of(r)[:50])
        if raiser is None:
            check(f"{label}: counted unavailable",
                  ar._STATS["autosplit_unavailable"] == 1)
        else:
            check(f"{label}: counted as an error",
                  ar._STATS["autosplit_errors"] == 1)


def test_real_bridge_import_is_absent_offline():
    print("\nBUG-157 the real bridge import is guarded (no vllm here)")
    reset_env()
    check("missing vllm._genesis_h119_api → None", ar._h119_bridge() is None)
    check("absence is cached as False", ar._H119_BRIDGE is False)


# ─── BUG-157: candidate filters ──────────────────────────────────────────────


def test_candidate_filters():
    print("\nBUG-157 pre-filters: no probe is spent on these")
    reset_env()
    cases = {
        "PN101 internal call":
            Request(chat_template_kwargs={ar._MARKER_KEY: True}),
        "PN100 internal call":
            Request(chat_template_kwargs={ar._PN100_MARKER_KEY: True}),
        "our own probe":
            Request(chat_template_kwargs={ar._AUTOSPLIT_MARKER: True}),
        "thinking off (mode b)":
            Request(chat_template_kwargs={"enable_thinking": False}),
        "caller already forced v5":
            Request(chat_template_kwargs={"pn102_force_v5": True}),
        "banner already rendered":
            Request(chat_template_kwargs={"pn_env_banner": "[envelope] ..."}),
        "tool request": Request(tools=[{"type": "function"}]),
        "structured request":
            Request(response_format={"type": "json_schema"}),
        "n>1 parallel sampling": Request(n=4),
        "1-token request": Request(max_tokens=1),
    }
    for label, req in cases.items():
        check(f"skipped: {label}", ar._autosplit_candidate(req) is False)
    check("a plain request IS a candidate",
          ar._autosplit_candidate(Request()) is True)


def test_promotion_obeys_skip_gates():
    print("\nBUG-157/156 the automatic promotion does NOT bypass the gates")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    # a structured request that somehow carries the promotion key must still
    # get NO banner — this is the guard that keeps BUG-156 fixed for guided.
    r = Request(response_format={"type": "json_schema"},
                chat_template_kwargs={"pn102_auto_v5": True})
    ar.maybe_add_answer_hint(r)
    check("structured + auto_v5 → no banner", banner_of(r) == "")
    # while the PN123 rerun's force_v5 keeps its deliberate bypass
    r2 = Request(chat_template_kwargs={"pn102_force_v5": True,
                                       ar._MARKER_KEY: True})
    ar.maybe_add_answer_hint(r2)
    check("PN123 rerun force_v5 still bypasses the marker gate",
          V5_TELL in banner_of(r2), banner_of(r2)[:50])


def test_unbounded_request_gets_no_banner():
    print("\nBUG-157 a request PN100 never sized still gets no banner")
    reset_env()
    os.environ["GENESIS_ENABLE_PN102_CONTRACT"] = "1"
    r = Request(thinking_token_budget=0,
                chat_template_kwargs={"pn102_auto_v5": True})
    ar.maybe_add_answer_hint(r)
    check("unbounded → no banner", banner_of(r) == "")


# ─── BUG-156: output-side banner echo net ────────────────────────────────────


def _rescue(request, result):
    class S:
        async def create_chat_completion(self, request, raw_request=None):
            raise AssertionError("no self-call expected here")
    return asyncio.run(ar.maybe_rescue_answer(S(), request, result))


def test_envelope_echo_stripped():
    print("\nBUG-156 leading [envelope] is stripped from the served answer")
    reset_env()
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] Work through"})
    res = Result(content='[envelope] {"facts": [1]}')
    _rescue(r, res)
    msg = res.choices[0].message
    check("marker gone", msg.content == '{"facts": [1]}', repr(msg.content))
    check("counted", ar._STATS["banner_echo_stripped"] == 1)


def test_envelope_echo_not_ours():
    print("\nBUG-156 an answer we never injected into is left alone")
    reset_env()
    r = Request()  # no pn_env_banner → not our injection
    res = Result(content="[envelope] verbatim from the user's own prompt")
    _rescue(r, res)
    check("untouched",
          res.choices[0].message.content.startswith("[envelope]"))
    check("not counted", ar._STATS["banner_echo_stripped"] == 0)


def test_envelope_only_answer_kept():
    print("\nBUG-156 an answer that is ONLY the marker is never emptied")
    reset_env()
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] x"})
    res = Result(content="[envelope]")
    _rescue(r, res)
    check("kept", res.choices[0].message.content == "[envelope]")


def test_step_echo_detect_only_by_default():
    print("\nBUG-156 numbered-step echo: detected, not stripped, by default")
    reset_env()
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] x"})
    body = "Step 1: read the chunk\nStep 2: assemble JSON\n\n{\"facts\": []}"
    res = Result(content=body)
    _rescue(r, res)
    check("seen", ar._STATS["banner_step_echo_seen"] == 1)
    check("not stripped", res.choices[0].message.content == body)
    check("not counted as stripped", ar._STATS["banner_step_echo_stripped"] == 0)


def test_step_echo_opt_in_strip():
    print("\nBUG-156 numbered-step echo: stripped when opted in")
    reset_env()
    os.environ["GENESIS_PN102_STRIP_STEP_ECHO"] = "1"
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] x"})
    res = Result(content="Step 1: read\nStep 2: assemble\n\n{\"facts\": []}")
    _rescue(r, res)
    check("stripped to the answer",
          res.choices[0].message.content == '{"facts": []}',
          repr(res.choices[0].message.content))
    check("counted", ar._STATS["banner_step_echo_stripped"] == 1)


def test_step_only_answer_never_emptied():
    print("\nBUG-156 a reply that is ONLY steps survives even when opted in")
    reset_env()
    os.environ["GENESIS_PN102_STRIP_STEP_ECHO"] = "1"
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] x"})
    body = "Step 1: first\nStep 2: second"
    res = Result(content=body)
    _rescue(r, res)
    check("kept whole", res.choices[0].message.content == body)


def test_strip_can_be_disabled():
    print("\nBUG-156 GENESIS_PN102_STRIP_ECHO=0 restores the old behaviour")
    reset_env()
    os.environ["GENESIS_PN102_STRIP_ECHO"] = "0"
    r = Request(chat_template_kwargs={"pn_env_banner": "[envelope] x"})
    res = Result(content="[envelope] answer")
    _rescue(r, res)
    check("untouched", res.choices[0].message.content == "[envelope] answer")


# ─── BUG-144: PN118 → PN123 ──────────────────────────────────────────────────


def test_closegate_flag_rename():
    print("\nBUG-144 close-gate: canonical PN123 flag + legacy PN118 alias")
    reset_env()
    check("both unset → off", ar._pn123_master_on() is False)
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    check("legacy flag still arms it", ar._pn123_master_on() is True)
    del os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"]
    os.environ["GENESIS_ENABLE_PN123_CLOSEGATE"] = "1"
    check("canonical flag arms it", ar._pn123_master_on() is True)
    os.environ["GENESIS_ENABLE_PN118_CLOSEGATE"] = "1"
    os.environ["GENESIS_ENABLE_PN123_CLOSEGATE"] = "0"
    check("canonical wins when both are set", ar._pn123_master_on() is False)
    check("legacy function name still resolves",
          ar._pn118_master_on is ar._pn123_master_on)


def test_closegate_knob_resolution():
    print("\nBUG-144 sub-knobs: canonical first, legacy honoured")
    reset_env()
    check("neither set → canonical name", ar._cg("FRAC") == "GENESIS_PN123_FRAC")
    os.environ["GENESIS_PN118_FRAC"] = "0.3"
    check("legacy set → legacy name", ar._cg("FRAC") == "GENESIS_PN118_FRAC")
    check("legacy value read", ar._env_float(ar._cg("FRAC"), 0.6) == 0.3)
    os.environ["GENESIS_PN123_FRAC"] = "0.4"
    check("canonical wins", ar._cg("FRAC") == "GENESIS_PN123_FRAC")
    check("canonical value read", ar._env_float(ar._cg("FRAC"), 0.6) == 0.4)


def test_legacy_symbols_present():
    print("\nBUG-144 the strings other files assert on are still exported")
    check("_PN118_MARKER alias", ar._PN118_MARKER == "pn118_internal")
    check("_PN118_DEFAULT_CUE alias",
          ar._PN118_DEFAULT_CUE == ar._PN123_DEFAULT_CUE)
    check("_pn118_answer_key alias",
          ar._pn118_answer_key is ar._pn123_answer_key)
    for k in ("pn118_skips", "pn118_fires", "pn118_errors",
              "pn118_shadow_would_fire", "pn118_attempts"):
        check(f"stat key {k} kept", k in ar._STATS)
    src = (MOD_DIR / "answer_rescue.py").read_text(encoding="utf-8")
    check("patch_id_lint marker string still present in the module",
          "GENESIS_ENABLE_PN118_CLOSEGATE" in src)


TESTS = [
    test_install_idempotent,
    test_flag_off_is_inert,
    test_deep_route_promotes_to_v5,
    test_probe_shape,
    test_lean_route_falls_through,
    test_route_inverted_by_env,
    test_unavailable_paths_fail_open,
    test_real_bridge_import_is_absent_offline,
    test_candidate_filters,
    test_promotion_obeys_skip_gates,
    test_unbounded_request_gets_no_banner,
    test_envelope_echo_stripped,
    test_envelope_echo_not_ours,
    test_envelope_only_answer_kept,
    test_step_echo_detect_only_by_default,
    test_step_echo_opt_in_strip,
    test_step_only_answer_never_emptied,
    test_strip_can_be_disabled,
    test_closegate_flag_rename,
    test_closegate_knob_resolution,
    test_legacy_symbols_present,
]


def main() -> int:
    for t in TESTS:
        t()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL BUG-157 / BUG-156 / BUG-144 TESTS PASSED")
    return 0


if __name__ == "__main__":
    reset_env()
    raise SystemExit(main())
