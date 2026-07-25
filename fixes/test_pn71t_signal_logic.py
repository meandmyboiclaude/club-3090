#!/usr/bin/env python3
"""CPU house test for PN71T — the truncation signal (detector + stamp + logging).

Imports the real sidecar module (no vLLM needed — it degrades to a stdlib logger)
and exercises it against stand-in choice objects.

    python3 fixes/test_pn71t_signal_logic.py     # prints PASS/FAIL, exit != 0 on FAIL
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pn71_truncation_signal as S  # noqa: E402
import patch_pn71t_truncation_signal as PT  # noqa: E402

FAILS: list[str] = []


class Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class Choice:
    def __init__(self, content=None, tool_calls=None, finish_reason="length"):
        self.message = Msg(content, tool_calls)
        self.finish_reason = finish_reason
        self.stop_reason = None


class StreamChoice:
    def __init__(self, content=None, tool_calls=None):
        self.delta = Msg(content, tool_calls)
        self.stop_reason = None


class Req:
    reasoning = "medium"
    reasoning_effort = None
    thinking_token_budget = 2048
    max_tokens = 2560
    max_completion_tokens = None


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAILS.append(name)


def main() -> int:
    for var in ("PN71T_ENABLE", "PN71T_STAMP", "PN71T_CONTENT_SENTINEL"):
        os.environ.pop(var, None)

    cap = Capture()
    S.log.addHandler(cap)
    S.log.setLevel(logging.DEBUG)

    print("PN71T — detector")
    T = S.is_truncated_in_think

    check("cut inside <think> with empty content IS the defect",
          T("<think>reasoning that never ends", "", "length", False))
    check("None content counts as empty",
          T("<think>reasoning that never ends", None, "length", False))
    check("whitespace-only content counts as empty",
          T("<think>reasoning that never ends", "   \n ", "length", False))

    check("a CLOSED think block is not the defect (answer just ran long)",
          not T("<think>done</think>partial answer", "partial answer", "length", False))
    check("closed-but-empty-content is not this defect either",
          not T("<think>done</think>", "", "length", False))
    check("finish_reason=stop is never the defect",
          not T("<think>reasoning", "", "stop", False))
    check("a non-empty content is never the defect",
          not T("<think>reasoning", "an answer", "length", False))
    check("a tool call is never the defect",
          not T("<think>reasoning", "", "length", True))
    # A length-finish with zero output is still an empty body at the token limit —
    # deliberately in-class, so the BUG-127-shaped "200 with nothing in it" is
    # counted rather than filtered out on a technicality.
    check("zero output at finish=length is flagged, not filtered away",
          T("", "", "length", False))

    # Whitespace-tolerant tag handling and non-default terminators.
    check("</ think > (spaced) still counts as closed",
          not T("<think>x</ think >answer", "answer", "length", False))
    check("a custom end tag is honoured when supplied",
          not T("<reason>x<|end_reason|>ans", "ans", "length", False,
                ("<|end_reason|>",)))
    check("a custom end tag that is absent still trips the detector",
          T("<reason>x", "", "length", False, ("<|end_reason|>",)))

    print("\nPN71T — stamp / logging / stats")
    before = S.get_stats()["truncated_in_think"]
    ch = Choice(content="")
    S.check_choice(None, Req(), ch, "<think>never closed", "", "length",
                   request_id="chatcmpl-test1")
    check("stop_reason stamped on the defective choice",
          ch.stop_reason == S.STAMP_VALUE, f"got {ch.stop_reason!r}")
    check("content is NOT rewritten by default", ch.message.content == "",
          f"got {ch.message.content!r}")
    check("counter advanced", S.get_stats()["truncated_in_think"] == before + 1)
    check("a structured PN71T-TRUNC line was emitted",
          any("PN71T-TRUNC" in ln for ln in cap.lines), str(cap.lines[-3:]))
    line = [ln for ln in cap.lines if "PN71T-TRUNC" in ln][-1]
    check("the line carries the resp_id join key", "chatcmpl-test1" in line, line)
    check("the line carries the budget", '"thinking_token_budget": 2048' in line, line)
    check("the line is machine-parseable JSON",
          _json_ok(line), line)

    ok = Choice(content="a real answer", finish_reason="stop")
    S.check_choice(None, Req(), ok, "<think>x</think>a real answer",
                   "a real answer", "stop", request_id="chatcmpl-ok")
    check("a healthy response is left completely alone",
          ok.stop_reason is None and ok.message.content == "a real answer")

    print("\nPN71T — env switches")
    os.environ["PN71T_STAMP"] = "0"
    ch = Choice(content="")
    S.check_choice(None, Req(), ch, "<think>never closed", "", "length")
    check("PN71T_STAMP=0 suppresses the stamp", ch.stop_reason is None)
    os.environ.pop("PN71T_STAMP")

    os.environ["PN71T_CONTENT_SENTINEL"] = "1"
    ch = Choice(content="")
    S.check_choice(None, Req(), ch, "<think>never closed", "", "length")
    check("PN71T_CONTENT_SENTINEL=1 writes the sentinel",
          ch.message.content == S.SENTINEL_TEXT, f"got {ch.message.content!r}")
    os.environ.pop("PN71T_CONTENT_SENTINEL")

    os.environ["PN71T_ENABLE"] = "0"
    ch = Choice(content="")
    n_before = S.get_stats()["truncated_in_think"]
    S.check_choice(None, Req(), ch, "<think>never closed", "", "length")
    check("PN71T_ENABLE=0 makes the whole hook inert",
          ch.stop_reason is None and S.get_stats()["truncated_in_think"] == n_before)
    os.environ.pop("PN71T_ENABLE")

    print("\nPN71T — fail-open + streaming")
    class Exploding:
        @property
        def message(self):
            raise RuntimeError("boom")
    err_before = S.get_stats()["errors"]
    S.check_choice(None, Req(), Exploding(), "<think>x", "", "length")
    check("an exploding choice object never propagates",
          S.get_stats()["errors"] == err_before + 1)

    sc = StreamChoice(content=None)
    S.check_choice(None, Req(), sc, "<think>never closed", None, "length",
                   request_id="chatcmpl-stream", streaming=True)
    check("streaming choices are stamped too", sc.stop_reason == S.STAMP_VALUE)
    line = [ln for ln in cap.lines if "chatcmpl-stream" in ln][-1]
    check("the streaming line is marked streaming", '"streaming": true' in line, line)

    sc = StreamChoice(content=None, tool_calls=[object()])
    S.check_choice(None, Req(), sc, "<think>never closed", None, "length",
                   streaming=True)
    check("a streaming tool call is not flagged", sc.stop_reason is None)

    print("\nPN71T — patcher shape")
    check("both anchors are declared", {n for n, _o, _n2 in PT.HUNKS} == {"F", "S"})
    check("resolve() rejects a missing anchor",
          PT.resolve("nothing here")[1] != [])
    dup = PT.F_OLD + PT.F_OLD + PT.S_OLD
    check("resolve() rejects an AMBIGUOUS anchor",
          any("AMBIGUOUS" in p for p in PT.resolve(dup)[1]))
    check("the marker is unique to this patch", PT.MARKER == "# PN71T:")

    print()
    if FAILS:
        print(f"FAIL — {len(FAILS)} check(s) failed: {FAILS}")
        return 1
    print("PASS — PN71T truncation signal")
    return 0


def _json_ok(line: str) -> bool:
    import json
    try:
        json.loads(line.split("PN71T-TRUNC ", 1)[1])
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
