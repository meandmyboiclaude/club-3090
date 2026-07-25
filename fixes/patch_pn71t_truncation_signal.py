#!/usr/bin/env python3
"""PN71T — wire the truncation signal into both chat-completion response paths.

Installs /fixes/pn71_truncation_signal.py as vllm/_genesis_pn71t.py and text-patches
vllm/entrypoints/openai/chat_completion/serving.py at two sites:

  F) chat_completion_full_generator, immediately before ``choices.append(choice_data)``
     — has ``output.text`` (raw), the parsed ``content``, and the finished choice.
  S) chat_completion_stream_generator, immediately after ``finish_reason_sent[i] = True``
     — has ``previous_texts[i]`` (the raw accumulated output) and the final choice chunk.

What it detects and why it is worth shipping on its own: see the sidecar's module
docstring. Short version — a generation cut inside ``<think>`` returns HTTP 200,
``finish_reason="length"`` and an empty ``content``; nothing downstream can count it.
This makes it a WARNING line, a counter and a ``stop_reason`` stamp.

Independent of the PN71 v3 budget restore. v3 makes the class rarer (the budget is
clamped so an answer always fits); this makes the residue measurable. Ship either
alone.

ORDERING vs PN101
-----------------
PN101's answer-rescue repair runs on the assembled response, AFTER this site, and is
default-OFF (``GENESIS_ENABLE_PN101_ANSWER_RESCUE``). With PN101 on, a PN71T line is
therefore a PRE-rescue observation: join it to PN101's own ``get_stats()`` before
concluding a caller actually saw an empty body. Deliberately not chained off PN101's
output — depending on another patch's edit is how a family patch goes phantom
(BUG-122).

Anchors are CONTENT-SNIFFED and COUNTED against post-sibling content. On this pin
four /fixes patches rewrite serving.py first (pn74 -> pn100 -> pn101 -> h119_route_api);
none of them touches either site, and the verifier proves that rather than assuming it:
    python3 fixes/verify_pn71_anchors.py

A drift fails LOUD (boxed stderr ERROR + logging.error) and writes nothing — never a
partial patch, never a silent skip.
"""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import sys

LOG = "[pn71t-truncation-signal]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM / "entrypoints/openai/chat_completion/serving.py"
SIDECAR_SRC = pathlib.Path("/fixes/pn71_truncation_signal.py")
SIDECAR_DST = VLLM / "_genesis_pn71t.py"
MARKER = "# PN71T:"

# (F) non-streaming. Anchor is the stock two-line tail of the per-output loop.
F_OLD = (
    "            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n"
    "\n"
    "            choices.append(choice_data)\n"
)
F_NEW = (
    "            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n"
    "\n"
    "            " + MARKER + " truncation signal — a generation cut inside <think>\n"
    "            # returns 200/finish=length with EMPTY content and nothing downstream\n"
    "            # can count it. Emits a structured WARNING + a stop_reason stamp.\n"
    "            # output.text is the raw authority: `reasoning` is dropped entirely\n"
    "            # when include_reasoning is false. Fail-open.\n"
    "            try:\n"
    "                from vllm._genesis_pn71t import check_choice as _pn71t_check\n"
    "                _pn71t_check(\n"
    "                    self, request, choice_data, output.text, content,\n"
    "                    choice_data.finish_reason, request_id=request_id,\n"
    "                    streaming=False,\n"
    "                )\n"
    "            except Exception:\n"
    "                import logging as _pn71t_logging\n"
    "                _pn71t_logging.getLogger(\n"
    "                    'genesis.pn71t'\n"
    "                ).debug('PN71T full-generator hook raised; ignored', exc_info=True)\n"
    "\n"
    "            choices.append(choice_data)\n"
)

# (S) streaming. Anchor is the stock end of the finish-chunk branch.
S_OLD = (
    "                        finish_reason_sent[i] = True\n"
)
S_NEW = (
    "                        finish_reason_sent[i] = True\n"
    "\n"
    "                        " + MARKER + " truncation signal (streaming). The raw\n"
    "                        # accumulated output is previous_texts[i]; if it carries no\n"
    "                        # reasoning-end tag then every token generated was reasoning\n"
    "                        # and content is necessarily empty. Fail-open.\n"
    "                        try:\n"
    "                            from vllm._genesis_pn71t import (\n"
    "                                check_choice as _pn71t_check,\n"
    "                            )\n"
    "                            _pn71t_check(\n"
    "                                self, request, choice_data, previous_texts[i],\n"
    "                                None, finish_reason_, request_id=request_id,\n"
    "                                streaming=True,\n"
    "                            )\n"
    "                        except Exception:\n"
    "                            import logging as _pn71t_logging\n"
    "                            _pn71t_logging.getLogger(\n"
    "                                'genesis.pn71t'\n"
    "                            ).debug(\n"
    "                                'PN71T stream hook raised; ignored', exc_info=True)\n"
)

HUNKS = (("F", F_OLD, F_NEW), ("S", S_OLD, S_NEW))

# Content sniff: (S) needs the raw accumulator to exist under that name, and both
# sites need `request_id` in scope. A pin that renames either would take the anchor
# with it, but sniffing says WHICH thing moved instead of just "not found".
REQUIRED_SYMBOLS = ("previous_texts[i] += delta_text", "request_id")


def counts(text: str) -> dict:
    return {name: text.count(old) for name, old, _new in HUNKS}


def resolve(text: str):
    """Return (hunks, problems). Any count != 1 is a problem."""
    problems = []
    for name, old, _new in HUNKS:
        n = text.count(old)
        if n == 0:
            problems.append(f"({name}) anchor NOT FOUND — re-anchor needed")
        elif n > 1:
            problems.append(f"({name}) anchor is AMBIGUOUS ({n} occurrences) — re-anchor needed")
    for sym in REQUIRED_SYMBOLS:
        if sym not in text:
            problems.append(f"expected symbol {sym!r} absent — the call site moved")
    return (HUNKS if not problems else ()), problems


def _shout(detail: str) -> None:
    bar = "=" * 78
    msg = (f"{bar}\n"
           f"{LOG} ERROR: the truncation signal is NOT INSTALLED.\n"
           f"{LOG} ERROR: {detail}\n"
           f"{LOG} ERROR: cut-inside-<think> completions keep returning HTTP 200 with an\n"
           f"{LOG} ERROR: empty body and NO signal — the exact caller-invisible class this\n"
           f"{LOG} ERROR: patch exists to make countable.\n"
           f"{LOG} ERROR: re-derive the anchors against POST-PATCH content:\n"
           f"{LOG} ERROR:   python3 fixes/verify_pn71_anchors.py\n"
           f"{bar}")
    print(msg, file=sys.stderr, flush=True)
    try:
        logging.getLogger("genesis.pn71t").error(msg.replace("\n", " | "))
    except Exception:  # noqa: BLE001
        pass


def _drift_rc() -> int:
    """Exit code for a drift.

    ZERO, unlike PN71's. The entrypoint runs under ``set -e`` and this patch is
    pure observability — a NEW capability, not a restore. Losing the signal is bad;
    refusing to serve because the signal could not be installed is worse. The boxed
    ERROR above is the alarm. Set ``PN71T_STRICT=1`` to make a drift abort the boot
    instead (use it in a pin-bump gate, where you do want the hard stop).
    """
    val = os.environ.get("PN71T_STRICT", "").strip().lower()
    return 1 if val in ("1", "true", "yes", "on") else 0


def main() -> int:
    if not TARGET.exists():
        _shout(f"{TARGET} not present on this pin")
        return _drift_rc()
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0

    hunks, problems = resolve(text)
    if problems:
        _shout("; ".join(problems))
        return _drift_rc()

    try:
        shutil.copy2(SIDECAR_SRC, SIDECAR_DST)
    except OSError as e:
        _shout(f"sidecar install failed: {e}")
        return _drift_rc()

    for name, old, new in hunks:
        text = text.replace(old, new, 1)
        print(f"{LOG} ({name}) wired")

    try:
        compile(text, str(TARGET), "exec")
    except SyntaxError as e:
        _shout(f"patched serving.py does not byte-compile ({e}) — refusing to write")
        return _drift_rc()

    TARGET.write_text(text, encoding="utf-8")
    print(f"{LOG} applied — grep container logs for PN71T-TRUNC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
