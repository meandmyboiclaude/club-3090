#!/usr/bin/env python3
"""PN101 — call-sites for answer rescue (PN100/PN71 companion, house-original).

Two anchored insertions in chat_completion/serving.py:
  1. hint site — sync `maybe_add_answer_hint(request)` just before the
     "# Streaming response" marker, i.e. AFTER PN100 (routed budgets get the
     hint too) and after PN16.
  2. repair site — wraps the outer create_chat_completion return value with
     `await maybe_rescue_answer(self, request, result)` (non-streaming
     responses only; the module gates everything else).

Module: vllm/_genesis/middleware/answer_rescue.py (mounted Genesis tree).
Master env flag GENESIS_ENABLE_PN101_ANSWER_RESCUE is DEFAULT OFF — behavioral
patches never default-on (house rule). With the flag off both call sites are
inert passthroughs. Fail-open: every exception is swallowed to debug.

MUST run AFTER patch_pn100_auto_thinking_budget.py in the entrypoint (the hint
anchor sits in the post-PN100 file; entrypoint order guarantees it).
"""
import pathlib
import sys

LOG = "[pn101-answer-rescue]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
MARKER = "# PN101"

HINT_ANCHOR = "        # Streaming response\n"
HINT_REPLACEMENT = (
    "        # PN101a: bounded-envelope answer-first hint (fail-open; default-OFF\n"
    "        # master flag; see _genesis/middleware/answer_rescue.py).\n"
    "        try:\n"
    "            from vllm._genesis.middleware.answer_rescue import (\n"
    "                maybe_add_answer_hint as _pn101_hint,\n"
    "            )\n"
    "            _pn101_hint(request)\n"
    "        except Exception:\n"
    "            import logging as _pn101a_logging\n"
    "            _pn101a_logging.getLogger(\n"
    "                'genesis.middleware.answer_rescue'\n"
    "            ).debug('PN101 hint raised; ignored', exc_info=True)\n"
    "        # Streaming response\n"
)

REPAIR_ANCHOR = (
    "        return await self._with_kv_transfer_rejection_cleanup(\n"
    "            self._create_chat_completion(request, raw_request), request, raw_request\n"
    "        )\n"
)
REPAIR_REPLACEMENT = (
    "        _pn101_result = await self._with_kv_transfer_rejection_cleanup(\n"
    "            self._create_chat_completion(request, raw_request), request, raw_request\n"
    "        )\n"
    "        # PN101b: answer-rescue post-pass (non-streaming, bounded-envelope,\n"
    "        # finish=length only; fail-open; default-OFF master flag).\n"
    "        try:\n"
    "            from vllm._genesis.middleware.answer_rescue import (\n"
    "                maybe_rescue_answer as _pn101_rescue,\n"
    "            )\n"
    "            _pn101_result = await _pn101_rescue(self, request, _pn101_result)\n"
    "        except Exception:\n"
    "            import logging as _pn101b_logging\n"
    "            _pn101b_logging.getLogger(\n"
    "                'genesis.middleware.answer_rescue'\n"
    "            ).debug('PN101 rescue raised; ignored', exc_info=True)\n"
    "        return _pn101_result\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0
    for name, anchor in (("hint", HINT_ANCHOR), ("repair", REPAIR_ANCHOR)):
        n = src.count(anchor)
        if n == 0:
            print(
                f"{LOG} FATAL: anchor-not-found ({name}) — serving.py drifted "
                "(or PN100 not applied first); re-derive anchors",
                file=sys.stderr,
            )
            return 1
        if n > 1:
            print(f"{LOG} FATAL: ambiguous anchor ({name}, {n} hits)", file=sys.stderr)
            return 1
    src = src.replace(HINT_ANCHOR, HINT_REPLACEMENT, 1)
    src = src.replace(REPAIR_ANCHOR, REPAIR_REPLACEMENT, 1)
    TARGET.write_text(src, encoding="utf-8")
    import py_compile

    py_compile.compile(str(TARGET), doraise=True)
    print(f"{LOG} applied: hint site + repair wrap (master flag default OFF)")
    return 0


sys.exit(main())
