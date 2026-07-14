# SPDX-License-Identifier: Apache-2.0
"""PN288 dev1060 probe — ``prev_tool_call_arr`` shim for the parser engine.

Added 2026-07-13 for the dev1060 port of PN288
(``pn288_tool_finish_reason_override.py``, same directory).

The PN288 middleware
(``sndr.engines.vllm.middleware.pn288_finish_reason_override``) decides
args validity by walking ``tool_parser.prev_tool_call_arr`` — the legacy
``ToolParser`` streaming surface. dev1060 removed that surface: qwen3
tool parsing flows through the new parser engine
(``vllm/parser/engine/parser_engine.py``), which tracks per-call state in
``ParserEngine._tool_slots`` and hands the serving layer a plain
``FunctionCall`` list. Without a shim, ``getattr(self, "tool_parser",
None)`` is ``None`` on dev1060 and the middleware permanently returns
KEPT_VALID — the override could never observe or fire.

This module rebuilds the minimal shape the middleware reads
(``.prev_tool_call_arr`` → list of ``{"arguments": str}`` dicts) from
what dev1060 actually exposes, WITHOUT touching the middleware — its
validity/length-band logic stays byte-identical across pins:

  * :func:`probe_from_tool_calls` — non-streaming leg. Probes the parsed
    ``tool_calls`` (``FunctionCall``) list from ``parser.parse(...)``:
    exactly the arguments the client receives in the response body.
  * :func:`probe_from_parser` — streaming leg. Probes
    ``parser._tool_slots[*].streamed_json``: the converted-JSON prefix
    actually streamed to the client so far. (On dev1060 this leg is
    currently pass-through in the middleware — ``auto_tools_called``
    does not exist in the streaming generator — but the probe keeps the
    call site truthful for a future trigger re-evaluation.)

Both helpers NEVER raise (PN288 contract: observability must not break
the request) and degrade to an empty ``prev_tool_call_arr``, which the
middleware treats as "no positive evidence of malformation" (upstream
verdict preserved).

Author: Sandermage (dev1060 port, 2026-07-13).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _entry(args: Any) -> dict:
    return {"arguments": args if isinstance(args, str) else ""}


def probe_from_tool_calls(tool_calls: Any) -> Any:
    """Non-streaming probe: ``FunctionCall`` list → legacy
    ``prev_tool_call_arr`` shape.

    ``tool_calls`` is the third element of ``parser.parse(...)`` in
    dev1060's ``chat_completion_full_generator`` (``list[FunctionCall]``
    or ``None``). Each entry's ``arguments`` string is what the client
    receives — the exact payload whose JSON validity PN288 cares about.
    """
    arr: list[dict] = []
    try:
        for tc in tool_calls or []:
            arr.append(_entry(getattr(tc, "arguments", None)))
    except Exception:
        arr = []
    return SimpleNamespace(prev_tool_call_arr=arr)


def probe_from_parser(parser: Any) -> Any:
    """Streaming probe: ``ParserEngine._tool_slots`` → legacy
    ``prev_tool_call_arr`` shape.

    Uses ``slot.streamed_json`` (the converted-JSON prefix already
    emitted to the client) rather than ``slot.args`` (raw XML parameter
    body for qwen3), because PN288 judges what the CLIENT accumulated,
    not the model's raw text.
    """
    arr: list[dict] = []
    try:
        for slot in getattr(parser, "_tool_slots", None) or []:
            arr.append(_entry(getattr(slot, "streamed_json", None)))
    except Exception:
        arr = []
    return SimpleNamespace(prev_tool_call_arr=arr)


__all__ = ["probe_from_tool_calls", "probe_from_parser"]
