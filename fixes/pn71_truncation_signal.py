#!/usr/bin/env python3
"""PN71T — truncation signal: turn a silent empty completion into a measurable one.

Installed by ``fixes/patch_pn71t_truncation_signal.py`` as
``vllm/_genesis_pn71t.py``; the two call sites in
``entrypoints/openai/chat_completion/serving.py`` are one guarded call each.

THE DEFECT THIS MAKES VISIBLE
-----------------------------
A request whose generation is cut while still inside ``<think>`` returns
``HTTP 200`` with ``finish_reason="length"`` and an EMPTY ``content``. The
reasoning is intact in ``reasoning_content``, so nothing looks wrong to the
server, to the trace, or to a bench runner that counts non-2xx responses. The
caller gets an empty assistant turn and no explanation.

This rig has a standing weakness for exactly this shape — BUG-127 is an
OOM-abort storm that returns ``HTTP 200`` with an empty body while the bench
runner reports zero errors. A defect you cannot count is a defect you cannot
fix, so this module converts the class into three signals, cheapest first:

  1. a structured, single-line, machine-parseable WARNING on the
     ``genesis.pn71t`` logger — ``PN71T-TRUNC {json}`` — carrying the join keys
     this rig already uses (``resp_id``), so container logs can be counted with
     ``grep -c PN71T-TRUNC`` and joined to bench rows the same way PN108 fires
     are;
  2. a process-local counter (``get_stats()``) for scrape//metrics wiring;
  3. a caller-visible ``stop_reason`` stamp on the affected choice.

WHY THE ``stop_reason`` STAMP IS DEFAULT-ON
-------------------------------------------
It is a behavioural change to the response, which normally means ship-dark on
this repo. Two things make it safe here, and both are load-bearing:

  * ``stop_reason`` is a vLLM-only field (``int | str | None``, "not part of the
    OpenAI spec, included for legacy reasons") that is ALREADY non-``None``
    whenever a stop string or stop token ends a generation. Every caller must
    already tolerate arbitrary values in it.
  * It is written ONLY onto a choice that is already an empty body. There is no
    working call site to regress — the response was defective before we touched
    it, and afterwards it says so.

Kill switch: ``PN71T_STAMP=0``. Detection/logging kill switch:
``PN71T_ENABLE=0`` (everything below no-ops and the sites cost one env read).
An optional content sentinel for callers that only ever read ``content`` is
available but OFF by default (``PN71T_CONTENT_SENTINEL=1``) — writing prose into
``content`` IS a real behavioural change and gets the normal dark treatment.

DETECTION
---------
``finish_reason == "length"`` AND the raw model output contains no reasoning-end
tag AND ``content`` is empty AND no tool call was emitted. The raw text is the
authority for the tag, never the parsed ``reasoning`` field — the server drops
``reasoning`` entirely when ``include_reasoning`` is false, and a detector that
read it would go blind on exactly those requests.

The end tag is read from the engine's own reasoning config when available so a
model with a non-``</think>`` terminator is handled; ``</think>`` is the
fallback. Everything is wrapped fail-open: this module must never be able to
turn a served response into a 500.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

try:  # vllm's logger prints INFO in-server; a plain root logger may not
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.pn71t")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.pn71t")

STAMP_VALUE = "pn71_truncated_in_think"
SENTINEL_TEXT = (
    "[pn71: no answer was produced — generation was cut inside the reasoning "
    "block at the token limit. Raise max_tokens, lower the thinking budget, or "
    "raise PN71_ANSWER_GRACE.]"
)

_DEFAULT_END_TAGS = ("</think>",)
_TAG_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)

_STATS: dict[str, int] = {
    "checked": 0,
    "truncated_in_think": 0,
    "stamped": 0,
    "sentinel_written": 0,
    "errors": 0,
}


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _enabled() -> bool:
    return _env_bool("PN71T_ENABLE", True)


def _end_tags(serving: Any) -> tuple[str, ...]:
    """The model's reasoning-end tag(s), from engine config when reachable.

    Falls back to ``</think>``. Never raises: a config shape we don't recognise
    must degrade to the default, not disable the detector.
    """
    try:
        cfg = getattr(getattr(serving, "model_config", None), "reasoning_config", None)
        if cfg is None:
            cfg = getattr(getattr(serving, "vllm_config", None), "reasoning_config", None)
        for attr in ("reasoning_end_token", "reasoning_end_tokens", "think_end_str"):
            v = getattr(cfg, attr, None)
            if isinstance(v, str) and v:
                return (v,)
            if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
                return tuple(v)
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_END_TAGS


def _has_end_tag(raw: str, tags: tuple[str, ...]) -> bool:
    if not raw:
        return False
    for t in tags:
        if t and t in raw:
            return True
    # Whitespace-tolerant fallback for the canonical tag only.
    return bool(_TAG_RE.search(raw)) if tags == _DEFAULT_END_TAGS else False


def is_truncated_in_think(
    raw_text: str | None,
    content: Any,
    finish_reason: str | None,
    has_tool_calls: bool,
    end_tags: tuple[str, ...] = _DEFAULT_END_TAGS,
) -> bool:
    """The whole classification, as a pure function (unit-testable, no vLLM)."""
    if finish_reason != "length":
        return False
    if has_tool_calls:
        return False
    if isinstance(content, str) and content.strip():
        return False
    if isinstance(content, list) and content:
        return False
    return not _has_end_tag(raw_text or "", end_tags)


def _emit(payload: dict) -> None:
    try:
        log.warning("PN71T-TRUNC %s", json.dumps(payload, sort_keys=True,
                                                 default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001 — logging must never break a response
        _STATS["errors"] += 1


def check_choice(
    serving: Any,
    request: Any,
    choice_data: Any,
    raw_text: str | None,
    content: Any,
    finish_reason: str | None,
    request_id: str | None = None,
    streaming: bool = False,
) -> None:
    """Non-streaming and streaming entry point. Fail-open, in-place, returns None."""
    try:
        if not _enabled():
            return
        _STATS["checked"] += 1
        has_tools = bool(getattr(choice_data, "message", None) is not None
                         and getattr(choice_data.message, "tool_calls", None))
        if not has_tools:
            delta = getattr(choice_data, "delta", None)
            has_tools = bool(delta is not None and getattr(delta, "tool_calls", None))
        tags = _end_tags(serving)
        if not is_truncated_in_think(raw_text, content, finish_reason, has_tools, tags):
            return

        _STATS["truncated_in_think"] += 1
        payload = {
            "resp_id": request_id,
            "streaming": streaming,
            "finish_reason": finish_reason,
            "reason": STAMP_VALUE,
            "raw_chars": len(raw_text or ""),
            "thinking_token_budget": getattr(request, "thinking_token_budget", None),
            "max_tokens": getattr(request, "max_tokens", None),
            "max_completion_tokens": getattr(request, "max_completion_tokens", None),
            "reasoning": getattr(request, "reasoning", None),
            "reasoning_effort": getattr(request, "reasoning_effort", None),
            "answer_grace": os.environ.get("PN71_ANSWER_GRACE", "1024"),
        }
        _emit(payload)

        if _env_bool("PN71T_STAMP", True):
            try:
                choice_data.stop_reason = STAMP_VALUE
                _STATS["stamped"] += 1
            except Exception:  # noqa: BLE001
                _STATS["errors"] += 1

        if _env_bool("PN71T_CONTENT_SENTINEL", False):
            try:
                target = getattr(choice_data, "message", None) or getattr(
                    choice_data, "delta", None)
                if target is not None:
                    target.content = SENTINEL_TEXT
                    _STATS["sentinel_written"] += 1
            except Exception:  # noqa: BLE001
                _STATS["errors"] += 1
    except Exception:  # noqa: BLE001 — never break a served response
        _STATS["errors"] += 1
        try:
            log.debug("PN71T check raised; ignored", exc_info=True)
        except Exception:  # noqa: BLE001
            pass
