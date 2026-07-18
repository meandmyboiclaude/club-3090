# SPDX-License-Identifier: Apache-2.0
"""PN101 — answer rescue (house-original, 2026-07-18). PN100/PN71 companion.

Bounded-envelope requests (thinking_token_budget set) must still land a usable
final answer instead of a mid-sentence max_tokens guillotine. Two legs, both
serving-layer, both fail-open, master flag DEFAULT OFF (behavioral patch):

  hint   — append one line to the last user message: reply window is limited,
           state the final answer in the FIRST sentence. Fixes answer ordering.
  repair — if the response still comes back finish_reason=length without a
           parseable answer: ONE tiny continuation request (original messages +
           truncated assistant text + "\\nFinal answer:", continue_final_message,
           thinking off, ~16 tokens, temp 0) spliced onto content. APC makes the
           continuation prefill nearly free; the model commits with its full
           derivation in context. No decode-path surgery — MTP/CUDA-graph/
           grammar interactions avoided by construction (see design doc
           ~/shared/DESIGN-pn101-answer-rescue-2026-07-18.md, incl. why the
           forced-token sibling of ThinkingBudgetStateHolder was rejected).

Env: GENESIS_ENABLE_PN101_ANSWER_RESCUE (master, default OFF),
     GENESIS_PN101_HINT / GENESIS_PN101_REPAIR (sub-toggles, default ON under
     master), GENESIS_PN101_REPAIR_TOKENS (16), GENESIS_PN101_TIMEOUT_S (15).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

try:
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.answer_rescue")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.middleware.answer_rescue")

_MARKER_KEY = "pn101_internal"
_PN100_MARKER_KEY = "pn100_internal"
_HINT_SENTINEL = "[reply-window note]"
_ANSWER_TAIL_RE = re.compile(r"(final\s+)?answer\s*[:\-]", re.IGNORECASE)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _master_on() -> bool:
    return _env_bool("GENESIS_ENABLE_PN101_ANSWER_RESCUE")


_STATS: dict[str, int] = {
    "hints_added": 0,
    "repairs_attempted": 0,
    "repairs_succeeded": 0,
    "repair_errors": 0,
}


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def _bounded(request: Any) -> bool:
    budget = getattr(request, "thinking_token_budget", None)
    return isinstance(budget, int) and budget > 0


def _has_structured_output(request: Any) -> bool:
    rf = getattr(request, "response_format", None)
    if rf is not None:
        rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
        if rf_type in ("json_object", "json_schema", "structural_tag"):
            return True
    for attr in ("guided_json", "guided_regex", "guided_grammar", "guided_choice",
                 "structured_outputs"):
        if getattr(request, attr, None):
            return True
    return False


def _skip_common(request: Any) -> bool:
    ctk = getattr(request, "chat_template_kwargs", None) or {}
    if ctk.get(_MARKER_KEY) or ctk.get(_PN100_MARKER_KEY):
        return True
    if getattr(request, "tools", None):
        return True
    if _has_structured_output(request):
        return True
    return False


def _completion_cap(request: Any) -> int | None:
    for attr in ("max_completion_tokens", "max_tokens"):
        v = getattr(request, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


# ─── Leg 1: answer-first envelope hint (sync, pre-render) ────────────────────


def maybe_add_answer_hint(request: Any) -> None:
    if not _master_on() or not _env_bool("GENESIS_PN101_HINT", True):
        return
    if not _bounded(request) or _skip_common(request):
        return
    messages = getattr(request, "messages", None)
    if not messages:
        return
    # find last user message
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            idx = i
            break
    if idx is None:
        return
    msg = messages[idx]
    if not isinstance(msg, dict):
        return
    cap = _completion_cap(request)
    budget = getattr(request, "thinking_token_budget", 0)
    allowance = (cap - budget) if (cap and cap > budget) else None
    approx = f" of roughly {allowance} tokens" if allowance else ""
    hint = (
        f"\n\n{_HINT_SENTINEL} Your reply after any hidden reasoning has a "
        f"limited window{approx}. State your final answer in the FIRST "
        "sentence, then add brief justification only if room remains."
    )
    content = msg.get("content")
    if isinstance(content, str):
        if _HINT_SENTINEL in content:
            return
        msg["content"] = content + hint
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and _HINT_SENTINEL in str(part.get("text", "")):
                return
        content.append({"type": "text", "text": hint})
    else:
        return
    _STATS["hints_added"] += 1
    log.debug("PN101: answer-first hint added (allowance=%s)", allowance)


# ─── Leg 2: post-hoc repair pass (async, post-response) ──────────────────────


def _extract_choice(result: Any):
    choices = getattr(result, "choices", None)
    if not choices:
        return None
    return choices[0]


async def maybe_rescue_answer(serving: Any, request: Any, result: Any) -> Any:
    if not _master_on() or not _env_bool("GENESIS_PN101_REPAIR", True):
        return result
    if hasattr(result, "__aiter__"):  # streaming generator — cannot repair
        return result
    if getattr(request, "stream", False):
        return result
    if not _bounded(request) or _skip_common(request):
        return result
    choice = _extract_choice(result)
    if choice is None or getattr(choice, "finish_reason", None) != "length":
        return result
    message = getattr(choice, "message", None)
    content = (getattr(message, "content", None) or "") if message else ""
    if not content.strip():
        return result  # guillotined inside think — nothing to continue from
    if getattr(message, "tool_calls", None):
        return result
    if _ANSWER_TAIL_RE.search(content[-200:]):
        return result  # commit already present; cut hit the justification

    _STATS["repairs_attempted"] += 1
    try:
        req_cls = type(request)
        fields = getattr(req_cls, "model_fields", {}) or {}
        partial = content.rstrip() + "\nFinal answer:"
        messages = list(getattr(request, "messages", None) or [])
        messages.append({"role": "assistant", "content": partial})
        kwargs: dict[str, Any] = {
            "model": getattr(request, "model", None),
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
                _MARKER_KEY: True,
            },
        }
        cap_field = (
            "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
        )
        kwargs[cap_field] = _env_int("GENESIS_PN101_REPAIR_TOKENS", 16)
        # continuation semantics: continue the final assistant message verbatim
        for fname, val in (("continue_final_message", True), ("add_generation_prompt", False)):
            if fname in fields:
                kwargs[fname] = val
            else:
                kwargs["chat_template_kwargs"][fname] = val
        synthetic = req_cls(**kwargs)
        timeout = _env_int("GENESIS_PN101_TIMEOUT_S", 15)
        resp = await asyncio.wait_for(
            serving.create_chat_completion(synthetic, raw_request=None), timeout
        )
        rchoice = _extract_choice(resp)
        tail = (getattr(getattr(rchoice, "message", None), "reasoning", None) or "") if rchoice else ""
        text = (getattr(getattr(rchoice, "message", None), "content", None) or "") if rchoice else ""
        text = (text or tail).strip()
        if not text:
            return result
        message.content = content.rstrip() + "\nFinal answer: " + text
        _STATS["repairs_succeeded"] += 1
        log.info(
            "PN101: rescued truncated answer (+%d chars, finish stays 'length')",
            len(text) + 15,
        )
    except Exception as exc:
        _STATS["repair_errors"] += 1
        log.warning("PN101: repair failed (%s) — returning original response", exc)
    return result
