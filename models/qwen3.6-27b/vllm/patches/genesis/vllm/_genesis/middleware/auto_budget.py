# SPDX-License-Identifier: Apache-2.0
"""PN100 — auto thinking-budget router (house-original, 2026-07-18).

The Qwen-side twin of the Claude Code effort router: the engine itself rates
how much hidden reasoning a request needs (one cheap thinking-off self-call
returning a single digit 0-3) and sets `thinking_token_budget` accordingly.
Dumb senders get per-prompt budgets with zero client changes.

Relationship to PN16 (lazy_reasoner, Genesis-original): PN16 is the sync
heuristic pre-pass — its variant-1 turns thinking OFF for trivial short
prompts before we ever spend a classify call. PN100 is the async sizing pass
for everything that keeps thinking. PN16's variant-4 (LogitsProcessor cap)
stayed upstream-blocked under spec-decode; PN100 instead uses the native
`thinking_token_budget` sampling param, which ThinkingCap quants respect
gracefully (validated 2026-07-18: budget-hit cut, zero leak into content).

Trigger policy — explicit intent always wins, auto covers the rest:
  - numeric request.thinking_token_budget                  -> untouched
  - any request.reasoning / reasoning_effort value (PN71)  -> untouched
  - explicit chat_template_kwargs.enable_thinking=False    -> untouched
    (covers PN16 variant-1 decisions too)
  - chat_template_kwargs.thinking_budget: "off" | 0        -> untouched
  - chat_template_kwargs.thinking_budget: <int>            -> that budget
  - chat_template_kwargs.thinking_budget: "auto"           -> classify now
  - else if GENESIS_PN100_AUTO_DEFAULT                     -> classify
  - tiny requests (completion cap <= GENESIS_PN100_MIN_MAX_TOKENS,
    default 128) never classify — no room to think, and it exempts
    probe/effort-router traffic by construction.
  - explicit chat_template_kwargs.enable_thinking=True     -> classify, but
    never disable (tier 0 clamps to tier 1).

Env knobs:
  GENESIS_ENABLE_PN100_AUTO_BUDGET   master gate (wired in compose)
  GENESIS_PN100_AUTO_DEFAULT         default-auto for budget-less requests
  GENESIS_PN100_TIER_BUDGETS         "0,1024,4096,10240" (tier 0 = thinking off)
  GENESIS_PN100_FALLBACK_TIER        classify failure/timeout tier (default 2 —
                                     the benchmarked 70%-accuracy point)
  GENESIS_PN100_MIN_MAX_TOKENS       tiny-request exemption (default 128)
  GENESIS_PN100_TIMEOUT_S            classify timeout (default 20)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from vllm._genesis.middleware.lazy_reasoner import _extract_text_from_message

try:  # vllm's logger prints INFO in-server; plain root logger may not
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.auto_budget")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.middleware.auto_budget")

_MARKER_KEY = "pn100_internal"
_CONTROL_KEY = "thinking_budget"

_RUBRIC = (
    "You rate how much hidden reasoning an AI assistant needs to answer a "
    "request well. Reply with ONE digit only, nothing else:\n"
    "0 = none: greetings, acknowledgments, formatting, copy/transform, "
    "single-fact lookup, classification with obvious answer.\n"
    "1 = brief: simple single-step tasks, short summaries, routine "
    "extraction, easy questions.\n"
    "2 = real reasoning: multi-step logic or math, code with edge cases, "
    "debugging, analysis, tool-use planning, anything ambiguous.\n"
    "3 = deep reasoning: hard math/proofs, tricky algorithms, "
    "multi-constraint planning, subtle correctness questions. Rare — "
    "maybe 1 request in 40.\n"
    "Judge the TASK's need, not the prompt's length; short prompts can be "
    "hard and long prompts can be trivial. When unsure pick 2."
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


def _is_enabled() -> bool:
    return _env_bool("GENESIS_ENABLE_PN100_AUTO_BUDGET")


def _tier_budgets() -> list[int]:
    raw = os.environ.get("GENESIS_PN100_TIER_BUDGETS", "0,1024,4096,10240")
    try:
        vals = [int(x) for x in raw.split(",")]
        if len(vals) == 4:
            return vals
    except ValueError:
        pass
    return [0, 1024, 4096, 10240]


_STATS: dict[str, int] = {
    "total_requests": 0,
    "classified": 0,
    "tier_0": 0,
    "tier_1": 0,
    "tier_2": 0,
    "tier_3": 0,
    "explicit_skip": 0,
    "tiny_skip": 0,
    "fallbacks": 0,
    "errors": 0,
}


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def _completion_cap(request: Any) -> int | None:
    for attr in ("max_completion_tokens", "max_tokens"):
        v = getattr(request, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _flatten_messages(request: Any, cap: int = 8000) -> str:
    parts: list[str] = []
    for msg in getattr(request, "messages", None) or []:
        role = (
            msg.get("role", "?") if isinstance(msg, dict) else getattr(msg, "role", "?")
        )
        text = _extract_text_from_message(msg)
        if text:
            parts.append(f"[{role}] {text}")
    joined = "\n".join(parts)
    if len(joined) > cap:
        half = cap // 2
        joined = f"{joined[:half]}\n...[truncated]...\n{joined[-half:]}"
    return joined


def _decide_mode(request: Any) -> tuple[str, bool]:
    """Return (mode, allow_disable). mode: 'skip' | 'classify' | direct int str."""
    ctk = getattr(request, "chat_template_kwargs", None) or {}
    control = ctk.pop(_CONTROL_KEY, None) if _CONTROL_KEY in ctk else None

    if getattr(request, "thinking_token_budget", None) is not None:
        return "skip", True
    if getattr(request, "reasoning_effort", None) is not None:
        return "skip", True
    if getattr(request, "reasoning", None) is not None:
        return "skip", True

    explicit_thinking = ctk.get("enable_thinking", None)

    if control is not None:
        if control in ("off", "0", 0, False):
            return "skip", True
        if isinstance(control, int) and control > 0:
            return str(control), True
        if isinstance(control, str) and control.isdigit() and int(control) > 0:
            return control, True
        if control == "auto":
            if explicit_thinking is False:
                return "skip", True
            return "classify", explicit_thinking is not True

    if explicit_thinking is False:
        return "skip", True

    cap = _completion_cap(request)
    if cap is not None and cap <= _env_int("GENESIS_PN100_MIN_MAX_TOKENS", 128):
        return "tiny", True

    if _env_bool("GENESIS_PN100_AUTO_DEFAULT"):
        return "classify", explicit_thinking is not True
    return "skip", True


async def _classify(serving: Any, request: Any) -> int | None:
    """One thinking-off self-call -> tier int, or None on any failure."""
    req_cls = type(request)
    fields = getattr(req_cls, "model_fields", {}) or {}
    kwargs: dict[str, Any] = {
        "model": getattr(request, "model", None),
        "messages": [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": _flatten_messages(request)},
        ],
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, _MARKER_KEY: True},
    }
    cap_field = "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
    kwargs[cap_field] = 6
    synthetic = req_cls(**kwargs)

    timeout = _env_int("GENESIS_PN100_TIMEOUT_S", 20)
    resp = await asyncio.wait_for(
        serving.create_chat_completion(synthetic, raw_request=None), timeout
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        return None
    content = getattr(choices[0].message, "content", "") or ""
    m = re.search(r"[0-3]", content)
    return int(m.group(0)) if m else None


def _apply_tier(request: Any, tier: int, allow_disable: bool) -> int:
    budgets = _tier_budgets()
    if tier == 0 and not allow_disable:
        tier = 1
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    if tier == 0:
        ctk["enable_thinking"] = False
        request.chat_template_kwargs = ctk
        return 0
    ctk["enable_thinking"] = True
    request.chat_template_kwargs = ctk
    request.thinking_token_budget = budgets[tier]
    return budgets[tier]


async def apply_hook_async(serving: Any, request: Any) -> None:
    """Mutate request in place per PN100 auto-budget policy.

    Failure mode: any exception is caught by the call-site's try/except and
    the request proceeds untouched (fail-open to current behavior)."""
    if not _is_enabled():
        return
    ctk = getattr(request, "chat_template_kwargs", None)
    if ctk and ctk.pop(_MARKER_KEY, None):
        return  # our own classify call — never recurse
    _STATS["total_requests"] += 1

    mode, allow_disable = _decide_mode(request)
    if mode == "skip":
        _STATS["explicit_skip"] += 1
        return
    if mode == "tiny":
        _STATS["tiny_skip"] += 1
        return
    if mode != "classify":  # direct numeric via chat_template_kwargs.thinking_budget
        budget = int(mode)
        ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
        ctk["enable_thinking"] = True
        request.chat_template_kwargs = ctk
        request.thinking_token_budget = budget
        log.info("PN100: direct budget=%d (client-pinned)", budget)
        return

    t0 = time.monotonic()
    tier = None
    try:
        tier = await _classify(serving, request)
    except Exception as exc:  # timeout, engine error, schema drift
        log.warning("PN100: classify failed (%s) — falling back", exc)
        _STATS["errors"] += 1
    if tier is None:
        tier = _env_int("GENESIS_PN100_FALLBACK_TIER", 2)
        _STATS["fallbacks"] += 1
    tier = max(0, min(3, tier))
    _STATS["classified"] += 1
    _STATS[f"tier_{tier}"] += 1
    applied = _apply_tier(request, tier, allow_disable)
    log.info(
        "PN100: tier=%d -> %s (%dms)",
        tier,
        "thinking off" if applied == 0 else f"budget={applied}",
        int((time.monotonic() - t0) * 1000),
    )
