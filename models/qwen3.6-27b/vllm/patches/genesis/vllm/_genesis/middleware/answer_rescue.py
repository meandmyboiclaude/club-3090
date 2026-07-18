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


# ─── Leg 1: PN102 Envelope Contract injector (sync, pre-render) ──────────────
# v2 (2026-07-18, post-sweep design rounds): the user-message hint is replaced
# by TEMPLATE-rendered contract pieces — a late-system banner (rules) + a
# think-seed (the model continues its own budget statement; self-emitted step
# numbers become the execution-time counter a transformer lacks). This function
# keeps its name (the deployed /fixes call site imports it) but now only sets
# chat_template_kwargs variables; chat_template_v2json.jinja renders them.
# Calibration: p75 tokens-per-step = 193 (436 real TC GPQA traces, 2026-07-18).
# Numbering clause gated to N <= GENESIS_PN102_NUMBER_MAX (default 24): at 53
# steps (tier 3) numbering is meta-noise — large budgets get a sizing banner.


def maybe_add_answer_hint(request: Any) -> None:
    if not _env_bool("GENESIS_ENABLE_PN102_CONTRACT"):
        return
    if not _bounded(request) or _skip_common(request):
        return
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    if ctk.get("pn_env_banner"):
        return  # idempotent
    if ctk.get("enable_thinking") is False:
        return
    budget = getattr(request, "thinking_token_budget", 0)
    tps = max(50, _env_int("GENESIS_PN102_TOKENS_PER_STEP", 193))
    planner_steps = ctk.pop("pn100_steps", None)  # planner-router suggestion
    if isinstance(planner_steps, int) and planner_steps > 0:
        steps = planner_steps
        # planner estimates per-item need; budget is the CAP — phrase them as
        # such instead of implying steps*193 == budget (they usually differ)
        size_clause = f"(budget allows up to ~{budget} thinking tokens)"
    else:
        steps = max(3, round(budget / tps))
        size_clause = f"(~{budget} tokens)"
    sentences = max(1, _env_int("GENESIS_PN102_SENTENCES", 3))
    number_max = _env_int("GENESIS_PN102_NUMBER_MAX", 24)
    answer_clause = (
        "Unless the user asked for longer form, put your final answer in the "
        f"FIRST sentence of your reply, then at most {sentences} sentences total."
    )
    if steps <= number_max:
        ctk["pn_env_banner"] = (
            f"[envelope] Thinking budget: about {steps} short steps "
            f"{size_clause}. Number your steps and conclude by "
            f"Step {steps} yourself — do not let the budget cut you off. "
            + answer_clause
        )
        ctk["pn_env_seed"] = f"Budget: ~{steps} short steps.\nStep 1:"
    else:
        ctk["pn_env_banner"] = (
            f"[envelope] You have a substantial thinking budget (~{budget} "
            "tokens). Conclude your reasoning yourself, well before the budget "
            "forces a cut. " + answer_clause
        )
        ctk["pn_env_seed"] = f"Budget: ~{budget} thinking tokens available.\n"
    request.chat_template_kwargs = ctk
    _STATS["hints_added"] += 1
    log.info("PN102: contract set (steps=%d budget=%d)", steps, budget)


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
    choice = _extract_choice(result)
    fr = getattr(choice, "finish_reason", None) if choice is not None else None
    if fr != "length":
        return result
    if getattr(request, "stream", False):
        log.info("PN101: guillotine observed on streaming request — cannot repair")
        return result
    if not _bounded(request):
        log.info("PN101: guillotine observed but request not bounded-shaped — skip")
        return result
    if _skip_common(request):
        log.info("PN101: guillotine observed but skip-gates hit (marker/tools/structured)")
        return result
    # From here every exit is logged — silent gate-outs on a guillotined
    # bounded response are exactly the failure mode we must be able to see.
    message = getattr(choice, "message", None)
    content = (getattr(message, "content", None) or "") if message else ""
    if not content.strip():
        log.info("PN101: guillotine observed but content empty — no repair basis")
        return result  # guillotined inside think — nothing to continue from
    if getattr(message, "tool_calls", None):
        log.info("PN101: guillotine observed but tool_calls present — skip")
        return result
    if _ANSWER_TAIL_RE.search(content[-200:]):
        log.info("PN101: guillotine observed but answer marker present — skip")
        return result
    log.info("PN101: guillotine observed (bounded, finish=length) — repairing")

    _STATS["repairs_attempted"] += 1
    try:
        req_cls = type(request)
        fields = getattr(req_cls, "model_fields", {}) or {}
        # TC's assistant format REQUIRES a think block — a continuation partial
        # without one is off-distribution and the model EOS's instantly
        # (live-diagnosed 2026-07-18: probe with no think -> empty @ temp 0;
        # with a minimal think block the model completes correctly). The
        # template drops EMPTY think blocks (jinja gate on truthy reasoning),
        # so the primer text is load-bearing, not cosmetic.
        # .strip() BOTH ends: the template lstrips newlines between </think>
        # and content — leading \n in content would make the rendered message
        # diverge from the raw one and trip vLLM's continue_final_message
        # containment check (live-diagnosed 2026-07-18).
        partial = (
            "<think>\nBudget spent; committing the final answer now.\n</think>\n\n"
            + content.strip()
            + "\nFinal answer:"
        )
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
        rmsg = getattr(rchoice, "message", None) if (rchoice := _extract_choice(resp)) else None
        # The continuation's output lands in message.reasoning on this stack:
        # the reasoning parser assumes generation starts inside <think> (the
        # normal generation prompt opens it), and a continue_final_message
        # request never emits the markers — so read all three fields.
        text = ""
        for attr in ("content", "reasoning", "reasoning_content"):
            text = (getattr(rmsg, attr, None) or "").strip() if rmsg else ""
            if text:
                break
        if not text:
            log.info("PN101: repair continuation returned empty — keeping original")
            return result
        text = text.split("\n")[0].strip()  # the committed answer line only
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
