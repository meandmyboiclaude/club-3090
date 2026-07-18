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
    "escalations_attempted": 0,
    "escalations_succeeded": 0,
    "escalation_errors": 0,
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
# v4 (2026-07-18, 30-round convergence): the banner is now STATIC — one string
# for every budgeted request, with no step arithmetic and no budget reference.
#
# Why v3's budget-sized banner had to go. The contract carried TWO functions
# fused together: numbering (self-location, which a transformer genuinely
# cannot do — it cannot feel token burn but can read labels it wrote itself)
# and a step TARGET derived from the budget. The target was the defect:
#   - auto scored ~4pt below a fixed arm AT IDENTICAL CAPS, purely because the
#     router's per-item step estimate ran below what the budget afforded and
#     the model complied with the smaller number ("compression").
#   - the arithmetic (steps = budget / p75) fed on a constant fitted to a
#     PREVIOUS quant, with no drift detection — a silent liability across
#     checkpoint swaps, and the origin of BUG-075's seed-path split.
# Externalize what the serving layer KNOWS (that reasoning happened, via the
# model's own step labels); never externalize what it merely GUESSES (how hard
# the task is — the classifier is the same model with less context and one shot).
#
# So N stops being a bound and becomes a CHECKPOINT: a scheduled moment to ask
# "am I done?", with continuing framed as normal rather than as an exception.
# Two consequences that make this safe to ship without calibration:
#   - N's value barely affects behaviour. Too low costs one cheap self-check;
#     too high and the item has already self-stopped. It is not a threshold.
#   - it self-targets. Easy requests finish before Step N and never reach the
#     checkpoint, so the banner is inert on exactly the traffic that must not
#     be perturbed and active on exactly the long reasoning that needs it.
# The number is kept (rather than "every few steps") deliberately: an immediate
# cadence asks "are you done?" at step two or three, biasing toward stopping —
# which IS the compression defect. "Around Step N" is a DELAY, not a target.
#
# Answer-shape steering ("answer-first, <=N sentences") is gone. Its job was
# prevention — keep the answer short enough to fit before the cap — and the
# escalation leg below replaces prevention with cure, which does not depend on
# the model complying with a number. Dropping it also retires the standing
# watch-item that bare prod callers were receiving response-format instructions
# they never asked for. The banner now governs reasoning cadence only and says
# nothing about answer form.
#
# COUPLING (do not ship half): this banner assumes a generous budget
# (GENESIS_PN100_TIER_BUDGETS=0,10240,10240,10240). Conversely the generous
# budget MUST NOT ship under the v3 banner — at 10240 the old arithmetic
# produced "wrap up around Step 53" with the headroom gate failing, i.e. an
# implied 53-step scope with no early-stop license on requests needing three.
# Rollback is both together: GENESIS_ENABLE_PN102_CONTRACT=0 + restore the
# tier-budget ladder. No redeploy required for either.
#
# INVARIANT (BUG-075): the seed MUST end mid-reasoning ("Step 1:"). A seed
# ending on a completed sentence reads as a natural stopping point and the
# model closes </think> instantly (31/37 rows rtok=0 @10240, proven from
# Phoenix rendered prompts). Now structurally safe: the seed no longer varies.


def _contract_v4_static(ctk: dict, budget: int) -> bool:
    """v4: one static banner for every budgeted request. Returns True if set."""
    ctk.pop("pn100_steps", None)  # planner estimate deliberately unused
    checkpoint = max(2, _env_int("GENESIS_PN102_CHECKPOINT_STEP", 10))
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. Around "
        f"Step {checkpoint} — and every few steps after — pause and check "
        "whether your answer is settled. If it is, stop reasoning and give "
        "it. If not, keep going."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v4 static, checkpoint=%d budget=%d)", checkpoint, budget)
    return True


def _contract_v3_sized(ctk: dict, budget: int) -> bool:
    """v3: budget/planner-sized banner. The validated prod path (072fff66)."""
    tps = max(50, _env_int("GENESIS_PN102_TOKENS_PER_STEP", 193))
    planner_steps = ctk.pop("pn100_steps", None)
    if isinstance(planner_steps, int) and planner_steps > 0:
        steps = planner_steps
        size_clause = f"budget allows up to ~{budget} thinking tokens"
    else:
        steps = max(3, round(budget / tps))
        size_clause = f"~{budget} tokens"
    sentences = max(1, _env_int("GENESIS_PN102_SENTENCES", 3))
    answer_clause = (
        "Unless the user asked for longer form, put your final answer in the "
        f"FIRST sentence of your reply, then at most {sentences} sentences total."
    )
    has_headroom = steps * tps < 0.7 * budget
    if budget >= _env_int("GENESIS_PN102_PERMISSION_MIN", 4096) and has_headroom:
        pace_clause = (
            f"Number your steps and wrap up around Step {steps} once your "
            "answer is settled; if the problem proves deeper than planned, "
            f"keep reasoning past Step {steps} — the budget is generous — "
            "and do not conclude while your answer is still uncertain. If "
            "you have genuinely exhausted your approaches, commit to your "
            "best answer. Do not let the budget cut you off. "
        )
        seed_label = "Plan"
    else:
        pace_clause = (
            f"Number your steps and wrap up around Step {steps} yourself — "
            "do not let the budget cut you off. "
        )
        seed_label = "Budget"
    ctk["pn_env_banner"] = (
        f"[envelope] Thinking budget: about {steps} short reasoning steps "
        f"({size_clause}). " + pace_clause + answer_clause
    )
    ctk["pn_env_seed"] = f"{seed_label}: ~{steps} short steps.\nStep 1:"
    log.info("PN102: contract set (v3 sized, steps=%d budget=%d)", steps, budget)
    return True


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
    if not isinstance(budget, int) or budget <= 0:
        return  # gated on "we actually assigned a thinking budget"
    # v4 ships OFF: it replaces a prod-validated banner and must not become the
    # live path until a bench window says so. It is also COUPLED to the
    # generous-budget env (see the v4 note above) — enable both or neither.
    if _env_bool("GENESIS_PN102_STATIC_BANNER", False):
        _contract_v4_static(ctk, budget)
    else:
        _contract_v3_sized(ctk, budget)
    request.chat_template_kwargs = ctk
    _STATS["hints_added"] += 1


# ─── Leg 2: post-hoc repair pass (async, post-response) ──────────────────────


def _extract_choice(result: Any):
    choices = getattr(result, "choices", None)
    if not choices:
        return None
    return choices[0]


def _read_reasoning(message: Any) -> str:
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None) or ""
        if val.strip():
            return val
    return ""


def _sum_usage(base: Any, extra: Any) -> None:
    """Fold the continuation's token counts into the returned response."""
    if base is None or extra is None:
        return
    for field in ("completion_tokens", "total_tokens"):
        b, e = getattr(base, field, None), getattr(extra, field, None)
        if isinstance(b, int) and isinstance(e, int):
            try:
                setattr(base, field, b + e)
            except Exception:  # pragma: no cover - frozen models
                pass


async def _maybe_escalate(serving: Any, request: Any, result: Any) -> bool:
    """Extend a request that consumed its whole budget still reasoning.

    This is the STARVATION half of the auto-budget system, and the signal it
    keys on has no false positives by construction: a response whose think
    block never closed did not choose to stop, it was stopped. Until v4 that
    case was detected, logged, and discarded ("no repair basis") — the single
    highest-value branch in the module was a no-op.

    Why a backstop and not the primary allocator: token spend is set by what an
    item NEEDS, not by what its cap allows (the model never learns its cap), so
    escalating from a small budget costs the same decode as one generous budget
    plus an extra prefill and, worse, a re-queue. Generous-first dominates. What
    escalation uniquely buys is reach ABOVE the generous cap for the ~8% that
    exhaust it — the accuracy curve was still monotone rising at 10240 with no
    knee, so that region is untested and cheap to probe only for items that
    prove they need it.

    Every failure path returns False and leaves the original response
    untouched, so the worst case is exactly today's behaviour. Non-streaming
    only (a streaming splice would mean owning SSE framing, finish_reason
    semantics and usage accounting across the splice); bounded to one pass.
    """
    if not _env_bool("GENESIS_PN101_ESCALATE", False):
        return False
    choice = _extract_choice(result)
    if choice is None:
        return False
    message = getattr(choice, "message", None)
    if message is None:
        return False
    content = (getattr(message, "content", None) or "").strip()
    reasoning = _read_reasoning(message)
    fr = getattr(choice, "finish_reason", None)
    # Two shapes, one cause — the answer never arrived:
    #   length + empty content = stopped mid-reasoning (classic starvation)
    #   stop   + empty content = closed the think block and emitted nothing
    if content or not reasoning.strip():
        return False
    if fr not in ("length", "stop"):
        return False

    _STATS["escalations_attempted"] += 1
    budget = _env_int("GENESIS_PN101_ESCALATE_BUDGET", 10240)
    try:
        req_cls = type(request)
        fields = getattr(req_cls, "model_fields", {}) or {}
        # Resume INSIDE the think region: an unclosed <think> is exactly the
        # grain this stack already runs with (the reasoning parser assumes
        # generation starts inside <think>, which is why PN101's continuations
        # land in message.reasoning). .strip() guards the containment check —
        # the template lstrips newlines, and any rendered-vs-raw divergence
        # makes vLLM reject continue_final_message.
        partial = "<think>\n" + reasoning.strip()
        messages = list(getattr(request, "messages", None) or [])
        messages.append({"role": "assistant", "content": partial})
        kwargs: dict[str, Any] = {
            "model": getattr(request, "model", None),
            "messages": messages,
            "temperature": getattr(request, "temperature", 0.0) or 0.0,
            "stream": False,
            "thinking_token_budget": budget,
            "chat_template_kwargs": {_MARKER_KEY: True},
        }
        cap_field = (
            "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
        )
        original_cap = _completion_cap(request)
        kwargs[cap_field] = max(original_cap or 0, budget + 512)
        for fname, val in (("continue_final_message", True), ("add_generation_prompt", False)):
            if fname in fields:
                kwargs[fname] = val
            else:
                kwargs["chat_template_kwargs"][fname] = val
        synthetic = req_cls(**kwargs)
        timeout = _env_int("GENESIS_PN101_ESCALATE_TIMEOUT_S", 180)
        resp = await asyncio.wait_for(
            serving.create_chat_completion(synthetic, raw_request=None), timeout
        )
        rchoice = _extract_choice(resp)
        rmsg = getattr(rchoice, "message", None) if rchoice else None
        if rmsg is None:
            log.info("PN101: escalation returned no choice — keeping original")
            return False
        new_content = (getattr(rmsg, "content", None) or "").strip()
        new_reasoning = _read_reasoning(rmsg)
        if not new_content and not new_reasoning.strip():
            log.info("PN101: escalation returned empty — keeping original")
            return False
        # The continuation may itself end inside the think region; in that case
        # its text lands in reasoning and there is still no answer. Hand that
        # to the repair leg rather than escalating again.
        for attr in ("reasoning", "reasoning_content"):
            if getattr(rmsg, attr, None) is not None:
                try:
                    setattr(message, attr, (reasoning + new_reasoning))
                except Exception:  # pragma: no cover
                    pass
                break
        if new_content:
            message.content = new_content
            try:
                choice.finish_reason = getattr(rchoice, "finish_reason", None) or "stop"
            except Exception:  # pragma: no cover
                pass
        _sum_usage(getattr(result, "usage", None), getattr(resp, "usage", None))
        _STATS["escalations_succeeded"] += 1
        log.info(
            "PN101: escalated starved request (prior_rtok~%d, +%d budget, answer=%s)",
            len(reasoning) // 4, budget, "yes" if new_content else "no",
        )
        return bool(new_content)
    except Exception as exc:
        _STATS["escalation_errors"] += 1
        log.warning("PN101: escalation failed (%s) — returning original response", exc)
        return False


async def maybe_rescue_answer(serving: Any, request: Any, result: Any) -> Any:
    if not _master_on() or not _env_bool("GENESIS_PN101_REPAIR", True):
        return result
    if hasattr(result, "__aiter__"):  # streaming generator — cannot repair
        return result
    if getattr(request, "stream", False):
        return result
    if _bounded(request) and not _skip_common(request):
        # Starvation first: a request that never reached an answer gets more
        # room before we consider forcing one out of a truncated think block.
        if await _maybe_escalate(serving, request, result):
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
