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


# v5 (2026-07-19, spend-band analysis): v4's checkpoint delay became a FLOOR.
# Measured: v4 produced ZERO items under 500 rtok vs 36 in the tier-ladder auto
# (median 698 -> 1658); the cheap band collapsed from 61 items / 18.2% of spend
# to 21 / 4.9% with NO accuracy gain there (86.9% -> 85.7%). The banner's
# "around Step 10" deferral meant nothing considered stopping before ~1900
# tokens. The v4 rationale (immediate cadence biases toward stopping = the
# compression defect) is refuted by the data: the delay did not protect the
# hard tail — it taxed the easy cohort. Meanwhile the deep band (3000+ rtok)
# doubled its population (19 -> 35) for +2.1pt within-band, because v4's
# "if not settled, keep going" is UNCONDITIONAL — an item that is uncertain
# because it is unsolvable grinds instead of committing.
#
# v5 = same static-banner architecture (no step arithmetic, no budget
# reference, self-targeting), two text changes:
#   - stop-side: settled means stop, from the FIRST step — no checkpoint delay.
#     The asymmetry does the work: cheap exits need confidence, not a schedule.
#   - continue-side: continuing is licensed by PROGRESS, not by uncertainty.
#     The exhaustion escape ("genuinely exhausted -> commit") returns from v3;
#     it was dropped in v4 and its absence is what fed the deep-band money pit
#     (73.4% of all tokens for 60% accuracy).
# Oracle says the budget is sufficient and misallocated: 87% @ 3139 mean rtok
# vs v4's 80% @ 3297. Move mass, don't add tokens.
# INVARIANT (BUG-075) unchanged: seed ends mid-reasoning ("Step 1:").


def _contract_v5_settled(ctk: dict, budget: int) -> bool:
    """v5: static banner, floorless stop + progress-conditional continue."""
    ctk.pop("pn100_steps", None)  # planner estimate deliberately unused
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. The moment "
        "your answer is settled — at any step, even the first — stop reasoning "
        "and give it; do not re-verify a settled answer. If it is not settled "
        "and you are still making real progress, keep going — there is room. "
        "If you have genuinely exhausted your approaches and are no longer "
        "making progress, stop and commit to your best answer."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v5 settled, budget=%d)", budget)
    return True


# v6 CANDIDATES (2026-07-19, post-v5 latency push — SHIP DARK, bench-gated):
# Target: hold v5's 88% while cutting mean rtok 2867 -> ~2000-2300 (mean lat
# 47s -> ~33-38s). The spend anatomy says the only lever that can move the
# MEAN is the deep band: n=33 items hold 74.0% of all tokens (~6.4K rtok
# ~105s each); cheap+mid together are 26%. The accuracy residual is the
# OPPOSITE cohort: 12 wrongs with median 1630 rtok — confident-wrong EARLY
# stops (gpqa-174/170 class), not deep grinders.
# Two candidates, one mechanism each (never both changes in one arm —
# unattributable):
#   v6a "prove-it exit": v5 text + (1) continue-license tightened from "real
#       progress" (self-certifiable indefinitely) to "each step must resolve
#       something new; a step that adds nothing means settled -> commit";
#       (2) quick answers must survive ONE short break-it step before being
#       given (taxes early exits ~100-200 rtok on ~1/3 of items ≈ +50 mean,
#       aims to convert the confident-wrong earlies).
#   v6b "named-unresolved": stop-side unchanged from v5; continue licensed
#       only while the model can NAME, in one line, what is still unresolved.
#       Naming is checkable-by-self each step; vaguer than-v5 grinding gets
#       no license. No break-check tax (isolates the deep-band lever).
# Pre-committed bars (V6-BANNER-CANDIDATES-RUNBOOK-20260719.md): survive =
# mean rtok <=2300 AND paired-net vs v5 >= -2; promote = <=2200 AND net >= 0;
# kill = net <= -4 or truncation/parse anomalies. INVARIANT (BUG-075): seed
# still ends mid-reasoning ("Step 1:"). Precedence v6a -> v6b -> v5 -> v4 -> v3.


def _contract_v6a_proveit(ctk: dict, budget: int) -> bool:
    """v6a: v5 + resolve-something-new continue license + break-it guard."""
    ctk.pop("pn100_steps", None)  # planner estimate deliberately unused
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. The moment "
        "your answer is settled — at any step, even the first — stop reasoning "
        "and give it. An answer that arrived quickly counts as settled only "
        "after one short step spent trying to break it; if it survives, give "
        "it and do not verify further. Keep going only while each step "
        "resolves something new; a step that adds nothing new means your "
        "current best answer IS settled — commit to it."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v6a prove-it, budget=%d)", budget)
    return True


def _contract_v6b_named(ctk: dict, budget: int) -> bool:
    """v6b: v5 stop-side + continue licensed only by naming the unresolved."""
    ctk.pop("pn100_steps", None)  # planner estimate deliberately unused
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. The moment "
        "your answer is settled — at any step, even the first — stop reasoning "
        "and give it; do not re-verify a settled answer. Continue only while "
        "you can state, in one line at the start of the step, what is still "
        "unresolved. When you can no longer name anything unresolved — or "
        "what you name would not change your answer — commit to your best "
        "answer."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v6b named-unresolved, budget=%d)", budget)
    return True


def _contract_v7_stateanswer(ctk: dict, budget: int) -> bool:
    """v7 (2026-07-22): v5 + report-9 stop-gate lever. The </think> gate fires
    when a STATED answer matches the model's internal computation (Termination
    Circuit, Qwen3: 94% vs 8% for a wrong value); "the most direct lever is to
    get the model to state its candidate answer earlier." So force an explicit
    running answer each step -> the gate can fire the moment the answer is
    computed, cutting the overthinking tail (median answer at ~30% of CoT).
    General mechanism, not benchmark-tuned. SHIP DARK (GENESIS_PN102_BANNER_V7)."""
    ctk.pop("pn100_steps", None)
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. At the START "
        "of each step, write one line 'Current answer: <your best answer now>'. "
        "The moment that stated answer stops changing from the previous step — "
        "at any step, even the first — stop reasoning and give it; do not "
        "re-verify a settled answer. Keep reasoning only while your stated "
        "answer is still changing or you are still making real progress. If you "
        "have genuinely exhausted your approaches, commit to your best answer."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v7 state-answer-early, budget=%d)", budget)
    return True


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
    # [2026-07-23 rec-4] per-band ANNOUNCED-N remap (concrete, not a ceiling):
    # "5:6,3:4" announces 6 where the planner said 5. Enforcement untouched.
    # Trace evidence: the N=5 pileup owns the premature-commit/skipped-verify
    # flips; a concrete bump keeps the anchor sharp (ceilings de-anchor).
    if isinstance(planner_steps, int) and planner_steps > 0:
        raw_map = os.environ.get("GENESIS_PN102_V3_N_BUMP_MAP", "").strip()
        if raw_map:
            try:
                bump = {int(k): int(v) for k, v in
                        (p.split(":") for p in raw_map.split(","))}
                planner_steps = bump.get(planner_steps, planner_steps)
            except ValueError:
                log.warning("PN102: bad V3_N_BUMP_MAP %r ignored", raw_map)
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
    # [2026-07-23 B2-S1/echo-anchor] optional Step-1 restatement opener: makes
    # the model re-echo the ask at Step 1 (Echoes-as-Anchors: a nearby echo
    # sharpens the numeric anchor). BUG-075 invariant holds (ends mid-reasoning).
    if _env_bool("GENESIS_PN102_V3_STEP1_ECHO", False):
        ctk["pn_env_seed"] = (f"{seed_label}: ~{steps} short steps.\n"
                              "Step 1 — what exactly is being asked:")
    else:
        ctk["pn_env_seed"] = f"{seed_label}: ~{steps} short steps.\nStep 1:"
    log.info("PN102: contract set (v3 sized, steps=%d budget=%d)", steps, budget)
    return True


def _contract_v8_hybrid(ctk: dict, budget: int) -> bool:
    """v8 (2026-07-23): numbered scaffold + GENEROUS announced step-ceiling +
    v5's behavioral stop/continue/exhausted clauses. Rationale (V3-SIZED-DEEP-
    ANALYSIS-20260723): the announced N is a hard behavioral anchor (72/100
    v3sizedfull2 traces land EXACTLY on Step N), so a lean N truncates —
    7/12 bad flips were premature commits from the classifier's N=5 pileup.
    v8 announces a CEILING ("up to ~N") the model should finish before, with
    v5's settled/progress/exhausted clauses making N a soft checkpoint, never
    a target. Announce in STEPS (behavioral), enforce in TOKENS (cost) —
    never derive both from one lean guess. Composes with
    GENESIS_PN102_ANNOUNCE_CEILING (dispatch inflates `budget` announce-side
    only). SHIP DARK (GENESIS_PN102_BANNER_V8)."""
    tps = max(50, _env_int("GENESIS_PN102_V8_TOKENS_PER_STEP", 260))
    planner_steps = ctk.pop("pn100_steps", None)
    n_ceil = max(3, round(budget / tps))
    if isinstance(planner_steps, int) and planner_steps > 0:
        n_ceil = max(n_ceil, planner_steps)
    n_ceil += max(0, _env_int("GENESIS_PN102_V8_STEP_HEADROOM", 4))
    # [2026-07-23 v8b] Arm-B measured: announcing the GENEROUS step count
    # (n_ceil) inflates spend — the anchor paces the model toward the big
    # number (wall +10% vs v5, dominated by v3+ceiling). Doctrine: generosity
    # belongs in the LICENSE + token clause; the STEP anchor stays lean.
    # V8_LEAN_ANCHOR=1 announces the lean planner N as a soft checkpoint with
    # v5's behavioral clauses around it; default off preserves Arm-B semantics.
    if _env_bool("GENESIS_PN102_V8_LEAN_ANCHOR", False):
        steps = planner_steps if isinstance(planner_steps, int) and planner_steps > 0 \
            else max(3, round(budget / tps))
        ctk["pn_env_banner"] = (
            f"[envelope] Thinking budget: about {steps} short reasoning steps "
            f"(budget allows up to ~{budget} thinking tokens). Work in "
            "numbered steps. The moment your answer is settled — at any step, "
            "even the first — stop reasoning and give it; do not re-verify a "
            f"settled answer. If it is not settled and you are still making "
            f"real progress, keep going past Step {steps} if you need to — "
            "there is room. If you have genuinely exhausted your approaches "
            "and are no longer making progress, stop and commit to your best "
            "answer."
        )
        ctk["pn_env_seed"] = f"Budget: ~{steps} short steps.\nStep 1:"
        log.info("PN102: contract set (v8b lean-anchor, steps=%d budget=%d)",
                 steps, budget)
        return True
    ctk["pn_env_banner"] = (
        "[envelope] Work through your reasoning in numbered steps. There is "
        f"room for up to ~{n_ceil} steps (~{budget} thinking tokens) — more "
        "than this should need. The moment your answer is settled — at any "
        "step, even the first — stop reasoning and give it; do not re-verify "
        "a settled answer. If it is not settled and you are still making real "
        "progress, keep going — there is room. If you have genuinely "
        "exhausted your approaches and are no longer making progress, stop "
        "and commit to your best answer."
    )
    ctk["pn_env_seed"] = "Step 1:"
    log.info("PN102: contract set (v8 hybrid, n_ceil=%d budget=%d)", n_ceil, budget)
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
    # [2026-07-22 USER hypothesis] announce-vs-grant decoupling: a too-low
    # ANNOUNCED number degrades reasoning itself (07-18: announced ~12 steps on
    # a 21-step budget = model complies, −4pt at identical caps). When growth
    # is available, size the BANNER from the reachable ceiling (grant +
    # growth allowance) while enforcement stays at the lean grant — the model
    # plans with room, self-stop keeps easy items short, growth delivers the
    # room only if actually used. Env: GENESIS_PN102_ANNOUNCE_CEILING=1.
    if _env_bool("GENESIS_PN102_ANNOUNCE_CEILING", False):
        allowance = _env_int("GENESIS_PN108_GROW_MAX_TOTAL", 0)
        if allowance > 0:
            budget = budget + allowance
    # v4 ships OFF: it replaces a prod-validated banner and must not become the
    # live path until a bench window says so. It is also COUPLED to the
    # generous-budget env (see the v4 note above) — enable both or neither.
    if _env_bool("GENESIS_PN102_BANNER_V8", False):
        _contract_v8_hybrid(ctk, budget)
    elif _env_bool("GENESIS_PN102_BANNER_V7", False):
        _contract_v7_stateanswer(ctk, budget)
    elif _env_bool("GENESIS_PN102_BANNER_V6A", False):
        _contract_v6a_proveit(ctk, budget)
    elif _env_bool("GENESIS_PN102_BANNER_V6B", False):
        _contract_v6b_named(ctk, budget)
    elif _env_bool("GENESIS_PN102_BANNER_V5", False):
        _contract_v5_settled(ctk, budget)
    elif _env_bool("GENESIS_PN102_STATIC_BANNER", False):
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
    # [2026-07-23 P-esc novelty gate, DARK] ledger 8G: re-grinding a PLATEAUED
    # trace wastes tokens and can hurt (trace review: heavy late second-
    # guessing regresses as often as it rescues). Text-side proxy for the
    # engine's novelty signal: new-trigram fraction of the reasoning TAIL vs
    # the body. Low tail novelty = the trace was looping when guillotined ->
    # return the forced answer instead of extending. Fail-open on any error.
    if _env_bool("GENESIS_PN101_NOVELTY_GATE", False):
        try:
            words = reasoning.split()
            if len(words) > 900:
                tail, body = words[-400:], words[:-400]
                seen = {tuple(body[i:i + 3]) for i in range(len(body) - 2)}
                fresh = sum(1 for i in range(len(tail) - 2)
                            if tuple(tail[i:i + 3]) not in seen)
                frac = fresh / max(1, len(tail) - 2)
                try:
                    thr = float(os.environ.get(
                        "GENESIS_PN101_NOVELTY_MIN", "") or 0.35)
                except ValueError:
                    thr = 0.35
                if frac < thr:
                    log.info(
                        "PN101: escalation skipped — tail novelty %.2f < %.2f "
                        "(plateaued; re-grind refused)", frac, thr,
                    )
                    return False
        except Exception:
            pass
    # [2026-07-22 grow-aware] escalation is a TOP-UP to a shared total-thinking
    # ceiling, never a fresh full budget stacked on grown spend (USER: no 2x).
    # A request that already thought >= the ceiling (a grown-to-ceiling rambler
    # that dodged the plateau detector) gets no further extension.
    total_ceil = _env_int("GENESIS_PN101_TOTAL_THINK_CEIL", 10240)
    prior_rtok = len(reasoning) // 4
    budget = min(_env_int("GENESIS_PN101_ESCALATE_BUDGET", 10240),
                 total_ceil - prior_rtok)
    if budget < 512:
        log.info(
            "PN101: escalation skipped — prior think ~%d already at/near total "
            "ceiling %d", prior_rtok, total_ceil,
        )
        return False
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
        # [2026-07-23 ultra-review #4] the engine's continue_thinking init
        # charges the prompt-resident reasoning against the budget already —
        # passing (ceil - prior) double-subtracted prior think and made prod
        # escalation (grants == ceil) a permanent no-op. Pass the TOTAL
        # ceiling; the engine performs the single subtraction.
        # [ultra-review #9] carry the client's sampling params — the
        # continuation used to silently run at default top_p/top_k/etc.
        kwargs: dict[str, Any] = {
            "model": getattr(request, "model", None),
            "messages": messages,
            "temperature": getattr(request, "temperature", 0.0) or 0.0,
            "stream": False,
            "thinking_token_budget": total_ceil,
            "chat_template_kwargs": dict(
                getattr(request, "chat_template_kwargs", None) or {},
                **{_MARKER_KEY: True},
            ),
        }
        for sf in ("top_p", "top_k", "min_p", "presence_penalty",
                   "frequency_penalty", "repetition_penalty", "stop",
                   "seed", "logit_bias"):
            if sf in fields:
                sv = getattr(request, sf, None)
                if sv is not None:
                    kwargs[sf] = sv
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
        # [2026-07-23 ultra-review #9] success = an ANSWER was delivered;
        # reasoning-only continuations are handed to the repair leg, not
        # counted as escalation wins.
        if new_content:
            _STATS["escalations_succeeded"] += 1
        else:
            _STATS["escalations_reasoning_only"] = (
                _STATS.get("escalations_reasoning_only", 0) + 1
            )
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
