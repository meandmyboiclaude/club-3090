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

Other legs living in this module (each with its own master flag, all OFF by
default):
  PN102  Leg 1  — the envelope-contract banner (GENESIS_ENABLE_PN102_CONTRACT)
  PN102  Leg 1b — server-side deep/lean BANNER autosplit, BUG-157
                  (GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT)
  PN102  Leg 4  — output-side banner-echo net, BUG-156 (gated on Leg 1's master
                  GENESIS_ENABLE_PN102_CONTRACT since BUG-168; sub-toggles
                  GENESIS_PN102_STRIP_ECHO / _STRIP_STEP_ECHO)
  PN123  Leg 3  — premature-close gate (GENESIS_ENABLE_PN123_CLOSEGATE; fka
                  PN118, renumbered 2026-07-26 for BUG-144 — see Leg 3's header)
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import time
from typing import Any

try:
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.answer_rescue")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.middleware.answer_rescue")

_MARKER_KEY = "pn101_internal"
_PN100_MARKER_KEY = "pn100_internal"
_HINT_SENTINEL = "[reply-window note]"

# ── PN102 force-v5 protocol (BUG-168, 2026-07-27) ────────────────────────────
# `pn102_force_v5` is the ONE ctk key that bypasses `_skip_common`, so a bare
# truthy from any caller put the envelope banner back on tools/structured
# requests — the exact population BUG-156 was fixed by excluding. A boolean
# cannot carry provenance, so the bypass now needs PROOF of internal origin:
#
#   * the value IS `_PN102_FORCE_V5_SENTINEL`, a documented shared constant,
#     OR
#   * the request already carries one of this module's own re-entry markers
#     (`pn101_internal` / `pn118_internal`), which only a synthetic self-call
#     has — this is the back-compat path for any in-tree setter still writing
#     `True`.
#
# A bare `pn102_force_v5: true` from a client is NOT provenance. It still
# selects the v5 banner SHAPE on a request that passes the normal gates (that
# choice was never dangerous and predates this key's bypass), but it can no
# longer unlock the bypass itself.
#
# The sentinel is deliberately a SHARED value, not a secret: qbench45's
# `thinkingcap_router_online*` arms are a legitimate external forcer (they call
# /v1/h119/score themselves and pass the route back), and their treatment
# `{ctk: {pn102_force_v5: <sentinel>}}` must keep working. Change this string
# and qbench45/config/arms.yaml must change with it.
_PN102_FORCE_V5_KEY = "pn102_force_v5"
_PN102_FORCE_V5_SENTINEL = "genesis-internal-force-v5"

# ── PN102 banner-injection marker (BUG-168, 2026-07-27) ──────────────────────
# Leg 4's echo net used to read `chat_template_kwargs["pn_env_banner"]` as proof
# that WE injected the banner. That key is client-settable and at least one real
# client protocol sets it, so the net rewrote served answers on requests we
# never touched — with every GENESIS_ENABLE_* flag OFF. This attribute is
# written ONLY by `maybe_add_answer_hint`, on the request object, which is never
# serialized to a caller and cannot be spoofed through the request body.
_BANNER_INJECTED_ATTR = "_genesis_pn102_banner_injected"
_ANSWER_TAIL_RE = re.compile(r"(final\s+)?answer\s*[:\-]", re.IGNORECASE)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
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
    # Leg 3 counters keep their pn118_* spelling on purpose — see the BUG-144
    # note in Leg 3's header: they are the recorded telemetry names and the
    # assertion surface of fixes/test_pn118_logic.py. Only the patch ID, its
    # env flags and its log lines were renumbered to PN123.
    "pn118_skips": 0,
    "pn118_shadow_would_fire": 0,
    "pn118_attempts": 0,
    "pn118_fires": 0,
    "pn118_errors": 0,
    # Leg 1b — server-side route→banner autosplit (BUG-157).
    "autosplit_probes": 0,
    "autosplit_deep": 0,
    "autosplit_lean": 0,
    "autosplit_unavailable": 0,
    "autosplit_errors": 0,
    # Leg 4 — output-side banner echo net (BUG-156).
    "banner_echo_stripped": 0,
    "banner_step_echo_seen": 0,
    "banner_step_echo_stripped": 0,
    # Leg 5 — PN155 budget-truth guard (BUG-155).
    "pn155_seen": 0,
    "pn155_stamped": 0,
    "pn155_fired": 0,
    "pn155_flagged": 0,
    "pn155_retries": 0,
    "pn155_retry_rescued": 0,
    # [BUG-167] a retry whose payload did not parse under the request's own
    # grammar — rejected, not served. Distinct from pn155_retry_rescued==0 with
    # an empty payload, which is the ORIGINAL BUG-155 shape.
    "pn155_retry_unparseable": 0,
    "pn155_errors": 0,
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
    # [2026-07-24 goal-80] v5 has no answer clause (v3's terse answers come
    # from its clause; v5 replies ramble ~4x the tokens). Opt-in graft of the
    # v3-validated wording — default-dark, GENESIS_PN102_V5_ANSWER_CLAUSE=1.
    # AC2: apply only to the budget mass — deep items close early under a
    # reply-level anchor (measured: gpqa-099 3151->986 rtok, right->wrong;
    # ACfull lost chronic-deep 127/142/174). Mass answers are worthless
    # overflow (cap-bound cohort: acc 6/9 at both cut depths, ans 467->1255).
    if (_env_bool("GENESIS_PN102_V5_ANSWER_CLAUSE", False)
            and budget <= _env_int("GENESIS_PN102_V5_AC_MAX_BUDGET", 2600)):
        sentences = max(1, _env_int("GENESIS_PN102_SENTENCES", 3))
        ctk["pn_env_banner"] += (
            " Unless the user asked for longer form, put your final answer in "
            f"the FIRST sentence of your reply, then at most {sentences} "
            "sentences total."
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


# [PN102 prefix-order 2026-07-25, backlog #4] GENESIS_PN102_BANNER_STATIC_FIRST:
# the number-carrying banners (v3 sized, v8 hybrid, v8b lean-anchor) open with
# per-request figures ("Thinking budget: about {N} steps (~{budget} tokens)"),
# so the banner's leading tokens differ on every request and any prefix-cache
# block containing the banner head can never be reused across requests
# (measured hit rate 0%). With the flag ON the SAME information is emitted in
# a stable order: all static instruction text first (referring to "Step N" /
# "the figures at the END of this notice"), then one trailing
# "Figures: N = ..." sentence carrying every per-request value. Semantic
# content is preserved field-for-field — only the order changes.
# QUALITY-AFFECTING (the prompt the model sees changes) -> default OFF =
# byte-identical current behaviour; flip to 1 for the 30-item A/B screen.
# The think-SEED is deliberately NOT reordered: it is the very tail of the
# prompt (never a prefix for anything) and BUG-075 requires it to end
# mid-reasoning ("Step 1:"), so numbers-last is impossible there anyway.
# v4/v5/v6a/v6b/v7 banners are already request-invariant — flag is a no-op.
def _banner_static_first() -> bool:
    return _env_bool("GENESIS_PN102_BANNER_STATIC_FIRST", False)


def _contract_v3_sized(ctk: dict, budget: int) -> bool:
    """v3: budget/planner-sized banner. The validated prod path (072fff66)."""
    tps = max(50, _env_int("GENESIS_PN102_TOKENS_PER_STEP", 193))
    planner_steps = ctk.pop("pn100_steps", None)
    # [2026-07-24 goal-80] per-REQUEST N override (router arm; default-dark —
    # fires only when the caller sends pn102_force_steps in chat_template_kwargs).
    # The chronic v3 losses close early on a too-small announced N; a correct
    # per-item N keeps the anchor sharp (concrete bump, not a ceiling).
    _forced = ctk.pop("pn102_force_steps", None)
    if isinstance(_forced, int) and _forced > 0:
        planner_steps = _forced
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
    static_first = _banner_static_first()
    if budget >= _env_int("GENESIS_PN102_PERMISSION_MIN", 4096) and has_headroom:
        if static_first:
            # [prefix-order] same clause, "Step N" symbolic — figures trail.
            pace_clause = (
                "Number your steps and wrap up around Step N once your "
                "answer is settled; if the problem proves deeper than "
                "planned, keep reasoning past Step N — the budget is "
                "generous — and do not conclude while your answer is still "
                "uncertain. If you have genuinely exhausted your approaches, "
                "commit to your best answer. Do not let the budget cut you "
                "off. "
            )
        else:
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
        if static_first:
            pace_clause = (
                "Number your steps and wrap up around Step N yourself — "
                "do not let the budget cut you off. "
            )
        else:
            pace_clause = (
                f"Number your steps and wrap up around Step {steps} yourself — "
                "do not let the budget cut you off. "
            )
        seed_label = "Budget"
    # [2026-07-23 D2 range-announce, dark] "about N–M steps": the low endpoint
    # keeps the precise-anchor pull (easy majority unchanged), the high endpoint
    # is a sanctioned in-anchor landing for deep items — numbers participate in
    # shallow-layer anchoring where license prose measurably cannot (lit:
    # tandem/both-anchor; unpublished for step announces). M = ceil(RANGE×N).
    range_mult = _env_float("GENESIS_PN102_V3_RANGE", 0.0)
    if range_mult > 1.0:
        hi = max(steps + 1, math.ceil(steps * range_mult))
        steps_txt = f"about {steps}–{hi} short reasoning steps"
    else:
        steps_txt = f"about {steps} short reasoning steps"
    # [2026-07-23 ANS-freeze, dark] running answer + structural stability
    # freeze (lit: Dynasor/ES-CoT/Answer-Convergence family): the re-verify
    # license exists only while the stated answer is unstable.
    ans_k = _env_int("GENESIS_PN102_V3_ANS_FREEZE", 0)
    ans_clause = ""
    if ans_k >= 2:
        ans_clause = (
            f" End every step with 'ANS: <your current answer>'. Once the "
            f"same answer has ended {ans_k} consecutive steps, it is settled "
            "— stop reasoning and give it."
        )
    if static_first:
        # [prefix-order] static instruction text first (byte-identical across
        # requests within a pace-clause branch), every per-request figure in
        # the single trailing "Figures:" sentence. Same fields, stable prefix.
        ctk["pn_env_banner"] = (
            "[envelope] You have a thinking budget for this request; its "
            "figures — the step count N and the token allowance — are stated "
            "at the END of this notice. " + pace_clause + answer_clause
            + ans_clause + f" Figures: N = {steps} ({steps_txt}; {size_clause})."
        )
    else:
        ctk["pn_env_banner"] = (
            f"[envelope] Thinking budget: {steps_txt} "
            f"({size_clause}). " + pace_clause + answer_clause + ans_clause
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
        if _banner_static_first():
            # [prefix-order] static text first, per-request figures trail.
            ctk["pn_env_banner"] = (
                "[envelope] You have a thinking budget for this request; its "
                "figures — the step count N and the token allowance — are "
                "stated at the END of this notice. Work in numbered steps. "
                "The moment your answer is settled — at any step, even the "
                "first — stop reasoning and give it; do not re-verify a "
                "settled answer. If it is not settled and you are still "
                "making real progress, keep going past Step N if you need to "
                "— there is room. If you have genuinely exhausted your "
                "approaches and are no longer making progress, stop and "
                "commit to your best answer. "
                f"Figures: N = {steps} (about {steps} short reasoning steps; "
                f"budget allows up to ~{budget} thinking tokens)."
            )
        else:
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
    if _banner_static_first():
        # [prefix-order] static text first, per-request figures trail.
        ctk["pn_env_banner"] = (
            "[envelope] Work through your reasoning in numbered steps. There "
            "is room for up to ~N steps — more than this should need; N and "
            "the token allowance are stated at the END of this notice. The "
            "moment your answer is settled — at any step, even the first — "
            "stop reasoning and give it; do not re-verify a settled answer. "
            "If it is not settled and you are still making real progress, "
            "keep going — there is room. If you have genuinely exhausted "
            "your approaches and are no longer making progress, stop and "
            "commit to your best answer. "
            f"Figures: N = {n_ceil} steps (~{budget} thinking tokens)."
        )
    else:
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


def _force_v5_is_internal(ctk: dict, raw: Any) -> bool:
    """Does this `pn102_force_v5` value prove internal origin? (BUG-168)

    Only an internal-origin force may bypass `_skip_common`. Two accepted
    proofs, both documented at `_PN102_FORCE_V5_SENTINEL`: the shared sentinel
    value, or a self-call re-entry marker already on the request. A bare `True`
    from a client body is neither.
    """
    if not raw:
        return False
    if isinstance(raw, str) and raw.strip() == _PN102_FORCE_V5_SENTINEL:
        return True
    # Back-compat: an in-tree setter writing `True` is authorised by the
    # suppression markers it necessarily carries (a client body that guessed
    # both marker keys has already forfeited the skip gates elsewhere).
    return bool(ctk.get(_MARKER_KEY) or ctk.get(_PN123_MARKER))


def _mark_banner_injected(request: Any, banner: Any) -> None:
    """Record on the REQUEST that Leg 1 injected this banner (BUG-168).

    Not a ctk key: ctk is caller-supplied and reaches the template. This
    attribute is the only thing Leg 4's echo net will accept as proof.
    """
    if not isinstance(banner, str) or not banner:
        return
    try:
        setattr(request, _BANNER_INJECTED_ATTR, banner)
    except Exception:  # pragma: no cover - frozen models
        log.debug("PN102: could not mark the injected banner", exc_info=True)


def maybe_add_answer_hint(request: Any) -> None:
    if not _env_bool("GENESIS_ENABLE_PN102_CONTRACT"):
        return
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    # [PN123 rerun 2026-07-23] a rerun forces the v5-shape banner for THIS one
    # synthetic call regardless of the live banner-version env chain. The
    # dispatch below reads env, not ctk, so this ctk key is the override seam.
    # Checked before the skip/marker gates: the rerun carries the PN123/PN101
    # markers (to suppress post-response re-entry), and those would otherwise
    # trip _skip_common and drop the banner it specifically needs.
    # [BUG-168 2026-07-27] but ONLY an internal-origin force gets that bypass —
    # see `_force_v5_is_internal`. A client-supplied bare `true` still picks the
    # v5 shape, it just no longer opens the tools/structured door (BUG-156).
    _force_raw = ctk.pop(_PN102_FORCE_V5_KEY, False)
    force_v5 = bool(_force_raw)
    force_internal = _force_v5_is_internal(ctk, _force_raw)
    # [BUG-157 2026-07-26] the ROUTE-driven promotion (Leg 1b below) is a
    # SEPARATE key on purpose. `pn102_force_v5` bypasses _skip_common because a
    # PN123 rerun is a synthetic call that carries the suppression
    # markers; an automatic promotion must NOT inherit that bypass, or it would
    # re-open BUG-156 by putting the banner back on tools/structured requests.
    # So `pn102_auto_v5` is popped here but only consulted after the gates.
    auto_v5 = bool(ctk.pop("pn102_auto_v5", False))
    if not _bounded(request) or (not force_internal and _skip_common(request)):
        return
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
    if force_v5:
        _contract_v5_settled(ctk, budget)  # PN123 rerun: v5-shape, this call only
    elif auto_v5:
        _contract_v5_settled(ctk, budget)  # Leg 1b: the H119 route said "deep"
    elif _env_bool("GENESIS_PN102_BANNER_V8", False):
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
    # [BUG-168] the ONLY place the echo net's proof-of-injection is written.
    _mark_banner_injected(request, ctk.get("pn_env_banner"))
    _STATS["hints_added"] += 1


# ─── Leg 1b: server-side deep/lean BANNER autosplit (BUG-157) ────────────────
# House-original, 2026-07-26. Master flag GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT,
# DEFAULT OFF: with it unset this whole leg is bytes on disk and nothing about
# the current behaviour changes.
#
# THE BUG. The design contract is two modes and no client cooperation: a normal
# request runs the FULL chain (route + auto-sized cap), or thinking is off and
# the machinery is bypassed. The BUDGET half honours that — PN100 sizes every
# request and the H119 consumer modulates it in-engine. The BANNER half did
# NOT: `pn102_force_v5` is written by production code in exactly one place
# (_pn123_rerun, the adjudicated-rerun path), and `maybe_add_answer_hint` only
# ever POPS it back out of the caller's chat_template_kwargs. So the deep/lean
# banner split existed only for a client that called /v1/h119/score itself,
# rendered its own banner, and sent it back — which is how the 07-24 champion
# was produced and why it was never reproducible from a plain request.
#
# THE ORDERING PROBLEM, AND WHY THE FIX HAS TO LIVE HERE. The lens route is
# derived FROM the prefill; the banner has to be in the prompt BEFORE prefill.
# One pass cannot do both (patch_h119_route_api.py, "THE HARD PART"). The only
# resolution is two passes, and the only question is who pays for the second
# one. Today it is the caller. This leg moves it into the server: score the
# request the caller actually sent, at max_tokens=1, read the route off the
# response, then run the real request with the banner that route implies. Same
# cost as the client-side protocol (one extra prefill, ~0.26 s measured, and
# APC makes the real request's prefill nearly free afterwards) — but universal
# and invisible, which is the whole point of the bug.
#
# WHY A WRAPPER AND NOT AN INLINE CALL. `maybe_add_answer_hint` is the only
# pre-render seat, and its call site (fixes/patch_pn101_answer_rescue.py, HINT
# site) is `_pn101_hint(request)` — SYNCHRONOUS, un-awaited, inside the running
# event loop. A probe from there cannot be awaited, cannot be run with
# `run_until_complete` (the loop is already running) and cannot be blocked on
# from another thread (we ARE the loop thread — the probe would need the loop
# we are blocking). So the probe has to happen one frame OUT, before
# `create_chat_completion` is entered, and this module reaches that frame by
# wrapping the bound method on the serving class the first time it is handed a
# serving instance. The wrap is installed at REQUEST time inside the live
# server process, not at patch time — the failure mode the h119 sidecar warns
# about ("apply_all runs standalone and then execs, so setattr is gone") does
# not apply here.
#
# COST OF THAT CHOICE, STATED NOT HIDDEN: the FIRST non-internal chat
# completion after a boot runs before the wrapper exists and is therefore never
# split. It is logged at INFO when the install lands. Everything after it is.
#
# WHAT IT DOES NOT DO. It does not touch the budget — PN100 keeps ownership of
# sizing and the H119 in-engine consumer keeps ownership of modulating it
# (H119_*_MULT=1.0 today = exact passthrough). This leg only chooses the
# BANNER, which is precisely the half that was missing.
#
# Env:
#   GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT   master, default OFF
#   GENESIS_PN102_AUTOSPLIT_V5_ROUTE       route that earns the v5 banner
#                                          (default "deep"; the other route
#                                          falls through to the normal env
#                                          dispatch chain, i.e. v3 in prod)
#   GENESIS_PN102_AUTOSPLIT_TIMEOUT_S      probe timeout (default 60)
# Requires (all live on :8021 today): GENESIS_ENABLE_PN102_CONTRACT=1,
# GENESIS_ENABLE_H119_LENS_ROUTER=1 and GENESIS_ENABLE_H119_ROUTE_API=1 — the
# route API bridge is what publishes the route into kv_transfer_params, and it
# works with PN119_MODE=shadow as well as enforce.

_AUTOSPLIT_MARKER = "pn102_autosplit_probe"
_AUTOSPLIT_INSTALLED_ATTR = "_pn102_autosplit_wrapped"

# Mirrors patch_h119_route_api.py::_H119_PROBE_OVERRIDES. Filtered against the
# request model's fields, so a pin that renames one degrades to "not
# overridden" rather than raising.
_AUTOSPLIT_PROBE_OVERRIDES = {
    "stream": False,
    "stream_options": None,
    "n": 1,
    "max_tokens": 1,
    "max_completion_tokens": 1,
    "logprobs": False,
    "top_logprobs": 0,
    "prompt_logprobs": None,
    "echo": False,
    "kv_transfer_params": None,
}

_H119_BRIDGE: Any = None


def _autosplit_on() -> bool:
    return _env_bool("GENESIS_ENABLE_PN102_ROUTE_AUTOSPLIT", False)


def _h119_bridge() -> Any:
    """`vllm._genesis_h119_api`, or None. Resolved once; False caches absence.

    This is the first import this module has ever taken from the H119 lane —
    BUG-157's evidence line "answer_rescue.py imports nothing from pn119/h119,
    so the banner code cannot see the route" names exactly this gap. Lazy and
    guarded: the bridge is written by /fixes/patch_h119_route_api.py and is
    simply absent on a boot where that patch soft-skipped.
    """
    global _H119_BRIDGE
    if _H119_BRIDGE is None:
        try:
            from vllm import _genesis_h119_api as _m

            _H119_BRIDGE = _m
        except Exception:
            _H119_BRIDGE = False
    return _H119_BRIDGE or None


def _autosplit_candidate(request: Any) -> bool:
    """Cheap pre-filters, evaluated BEFORE any GPU work is spent on a probe.

    NB `thinking_token_budget` is assigned by PN100's hook INSIDE
    `_create_chat_completion`, i.e. after this point, so `_bounded()` cannot be
    used here — a request that PN100 has not sized yet looks unbounded. These
    gates are the subset that is already decided at the API boundary.
    """
    ctk = getattr(request, "chat_template_kwargs", None) or {}
    if not isinstance(ctk, dict):
        return False
    # internal calls (PN101 repair/escalate, PN123 margin/continue/rerun, and
    # this leg's own probe) — never re-enter.
    if (ctk.get(_MARKER_KEY) or ctk.get(_PN100_MARKER_KEY)
            or ctk.get(_AUTOSPLIT_MARKER)):
        return False
    # mode (b): thinking off bypasses the thinking machinery entirely.
    if ctk.get("enable_thinking") is False:
        return False
    # the caller already made the decision itself — do not overrule it.
    if ctk.get("pn102_force_v5") or ctk.get("pn102_auto_v5"):
        return False
    if ctk.get("pn_env_banner"):
        return False
    # the same gates Leg 1 applies: a banner these never receive is a banner
    # not worth a probe (and structured/tool requests are BUG-156's fix).
    if getattr(request, "tools", None) or _has_structured_output(request):
        return False
    # n>1 parallel sampling publishes one of n routes, not a summary
    # (patch_h119_route_api.py, KNOWN WEAKNESSES) — do not act on it.
    n = getattr(request, "n", None)
    if isinstance(n, int) and n > 1:
        return False
    # a 1-token request has no reasoning to shape, and skipping it is also what
    # keeps /v1/h119/score's own probe from recursing back through here.
    cap = _completion_cap(request)
    if cap is not None and cap <= 1:
        return False
    return True


def _build_probe(request: Any) -> Any:
    """A max_tokens=1 copy of `request` — the prefill IS the score."""
    req_cls = type(request)
    fields = set(getattr(req_cls, "model_fields", {}) or {})
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    ctk[_MARKER_KEY] = True          # suppress Leg 1's banner on the probe
    ctk[_AUTOSPLIT_MARKER] = True    # and this leg, belt-and-braces
    update = {k: v for k, v in _AUTOSPLIT_PROBE_OVERRIDES.items()
              if k in fields}
    update["chat_template_kwargs"] = ctk
    copy = getattr(request, "model_copy", None)
    if callable(copy):
        return copy(update=update)
    raise TypeError("request model has no model_copy()")


async def _h119_route(orig: Any, serving: Any, request: Any,
                      raw_request: Any) -> str | None:
    """Score the prompt and return "deep"/"lean", or None if unavailable.

    Calls the UNWRAPPED `create_chat_completion` directly, so the probe cannot
    re-enter this leg no matter what the marker gates do.
    """
    bridge = _h119_bridge()
    if bridge is None:
        _STATS["autosplit_unavailable"] += 1
        log.info("PN102: autosplit — H119 route bridge not installed on this "
                 "boot (need /fixes/patch_h119_route_api.py); banner unchanged")
        return None
    try:
        if not bridge.enabled():
            _STATS["autosplit_unavailable"] += 1
            log.info("PN102: autosplit — H119 route API disabled "
                     "(GENESIS_ENABLE_H119_ROUTE_API=1 required); "
                     "banner unchanged")
            return None
    except Exception:
        pass
    probe = _build_probe(request)
    timeout = _env_int("GENESIS_PN102_AUTOSPLIT_TIMEOUT_S", 60)
    _STATS["autosplit_probes"] += 1
    resp = await asyncio.wait_for(orig(serving, probe, raw_request), timeout)
    payload = bridge.payload(resp)
    if not payload:
        _STATS["autosplit_unavailable"] += 1
        log.info("PN102: autosplit — probe returned no route (router not live "
                 "in the engine process, or it refused this prompt); "
                 "banner unchanged")
        return None
    route = str(payload.get("route") or "").strip().lower()
    if not route:
        _STATS["autosplit_unavailable"] += 1
        return None
    log.info("PN102: autosplit probe route=%s score=%s source=%s req=%s",
             route, payload.get("score"), payload.get("source"),
             payload.get("req_id"))
    return route


def _autosplit_wrapper(orig: Any) -> Any:
    """Wrap `OpenAIServingChat.create_chat_completion` with score-then-generate.

    Fail-open by construction: every failure path falls through to `orig` with
    the request unmodified, so the worst case is exactly today's behaviour plus
    one wasted prefill.
    """

    @functools.wraps(orig)
    async def _pn102_autosplit(serving, *args, **kwargs):
        request = kwargs.get("request") if "request" in kwargs else (
            args[0] if args else None)
        raw_request = kwargs.get("raw_request") if "raw_request" in kwargs else (
            args[1] if len(args) > 1 else None)
        try:
            if (request is not None and _autosplit_on()
                    and _env_bool("GENESIS_ENABLE_PN102_CONTRACT")
                    and _autosplit_candidate(request)):
                route = await _h119_route(orig, serving, request, raw_request)
                if route:
                    v5_route = (os.environ.get(
                        "GENESIS_PN102_AUTOSPLIT_V5_ROUTE", "")
                        or "deep").strip().lower()
                    if route == v5_route:
                        ctk = dict(
                            getattr(request, "chat_template_kwargs", None) or {})
                        ctk["pn102_auto_v5"] = True
                        request.chat_template_kwargs = ctk
                        _STATS["autosplit_deep"] += 1
                        log.info("PN102: autosplit route=%s → v5 banner "
                                 "(server-side, no client involvement)", route)
                    else:
                        _STATS["autosplit_lean"] += 1
                        log.info("PN102: autosplit route=%s → default banner "
                                 "chain", route)
        except Exception as exc:
            _STATS["autosplit_errors"] += 1
            log.warning("PN102: autosplit failed (%s) — request served "
                        "unsplit", exc)
        return await orig(serving, *args, **kwargs)

    return _pn102_autosplit


def install_route_autosplit(serving: Any) -> bool:
    """Install the wrapper on `type(serving)`. Idempotent; True iff it landed.

    Deliberately installed even when the master flag is OFF: the wrapper is
    inert in that case (it re-reads the flag per request), and installing
    unconditionally means flipping the flag needs no redeploy — only that the
    process has served one request, which by construction it has.
    """
    try:
        cls = type(serving)
        current = getattr(cls, "create_chat_completion", None)
        if current is None or getattr(current, _AUTOSPLIT_INSTALLED_ATTR, False):
            return False
        wrapped = _autosplit_wrapper(current)
        setattr(wrapped, _AUTOSPLIT_INSTALLED_ATTR, True)
        cls.create_chat_completion = wrapped
        log.info("PN102: autosplit wrapper installed on %s.%s "
                 "(master flag %s) — the request that installed it is the one "
                 "request this boot that cannot be split",
                 cls.__module__, cls.__name__,
                 "ON" if _autosplit_on() else "OFF")
        return True
    except Exception as exc:
        log.warning("PN102: autosplit wrapper install failed (%s) — the "
                    "deep/lean banner split stays client-only this boot", exc)
        return False


# ─── Leg 4: output-side banner echo net (BUG-156) ────────────────────────────
# The PN102 banner is scaffolding for the THINK channel. Two measured shapes
# leak it into the ANSWER channel instead: a reply that opens with the literal
# "[envelope]" marker (2/40 prod, 1/101 on one gpqa arm), and a reply that
# emits the banner's numbered steps ahead of the real answer ("Step 1: …" …
# "Step 12: Assemble JSON…"). `_has_structured_output` already suppresses the
# banner entirely for requests carrying response_format/guided_* — a guided arm
# measured 0/40 of either shape — but the prod caller asks for JSON in PROSE,
# so that guard never fires for it.
#
# The banner TEXT is not touched here: it was A/B-tuned and nothing in
# production reads the "[envelope]" token. This is a net under the answer
# channel, and it is deliberately narrow:
#
#   * it fires only when THIS request carried a banner WE injected, and only
#     strips the marker that banner itself opens with (read off the banner, not
#     hardcoded) — so a caller whose legitimate answer starts with "[envelope]"
#     is only ever touched on a request we were injecting into anyway.
#     [BUG-168 2026-07-27] that claim did not used to be TRUE: proof-of-
#     injection was `chat_template_kwargs["pn_env_banner"]`, a key a client can
#     set (and one real client protocol does — see the BUG-157 narrative
#     above), and the call site sat above every master-flag return, so with
#     every GENESIS_ENABLE_* unset a client that sent its own banner still had
#     its served answer rewritten. Now the net requires BOTH
#     GENESIS_ENABLE_PN102_CONTRACT (the master that owns injection: if we
#     never injected, we never rewrite) and `_BANNER_INJECTED_ATTR`, which only
#     `maybe_add_answer_hint` writes and no request body can carry;
#   * the numbered-step shape is DETECTED and counted but NOT stripped by
#     default. A leading "Step 1: … Step N: …" block is indistinguishable from
#     a caller legitimately asking for worked steps, and this net has no way to
#     tell the two apart. GENESIS_PN102_STRIP_STEP_ECHO=1 opts in for callers
#     that know they never want it.
#
# Env: GENESIS_ENABLE_PN102_CONTRACT (the master — the net does not run at all
#      without it, BUG-168), GENESIS_PN102_STRIP_ECHO (default ON under that
#      master — it can only fire on a request we injected into),
#      GENESIS_PN102_STRIP_STEP_ECHO (default OFF).

_BANNER_MARKER_RE = re.compile(r"^\s*(\[[a-z][a-z0-9 _-]{0,30}\])")
_STEP_LINE_RE = re.compile(r"^\s*Step\s+(\d+)\s*[:.\-]", re.IGNORECASE)


def _injected_banner_marker(request: Any) -> str | None:
    """The leading "[…]" token of the banner WE put on this request, if any.

    [BUG-168 2026-07-27] reads the INTERNAL injection marker, not
    `chat_template_kwargs["pn_env_banner"]`. That ctk key is client-settable —
    and per this file's own BUG-157 narrative a real client protocol sets it —
    so keying on it made the net rewrite served answers on requests we never
    injected into. `_BANNER_INJECTED_ATTR` is written in exactly one place,
    `maybe_add_answer_hint`, after it has actually written a banner.
    """
    banner = getattr(request, _BANNER_INJECTED_ATTR, None)
    if not isinstance(banner, str) or not banner:
        return None
    m = _BANNER_MARKER_RE.match(banner)
    return m.group(1) if m else None


def _strip_leading_steps(content: str) -> str | None:
    """Drop a leading contiguous block of numbered "Step N:" lines.

    Returns None unless the block is >= 2 steps AND real content survives it —
    a reply that is ONLY numbered steps is the answer, however unwelcome its
    shape, and discarding it would be exactly the silent data loss this module
    exists to prevent.
    """
    lines = content.splitlines()
    i = 0
    steps = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if _STEP_LINE_RE.match(lines[i]):
            steps += 1
            i += 1
            continue
        break
    if steps < 2:
        return None
    rest = "\n".join(lines[i:]).strip()
    return rest or None


def _maybe_strip_banner_echo(request: Any, result: Any) -> None:
    """Remove PN102 scaffolding that landed on the answer side of </think>."""
    marker = _injected_banner_marker(request)
    if marker is None:
        return
    choice = _extract_choice(result)
    message = getattr(choice, "message", None) if choice is not None else None
    if message is None:
        return
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        return

    if _env_bool("GENESIS_PN102_STRIP_ECHO", True):
        stripped = content.lstrip()
        if stripped.startswith(marker):
            new = stripped[len(marker):].lstrip()
            if new:
                try:
                    message.content = new
                    _STATS["banner_echo_stripped"] += 1
                    log.info("PN102: stripped echoed banner marker %s from the "
                             "served answer (BUG-156)", marker)
                    content = new
                except Exception:  # pragma: no cover - frozen models
                    return

    rest = _strip_leading_steps(content)
    if rest is not None:
        _STATS["banner_step_echo_seen"] += 1
        if not _env_bool("GENESIS_PN102_STRIP_STEP_ECHO", False):
            log.info("PN102: banner numbered-steps echoed into the served "
                     "answer (BUG-156, detect-only — set "
                     "GENESIS_PN102_STRIP_STEP_ECHO=1 to strip)")
            return
        try:
            message.content = rest
            _STATS["banner_step_echo_stripped"] += 1
            log.info("PN102: stripped echoed numbered-step block from the "
                     "served answer (BUG-156)")
        except Exception:  # pragma: no cover - frozen models
            pass


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


# ─── Leg 3: PN123 premature-close gate (async, post-response) ────────────────
# [BUG-144, renumbered 2026-07-26] This leg shipped as "PN118". That number was
# already taken: the vendored lane-2 sndr registry owns PN118_V2_MD5_WORKSPACE
# and PN118_V2_MD5_TURBOQUANT_ATTN (TurboQuant), which the boot recorder
# truncates to "PN118", and the live container also sets a bare
# GENESIS_ENABLE_PN118=1 for that lane. PN119 is NOT the escape (it is lane-2's
# TurboQuant k8v4 GQA kernel — which is why the lens router had to become
# H119), so this leg takes PN123: unclaimed in lane-1, lane-2, /fixes and the
# tracker, and legal under `_genesis/utils/patch_id_lint.py`'s recorder shape.
#
# What was renamed: the patch id, its env flags and its log lines. What was
# NOT, and why — three strings are load-bearing outside this file and this
# change is meant to be cosmetic, not a live-behaviour change:
#   * GENESIS_ENABLE_PN118_CLOSEGATE stays a working LEGACY ALIAS. It is the
#     marker `patch_id_lint.py::HOUSE_IDS_OUTSIDE_FIXES["PN118"]` asserts is
#     present in this file — dropping it turns a green gate red — and it is
#     what any existing compose/env would set. Same for every GENESIS_PN118_*
#     sub-knob: canonical name first, legacy honoured when the canonical one is
#     unset (see `_cg`).
#   * the ctk marker value "pn118_internal" (a private tag, never read back
#     outside this module, asserted by fixes/test_pn118_logic.py:691).
#   * the `_STATS` keys pn118_* (the recorded telemetry names, asserted
#     throughout fixes/test_pn118_logic.py).
# Finishing the rename means editing patch_id_lint.py's table and that test —
# both outside this file's ownership.
#
# House-original, 2026-07-23. The INVERSE of the dead ending-detection lane
# (which CUT thinking early — never rebuild that): PN123 catches the model
# VOLUNTARILY closing </think> far under its assigned budget (obeying the
# announced step number in the v3-sized banner) and answering with weak
# confidence — the premature-commit class that owns 5-7 of the 12 accuracy
# losses vs the numberless champion. It resumes generation INSIDE the think
# region with a first-person own-voice cue and the leftover budget.
#
# Trigger (ALL must hold), see A10 RECOURSE-GATED TRIM + s12-P17 arm rules:
#   1. thinking_token_budget set > 0 (bounded request)
#   2. VOLUNTARY close, not force/cap-bound: finish=stop with an emitted answer,
#      AND reasoning_tokens < budget - grace (a cap-bound close has no leftover)
#   3. PREMATURE: reasoning_tokens < FRAC * budget (budget left on the table)
#   4. WEAK confidence: letter-margin echo read < MARGIN (the A10 answer-token
#      margin; a post-close 1-token echo self-call, not a mid-trace level gate)
# Every failure path leaves the original response untouched (PN101 fail-open).
# One fire per request. Modes: shadow (log would-fire incl. margin, change
# nothing) | enforce (splice the continuation). Master default OFF.

_PN123_MARKER = "pn118_internal"  # wire value frozen — see the BUG-144 note
_PN123_DEFAULT_CUE = (
    "Wait — that felt too quick; let me actually check the remaining cases "
    "before I commit."
)
_PN123_LETTERS = "ABCDEFGHIJ"

# Legacy aliases for fixes/test_pn118_logic.py, which imports these by name.
_PN118_MARKER = _PN123_MARKER
_PN118_DEFAULT_CUE = _PN123_DEFAULT_CUE
_PN118_LETTERS = _PN123_LETTERS


def _cg(suffix: str) -> str:
    """Close-gate env NAME: canonical GENESIS_PN123_<suffix>, falling back to
    the legacy GENESIS_PN118_<suffix> when the canonical one is unset.

    Returns a name rather than a value so the existing `_env_bool/_env_int/
    _env_float` readers (and their defaults) keep owning the parsing.
    """
    new = "GENESIS_PN123_" + suffix
    if os.environ.get(new, "").strip():
        return new
    old = "GENESIS_PN118_" + suffix
    return old if os.environ.get(old, "").strip() else new


def _pn123_master_on() -> bool:
    for name in ("GENESIS_ENABLE_PN123_CLOSEGATE",
                 "GENESIS_ENABLE_PN118_CLOSEGATE"):  # legacy, still honoured
        if os.environ.get(name, "").strip():
            return _env_bool(name, False)
    return False


_pn118_master_on = _pn123_master_on  # legacy alias


def _reasoning_tokens(result: Any, reasoning: str) -> int:
    """Spend already consumed inside the think block. Prefer the engine's
    reasoning_tokens count; fall back to the ~4-char/token estimate PN101 uses."""
    usage = getattr(result, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    if details is not None:
        rt = getattr(details, "reasoning_tokens", None)
        if isinstance(rt, int) and rt > 0:
            return rt
    return len(reasoning) // 4


def _letter_posterior_margin(choice: Any) -> float | None:
    """Margin between the top-1 and top-2 single-letter posteriors at the echo
    position. High margin = the model is confident which option letter it means;
    low margin = weak commitment (the premature-commit signal). None if no
    letter mass is present (open-ended answer — margin read does not apply)."""
    lp = getattr(choice, "logprobs", None)
    toks = getattr(lp, "content", None) if lp else None
    if not toks:
        return None
    top = getattr(toks[0], "top_logprobs", None) or []
    probs: list[float] = []
    for t in top:
        tok = (getattr(t, "token", "") or "").strip()
        if len(tok) == 1 and tok.upper() in _PN123_LETTERS:
            probs.append(math.exp(getattr(t, "logprob", -99.0)))
    if not probs:
        return None
    probs.sort(reverse=True)
    top1 = probs[0]
    top2 = probs[1] if len(probs) > 1 else 0.0
    return top1 - top2


async def _letter_margin(serving: Any, request: Any, content: str) -> float | None:
    """One 1-token echo self-call reading the answer's letter posterior margin.
    messages = original + the emitted answer as an assistant prefix ending
    'Answer: (', continue_final_message + logprobs → letter margin. The
    alternative to a low-margin read is submitting that same guess anyway, so
    the read can only add accuracy (A10). Thinking OFF, temp 0, marker-tagged to
    avoid re-entry. Returns None on any structural miss (fail-open upstream)."""
    req_cls = type(request)
    fields = getattr(req_cls, "model_fields", {}) or {}
    if "logprobs" not in fields:
        return None
    partial = content.strip() + "\nAnswer: ("
    messages = list(getattr(request, "messages", None) or [])
    messages.append({"role": "assistant", "content": partial})
    kwargs: dict[str, Any] = {
        "model": getattr(request, "model", None),
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {
            "enable_thinking": False,
            _MARKER_KEY: True,
            _PN123_MARKER: True,
        },
    }
    cap_field = (
        "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
    )
    kwargs[cap_field] = 1
    for fname, val in (("continue_final_message", True), ("add_generation_prompt", False)):
        if fname in fields:
            kwargs[fname] = val
        else:
            kwargs["chat_template_kwargs"][fname] = val
    synthetic = req_cls(**kwargs)
    timeout = _env_int(_cg("MARGIN_TIMEOUT_S"), 30)
    resp = await asyncio.wait_for(
        serving.create_chat_completion(synthetic, raw_request=None), timeout
    )
    choice = _extract_choice(resp)
    if choice is None:
        return None
    return _letter_posterior_margin(choice)


# ─── PN123 gate mode: engine-side c_mean bridge (2026-07-23) ─────────────────
# The letter-margin echo call (above) is a post-close OUTPUT signal; every
# post-close output signal measured AUC ~0.5. The ONLY signal that discriminates
# premature-vs-settled is pn112's per-step sampling confidence C — but that is
# computed in the EngineCore process. pn112_export.py drops each request's
# rolling C mean into /tmp/genesis_pn112_conf.json; this reads it back and uses
# it as an alternative close-gate. GENESIS_PN123_GATE = margin|cmean|both.
#
# Join key: the engine keys the file by its InputBatch req_id — vLLM's request
# id string, form "chatcmpl-<uuid>". The serving layer sees that identical
# string as the ChatCompletionResponse.id (result.id). For n>1 sampling vLLM
# may append a per-sequence "-<k>" suffix engine-side; _normalize_req_id strips
# it so both forms join. (request.request_id is NOT reliably the chatcmpl id and
# falls back to id(request); result.id is authoritative.)
_PN112_CONF_PATH = "/tmp/genesis_pn112_conf.json"
_PARALLEL_SUFFIX_RE = re.compile(r"-\d+$")


def _normalize_req_id(rid: Any) -> str:
    """Strip vLLM's per-sequence parallel-sample suffix ('-<k>') so the engine
    req_id and the serving-side result.id join. The chatcmpl uuid itself carries
    no trailing '-<digits>' (uuid4 hex), so this only removes the sample index."""
    return _PARALLEL_SUFFIX_RE.sub("", str(rid))


def _pn123_join_id(request: Any, result: Any) -> Any:
    """The id to look up in the conf file: the ChatCompletionResponse.id, which
    equals the engine's InputBatch req_id. request.request_id is a defensive
    fallback (it is usually absent → id(request), which will simply miss)."""
    return getattr(result, "id", None) or getattr(request, "request_id", None)


def _pn112_conf_lookup(join_id: Any) -> dict[str, Any] | None:
    """Read this request's exported rolling conf entry. Fail-open: a missing
    file / torn read / malformed json all return None (→ conservative no-fire)."""
    if join_id is None:
        return None
    try:
        with open(_PN112_CONF_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    key = _normalize_req_id(join_id)
    entry = data.get(str(join_id))
    if entry is None:
        entry = data.get(key)
    if entry is None:
        for k, v in data.items():  # defensive: stored key may carry '-<k>'
            if _normalize_req_id(k) == key:
                entry = v
                break
    if entry is None:
        # [2026-07-23 live canary] the engine appends an 8-hex internal suffix
        # ("chatcmpl-<uuid>-<8hex>") that is NOT the integer sample index —
        # prefix-match against the serving id (file holds ≤128 keys).
        pref = key + "-"
        for k, v in data.items():
            if k.startswith(pref) or _normalize_req_id(k).startswith(pref):
                entry = v
                break
    return entry if isinstance(entry, dict) else None


def _pn123_cmean_decision(entry: dict[str, Any] | None, req_id: Any,
                          join_id: Any) -> tuple[bool, float | None]:
    """Truth table for the c_mean gate. Fire iff entry exists AND n >= MINN AND
    NOT stale (ts within TTL) AND c_last < CMEAN. Every miss → (False, c_last).
    Conservative by construction: missing/stale/low-n never fires."""
    if not entry:
        log.info("PN123: cmean skip req=%s — no conf entry (join=%s)", req_id, join_id)
        return False, None
    # [2026-07-23 field fix] flush-on-close made c_last = the COMMITMENT-moment
    # window, which is high for everyone (the model commits confidently; the
    # 9-vs-11 discrimination lives mid-trace). Default gate field is therefore
    # c_trace (whole-trace mean, flush-proof); c_last remains selectable.
    field = os.environ.get(_cg("CMEAN_FIELD"), "c_trace").strip() or "c_trace"
    c_last = entry.get(field, entry.get("c_last"))
    n = entry.get("n")
    ts = entry.get("ts")
    if not isinstance(c_last, (int, float)) or not isinstance(n, int):
        log.info("PN123: cmean skip req=%s — malformed entry %r", req_id, entry)
        return False, c_last if isinstance(c_last, (int, float)) else None
    if not isinstance(ts, (int, float)):
        log.info("PN123: cmean skip req=%s — entry has no ts (conservative)", req_id)
        return False, c_last
    ttl = _env_float(_cg("CMEAN_TTL_S"), 600.0)
    age = time.monotonic() - ts
    if age > ttl:
        log.info("PN123: cmean skip req=%s — stale (age=%.1fs > %.1fs)", req_id, age, ttl)
        return False, c_last
    minn = _env_int(_cg("CMEAN_MINN"), 64)
    if n < minn:
        log.info("PN123: cmean skip req=%s — n=%d < MINN=%d", req_id, n, minn)
        return False, c_last
    thr = _env_float(_cg("CMEAN"), 10.0)
    if c_last >= thr:
        log.info("PN123: cmean skip req=%s — confident (c_last=%.3f >= %.3f)",
                 req_id, c_last, thr)
        return False, c_last
    return True, c_last


async def _pn123_continue(serving: Any, request: Any, result: Any, message: Any,
                          choice: Any, reasoning: str, spent: int, budget: int,
                          req_id: Any, signal: float) -> bool:
    """The action leg: ONE continuation resuming inside the think region with the
    first-person cue and the leftover budget. Clones PN101 escalate's resume-
    inside-think construction (unclosed <think>, continue_final_message). Any
    failure returns False and leaves the original response untouched."""
    try:
        req_cls = type(request)
        fields = getattr(req_cls, "model_fields", {}) or {}
        cue = os.environ.get(_cg("CUE"), "").strip() or _PN123_DEFAULT_CUE
        min_c = _env_int(_cg("MIN_CONT"), 512)
        max_c = _env_int(_cg("MAX_CONT"), 6144)
        # [REVIEW C1 2026-07-23] the prefilled reasoning is RE-CHARGED against
        # the continuation's budget by the engine (same double-subtraction the
        # PN101 escalate leg fixed, ultra-review #4): pass spent + room as the
        # TOTAL so the continuation actually has `room` tokens of new thinking.
        room = max(min_c, min(max_c, budget - spent))
        cont_budget = spent + room
        # Resume INSIDE the think region: original think content WITHOUT
        # </think> + the own-voice cue, ending mid-flow so the model continues
        # reasoning (never anything resembling </think>; s12-P17: first-person
        # self-talk is internalized as own-voice, imperatives/brackets derail).
        partial = "<think>\n" + reasoning.strip() + "\n" + cue
        messages = list(getattr(request, "messages", None) or [])
        messages.append({"role": "assistant", "content": partial})
        kwargs: dict[str, Any] = {
            "model": getattr(request, "model", None),
            "messages": messages,
            "temperature": getattr(request, "temperature", 0.0) or 0.0,
            "stream": False,
            "thinking_token_budget": cont_budget,
            "chat_template_kwargs": dict(
                getattr(request, "chat_template_kwargs", None) or {},
                **{_MARKER_KEY: True, _PN123_MARKER: True},
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
        kwargs[cap_field] = max(original_cap or 0, cont_budget + 512)
        for fname, val in (("continue_final_message", True), ("add_generation_prompt", False)):
            if fname in fields:
                kwargs[fname] = val
            else:
                kwargs["chat_template_kwargs"][fname] = val
        synthetic = req_cls(**kwargs)
        timeout = _env_int(_cg("TIMEOUT_S"), 180)
        resp = await asyncio.wait_for(
            serving.create_chat_completion(synthetic, raw_request=None), timeout
        )
        rchoice = _extract_choice(resp)
        rmsg = getattr(rchoice, "message", None) if rchoice else None
        if rmsg is None:
            log.info("PN123: continuation returned no choice req=%s — original kept", req_id)
            return False
        new_content = (getattr(rmsg, "content", None) or "").strip()
        new_reasoning = _read_reasoning(rmsg)
        if not new_content and not new_reasoning.strip():
            log.info("PN123: continuation returned empty req=%s — original kept", req_id)
            return False
        # Fold the continuation's reasoning onto the original think content.
        for attr in ("reasoning", "reasoning_content"):
            if getattr(rmsg, attr, None) is not None:
                try:
                    setattr(message, attr, (reasoning + new_reasoning))
                except Exception:  # pragma: no cover - frozen models
                    pass
                break
        if not new_content:
            # Continuation itself ended still inside think — no answer to splice.
            log.info("PN123: continuation ended in-think req=%s — original kept", req_id)
            return False
        message.content = new_content
        try:
            choice.finish_reason = getattr(rchoice, "finish_reason", None) or "stop"
        except Exception:  # pragma: no cover
            pass
        _sum_usage(getattr(result, "usage", None), getattr(resp, "usage", None))
        _STATS["pn118_fires"] += 1
        log.info(
            "PN123: FIRED req=%s spent=%d budget=%d signal=%.3f +cont_budget=%d",
            req_id, spent, budget, signal, cont_budget,
        )
        return True
    except Exception as exc:
        _STATS["pn118_errors"] += 1
        log.warning("PN123: continuation failed req=%s (%s) — original kept", req_id, exc)
        return False


# ─── PN123 action: adjudicated rerun (Leg 3b, Fable R2) ──────────────────────
# GENESIS_PN123_ACTION = continue (default, current behaviour) | rerun.
# rerun does ONE fresh solve of the ORIGINAL request under the v5-shape banner
# (no poisoned-trace inheritance) and keeps the original answer UNLESS the rerun
# disagrees AND is the more confident trace (its own c_last, exported by its
# engine pass via the flush hook). Strictly dominates plain rerun: corruption
# now needs disagree AND rerun-wrong AND rerun-more-confident. Fail-open at
# every step to the original response.

_PN123_ANS_RE = re.compile(
    r"(?:answer|option|choice)\b[^A-Za-z0-9]{0,6}(?:is|:|=|\bis\b)?[^A-Za-z0-9(]{0,4}"
    r"\(?\s*([A-J])\b",
    re.IGNORECASE,
)


def _pn123_answer_key(content: str) -> str:
    """Normalized answer for agreement comparison. Prefer the LAST letter-answer
    (MCQ final commit); else the whitespace-normalized lowercased content."""
    if not content:
        return ""
    last = None
    for last in _PN123_ANS_RE.finditer(content):
        pass
    if last is not None:
        return last.group(1).upper()
    return " ".join(content.lower().split())


_pn118_answer_key = _pn123_answer_key  # legacy alias (test_pn118_logic.py)


async def _pn123_conf_wait(join_id: Any) -> dict[str, Any] | None:
    """Look up the rerun's exported conf entry, retrying ≤ WAIT_S for the
    engine-side flush to land (the rerun's </think> flush races this read)."""
    wait_s = _env_float(_cg("RERUN_CONF_WAIT_S"), 2.0)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        entry = _pn112_conf_lookup(join_id)
        if entry and isinstance(entry.get("c_last"), (int, float)):
            return entry
        if time.monotonic() >= deadline:
            return entry
        await asyncio.sleep(0.2)


async def _pn123_rerun(serving: Any, request: Any, result: Any, message: Any,
                       choice: Any, content: str, spent: int, budget: int,
                       req_id: Any, signal: float,
                       c_last_orig: float | None) -> bool:
    """Adjudicated rerun action. ONE fresh v5-shape solve of the original
    request; keep the original answer unless the rerun disagrees and its trace
    is at least as confident. Splice (content + usage fold) on swap; never
    expose two answers. Any failure returns False (original kept)."""
    try:
        req_cls = type(request)
        fields = getattr(req_cls, "model_fields", {}) or {}
        rerun_budget = _env_int(_cg("RERUN_BUDGET"), 10240)
        # Fresh solve: ORIGINAL messages (no <think> prefill), v5 banner forced
        # for this call only. Strip any banner the first pass injected so
        # maybe_add_answer_hint re-applies v5 (it early-returns on a present
        # pn_env_banner). Markers suppress PN123/PN101 re-entry.
        base_ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
        base_ctk.pop("pn_env_banner", None)
        base_ctk.pop("pn_env_seed", None)
        # [BUG-168] the SENTINEL, not True: `pn102_force_v5` is the one key that
        # bypasses _skip_common, and the bypass is now provenance-gated. The
        # markers below would authorise it too (back-compat path), but writing
        # the sentinel is what makes this call site self-evidently internal.
        base_ctk[_PN102_FORCE_V5_KEY] = _PN102_FORCE_V5_SENTINEL
        base_ctk[_MARKER_KEY] = True
        base_ctk[_PN123_MARKER] = True
        messages = list(getattr(request, "messages", None) or [])
        kwargs: dict[str, Any] = {
            "model": getattr(request, "model", None),
            "messages": messages,
            "temperature": getattr(request, "temperature", 0.0) or 0.0,
            "stream": False,
            "thinking_token_budget": rerun_budget,
            "chat_template_kwargs": base_ctk,
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
        kwargs[cap_field] = max(original_cap or 0, rerun_budget + 512)
        synthetic = req_cls(**kwargs)
        timeout = _env_int(_cg("TIMEOUT_S"), 180)
        resp = await asyncio.wait_for(
            serving.create_chat_completion(synthetic, raw_request=None), timeout
        )
        rchoice = _extract_choice(resp)
        rmsg = getattr(rchoice, "message", None) if rchoice else None
        if rmsg is None:
            log.info("PN123: rerun returned no choice req=%s — original kept", req_id)
            return False
        new_content = (getattr(rmsg, "content", None) or "").strip()
        if not new_content:
            log.info("PN123: rerun produced no answer req=%s — original kept", req_id)
            return False

        # ADJUDICATION. Agree → keep original (cheap, done).
        if _pn123_answer_key(content) == _pn123_answer_key(new_content):
            _STATS["pn118_rerun_agree"] = _STATS.get("pn118_rerun_agree", 0) + 1
            log.info("PN123: rerun AGREES req=%s — original kept (no swap)", req_id)
            return False

        # [2026-07-23 R1 fold] On GPQA-class MCQ, confidence-comparing two
        # DISAGREEING answers is a measured coin flip (arXiv 2607.17531:
        # 10/10 split; every training-free selector net-negative). Our case
        # is asymmetric — the rerun is the STRONGER config on a wrong-enriched
        # pool — so the default rule is rerun-wins-on-disagree. The
        # confidence-compare survives behind GENESIS_PN123_ADJUDICATE=confidence.
        if os.environ.get(_cg("ADJUDICATE"), "rerun_wins").strip() \
                != "confidence":
            log.info("PN123: rerun DISAGREES req=%s — rerun wins (asymmetric "
                     "escalation, R1 rule)", req_id)
            for attr in ("reasoning", "reasoning_content"):
                if getattr(rmsg, attr, None) is not None:
                    try:
                        setattr(message, attr, _read_reasoning(rmsg))
                    except Exception:  # pragma: no cover - frozen models
                        pass
                    break
            message.content = new_content
            try:
                choice.finish_reason = getattr(rchoice, "finish_reason", None) or "stop"
            except Exception:  # pragma: no cover
                pass
            _sum_usage(getattr(result, "usage", None), getattr(resp, "usage", None))
            _STATS["pn118_rerun_swap"] = _STATS.get("pn118_rerun_swap", 0) + 1
            return True

        # Disagree → prefer the rerun only if its trace is >= as confident.
        rerun_join = _pn123_join_id(synthetic, resp)
        rerun_entry = await _pn123_conf_wait(rerun_join)
        c_last_rerun = (rerun_entry or {}).get("c_last")
        if c_last_orig is None:
            orig_entry = _pn112_conf_lookup(_pn123_join_id(request, result))
            c_last_orig = (orig_entry or {}).get("c_last")
        if not isinstance(c_last_rerun, (int, float)) or \
                not isinstance(c_last_orig, (int, float)):
            _STATS["pn118_rerun_confmiss"] = _STATS.get("pn118_rerun_confmiss", 0) + 1
            log.info("PN123: rerun DISAGREES req=%s but conf lookup missed "
                     "(orig=%s rerun=%s) — original kept",
                     req_id, c_last_orig, c_last_rerun)
            return False
        if c_last_rerun < c_last_orig:
            _STATS["pn118_rerun_keep"] = _STATS.get("pn118_rerun_keep", 0) + 1
            log.info("PN123: rerun DISAGREES req=%s but less confident "
                     "(c_last rerun=%.3f < orig=%.3f) — original kept",
                     req_id, c_last_rerun, c_last_orig)
            return False

        # Swap: rerun disagrees AND is at least as confident.
        for attr in ("reasoning", "reasoning_content"):
            if getattr(rmsg, attr, None) is not None:
                try:
                    setattr(message, attr, _read_reasoning(rmsg))
                except Exception:  # pragma: no cover - frozen models
                    pass
                break
        message.content = new_content
        try:
            choice.finish_reason = getattr(rchoice, "finish_reason", None) or "stop"
        except Exception:  # pragma: no cover
            pass
        _sum_usage(getattr(result, "usage", None), getattr(resp, "usage", None))
        _STATS["pn118_fires"] += 1
        _STATS["pn118_rerun_swap"] = _STATS.get("pn118_rerun_swap", 0) + 1
        log.info("PN123: rerun SWAP req=%s — rerun more confident "
                 "(c_last rerun=%.3f >= orig=%.3f, budget=%d)",
                 req_id, c_last_rerun, c_last_orig, rerun_budget)
        return True
    except Exception as exc:
        _STATS["pn118_errors"] += 1
        log.warning("PN123: rerun failed req=%s (%s) — original kept", req_id, exc)
        return False


async def _maybe_pn123_closegate(serving: Any, request: Any, result: Any) -> bool:
    """PN123 premature-close gate. Returns True iff a continuation was spliced
    in (enforce mode). Shadow mode logs the would-fire and returns False."""
    if _skip_common(request) or not _bounded(request):
        return False
    choice = _extract_choice(result)
    if choice is None:
        return False
    # One fire per request: mark the choice so a re-invocation short-circuits.
    if getattr(choice, "_pn123_seen", False):
        return False
    try:
        setattr(choice, "_pn123_seen", True)
    except Exception:  # pragma: no cover - frozen models
        pass
    message = getattr(choice, "message", None)
    if message is None:
        return False
    content = (getattr(message, "content", None) or "").strip()
    reasoning = _read_reasoning(message)
    fr = getattr(choice, "finish_reason", None)
    budget = getattr(request, "thinking_token_budget", 0)
    req_id = getattr(request, "request_id", None) or id(request)

    # (1) voluntary close = an answer arrived after a real think block.
    if not content or not reasoning.strip():
        return False
    if fr != "stop" or getattr(message, "tool_calls", None):
        return False
    spent = _reasoning_tokens(result, reasoning)
    # (2) NOT a force/cap-bound close — a cap-bound close has no leftover budget.
    grace = _env_int(_cg("GRACE"), 256)
    if spent >= budget - grace:
        log.info("PN123: skip req=%s cap-bound close (spent=%d budget=%d grace=%d)",
                 req_id, spent, budget, grace)
        _STATS["pn118_skips"] += 1
        return False
    # (3) premature — spent well under the assigned budget.
    frac = _env_float(_cg("FRAC"), 0.6)
    if spent >= frac * budget:
        log.info("PN123: skip req=%s not premature (spent=%d >= %.2f*%d)",
                 req_id, spent, frac, budget)
        _STATS["pn118_skips"] += 1
        return False
    # (4) WEAK-confidence gate. GENESIS_PN123_GATE selects the discriminator:
    #   margin (default) — the post-close letter-margin echo self-call (current)
    #   cmean            — pn112's engine-side rolling confidence via the /tmp
    #                      bridge; NO echo call. The only signal with AUC > 0.5.
    #   both             — cmean AND margin must both pass.
    gate = (os.environ.get(_cg("GATE"), "margin") or "margin").strip().lower()
    if gate not in ("margin", "cmean", "both"):
        gate = "margin"

    c_last: float | None = None
    if gate in ("cmean", "both"):
        join_id = _pn123_join_id(request, result)
        entry = _pn112_conf_lookup(join_id)
        cmean_ok, c_last = _pn123_cmean_decision(entry, req_id, join_id)
        # shadow-log the looked-up c_last either way (calibration visibility).
        log.info("PN123: cmean lookup req=%s join=%s c_last=%s n=%s pass=%s",
                 req_id, join_id,
                 ("%.3f" % c_last) if c_last is not None else "na",
                 (entry or {}).get("n", "na"), cmean_ok)
        if not cmean_ok:
            _STATS["pn118_skips"] += 1
            return False

    margin: float | None = None
    if gate in ("margin", "both"):
        try:
            margin = await _letter_margin(serving, request, content)
        except Exception as exc:
            log.warning("PN123: margin read failed req=%s (%s) — skip (fail-open)",
                        req_id, exc)
            _STATS["pn118_errors"] += 1
            return False
        if margin is None:
            log.info("PN123: skip req=%s no letter margin (open-ended answer)", req_id)
            _STATS["pn118_skips"] += 1
            return False
        margin_thr = _env_float(_cg("MARGIN"), 0.5)
        if margin >= margin_thr:
            log.info("PN123: skip req=%s confident (margin=%.3f >= %.3f)",
                     req_id, margin, margin_thr)
            _STATS["pn118_skips"] += 1
            return False

    # the fire signal logged / passed downstream: margin when read, else c_last.
    signal = margin if margin is not None else (c_last if c_last is not None else -1.0)

    mode = (os.environ.get(_cg("MODE"), "shadow") or "shadow").strip().lower()
    if mode != "enforce":
        _STATS["pn118_shadow_would_fire"] += 1
        log.info("PN123: WOULD-FIRE (shadow) req=%s gate=%s spent=%d budget=%d signal=%.3f",
                 req_id, gate, spent, budget, signal)
        return False

    _STATS["pn118_attempts"] += 1
    action = (os.environ.get(_cg("ACTION"), "continue")
              or "continue").strip().lower()
    if action == "rerun":
        return await _pn123_rerun(serving, request, result, message, choice,
                                  content, spent, budget, req_id, signal, c_last)
    return await _pn123_continue(serving, request, result, message, choice,
                                 reasoning, spent, budget, req_id, signal)


# ─── Leg 5: PN155 budget-truth guard (BUG-155) ───────────────────────────────
# Under a JSON grammar the cheapest completion the parser accepts is the empty
# container — `{"facts": []}`, ~8 answer tokens. A request whose thinking budget
# expires is force-closed by the holder and then takes that exit, and the caller
# receives a well-formed, schema-valid, `finish_reason="stop"` response carrying
# no data. Measured 2026-07-26 on prod_mixed_v3/guided: `prod-016` returned 0
# facts after rtok=3899 against its 3900 grant (atok=8) on a chunk the unguided
# arm mined 9 facts from. 25/40 guided rows sit exactly on a PN100 ceiling
# (2098-2100 x14, 3099 x3, 3899 x8); the unguided control pins 2/40.
#
# NOTHING ELSE IN THIS MODULE CAN SEE IT, by construction:
#   * `_skip_common()` returns True for `_has_structured_output(request)`, and
#     every PN101 leg is gated behind `not _skip_common(request)` — the one
#     component that exists to rescue empty answers excludes exactly the
#     requests this hits;
#   * the PN101 guillotine path needs `finish_reason == "length"` and these rows
#     finish `"stop"` — the grammar closed the JSON legally.
# So this is a NEW leg that runs ON the requests the existing gates skip, with
# its own master flag, independent of PN101's.
#
# THE FIX IS ONE LINE OF SEMANTICS: report `finish_reason="length"`. That is the
# truthful reason (the response ended because it ran out of budget, not because
# the model was done), every OpenAI-compatible client already treats it as
# incomplete, and no caller has to learn a new field to stop trusting the empty
# array. The original is preserved on the choice, never destroyed.
#
# NOT DONE, deliberately (spec §3c): the grammar is NOT changed to reject `[]`
# — an empty array is legal in the caller's schema and legitimately occurs
# (prod-038's chunk is literally `<local-command-stdout>Bye!</local-command-
# stdout>`, empty in three separate runs); rejecting it would convert a
# detectable failure into a WRONG answer. And the forced close is not suppressed
# — PN122's graft already keeps the forced `</think>` out of the constrained
# region, which fixes malformed JSON, not this; a request that reaches its cap
# still has to end somewhere and the grammar still offers the cheap exit.
#
# Env (all inert while GENESIS_ENABLE_PN155_BUDGET_TRUTH is unset):
#   GENESIS_ENABLE_PN155_BUDGET_TRUTH   master, DEFAULT OFF (house rule:
#                                       behavioural patches never default on)
#   GENESIS_PN155_MODE                  observe | flag | retry (default observe;
#                                       `flag` is the ship target — an unflagged
#                                       empty IS the bug)
#   GENESIS_PN155_STAMP_BUDGET          default 1 — pure addition, see below
#   GENESIS_PN155_MARGIN                default 16 (see _pn155_spend)
#   GENESIS_PN155_RETRY_MULT            default 2
#   GENESIS_PN155_RETRY_CEIL            default = GENESIS_PN101_TOTAL_THINK_CEIL
#   GENESIS_PN155_TIMEOUT_S             default 180

_PN155_MARKER_KEY = "pn155_internal"
# usage/choice field names. `thinking_token_budget` is one of the four names the
# qbench45 client already probes (bench/client.py:OpenAIArm._granted_budget) at
# top level, on the choice, in usage and in usage.completion_tokens_details — so
# stamping it upgrades the harness guard from an INFERRED cap to an exact one
# with no client change.
_PN155_BUDGET_FIELD = "thinking_token_budget"
_PN155_EMPTY_FIELD = "budget_empty"
_PN155_FORCED_FIELD = "budget_forced"
_PN155_ORIG_FR_FIELD = "genesis_finish_reason_original"
# Answer-token estimate for the BUG-158 fallback in _pn155_spend. 4 chars/token
# is the same constant PN101 uses for its reasoning estimate.
_PN155_CHARS_PER_TOKEN = 4


def _pn155_master_on() -> bool:
    return _env_bool("GENESIS_ENABLE_PN155_BUDGET_TRUTH")


def _pn155_mode() -> str:
    mode = (os.environ.get("GENESIS_PN155_MODE", "") or "observe").strip().lower()
    return mode if mode in ("observe", "flag", "retry") else "observe"


def _pn155_is_empty(content: Any) -> bool:
    """The grammar's cheapest legal completion.

    Schema-free on purpose: any container with zero entries counts, and anything
    unparseable does NOT — that is a different failure and it is already visible
    to the caller as broken JSON.
    """
    if not isinstance(content, str):
        return False
    try:
        obj = json.loads(content.strip())
    except Exception:
        return False
    if isinstance(obj, list):
        return len(obj) == 0
    if isinstance(obj, dict):
        if not obj:
            return True
        return all(isinstance(v, (list, dict)) and not v for v in obj.values())
    return False


def _pn155_budget(request: Any, result: Any) -> int | None:
    """The thinking budget this request was actually GRANTED, or None.

    `request.thinking_token_budget` is PN100's grant (auto_budget `_apply_budget`
    / `_apply_tier`) and this pin's chat_completion/protocol.py carries it into
    SamplingParams natively, so it is the number the holder enforces — EXCEPT in
    one case. H119's budget consumer (fixes/pn119_router.py `_h119_route_budget`)
    MODULATES the caller's prior by H119_LEAN_MULT / H119_DEEP_MULT; with no
    prior at all it substitutes the flat H119_LEAN_BUDGET / H119_DEEP_BUDGET
    constants, and only then is the request-side value (absent) wrong rather than
    merely approximate. That flat case is recovered here from the route the API
    bridge publishes on the response. A non-1.0 multiplier makes the request-side
    number an approximation; the error is bounded by the ladder snap and biases
    this leg toward NOT firing, which is the safe direction.
    """
    budget = getattr(request, "thinking_token_budget", None)
    if isinstance(budget, int) and budget > 0:
        return budget
    if not _env_bool("GENESIS_ENABLE_H119_ROUTE_BUDGET"):
        return None
    kvt = getattr(result, "kv_transfer_params", None)
    h119 = kvt.get("h119") if isinstance(kvt, dict) else None
    if not isinstance(h119, dict) or h119.get("mode") != "enforce":
        return None
    route = h119.get("route")
    if route == "deep":
        return _env_int("H119_DEEP_BUDGET", 10240)
    if route == "lean":
        return _env_int("H119_LEAN_BUDGET", 1600)
    return None


def _pn155_spend(result: Any, message: Any, content: str) -> tuple[int, str]:
    """(thinking tokens spent, where the number came from).

    `usage.completion_tokens_details.reasoning_tokens` is authoritative when the
    accounting path populates it. On this deployment it does not: BUG-158 —
    the chat template opens `<think>` in PROMPT space (chat_template.jinja:147),
    so the usage path never finds an opener in the output and reports
    `reasoning_tokens: 0` / `reasoning_content: null` for every request,
    verified live on :8021 (879 completion tokens against a ~290-char answer,
    reasoning_tokens 0). Keying condition 3 on it alone would make this leg
    UNFIREABLE on the exact endpoint the bug was measured on.

    So the fallback is `completion_tokens` minus the answer's own length. It is
    an estimate, but only its DIFFERENCE from the budget is used, the answer it
    subtracts is by construction tiny (the empty container is ~8 tokens), and
    over-estimating the spend errs toward firing on a row that is already empty
    — never toward touching a row that carried data.
    """
    usage = getattr(result, "usage", None)
    det = getattr(usage, "completion_tokens_details", None) if usage else None
    rt = getattr(det, "reasoning_tokens", None) if det is not None else None
    if isinstance(rt, int) and rt > 0:
        return rt, "usage"
    reasoning = _read_reasoning(message) if message is not None else ""
    if reasoning.strip():
        return len(reasoning) // _PN155_CHARS_PER_TOKEN, "reasoning_text"
    ctok = getattr(usage, "completion_tokens", None) if usage else None
    if not isinstance(ctok, int) or ctok <= 0:
        return 0, "unavailable"
    answer_tokens = len(content or "") // _PN155_CHARS_PER_TOKEN
    return max(0, ctok - answer_tokens), "derived"


def _pn155_forced(result: Any) -> bool | None:
    """The holder's own "I forced `</think>` on this request" bit, if anyone has
    published it to the API process yet. None = not available.

    The holder (`v1/sample/thinking_budget_state.py`) knows this exactly, and
    fixes/pn119_router.py already latches it engine-side as `censor_forced`. It
    is NOT readable here: the holder lives in the `VLLM::EngineCore` process and
    this middleware runs in the API server process, so publishing the bit means
    a new field on EngineCoreOutput — the same protocol hop the H119 route rides
    via `kv_transfer_params`. That is a separate patch. This reader exists so
    that the day it lands, the exact signal replaces the threshold below with no
    change to this leg.
    """
    usage = getattr(result, "usage", None)
    det = getattr(usage, "completion_tokens_details", None) if usage else None
    for holder in (det, usage):
        if holder is None:
            continue
        for name in (_PN155_FORCED_FIELD, "censor_forced"):
            val = getattr(holder, name, None)
            if isinstance(val, bool):
                return val
    kvt = getattr(result, "kv_transfer_params", None)
    h119 = kvt.get("h119") if isinstance(kvt, dict) else None
    if isinstance(h119, dict) and isinstance(h119.get("censor_forced"), bool):
        return h119["censor_forced"]
    return None


def _pn155_stamp(result: Any, **fields: Any) -> bool:
    """Additive stamps on the usage block. OpenAI clients ignore unknown keys.

    `CompletionTokenUsageInfo`/`UsageInfo` are `OpenAIBaseModel`, i.e. pydantic
    with `extra="allow"`, so an attribute set here lands in `__pydantic_extra__`
    and IS serialized (verified against the live pin's protocol module). Falls
    back to the usage object when the details block is absent — the qbench45
    client probes both locations.
    """
    usage = getattr(result, "usage", None)
    if usage is None:
        return False
    det = getattr(usage, "completion_tokens_details", None)
    target = det if det is not None else usage
    written = False
    for key, val in fields.items():
        try:
            setattr(target, key, val)
            written = True
        except Exception as exc:  # pragma: no cover - frozen models
            log.warning("PN155: could not stamp %s (%s)", key, exc)
    return written


async def _pn155_retry(serving: Any, request: Any, result: Any, message: Any,
                       choice: Any, budget: int) -> bool:
    """One bounded re-generate with more thinking room. True iff it delivered a
    payload we are willing to SERVE (which then replaces the empty one).

    NOT `_maybe_escalate`: that leg continues an UNCLOSED think block from the
    reasoning text, and a BUG-155 row has neither (its think block closed, and
    `reasoning_content` is null here anyway — BUG-158). This is a fresh
    generation of the same request, carrying the caller's structured-output
    fields so the retry is answering the same question under the same grammar.
    Never a second retry: the synthetic carries `_PN155_MARKER_KEY`.

    [BUG-167 2026-07-27] ACCEPTANCE is not "non-empty". `_pn155_is_empty`
    returns False for anything that does not parse — deliberate, and correct
    for what it is FOR (deciding whether the model produced the grammar's
    cheapest legal completion), but it is not a validity check. Reused as one,
    it accepted a grammar-constrained retry that hit its cap mid-schema: a
    legal PREFIX of the schema, non-blank, unparseable, served in place of the
    well-formed `{"facts": []}` this leg exists to flag — and served under the
    FIRST pass's `finish_reason="stop"`, because the retry's own reason was
    never read. Strictly worse than the row it replaced, and invisible.

    Two gates close it, both cheap because we are already inside the guided
    path: the payload must PARSE (`json.loads`) when the request was
    structured, and the retry's OWN `finish_reason` is propagated onto the
    served choice. A payload that fails the parse gate is NOT served: this
    returns False and the caller applies the normal PN155 flag semantics to the
    ORIGINAL response (`finish_reason="length"` +
    `genesis_finish_reason_original`), so the failure stays visible.
    """
    ceil = _env_int("GENESIS_PN155_RETRY_CEIL",
                    _env_int("GENESIS_PN101_TOTAL_THINK_CEIL", 10240))
    mult = _env_float("GENESIS_PN155_RETRY_MULT", 2.0)
    new_budget = min(ceil, int(budget * mult))
    if new_budget <= budget:
        log.info("PN155: retry skipped — budget %d already at ceiling %d",
                 budget, ceil)
        return False

    _STATS["pn155_retries"] += 1
    req_cls = type(request)
    fields = getattr(req_cls, "model_fields", {}) or {}
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    ctk["enable_thinking"] = True
    ctk[_MARKER_KEY] = True
    ctk[_PN155_MARKER_KEY] = True
    kwargs: dict[str, Any] = {
        "model": getattr(request, "model", None),
        "messages": list(getattr(request, "messages", None) or []),
        "temperature": getattr(request, "temperature", 0.0) or 0.0,
        "stream": False,
        # An explicit positive budget makes PN100 take its "explicit ON intent"
        # skip branch, so our number survives the hook.
        "thinking_token_budget": new_budget,
        "chat_template_kwargs": ctk,
    }
    # The grammar is the whole point of the retry — losing it would compare an
    # unguided answer against a guided one.
    for sf in ("response_format", "guided_json", "guided_regex", "guided_grammar",
               "guided_choice", "guided_decoding_backend", "structured_outputs",
               "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
               "repetition_penalty", "stop", "seed", "logit_bias"):
        if sf in fields:
            sv = getattr(request, sf, None)
            if sv is not None:
                kwargs[sf] = sv
    cap_field = (
        "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
    )
    kwargs[cap_field] = max(_completion_cap(request) or 0, new_budget + 512)
    synthetic = req_cls(**kwargs)
    resp = await asyncio.wait_for(
        serving.create_chat_completion(synthetic, raw_request=None),
        _env_int("GENESIS_PN155_TIMEOUT_S", 180),
    )
    rchoice = _extract_choice(resp)
    rmsg = getattr(rchoice, "message", None) if rchoice is not None else None
    new_content = (getattr(rmsg, "content", None) or "") if rmsg else ""
    new_finish = getattr(rchoice, "finish_reason", None) if rchoice is not None else None
    if not new_content.strip() or _pn155_is_empty(new_content):
        log.info("PN155: retry at budget=%d empty again — falling through to flag",
                 new_budget)
        return False
    # [BUG-167] gate (a): under a grammar, "not empty" is not "valid". Run the
    # same parse the caller is about to run, one string earlier.
    if _has_structured_output(request):
        try:
            json.loads(new_content)
        except Exception as exc:
            _STATS["pn155_retry_unparseable"] += 1
            log.warning(
                "PN155: retry at budget=%d returned an UNPARSEABLE payload "
                "(%s; finish=%s, %d chars) — refusing to serve it, falling "
                "through to flag (BUG-167)",
                new_budget, exc.__class__.__name__, new_finish, len(new_content),
            )
            return False
    message.content = new_content
    # [BUG-167] gate (b): the served payload is the RETRY's, so it must carry
    # the retry's label. Keeping the original "stop" on a retry that ended at
    # `length` is exactly how a truncated retry looked complete. The original is
    # preserved on its own field, never destroyed — same contract as the flag
    # path below.
    if new_finish is not None and new_finish != getattr(choice, "finish_reason", None):
        try:
            prior = getattr(choice, "finish_reason", None)
            if not hasattr(choice, _PN155_ORIG_FR_FIELD):
                setattr(choice, _PN155_ORIG_FR_FIELD, prior)
            choice.finish_reason = new_finish
            log.warning(
                "PN155: retry finish_reason %s -> %s propagated onto the served "
                "choice (BUG-167)", prior, new_finish,
            )
        except Exception as exc:  # pragma: no cover - frozen models
            _STATS["pn155_errors"] += 1
            log.warning("PN155: could not propagate retry finish_reason (%s)", exc)
    _sum_usage(getattr(result, "usage", None), getattr(resp, "usage", None))
    _pn155_stamp(result, pn155_retry_budget=new_budget)
    _STATS["pn155_retry_rescued"] += 1
    log.info("PN155: retry at budget=%d recovered a servable payload "
             "(%d chars, finish=%s)", new_budget, len(new_content), new_finish)
    return True


async def _maybe_pn155_budget_truth(serving: Any, request: Any,
                                    result: Any) -> None:
    """Stamp the granted budget (3a) and guard the empty-at-cap exit (3b)."""
    ctk = getattr(request, "chat_template_kwargs", None) or {}
    if isinstance(ctk, dict) and ctk.get(_PN155_MARKER_KEY):
        return  # our own retry — never recurse

    budget = _pn155_budget(request, result)
    if budget is None:
        return
    _STATS["pn155_seen"] += 1

    # ── 3a: observability. Pure addition, so it runs for every budgeted
    # request, structured or not, and independently of the mode below. Today
    # the grant is reported NOWHERE on the response, which is why the harness
    # has to infer the cap from the PN100 grid.
    if _env_bool("GENESIS_PN155_STAMP_BUDGET", True):
        if _pn155_stamp(result, **{_PN155_BUDGET_FIELD: budget}):
            _STATS["pn155_stamped"] += 1

    # ── 3b: detection. All four conditions, in cost order.
    if not _has_structured_output(request):
        return
    choice = _extract_choice(result)
    message = getattr(choice, "message", None) if choice is not None else None
    if message is None:
        return
    content = getattr(message, "content", None)
    if not _pn155_is_empty(content):
        return
    forced = _pn155_forced(result)
    spend, spend_src = _pn155_spend(result, message, content or "")
    margin = _env_int("GENESIS_PN155_MARGIN", 16)
    if forced is False:
        return  # the holder says it did not force this one — believe it
    if forced is None and spend < budget - margin:
        return  # ended well short of its cap: a genuinely empty chunk
    if forced is None and spend_src == "unavailable":
        return  # no spend signal at all — refuse to guess

    _STATS["pn155_fired"] += 1
    mode = _pn155_mode()
    log.warning(
        "PN155: structured request emptied out at its thinking budget "
        "(budget=%d spend=%d[%s] forced=%s finish=%s mode=%s) — BUG-155",
        budget, spend, spend_src, forced,
        getattr(choice, "finish_reason", None), mode,
    )
    stamps: dict[str, Any] = {_PN155_EMPTY_FIELD: True}
    if forced is not None:  # absent means "nobody published it", not "False"
        stamps[_PN155_FORCED_FIELD] = forced
    _pn155_stamp(result, **stamps)

    if mode == "observe":
        return
    if mode == "retry":
        try:
            if await _pn155_retry(serving, request, result, message, choice,
                                  budget):
                _pn155_stamp(result, **{_PN155_EMPTY_FIELD: False})
                return
        except Exception as exc:
            _STATS["pn155_errors"] += 1
            log.warning("PN155: retry failed (%s) — falling through to flag", exc)

    # `flag` — and the terminal state of `retry`. `length` is the truthful
    # finish reason and the one every OpenAI-compatible client already reads as
    # "incomplete". The original is preserved, never destroyed: `stop_reason`
    # keeps its upstream meaning and the old value lands on its own field.
    original = getattr(choice, "finish_reason", None)
    try:
        setattr(choice, _PN155_ORIG_FR_FIELD, original)
        choice.finish_reason = "length"
    except Exception as exc:  # pragma: no cover - frozen models
        _STATS["pn155_errors"] += 1
        log.warning("PN155: could not rewrite finish_reason (%s) — original kept", exc)
        return
    _STATS["pn155_flagged"] += 1
    log.warning("PN155: finish_reason %s -> length (budget-enforced empty payload)",
                original)


async def maybe_rescue_answer(serving: Any, request: Any, result: Any) -> Any:
    # [BUG-157] This is the only seat in the module that is handed the serving
    # instance, so it is where Leg 1b's wrapper gets installed. Unconditional
    # and idempotent; it runs before every master-flag gate below because the
    # wrapper it installs has its OWN flag and must be armable without a
    # redeploy. Cannot help the request that installs it (that request is
    # already inside create_chat_completion) — every later one, yes.
    install_route_autosplit(serving)
    # PN123 premature-close gate: an independent leg in the same post-response
    # seat, its own master flag, unaffected by the PN101 master/repair toggles.
    if (_pn123_master_on() and not hasattr(result, "__aiter__")
            and not getattr(request, "stream", False)):
        try:
            await _maybe_pn123_closegate(serving, request, result)
        except Exception as exc:  # pragma: no cover - belt-and-suspenders fail-open
            _STATS["pn118_errors"] += 1
            log.warning("PN123: closegate failed (%s) — original kept", exc)
    # [BUG-156] Output-side banner-echo net. Non-streaming only (a streaming
    # splice would mean owning SSE framing) and independent of the PN101 master
    # flag: the banner it cleans up after is PN102's, not PN101's.
    # Placed AFTER the close gate (whose rerun/continue can replace `content`
    # wholesale) and BEFORE the PN101 legs (escalate only fires on EMPTY
    # content, and repair APPENDS to content this has already cleaned) — so
    # every path that can put text in the answer channel is covered exactly
    # once.
    # [BUG-168 2026-07-27] gated on GENESIS_ENABLE_PN102_CONTRACT — the master
    # that owns banner injection. The net is a cleanup pass for OUR banner; with
    # the injector dark there is nothing of ours to clean up, and running anyway
    # broke identity-when-dark (it was the one leg with no GENESIS_ENABLE_* of
    # its own). Belt and braces: `_injected_banner_marker` additionally requires
    # the internal injection marker, so even with the master ON a client that
    # sends its own `pn_env_banner` is not rewritten.
    if (_env_bool("GENESIS_ENABLE_PN102_CONTRACT")
            and not hasattr(result, "__aiter__")
            and not getattr(request, "stream", False)):
        try:
            _maybe_strip_banner_echo(request, result)
        except Exception as exc:  # pragma: no cover - fail-open
            log.warning("PN102: banner-echo net failed (%s) — original kept", exc)
    # [BUG-155] PN155 budget-truth guard. Own master flag, independent of the
    # PN101 master/repair toggles for the same reason the close gate is: it is
    # the ONLY leg that runs on the requests `_skip_common()` excludes, which is
    # exactly the structured population this bug lives in. Placed AFTER the
    # close gate and the banner-echo net (both may rewrite `content`, and the
    # emptiness test has to read the content the CALLER will get) and BEFORE the
    # `_master_on()` early return. Non-streaming only.
    if (_pn155_master_on() and not hasattr(result, "__aiter__")
            and not getattr(request, "stream", False)):
        try:
            await _maybe_pn155_budget_truth(serving, request, result)
        except Exception as exc:  # pragma: no cover - fail-open, always
            _STATS["pn155_errors"] += 1
            log.warning("PN155: guard failed (%s) — original kept", exc)
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
