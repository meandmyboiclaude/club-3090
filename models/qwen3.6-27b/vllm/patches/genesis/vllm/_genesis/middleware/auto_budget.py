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

EXPLICIT THINKING-OFF IS A HARD BYPASS (2026-07-26, USER directive)
-------------------------------------------------------------------
"prompts with reasoning / thinking off in request correctly bypasses all
thinking things while something that doesnt have that runs on thinking."

A request that ASKS for thinking off must exit every thinking layer, not just
this one. `_decide_mode` already returned "skip" for all five off-forms, so
PN100 itself never classified them and never budgeted them — but "skip" is
silence, and three layers downstream read silence as "nothing was decided":

  1. `chat_template_kwargs.thinking_budget: "off"` and `reasoning: off` left
     `enable_thinking` UNSET. The Qwen3.6 template defaults it ON, so the
     caller's off-intent never reached the render — the model thought, with
     no budget at all (the worst of both).
  2. `thinking_token_budget: 0` (the house-preferred off form — callers use
     `thinking_token_budget: N`, NEVER `reasoning_effort`) reached the worker
     as a REAL budget: ThinkingBudgetStateHolder built a state entry for it,
     PN108's plateau cap and the forced-`</think>` machinery attached to that
     entry, and the template still opened `<think>`.
  3. No `h119_overridable` stamp was written, so the H119 route consumer read
     the row as "caller said nothing" and installed a provisional DEEP budget
     at batch-add — measured on the live boot: 101 thinking-OFF requests got
     an H119 budget. The consumer acts in sync_batch; the router's own
     thinking-off tap (`_prompt_thinking`) only runs later, in the prefill
     postprocess, so it can never get there first.

`_explicit_thinking_off()` + `_apply_explicit_off()` close all three: one
short-circuit at the TOP of the hook that renders the off decision explicit
(`enable_thinking=False`), clears the zero budget so no holder entry is ever
built, and stamps `h119_overridable=0` so the worker keeps out. No classify
call is made (that is a whole extra chat completion — 0.3-0.6s typical, 2.7s
worst), no banner can attach (PN102 gates on both `enable_thinking is False`
and a positive budget), and no forced-close machinery has a state entry to
attach to.

REQUEST PARAMS ONLY. Like `_shape_tier0`, this never reads prompt TEXT — the
PN16-regex lesson. A request that does not carry one of the off forms runs on
thinking exactly as before; the ~95-98% tier-0/engage figure is a property of
requests that specified NOTHING (GENESIS_PN100_AUTO_DEFAULT=1 classifies
those), not of requests that asked for off.

Gate: GENESIS_THINKING_OFF_STRICT (DEFAULT ON — this enforces the caller's
own choice rather than overriding it; set 0 to restore the pre-2026-07-26
silence). Runs even when GENESIS_ENABLE_PN100_AUTO_BUDGET is 0: the bypass is
a request contract, not a budget policy, and the H119 stamp has to be written
whether or not PN100 is allocating this boot.
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
  GENESIS_PN100_SHAPE_TIER0          DEFAULT-DARK [2026-07-25, tier-0/engage
                                     defect]: request-SHAPE tier-0 rules that
                                     short-circuit the classify call — see
                                     _shape_tier0(). OFF (default) = classify
                                     path byte-identical to before.
  GENESIS_PN100_SHAPE_SMALL_MAX_TOKENS  small-completion-cap threshold used by
                                     the temp-0 and forced-tool rules (512)
  GENESIS_PN100_TOTAL_CEIL_F         DEFAULT-DARK (0.0 = off): total-completion
                                     ceiling multiplier. Bounds think+answer at
                                     max(F x budget, budget) + slack — see
                                     _apply_total_ceiling() for the 2026-07-26
                                     fit. Fitted ship value: 1.0.
  GENESIS_PN100_TOTAL_CEIL_SLACK     answer allowance for that ceiling (default
                                     1024, floor 256). Fitted ship value: 3072
                                     = prod_mixed_v2 natural-stop answer max
                                     2740 + margin. 1024 CLIPS REAL PROD
                                     ANSWERS (prod p95 is 1802).
  GENESIS_PN100_SHAPE_MAX_TEMP       "very low" temperature ceiling for the
                                     temp-0 rule (default 0.0 = exactly zero)

DO NOT wire the H119 lens-router here (settled 2026-07-25).
--------------------------------------------------------------
The tempting last mile — read `_genesis_pn119.ROUTES[req_id]` in _decide_mode
and pick a deep vs lean budget — cannot work, and fails SILENTLY (every request
reads as a miss, takes PN119_FALLBACK_ROUTE=deep, and you pay champion cost for
zero saving). This hook runs at the TOP of create_chat_completion: before the
prompt is rendered, before `request_id` exists, and in the API-SERVER process,
whereas ROUTES is a module-global in the out-of-process EngineCore/worker that
only writes it during prefill. Proof (source-level, no boot):
fixes/test_h119_route_consumer_timing.py. The viable site is worker-side —
vllm/v1/sample/thinking_budget_state.py — see the H119 router's docstring.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import time
from collections import OrderedDict
from typing import Any

from vllm._genesis.middleware.lazy_reasoner import _extract_text_from_message

try:  # vllm's logger prints INFO in-server; plain root logger may not
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.auto_budget")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.middleware.auto_budget")

# [PN162 2026-07-27] closed-loop budget calibrator (GENESIS_ENABLE_PN162_
# BUDGET_CAL, default OFF). Sibling module, loaded two ways because this file
# is imported BOTH as a package member in-container and by absolute path from
# the offline tests (which stub `vllm.*`, so the package import cannot resolve).
# A missing/broken calibrator leaves `_pn162 = None` -> every leg below is a
# no-op and PN100 behaves exactly as it does today.
try:
    from vllm._genesis.middleware import pn162_budget_cal as _pn162
except Exception:  # pragma: no cover — path fallback, see above
    try:
        import importlib.util as _ilu

        _p162 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pn162_budget_cal.py")
        _s162 = _ilu.spec_from_file_location("_genesis_pn162_budget_cal", _p162)
        _pn162 = _ilu.module_from_spec(_s162)
        _s162.loader.exec_module(_pn162)
    except Exception:
        _pn162 = None

_MARKER_KEY = "pn100_internal"
_CONTROL_KEY = "thinking_budget"
_LAST_CONF: dict = {"v": "h"}

_RUBRIC = (
    "You rate how much hidden reasoning an AI assistant needs to answer a "
    "request well. Reply with the tier digit, a pipe, and your estimate of "
    "how many short reasoning steps (each a few sentences) it needs — "
    "nothing else. Example replies: 0|0  1|4  2|12  3|30\n"
    "Tiers:\n"
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

# Depth-variant rubric (GENESIS_PN100_RUBRIC=depth): the rewindow bench showed
# the default rubric rates task CATEGORY, not per-item DEPTH — on uniformly-hard
# GPQA it sent 91% tier-2 / 10% tier-3 where the top tier earns +6pt. This
# variant drops the "1 in 40" frequency prior entirely and rates depth signals
# per item. Default rubric unchanged (prod traffic is mixed; rarity prior is
# right there) — flip only after the A/B leg validates.
_RUBRIC_DEPTH = (
    "You rate how much hidden reasoning an AI assistant needs to answer THIS "
    "specific request well. Reply with the tier digit, a pipe, and your "
    "estimate of how many short reasoning steps (each a few sentences) it "
    "needs — nothing else. Example replies: 0|0  1|4  2|12  3|30\n"
    "Tiers — judge by the DEPTH signals in the item itself:\n"
    "0 = none: greetings, formatting, copy/transform, single-fact lookup.\n"
    "1 = brief: one clean inference or lookup-plus-arithmetic; you can see "
    "the whole solution path immediately.\n"
    "2 = real reasoning: a clear multi-step path exists but needs care — "
    "standard methods, moderate case analysis, routine debugging.\n"
    "3 = deep: ANY of — multiple interacting constraints; a proof or "
    "derivation chain longer than a few steps; specialized domain knowledge "
    "combined with calculation; competing candidate answers that each need "
    "checking; you cannot see the full solution path from the question "
    "alone. If genuinely torn between 2 and 3, pick 3 — an unused budget "
    "costs nothing when the answer comes early.\n"
    "Judge the TASK's need, not the prompt's length. When unsure between "
    "0/1/2 pick 2."
)


def _env_float(name: str, default: float) -> float:
    # [2026-07-23] was referenced by the LOWCONF path inside a swallowed
    # try/except but never defined — the P-pess edit exposed it on the hot
    # path (NameError -> PN100 fail-open -> bare mode; killed run 1).
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


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
    "explicit_off": 0,
    "tiny_skip": 0,
    "shape_tier0": 0,
    "prefilter_default": 0,
    "cache_hits": 0,
    "fallbacks": 0,
    "errors": 0,
    "total_ceiling": 0,
}

# Tier cache: retries and template-identical callers skip the classify call.
# Keyed by sha256 of the flattened conversation; asyncio single-thread safe.
_TIER_CACHE: OrderedDict[str, int] = OrderedDict()
_TIER_CACHE_MAX = 256


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
    tools = getattr(request, "tools", None)
    if tools:
        # Tool orchestration is a reasoning signal the raw text may not carry.
        parts.append(f"[note] request includes {len(tools)} callable tool(s)")
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


# ── explicit request-level thinking-off (USER directive, 2026-07-26) ────────
# The four wire forms an "off" can arrive in. Everything else — including a
# missing flag — means "unspecified", which runs on thinking.
_OFF_WORDS = frozenset(("off", "none", "no", "false", "disable", "disabled",
                        "0"))


def _strict_off() -> bool:
    """DEFAULT ON. Honouring the caller's own off is not an override."""
    return _env_bool("GENESIS_THINKING_OFF_STRICT", True)


def _is_off_value(val: Any) -> bool:
    """True iff `val` is one of the recognised OFF spellings.

    Bools first: `False` is an int subtype and `True == 1`, so an unguarded
    numeric test would read `enable_thinking=True` as a zero-ish off (the same
    class of bug the 2026-07-23 ultra-review #10 found in `_decide_mode`).
    Also unwraps the OpenAI Responses-API object form `{"effort": ...}` that
    PN71 accepts on `reasoning`.
    """
    if isinstance(val, bool):
        return val is False
    if isinstance(val, (int, float)):
        return val == 0
    if isinstance(val, str):
        return val.strip().lower() in _OFF_WORDS
    if isinstance(val, dict):
        return _is_off_value(val.get("effort"))
    return False


def _explicit_thinking_off(request: Any) -> str | None:
    """Name the wire form that turned thinking off, or None if unspecified.

    REQUEST PARAMS ONLY — never prompt text. Ordered most-specific first so
    the log line names the field the caller actually used.
    """
    ctk = getattr(request, "chat_template_kwargs", None) or {}
    if isinstance(ctk, dict):
        if ctk.get("enable_thinking", None) is False:
            return "chat_template_kwargs.enable_thinking=false"
        if _CONTROL_KEY in ctk and _is_off_value(ctk.get(_CONTROL_KEY)):
            return f"chat_template_kwargs.{_CONTROL_KEY}=off"
    # The house-preferred form. `is not None` matters: absent != 0.
    budget = getattr(request, "thinking_token_budget", None)
    if isinstance(budget, int) and not isinstance(budget, bool) and budget <= 0:
        return "thinking_token_budget=0"
    # PN71 aliases. Only the OFF spellings — low/medium/high/max are ON.
    for attr in ("reasoning_effort", "reasoning"):
        val = getattr(request, attr, None)
        if val is not None and _is_off_value(val):
            return f"{attr}=off"
    return None


def _apply_explicit_off(request: Any) -> None:
    """Make the caller's off decision explicit to every layer downstream.

    Three writes, each closing one leak documented in the module header:
      * `enable_thinking=False` — the chat template pre-closes `<think></think>`
        (so the model spends zero reasoning tokens), PN16 variant-3 defers,
        PN102's banner injector early-returns, and PN114's seed stripper
        declines. This is the one signal every downstream layer already reads.
      * clear a zero `thinking_token_budget` — a budget of 0 is still a budget
        to `ThinkingBudgetStateHolder`, which builds a state entry for it and
        hands that entry to the forced-close path and to PN108's plateau cap.
        None makes the holder pop the row instead of tracking it.
      * `h119_overridable=0` — the ownership stamp the H119 route consumer
        reads in `sync_batch`. Without it an unbudgeted row looks unclaimed and
        the consumer installs a provisional DEEP budget before the router's own
        thinking-off tap has run (it runs in the prefill postprocess, a whole
        phase later — this is why the frontend has to say it).

    The control key is popped for the same reason `_decide_mode` pops it: it is
    our protocol, not the chat template's, and it must not reach the render.
    """
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    ctk.pop(_CONTROL_KEY, None)
    ctk["enable_thinking"] = False
    # A banner/steps pair can only be here if something upstream already ran;
    # neither is legal on a request that will not think.
    ctk.pop("pn_env_banner", None)
    ctk.pop("pn100_steps", None)
    request.chat_template_kwargs = ctk
    budget = getattr(request, "thinking_token_budget", None)
    if isinstance(budget, int) and not isinstance(budget, bool) and budget <= 0:
        try:
            request.thinking_token_budget = None
        except Exception:  # noqa: BLE001 — frozen model: the stamp still lands
            log.debug("PN100: could not clear zero thinking_token_budget",
                      exc_info=True)
    _stamp_h119(request, 0)


def _shape_tier0(request: Any) -> bool:
    """[2026-07-25 tier-0/engage defect, showdown verdict #1] Request-SHAPE
    hints that mark a structured-output call — land tier 0 WITHOUT spending a
    classify call. DEFAULT-DARK behind GENESIS_PN100_SHAPE_TIER0.

    Why: when the server decides (thinking_budget:"auto"), classify engages
    thinking on ~95% of rows prod actually runs thinking-OFF (prod_mixed_v2:
    the hindsight extract/consolidate class). Those calls are identifiable
    from the REQUEST alone: they carry response_format json_schema (grammar-
    enforced output) — the caller already declared "short structured answer,
    no prose". Rules are deliberately explicit and narrow (PN16-regex lesson:
    never infer triviality from prompt TEXT — only from request PARAMS):

      A. structured-output constraint present: response_format type in
         {json_object, json_schema, structural_tag}, or any vLLM guided-
         decoding param (guided_json/regex/choice/grammar, structured_outputs).
      B. temperature <= SHAPE_MAX_TEMP (default: exactly 0) AND completion
         cap <= SHAPE_SMALL_MAX_TOKENS (default 512) — deterministic tiny
         answers (label/verdict probes).
      C. forced NAMED tool call (tool_choice={"type":"function",...}) AND the
         same small completion cap — the answer is one short tool invocation
         by construction.

    Deliberately NOT claimed (false-tier-0 costs quality): temp-0 with a
    large cap (temp-0 reasoning evals live there); tool_choice "auto"/
    "required" (tool SELECTION can need thinking); short prompts; any
    prompt-content heuristic; requests with explicit enable_thinking=True
    (caller wins — handled at the call sites, which skip this check).
    """
    if not _env_bool("GENESIS_PN100_SHAPE_TIER0"):
        return False
    # Rule A — structured-output constraints.
    rf = getattr(request, "response_format", None)
    if rf is not None:
        rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
        if rf_type in ("json_object", "json_schema", "structural_tag"):
            return True
    for attr in ("guided_json", "guided_regex", "guided_choice",
                 "guided_grammar", "structured_outputs"):
        if getattr(request, attr, None) is not None:
            return True
    cap = _completion_cap(request)
    small = cap is not None and cap <= _env_int(
        "GENESIS_PN100_SHAPE_SMALL_MAX_TOKENS", 512)
    if not small:
        return False
    # Rule B — deterministic + tiny answer.
    temp = getattr(request, "temperature", None)
    if (isinstance(temp, (int, float)) and not isinstance(temp, bool)
            and temp <= _env_float("GENESIS_PN100_SHAPE_MAX_TEMP", 0.0)):
        return True
    # Rule C — forced named tool call, short by construction.
    tc = getattr(request, "tool_choice", None)
    if tc is not None and not isinstance(tc, str):
        tc_type = tc.get("type") if isinstance(tc, dict) else getattr(tc, "type", None)
        if tc_type == "function":
            return True
    return False


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
        # [2026-07-23 ultra-review #10] JSON `true` is an int subtype and >0
        # -> str(True) -> int("True") ValueError -> whole hook failed open
        # SILENTLY with the client's control key already popped. Bools first.
        if control is True:
            control = "auto"
        if control in ("off", "0", 0, False):
            return "skip", True
        if isinstance(control, int) and control > 0:
            return str(control), True
        if isinstance(control, str) and control.isdigit() and int(control) > 0:
            return control, True
        if control == "auto":
            if explicit_thinking is False:
                return "skip", True
            # [2026-07-25 shape-tier0, DARK] explicit enable_thinking=True
            # still wins (rule skipped -> classify clamps tier 0 to 1).
            if explicit_thinking is not True and _shape_tier0(request):
                return "shape0", True
            return "classify", explicit_thinking is not True

    if explicit_thinking is False:
        return "skip", True

    cap = _completion_cap(request)
    if cap is not None and cap <= _env_int("GENESIS_PN100_MIN_MAX_TOKENS", 128):
        return "tiny", True

    if _env_bool("GENESIS_PN100_AUTO_DEFAULT"):
        # [2026-07-25 shape-tier0, DARK] checked BEFORE the length prefilter:
        # a long structured-output prompt (the prod extract class) must land
        # tier 0, not the prefilter's default tier.
        if explicit_thinking is not True and _shape_tier0(request):
            return "shape0", True
        if _skip_classify_by_length(request):
            return "default", explicit_thinking is not True
        return "classify", explicit_thinking is not True
    return "skip", True


def _default_tier() -> int:
    """Tier used when we skip the classify call. Same value the classifier
    falls open to, so a skipped question and a failed one behave identically."""
    return max(0, min(3, _env_int("GENESIS_PN100_FALLBACK_TIER", 2)))


def _prompt_chars(request: Any) -> int:
    total = 0
    for m in (getattr(request, "messages", None) or []):
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):  # multimodal content parts
            for part in c:
                t = part.get("text") if isinstance(part, dict) else None
                if isinstance(t, str):
                    total += len(t)
    return total


def _skip_classify_by_length(request: Any) -> bool:
    """Long prompts skip the classify call and take the default budget.

    The classify call only ever changes the outcome by detecting TRIVIALITY
    (thinking off). Measured on real traffic that is ~5% of requests, while the
    call is charged on 100% — 0.18-0.60s typical, 2.7s worst, plus a second
    prefill and double trace volume. It does not pay for itself.

    Inverting who pays works because the call is a prefill, so its cost scales
    with prompt length — and length also predicts the answer:
      long  -> expensive to classify AND never trivial  -> skip
      short -> cheap to classify AND the only place triviality lives -> ask
    The expensive calls are exactly the ones dropped.

    This is NOT the PN16 regex failure mode. A regex that decides the OUTCOME
    kills thinking on short-hard prompts; this decides only WHO GETS ASKED, so
    a wrong threshold costs at most one unnecessary (cheap) question, never a
    wrong budget. "Prove there are infinitely many primes of the form 4k+3" is
    short, gets asked, and is correctly rated non-trivial.

    Accepted trade: a long-but-trivial request ("reformat this JSON") skips the
    question, takes a budget, and self-stops after brief thinking.
    """
    limit = _env_int("GENESIS_PN100_CLASSIFY_MAX_CHARS", 0)
    if limit <= 0:
        return False  # ships off; opt in per deployment
    return _prompt_chars(request) > limit


async def _classify(serving: Any, request: Any,
                    temperature: float = 0.0) -> tuple[int, int | None] | None:
    """One thinking-off self-call -> (tier, steps|None), or None on failure."""
    req_cls = type(request)
    fields = getattr(req_cls, "model_fields", {}) or {}
    rubric = (
        _RUBRIC_DEPTH
        if os.environ.get("GENESIS_PN100_RUBRIC", "").strip().lower() == "depth"
        else _RUBRIC
    )
    kwargs: dict[str, Any] = {
        "model": getattr(request, "model", None),
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": _flatten_messages(request)},
        ],
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, _MARKER_KEY: True},
    }
    cap_field = "max_completion_tokens" if "max_completion_tokens" in fields else "max_tokens"
    kwargs[cap_field] = 12
    projn = _env_bool("GENESIS_PN100_PROJECTED_N", False) and "logprobs" in fields
    if projn:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 20
    synthetic = req_cls(**kwargs)

    timeout = _env_int("GENESIS_PN100_TIMEOUT_S", 20)
    resp = await asyncio.wait_for(
        serving.create_chat_completion(synthetic, raw_request=None), timeout
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        return None
    content = getattr(choices[0].message, "content", "") or ""
    # planner forms: "T|S|C" (rubric v3, C in {h,l}), "T|S", bare "T"
    m = re.search(r"([0-3])\s*\|\s*(\d{1,3})\s*\|\s*([hl])", content)
    if not m:
        m2 = re.search(r"([0-3])\s*\|\s*(\d{1,3})", content)
    else:
        m2 = None
    hit = m or m2
    if hit:
        _LAST_CONF["v"] = m.group(3) if m else "h"
        tier = int(hit.group(1))
        steps = max(0, min(120, int(hit.group(2))))
        # menu gate: the posterior read is only calibrated for the GPQA-band
        # step menu; off-menu sampled values (0,1,2,6,...) make the digit
        # read ambiguous ("2" = two or twenty-five) — fall back to sampled.
        if projn and steps in (3, 4, 5, 8, 12, 15, 25):
            try:
                proj = _projected_steps(choices[0])
                if proj is not None:
                    log.info("PN100: projected-N %.2f -> %d (sampled %d)",
                             proj[0], proj[1], steps)
                    steps = proj[1]
            except Exception as exc:  # noqa: BLE001 — fail-open to sampled
                log.warning("PN100: projected-N read failed (%s); sampled used", exc)
        return tier, steps
    m = re.search(r"[0-3]", content)
    _LAST_CONF["v"] = "l"
    return (int(m.group(0)), None) if m else None


# [2026-07-23 A1-live] Projected-N: read the steps-token POSTERIOR from the
# classify call's own logprobs instead of trusting the sampled point.
# Offline gate passed (a1_logit_menu.py, n=100): expected-steps Spearman 0.813
# vs 0.785 sampled against realized v5-need rank; declusters the N=5 pileup
# (46/100 items) that owned 7/12 premature-commit flips (V3-SIZED analysis).
# Menu first-tokens are single-digit under Qwen3.6's digit-split vocab; the
# "1x"/"2x" branches need a second-position read (emitted branch only; the
# non-emitted branch mass goes to its midpoint). Calibration: measured curve
# E[steps]->realized-need-steps is ~identity low, UNDER ~1.4x deep — encoded
# as a piecewise-linear map (env-overridable), clamp [3, 30].
_PROJN_MENU = {"3": 3.0, "4": 4.0, "5": 5.0, "8": 8.0}
_PROJN_STAGE2 = {"1": {"2": 12.0, "5": 15.0}, "2": {"5": 25.0}}
_PROJN_MID = {"1": 13.5, "2": 25.0}


def _projn_calibrate(e: float) -> int:
    raw = os.environ.get(
        "GENESIS_PN100_PROJN_CAL",
        "4.75:3.2,6:5.1,7.25:5.3,9.5:10.2,13.5:17.3")
    try:
        pts = sorted((float(a), float(b)) for a, b in
                     (p.split(":") for p in raw.split(",")))
    except ValueError:
        return max(3, round(e))
    if e <= pts[0][0]:
        y = pts[0][1]
    elif e >= pts[-1][0]:
        y = pts[-1][1] + (e - pts[-1][0]) * 1.4  # keep deep stretch past last anchor
    else:
        y = pts[-1][1]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x1 <= e <= x2:
                y = y1 + (y2 - y1) * (e - x1) / (x2 - x1)
                break
    return max(3, min(30, round(y)))


def _projected_steps(choice: Any) -> tuple[float, int] | None:
    """(expected_steps, calibrated_steps) from the classify choice logprobs."""
    lp = getattr(choice, "logprobs", None)
    toks = getattr(lp, "content", None) if lp else None
    if not toks:
        return None
    strs = [getattr(t, "token", "") for t in toks]
    try:
        sp = strs.index("|") + 1
    except ValueError:
        return None
    if sp >= len(toks):
        return None
    top = {getattr(t, "token", ""): math.exp(getattr(t, "logprob", -99.0))
           for t in (getattr(toks[sp], "top_logprobs", None) or [])}
    mass: dict[float, float] = {}
    for tk, p in top.items():
        if tk in _PROJN_MENU:
            mass[_PROJN_MENU[tk]] = mass.get(_PROJN_MENU[tk], 0.0) + p
        elif tk in _PROJN_STAGE2:
            if tk == strs[sp] and sp + 1 < len(toks):
                t2 = {getattr(t, "token", ""): math.exp(getattr(t, "logprob", -99.0))
                      for t in (getattr(toks[sp + 1], "top_logprobs", None) or [])}
                tot2 = sum(t2.get(k, 0.0) for k in _PROJN_STAGE2[tk]) or 1.0
                for k, v in _PROJN_STAGE2[tk].items():
                    mass[v] = mass.get(v, 0.0) + p * (t2.get(k, 0.0) / tot2)
            else:
                mass[_PROJN_MID[tk]] = mass.get(_PROJN_MID[tk], 0.0) + p
    tot = sum(mass.values())
    if tot <= 0.05:  # no meaningful menu mass — fall back to sampled
        return None
    e = sum(k * v for k, v in mass.items()) / tot
    return e, _projn_calibrate(e)


def _stash_steps(request: Any, steps: int) -> None:
    """Hand the planner's step estimate to the PN102 contract injector
    (which pops it from chat_template_kwargs downstream)."""
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    ctk["pn100_steps"] = steps
    request.chat_template_kwargs = ctk


def _stamp_h119(request: Any, overridable: int) -> None:
    """Tell the H119 route consumer whether this budget is ours to override.

    [2026-07-25] H119's consumer defers to any non-None thinking_token_budget so
    an explicit CLIENT budget always wins. But PN100 runs with AUTO_DEFAULT=1 and
    budgets ~every request, so from the worker's side PN100's grant is
    indistinguishable from a client's — and the consumer deferred 100% of the
    time. Measured: a GPQA-30 with the consumer correctly installed on all seven
    sites was byte-identical to the control on every compared row.

    The stamp rides SamplingParams.extra_args, which is where vllm_xargs lands
    (chat_completion/protocol.py: extra_args = self.vllm_xargs) and which
    reaches the worker inside the same params object BatchUpdate.added hands the
    consumer. 1 = "PN100 chose this, H119 may re-decide it"; 0 = "keep out"
    (tier-0/thinking-off, and a client-pinned numeric).

    vllm_xargs is typed dict[str, str|int|float|list] — an int, never a bool.
    Fail-open: PN100's contract is that it never breaks a request, and an
    unstamped budget simply reads as a caller's, which is today's behaviour.
    """
    try:
        xargs = dict(getattr(request, "vllm_xargs", None) or {})
        xargs["h119_overridable"] = int(overridable)
        request.vllm_xargs = xargs
    except Exception:  # noqa: BLE001 — a stamp is never worth a failed request
        log.debug("PN100: could not stamp h119_overridable", exc_info=True)


_TOTAL_CEIL_MIN_SLACK = 256


def _apply_total_ceiling(request: Any, budget: int) -> int | None:
    """Bound TOTAL completion (think + answer) for a budgeted request.

    [2026-07-23 total-completion ceiling, DARK; corrected + refitted
    2026-07-26] The answer channel is unbounded and capped-think grinders
    relocate their burn there. Measured on aibox-20260726-clean-100 (n=100,
    joined 100/100 to the PN108 observe line, so every row's grant is known):

      * 38/100 rows were force-closed, in TWO signatures — 28 at `cap-5`
        (the PN100-grant path) and 10 at `cap-13` (the client-budget path).
        The earlier review counted only the cap-5 signature.
      * cap-5 rows: median 908 answer tokens (max 4279) against a median 56
        on rows that stopped naturally. cap-13 rows do NOT relocate
        (median 82) — the relocation grind is a PN100-grant-path effect.
      * median total completion 3568 forced vs 522 self-stopped.

    WHAT THIS ACTUALLY BOUNDS: the request's completion cap. It is a HARD
    sampling stop — vLLM ends the sequence at the limit, `finish_reason` is
    `"length"`, and the client receives the answer CUT MID-EMISSION with no
    repair. There is no soft landing. Hence three safety rules below.

    1. It bounds the EFFECTIVE cap. The old code wrote `max_tokens` only,
       but vLLM's `ChatCompletionRequest` prefers `max_completion_tokens`
       when set — so against any OpenAI-modern caller (which is what the
       hindsight prod client sends) the ceiling was silently INERT.
    2. It never cuts inside the think block. `ceil_total` is floored at
       `budget + slack`, so an F < 1 (or a slack typo) can no longer produce
       a request that dies before it can answer at all.
    3. It only ever LOWERS. A caller who already asked for less keeps their
       own smaller cap; we never raise a cap the caller set.

    FITTED VALUES (banked data, 2026-07-26 — see the F trade curve in the
    commit message). Answer-length p99 on rows that stopped NATURALLY (the
    only answers uncontaminated by relocation):
        GPQA clean-100, n=62 : p95 474, p99 1311, max 1311
        prod_mixed_v2, n=1043 (89 trace-corrupted rows excluded):
                               p95 1802, p99 2173, p99.9 2608, max 2740
    Real traffic answers are ~2x longer than GPQA's, so SLACK IS SIZED OFF
    PROD, not off the bench: 3072 clears the observed prod max (2740) with
    12% margin and GPQA's max with 2.3x. F=1.0 with that slack binds 2/100
    GPQA rows, both already-force-closed grinders, zero natural rows, zero
    correct answers lost: -1873 ctok (-1.0%) and -29.0s wall (-1.0%).

    F=2.0 — the value the original comment proposed — binds ZERO rows on
    clean-100 at any slack. It was fitted on the older 8.4K-answer traces
    and is inert against current traffic; do not ship it as "the safe one",
    it is the OFF one.
    """
    tc_f = _env_float("GENESIS_PN100_TOTAL_CEIL_F", 0.0)
    if tc_f <= 0 or budget <= 0:
        return None
    slack = max(_env_int("GENESIS_PN100_TOTAL_CEIL_SLACK", 1024),
                _TOTAL_CEIL_MIN_SLACK)
    ceil_total = max(int(budget * tc_f) + slack, budget + slack)
    # The EFFECTIVE cap the caller already asked for: max_completion_tokens
    # wins over max_tokens where both are set, but a caller may set either,
    # so respect the tighter of the two and never raise it.
    caps = [
        v for v in (getattr(request, "max_completion_tokens", None),
                    getattr(request, "max_tokens", None))
        if isinstance(v, int) and not isinstance(v, bool) and v > 0
    ]
    if caps and min(caps) <= ceil_total:
        return None  # caller is already tighter — leave the request alone
    applied = False
    for attr in ("max_tokens", "max_completion_tokens"):
        if not hasattr(request, attr):
            continue
        try:
            setattr(request, attr, ceil_total)
            applied = True
        except Exception:  # noqa: BLE001 — frozen model: fail open, no ceiling
            log.debug("PN100: could not set %s for total ceiling", attr,
                      exc_info=True)
    if not applied:
        return None
    _STATS["total_ceiling"] += 1
    log.info("PN100: total-completion ceiling %d (budget=%d F=%.2f slack=%d)",
             ceil_total, budget, tc_f, slack)
    return ceil_total


def _apply_tier(request: Any, tier: int, allow_disable: bool) -> int:
    budgets = _tier_budgets()
    if tier == 0 and not allow_disable:
        tier = 1
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    # tier 0 disables thinking only when budgets[0]==0 (the default); an
    # operator who sets a non-zero tier-0 budget gets a small budget instead.
    if tier == 0 and budgets[0] <= 0:
        ctk["enable_thinking"] = False
        request.chat_template_kwargs = ctk
        # Thinking is OFF and no budget is set. H119 must not install a
        # provisional entry here: PN100 keeps absolute authority over tier 0.
        _stamp_h119(request, 0)
        return 0
    ctk["enable_thinking"] = True
    request.chat_template_kwargs = ctk
    request.thinking_token_budget = budgets[tier]
    _stamp_h119(request, 1)
    # The tier path relocates exactly like the continuous path — the ceiling
    # was only ever wired into _apply_budget, so a tier-map boot got none.
    _apply_total_ceiling(request, budgets[tier])
    return budgets[tier]


def _continuous_budget(tier: int, steps: int | None,
                       ptok: int | None = None) -> int | None:
    """Continuous per-task budget from the classifier's own step estimate.

    [2026-07-22 USER] The tier bucket is too coarse — real traffic wants a
    flexible budget (~100-tok granularity in the 500-1500 band where most
    requests live). When GENESIS_PN100_CONTINUOUS=1 and the planner returned a
    step count, the budget is steps x tok_per_step, rounded to 100 and clamped.
    The tier is used ONLY to gate thinking off (tier 0). Fail-safe: returns
    None when disabled or no step estimate (caller falls back to the tier map).
    """
    if not _env_bool("GENESIS_PN100_CONTINUOUS", False):
        return None
    if not steps or steps <= 0:
        return None
    # [2026-07-22 USER] per-bucket quantile map: budget = what THIS step-value
    # historically needed (fitted from banked L-data), piecewise-linear between
    # fitted points. Beats any single multiplier because true need per step
    # value is nonlinear. Format: "steps:budget,steps:budget,..." sorted keys.
    map_env = os.environ.get("GENESIS_PN100_STEP_BUDGET_MAP", "").strip()
    if map_env:
        try:
            pts = sorted(
                (int(a), int(b))
                for a, b in (p.split(":") for p in map_env.split(","))
            )
            if steps <= pts[0][0]:
                raw = pts[0][1]
            elif steps >= pts[-1][0]:
                raw = pts[-1][1]
            else:
                raw = None
                for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
                    if s0 <= steps <= s1:
                        f = (steps - s0) / max(1, s1 - s0)
                        raw = b0 + f * (b1 - b0)
                        break
            if raw is not None:
                if _LAST_CONF.get("v") == "l":
                    raw *= _env_float("GENESIS_PN100_LOWCONF_MULT", 1.5)
                floor_m = _env_int("GENESIS_PN100_BUDGET_FLOOR", 128)
                ceil_m = _env_int("GENESIS_PN100_BUDGET_CEIL", 10240)
                return max(floor_m, min(ceil_m, int(round(raw / 100.0)) * 100))
        except (ValueError, IndexError):
            pass  # malformed map -> fall through to k*steps
    k = _env_int("GENESIS_PN100_TOK_PER_STEP", 150)
    # [2026-07-22 USER] floor 128: prod trivials should be able to land tiny
    # grants (512 and below); grow-on-progress makes a low floor harmless.
    floor = _env_int("GENESIS_PN100_BUDGET_FLOOR", 128)
    ceil = _env_int("GENESIS_PN100_BUDGET_CEIL", 10240)
    raw = steps * k
    # [2026-07-23 P-pess] high-step pessimism: the k260 audit showed ALL
    # budget-caused misses were UNDER-grants on high-step items (099/097/127
    # near-miss class, 76-95% of need). ENFORCED budget only — the banner
    # renders steps, so this is invisible to the model (no P1 anchoring risk).
    # Fat is free (self-stop). Dark unless GENESIS_PN100_HIGHSTEP_MULT is set.
    hs_mult = _env_float("GENESIS_PN100_HIGHSTEP_MULT", 1.0)
    hs_min = _env_int("GENESIS_PN100_HIGHSTEP_MIN", 10)
    if hs_mult > 1.0 and steps >= hs_min:
        raw *= hs_mult
    # [PN162 2026-07-27, DARK] closed-loop calibration: multiply in the learned
    # per-bucket k BEFORE rounding, so grant' = round100(steps x k_step x k).
    # Identity (k == 1.0) unless GENESIS_ENABLE_PN162_BUDGET_CAL=1 and a
    # well-formed ledger is readable; the lookup never blocks and never raises.
    # NOT reached on the STEP_BUDGET_MAP branch above (it returns early) and
    # refused there too — see pn162_budget_cal.budget_multiplier.
    if _pn162 is not None:
        raw *= _pn162.budget_multiplier(steps, ptok)
    rounded = int(round(raw / 100.0)) * 100
    return max(floor, min(ceil, rounded))


def _pn162_ptok(request: Any) -> int | None:
    """Rough prompt-token count for PN162's composite key schemas.

    Only read when the ledger declares a composite `key_schema` (default
    "steps" ignores it). `_prompt_chars` is uncapped and already on this path;
    the sink's exact `prompt_tok` is what the updater bands on, so the two
    agree only approximately — the reason a composite schema needs its own
    screened boot before being switched on.
    """
    if _pn162 is None:
        return None
    try:
        cpt = _env_float("GENESIS_PN162_CHARS_PER_TOK",
                         _pn162.DEFAULT_CHARS_PER_TOK)
        return int(_prompt_chars(request) / max(0.5, cpt))
    except Exception:  # noqa: BLE001
        return None


def _pn162_arm(request: Any, steps: int | None) -> tuple[int | None, int | None]:
    """(steps_to_announce, steps_to_size) + the arm stamp. Dark no-op today.

    PN162's exploration leg is the only counterfactual for "the announced N was
    too low even though nothing bound" — the LEAN lane renders N into the
    prompt (`_contract_v3_sized`), so an under-announced N self-fulfils and the
    bound/slack loop never sees it. Disarmed unless GENESIS_ENABLE_PN162_
    EXPLORE=1 with PN162_EXPLORE_EPS > 0.
    """
    if _pn162 is None:
        return steps, steps
    try:
        ann, size, arm = _pn162.explore_arm(steps)
        if arm is not None:
            _pn162.stamp_xargs(request, arm=arm)
        return ann, size
    except Exception:  # noqa: BLE001 — never break a request for telemetry
        log.debug("PN162: explore arm failed — identity", exc_info=True)
        return steps, steps


def _pn162_exact(request: Any, budget: int | None, key: str) -> int | None:
    """Raise `budget` to this exact prompt's remembered bound grant.

    `key` is PN100's own tier-cache key — sha256 of the flattened messages —
    truncated to 32 hex chars; the middleware already computes it, so the hash
    is free. Nothing populates the ledger's `exact` map today (the PN119 sink
    carries no prompt hash), so this is inert in practice; the stamp below is
    what makes wiring it a one-line router change. Separate flag, USER ruling
    pending — it is a per-item history, which the bench rule forbids for bench
    traffic.
    """
    if _pn162 is None or budget is None:
        return budget
    try:
        if not (_pn162.is_enabled() and _pn162.exact_enabled()):
            return budget
        phash = key[:32]
        _pn162.stamp_xargs(request, prompt_hash=phash)
        return _pn162.exact_floor(phash, int(budget))
    except Exception:  # noqa: BLE001
        log.debug("PN162: exact leg failed — identity", exc_info=True)
        return budget


def _apply_budget(request: Any, budget: int) -> int:
    """Set an explicit thinking budget (continuous path)."""
    ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
    ctk["enable_thinking"] = True
    request.chat_template_kwargs = ctk
    request.thinking_token_budget = budget
    _stamp_h119(request, 1)
    _apply_total_ceiling(request, budget)
    return budget


async def apply_hook_async(serving: Any, request: Any) -> None:
    """Mutate request in place per PN100 auto-budget policy.

    Failure mode: any exception is caught by the call-site's try/except and
    the request proceeds untouched (fail-open to current behavior)."""
    ctk = getattr(request, "chat_template_kwargs", None)
    if ctk and ctk.pop(_MARKER_KEY, None):
        return  # our own classify call — never recurse (checked FIRST: that
        # call carries enable_thinking=False and must not take the bypass)
    enabled = _is_enabled()
    if enabled:
        _STATS["total_requests"] += 1

    # ── explicit request-level thinking-off: hard bypass ──────────────────
    # Deliberately ahead of the master gate — see the module header. The
    # h119_overridable=0 stamp is a worker-side contract that has to be
    # written whether or not PN100 is allocating budgets this boot.
    if _strict_off():
        off_form = _explicit_thinking_off(request)
        if off_form is not None:
            _STATS["explicit_off"] += 1
            _apply_explicit_off(request)
            log.info("PN100: explicit thinking-off (%s) — bypassed: no "
                     "classify, no budget, no banner, no H119 grant", off_form)
            return
    if not enabled:
        return

    # NB with GENESIS_THINKING_OFF_STRICT on (the default) the off-forms never
    # reach here, so "explicit_skip" now counts explicit ON intent only:
    # a positive thinking_token_budget, or a non-off reasoning/reasoning_effort.
    mode, allow_disable = _decide_mode(request)
    if mode == "skip":
        _STATS["explicit_skip"] += 1
        return
    if mode == "tiny":
        _STATS["tiny_skip"] += 1
        return
    if mode == "shape0":
        # [2026-07-25 shape-tier0, DARK] structured-output request shape —
        # tier 0 with zero classify spend. Log line keeps the
        # "PN100: tier=N -> ..." grammar prod_footprint.py parses.
        _STATS["shape_tier0"] += 1
        applied = _apply_tier(request, 0, allow_disable)
        log.info(
            "PN100: tier=0 -> %s (shape-hint, no classify)",
            "thinking off" if applied == 0 else f"budget={applied}",
        )
        return
    if mode == "default":
        # Prefilter: too long to be trivial, so the classify call could only
        # have confirmed what we already assume. Take the default tier without
        # paying for the question.
        _STATS["prefilter_default"] += 1
        applied = _apply_tier(request, _default_tier(), allow_disable)
        log.info("PN100: prefilter -> default tier (budget=%d, no classify)", applied)
        return
    if mode != "classify":  # direct numeric via chat_template_kwargs.thinking_budget
        budget = int(mode)
        ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
        ctk["enable_thinking"] = True
        request.chat_template_kwargs = ctk
        request.thinking_token_budget = budget
        # Client-pinned via chat_template_kwargs.thinking_budget: this is the
        # caller's number, not ours. H119 must leave it entirely alone.
        _stamp_h119(request, 0)
        log.info("PN100: direct budget=%d (client-pinned)", budget)
        return

    t0 = time.monotonic()
    key = hashlib.sha256(_flatten_messages(request).encode()).hexdigest()
    cached = _TIER_CACHE.get(key)
    if cached is not None:
        _TIER_CACHE.move_to_end(key)
        _STATS["cache_hits"] += 1
        c_tier, c_steps = cached
        c_ann, c_size = _pn162_arm(request, c_steps)      # PN162, dark no-op
        cont = _pn162_exact(
            request,
            _continuous_budget(c_tier, c_size, _pn162_ptok(request)), key)
        if cont is not None and not (c_tier == 0 and allow_disable):
            applied = _apply_budget(request, cont)
        else:
            applied = _apply_tier(request, c_tier, allow_disable)
        if applied > 0 and c_ann:
            _stash_steps(request, c_ann)
        log.info(
            "PN100: tier=%d -> %s (cached)",
            c_tier,
            "thinking off" if applied == 0 else f"budget={applied}",
        )
        return

    tier = None
    steps = None
    try:
        # [2026-07-23 P-decode resample, DARK] GENESIS_PN100_RESAMPLE_N>1:
        # N concurrent classify calls (1 at temp 0 + N-1 at 0.7); steps =
        # rounded mean, tier = max (safety), and the SPREAD becomes the real
        # confidence signal (spread > 40% of mean -> conf 'l', feeding
        # LOWCONF_MULT) — replaces the dead verbalized h/l flag. De-quantizes
        # the estimator at decode time without touching the frozen rubric.
        n = _env_int("GENESIS_PN100_RESAMPLE_N", 1)
        if n > 1:
            # MODEL temperature for the diversity calls. Default = our
            # verified-clean serving temp 0.6; the card's native
            # recommendation is 1.0 (documented high reasoning variability
            # there — for a 12-token structured reply 0.6 keeps parse
            # compliance while still spreading step estimates).
            rt = _env_float("GENESIS_PN100_RESAMPLE_TEMP", 0.6)
            calls = [_classify(serving, request)]
            calls += [_classify(serving, request, temperature=rt)
                      for _ in range(n - 1)]
            results = await asyncio.gather(*calls, return_exceptions=True)
            oks = [r for r in results
                   if isinstance(r, tuple) and r[0] is not None]
            if oks:
                tier = max(t for t, _ in oks)
                svals = [s for _, s in oks if s]
                if svals:
                    steps = int(round(sum(svals) / len(svals)))
                    spread = max(svals) - min(svals)
                    _LAST_CONF["v"] = (
                        "l" if spread > 0.4 * max(1, steps) else "h"
                    )
                    log.info(
                        "PN100: resample n=%d steps=%s -> %d spread=%d "
                        "conf=%s", len(oks), svals, steps, spread,
                        _LAST_CONF["v"],
                    )
        else:
            verdict = await _classify(serving, request)
            if verdict is not None:
                tier, steps = verdict
    except Exception as exc:  # timeout, engine error, schema drift
        log.warning("PN100: classify failed (%s) — falling back", exc)
        _STATS["errors"] += 1
    if tier is None:
        tier = _env_int("GENESIS_PN100_FALLBACK_TIER", 2)
        _STATS["fallbacks"] += 1
    else:
        _TIER_CACHE[key] = (max(0, min(3, tier)), steps)
        if len(_TIER_CACHE) > _TIER_CACHE_MAX:
            _TIER_CACHE.popitem(last=False)
    tier = max(0, min(3, tier))
    _STATS["classified"] += 1
    _STATS[f"tier_{tier}"] += 1
    ann_steps, size_steps = _pn162_arm(request, steps)    # PN162, dark no-op
    cont = _pn162_exact(
        request, _continuous_budget(tier, size_steps, _pn162_ptok(request)),
        key)
    if cont is not None and not (tier == 0 and allow_disable):
        applied = _apply_budget(request, cont)
    else:
        applied = _apply_tier(request, tier, allow_disable)
    if applied > 0 and ann_steps:
        _stash_steps(request, ann_steps)
    # [2026-07-23 T-sched SJF, DARK — PROD lane only] classify already knows
    # the expected length; expose it as vLLM's per-request priority (lower =
    # earlier) so an OPEN-arrival queue schedules short-first. Requires the
    # server to run --scheduling-policy priority. NO effect on the bench
    # (closed-loop client keeps the server queue empty — measured 07-23).
    if _env_bool("GENESIS_PN100_SJF_PRIORITY", False):
        try:
            request.priority = int(steps or 8)
        except Exception:
            pass
    log.info(
        "PN100: tier=%d -> %s%s (%dms)",
        tier,
        "thinking off" if applied == 0 else f"budget={applied}",
        f" steps~{steps}" if steps else "",
        int((time.monotonic() - t0) * 1000),
    )
