"""PN121 — soft-landing thinking-budget close (2026-07-26).

WHAT THIS FIXES (measured, aibox-20260726-clean-100, n=100)
  38/100 requests are force-closed at their thinking budget with a BARE
  </think>, mid-sentence. Those rows are 39% wrong vs 15% for rows that stop
  naturally, and the error rate rises monotonically with how badly they were
  under-capped (0% wrong at cap/need <= 1.1x, 67% above 1.5x). They then keep
  reasoning in the ANSWER channel: median 908 answer tokens vs ~7 for a real
  answer, worst 4279. The reasoning is RELOCATED, not saved.

WHAT THIS IS NOT
  This is NOT GENESIS_PN112_WRAPUP_AT_CAP, which was benchmarked
  2026-07-23 (aibox-20260723-wrapcapfull) at 70/100 vs an 83/100 baseline
  (e-value M=34.64, KILL). That arm fired at `think >= budget - 512`: it
  PRE-EMPTED the guillotine by 512 tokens and shortened deep items
  (rtok 1746 -> 1414). It tested EARLY WRAP-UP, not SOFT LANDING.

  PN121 never shortens. It parks the budget at `budget - PARK_AT` (default
  16 tokens — just ahead of the holder's own spec-window trigger, which
  fires at think_count + len(spec) + 1 > budget, i.e. ~cap-5 under mtp/3),
  then EXTENDS by up to PN121_HARD_MARGIN (384) waiting for a newline to
  land on. Worst case it closes 16 tokens earlier than the stock guillotine
  already does; typical case it closes LATER, at a sentence boundary,
  through a transition phrase instead of a bare cut. Direction of the token
  delta is the falsifiable difference from the killed arm: that one was
  -332 rtok, this one is >= 0.

DESIGN (P7 research doc §4, ~/shared/folderX/research/v2/compass_artifact_
wf-5886c2cd-0004-5140-bd48-ae2757fee8e1_text_markdown.md)
  soft phase   think >= budget - PN121_SOFT_RESERVE (320):
               upweight the newline ids and </think> in the logits
               (Mueller's >95%-of-budget nudge). NO forcing, NO budget
               change — the model may still stop naturally.
  park         think >= budget - PN121_PARK_AT (16): stash the real budget,
               set the holder's budget to a sentinel so its own guillotine
               cannot fire, and start watching for a boundary.
  land         the last LANDED token is a newline-ending token -> restore
               the real budget and force the span
               "\\nConsidering the limited time, ...\\n</think>\\n\\n".
  hard force   think >= budget + PN121_HARD_MARGIN (384): force it anyway.
               (Nemotron's proven rule is soft target at cap, hard force at
               cap+500; the doc recommends 320/384 for our traffic.)

  Newline first, then </think>: the span's first token IS a newline
  ("for a 'true stop' the model expects [newline] followed by [</think>], it
  can't be just [</think>]" — Mueller, quoted verbatim in the doc §2).
  </think> is the single token 248069 on this tokenizer.

TOOL / JSON GUARD (upstream #44676, UNMERGED)
  PN121 refuses to park or land once a <tool_call> opener appears in the
  live think slice (the implicit reasoning end that the stock holder does
  not know about), and refuses entirely while a constrained-decoding
  grammar is active for the row (rows stamped by patch_pn121_softland.py
  graft X from the gpu_model_runner grammar seat). Both are independent of
  GENESIS_ENABLE_PR44812_TOOL_GUARD — that graft fixes the holder's own
  accounting, this one keeps OUR injection out of the same window.

HEADROOM AND COUNTER-EVIDENCE (read before believing this will win)
  Ceiling: ThinkBrake (arXiv 2510.00546) measures ORACLE boundary stopping
  — </think> injected at every sentence boundary, best point chosen in
  hindsight — at +8% accuracy for -72% thinking tokens. That is a ceiling
  chosen with hindsight, not a target, but it says boundary-based stopping
  has real headroom.
  Against it: our own ledger says the soft landing may not be needed at
  all. "Trace review weakens the need: answer body reliably finishes the
  job after force-close"; "cut items KEEP their answers (8/8 fired+correct
  items stayed correct incl. a 3895->857 cut)"; "0 parse failures observed
  so far, n=230 cut requests cumulative". So the honest claim for PN121 is
  ctok/wall relocation and mid-derivation quality — NOT parse failures,
  which are already near zero. Untested is untested: this ships OFF.

MTP
  Span emission is whole-window masked, not counter-only. Graft S masks
  EVERY row the sequence owns each step (n draft positions, or the bonus
  row at end_count + n_spec in the all-accept case) with the positionally
  correct span token; graft B recomputes the emitted position from the
  LANDED output alone and never credits a draft. That is the invariant
  vLLM PR #14702 states for grammars ("draft tokens ... should not modify
  the matcher state") and it is what makes a mid-span rejection a no-op
  instead of a break: on rejection the position is re-sampled and the same
  forced token is still massed there. This matters because rejection is the
  COMMON path — at mtp/3 with acceptance 0.92/0.77/0.62 the all-accept
  probability is ~0.44 and our span is 8 tokens. test_pn121_softland.py T10
  drives a rejection at every single step and asserts the span is exact.
  Span length is fixed and never changes the speculation length, so no
  CUDA-graph re-capture is triggered (masks are data, not shape).
  The #34650 class (delta window computed from num_computed_tokens, which
  already includes unverified spec tokens, leaving the slice empty) does
  not apply: the holder scans output_tok_ids from scan_offset, and this
  base already prefers the new_token_ids window in
  v1/structured_output/__init__.py (the placeholder fallback is documented
  there as the broken path, #43388).

  Depth comes from pn108's spec-aware live slice (never think_count, which
  is frozen under budget — BUG-120), so drafts in flight are counted. The
  forced span itself rides PN114's span-sound walk (patch_pn114_forced_span
  sites B/S), whose position authority is the LANDED output — an MTP
  rejection cannot desync it. The base engine carries PR #34668
  (apply_to_logits takes predict_bonus_token + spec_token_ids), so the
  budget is not the pre-v0.21.0 silent no-op under MTP.

NOT A GRACE CHANGE
  PN121 does not touch GENESIS_PN112_GRACE (live 64) or
  H119_ROUTE_GRACE_TOKENS (8, a different knob on the router's budget-cap
  path). PN121_HARD_MARGIN is an EXTENSION CEILING for the boundary hunt,
  not a post-close answer grace, and it is spent only when no newline
  arrives. This matters because a 384-token GRACE has a live kill on this
  box ("grace-384 (relocation erased envelope live)") — measured at the
  PN112 FIRE seat (early, confidence-triggered close). A 256-384 grace at
  the ENFORCE-CAP seat is proposed and has never been screened. Do not read
  this arm as a re-run of that dead one, and do not bundle a grace change
  into it.

EXPECTED UPSIDE, HONESTLY CAPPED
  Relapse is plausibly a TRAINING property, not an inference bug. Nemotron,
  verbatim: "without truncation-trained SFT, the model 'compensates' by
  using more answer tokens; with it, the compensation effect is absent."
  Qwen3.6-27B is not truncation-trained, and our own measurement agrees —
  a forced close relocates rather than saves (true saving tau13 = -288
  ctok / -12% wall, against the -24% that rtok alone suggested). So SCREEN
  THIS ON ctok + WALL, never on rtok: rtok flatters every close mechanism
  because it cannot see the relocation. And judge against the
  NATURAL-STOP cohort, not only the bare-force cohort — at our serving
  temperature (1.0, the model's calibration temp) native self-stopping
  already scores 80% uncapped vs 70% at 0.6, so the bar is the natural
  stop, not the guillotine.

Flag: GENESIS_ENABLE_PN121_SOFTLAND (default OFF). Distinct from the killed
GENESIS_PN112_WRAPUP_AT_CAP; the two are mutually exclusive at runtime
(PN121 stands down if the killed flag is on, and says so once).
"""
from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

log = init_logger("vllm.genesis.pn121")

_KEY = "_pn121"
# budget sentinel while parked; big enough that the holder's countdown can
# never reach it, small enough to be obviously synthetic in a log line.
_PARK_SENTINEL = 10_000_000
_STATS = {"parked": 0, "landed_nl": 0, "landed_hard": 0, "aborted": 0,
          "tool_suppressed": 0, "grammar_suppressed": 0, "unparked": 0}
_WARNED_BOTH = False


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def enabled() -> bool:
    """OFF by default. Also OFF whenever the KILLED early-fire arm is on —
    they target the same close and would fight over the budget."""
    if not _env_bool("GENESIS_ENABLE_PN121_SOFTLAND", False):
        return False
    if _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False):
        global _WARNED_BOTH
        if not _WARNED_BOTH:
            _WARNED_BOTH = True
            log.warning(
                "PN121: GENESIS_PN112_WRAPUP_AT_CAP is ALSO set — that arm "
                "was KILLED 2026-07-23 (70/100 vs 83/100, M=34.64). PN121 "
                "stands down; unset it to screen the soft landing."
            )
        return False
    return True


def cfg() -> dict[str, Any]:
    return {
        # soft nudge starts here (tokens before the cap). 320 = the doc's
        # recommended reserve (mid-point of the supported 256-384 band).
        "soft_reserve": _env_int("PN121_SOFT_RESERVE", 320),
        # park here. 16 > the holder's own trigger distance under mtp/3
        # (len(spec)+1 = 4) and > H119_ROUTE_GRACE_TOKENS (8), so we get the
        # decision instead of the guillotine — and never earlier than that.
        "park_at": _env_int("PN121_PARK_AT", 16),
        # unconditional force at cap + this. Nemotron ships +500; the doc
        # recommends +384 for our traffic mix.
        "hard_margin": _env_int("PN121_HARD_MARGIN", 384),
        # additive logit bumps during the soft phase (logit space, applied
        # at the raw-logits seat before temperature).
        "nudge_nl": _env_float("PN121_NUDGE_NL", 2.0),
        "nudge_end": _env_float("PN121_NUDGE_END", 1.0),
        # refuse to act on budgets too small for the machinery to make sense
        # (a 128-token cap has no room for a 320 soft phase).
        "min_budget": _env_int("PN121_MIN_BUDGET", 512),
    }


def stats_line() -> str:
    return "PN121 " + " ".join(f"{k}={v}" for k, v in _STATS.items())


def _st(state: dict[str, Any]) -> dict[str, Any]:
    st = state.get(_KEY)
    basis = state.get("start_thinking", -1)
    if st is None or st.get("basis") != basis:
        # reset per think block (pn108/pn112/pn114 house rule — a stale
        # phase silently disabled block 2+ in the pn114 lineage).
        st = {"basis": basis, "phase": None, "budget": None, "nudge": False,
              "done": False, "reason": None}
        state[_KEY] = st
    return st


def _contains(hay: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(hay):
        return False
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return True
    return False


def _tool_call_seen(state: dict[str, Any], ids: dict[str, list[int]]) -> bool:
    """#44676 guard: a tool-call opener inside the think block is an IMPLICIT
    reasoning end. Everything after it is JSON arguments — injecting a
    transition phrase + </think> there is exactly the upstream corruption.
    Scanned over the same post-scan_offset slice the holder uses, so a stale
    opener from an earlier segment cannot suppress us forever."""
    tc = ids.get("tool_call")
    if not tc:
        return False
    out = state.get("output_tok_ids", []) or []
    off = state.get("scan_offset", 0) or 0
    return _contains(list(out[off:]), list(tc))


def _grammar_active(state: dict[str, Any], seq_idx: int) -> bool:
    """Constrained decoding / structured output in flight for this row.
    Rows are stamped each step by graft X. A forced span would fight the
    grammar bitmask (both write the same logits rows) and the doc's own
    number for reasoning overflow WITHOUT structured output is ~30%, so the
    structured path is the one place we must not touch."""
    rows = state.get("_pn121_grammar_rows")
    if rows is None:
        return False
    return seq_idx in rows


def _unpark(state: dict[str, Any], st: dict[str, Any], why: str,
            think: int | None = None) -> None:
    """Give the real budget back and stand down for this think block."""
    b = st.get("budget")
    if b is not None and state.get("thinking_token_budget") == _PARK_SENTINEL:
        state["thinking_token_budget"] = b
        spent = think if think is not None else 0
        state["check_count_down"] = max(1, b - spent)
        _STATS["unparked"] += 1
    st["phase"] = None
    st["nudge"] = False
    st["done"] = True
    st["reason"] = why


def release(state: dict[str, Any]) -> None:
    """Hand the real budget back when the think block ended by any route
    other than our own landing (natural </think>, a PN112 cut, the slice
    going non-think). Without this a later think block in the same request
    inherits the park sentinel and is effectively uncapped."""
    st = state.get(_KEY)
    if not st or st.get("phase") != "wait":
        return
    _unpark(state, st, "released")
    log.info("PN121: released park (think block ended elsewhere)")


def observe(state: dict[str, Any], think: int, seq_idx: int = -1,
            req_id: str | None = None) -> None:
    """Called once per tracked request per step from pn114.observe_state,
    BEFORE the holder's _update_think_state — which is the only window in
    which we can park the budget ahead of its own guillotine.

    `think` is the LIVE spec-aware think depth (pn108 slice). Never
    think_count: that is frozen near its init value while under budget
    (BUG-120), so anything keyed on it can never fire mid-think.
    """
    if not enabled():
        return
    try:
        from vllm._genesis.plateau import pn114 as _pn114
    except Exception:
        return
    ids = _pn114._ids() or {}
    if not ids.get("softland_close"):
        return
    st = _st(state)
    if st.get("done"):
        return
    c = cfg()

    # The real budget: ours while parked, the holder's otherwise.
    parked = st.get("phase") == "wait"
    budget = st["budget"] if parked else state.get("thinking_token_budget", -1)
    if budget is None or budget < c["min_budget"]:
        return

    # --- suppression gates (checked every step, both directions) ----------
    if _grammar_active(state, seq_idx):
        _STATS["grammar_suppressed"] += 1
        _unpark(state, st, "grammar", think)
        log.info("PN121: suppressed (structured-output grammar active) "
                 "req=%s", req_id)
        return
    if _tool_call_seen(state, ids):
        _STATS["tool_suppressed"] += 1
        _unpark(state, st, "tool_call", think)
        log.info("PN121: suppressed (<tool_call> in think slice — implicit "
                 "reasoning end, #44676) req=%s think=%d", req_id, think)
        return

    if parked:
        # Someone else (PN112 cut, PN114 confirm, a natural close) took the
        # budget or ended the block while we were parked — hand it back.
        if state.get("thinking_token_budget") != _PARK_SENTINEL:
            _STATS["aborted"] += 1
            st["phase"] = None
            st["nudge"] = False
            st["done"] = True
            st["reason"] = "budget-stolen"
            log.info("PN121: park aborted (budget changed under us) req=%s",
                     req_id)
            return
        if state.get("in_end") or not state.get("in_think", True):
            _unpark(state, st, "closed-elsewhere", think)
            return
        if _pn114.phase_active(state):
            return  # a PN114 span is running; wait it out
        # --- landing decision -------------------------------------------
        out = state.get("output_tok_ids", []) or []
        last = out[-1] if out else None
        nl_end = ids.get("nl_end") or ids.get("newline") or []
        if last is not None and last in nl_end:
            _land(state, st, _pn114, ids, think, budget, "newline", req_id)
            return
        if think >= budget + c["hard_margin"]:
            _land(state, st, _pn114, ids, think, budget, "hard", req_id)
        return

    # --- not parked yet ---------------------------------------------------
    if state.get("in_end") or not state.get("in_think", True):
        return
    if _pn114.phase_active(state):
        return
    # soft phase: nudge only. This cannot shorten by more than soft_reserve
    # and only where the model was already near a boundary; it never forces
    # and never touches the budget.
    st["nudge"] = think >= budget - c["soft_reserve"]
    if think < budget - c["park_at"]:
        return
    # AT the cap (park_at ahead of the holder's own spec-window trigger).
    st["phase"] = "wait"
    st["budget"] = budget
    state["thinking_token_budget"] = _PARK_SENTINEL
    state["check_count_down"] = _PARK_SENTINEL
    _STATS["parked"] += 1
    log.info("PN121: parked at think=%d budget=%d (hard force at %d) req=%s",
             think, budget, budget + c["hard_margin"], req_id)


def _land(state: dict[str, Any], st: dict[str, Any], _pn114: Any,
          ids: dict[str, list[int]], think: int, budget: int, why: str,
          req_id: str | None) -> None:
    """Restore the real budget, then force the soft-landing span.

    Order matters: pn114._arm() stashes the CURRENT budget and re-parks it
    behind its own sentinel, and its wrapup completion path restores that
    stash. Handing it the real budget here is what keeps a second think
    block in the same request from inheriting our sentinel.
    """
    state["thinking_token_budget"] = budget
    state["check_count_down"] = max(1, budget - think)
    # pn114's own saved_budget must be clear or _arm() will not re-stash.
    p114 = state.get("_pn114")
    if isinstance(p114, dict):
        p114["saved_budget"] = None
    st["nudge"] = False
    if state.get("in_end"):
        # an end-forcing is already in flight; arming would mangle it.
        _unpark(state, st, "end-in-flight", think)
        return
    ok = _pn114.arm_wrapup(state, req_id, key="softland_close")
    if not ok:
        _STATS["aborted"] += 1
        st["phase"] = None
        st["done"] = True
        st["reason"] = "arm-failed"
        log.warning("PN121: arm failed at think=%d — stock guillotine takes "
                    "it from here req=%s", think, req_id)
        return
    _STATS["landed_nl" if why == "newline" else "landed_hard"] += 1
    st["phase"] = None
    st["done"] = True
    st["reason"] = why
    log.info("PN121: LAND (%s) think=%d budget=%d overshoot=%+d req=%s",
             why, think, budget, think - budget, req_id)


def nudge_rows(holder: Any) -> list[int]:
    """Row indices currently in the soft phase (for the logits seat)."""
    rows = []
    for idx, state in getattr(holder, "_state", {}).items():
        st = state.get(_KEY)
        if st and st.get("nudge"):
            rows.append(idx)
    return rows
