"""PN114-SEED — the PN102 think-seed as a FORCED SPAN (installed by
/fixes/patch_pn114_seed_span.py as vllm/_genesis_pn114_seed.py).

WHY THIS EXISTS
---------------
The H119 lens router scores a request from its own prefill hidden states.
Everything the route can steer therefore has to be applied AFTER prefill —
which today means the thinking-token CAP and nothing else. That ceiling is
low: the zero-risk cap-only oracle saving is ~11.5%, and on the deep side
``cap_hit`` is 0/31, so raising the deep cap does literally nothing. The
headline result came from TREATMENT SELECTION, not from capping.

The PN102 treatment has two halves: the banner (a trailing system turn) and
the seed (``Budget: ~N short steps.\\nStep 1:``). The banner is out of the
model's voice and out of scope for this pass. The SEED, however, is rendered
by the chat template INSIDE ``<think>`` — it is already the model's own voice
at the model's own position. Moving it from the prompt into forced output is
therefore position- and voice-identical: the same token ids land at the same
absolute sequence positions, and the holder's own accounting stays exact
(see ACCOUNTING below). That makes the seed the one half of the treatment
that a post-prefill decision can still choose. ~12-13 tokens; at MTP n=3 that
is 4-5 engine steps, all of them forced.

MECHANISM
---------
The forcer is PN114's generalized span machine (fixes/patch_pn114_forced_span
.py): ``state["force_seq"]`` is a list of token ids, the walk in site B
recomputes emitted progress positionally from the LANDED output, and site S
masks every row the seq owns this step. Drafts are never credited and a
rejection's recovery token IS the forced token, so MTP cannot desync a span.
This module only decides WHICH span and WHEN; it owns no forcing code.

Three seats (all grafted by the patch):
  1. ``note_params``      — sync_batch, stashes the seed the API server
                            stripped out of the prompt for this request.
  2. ``maybe_arm``        — update_state, per seq, AFTER H119 has resolved the
                            routed budget and BEFORE _update_think_state, on
                            the step where the request has produced no token
                            yet. That is the only instant at which a forced
                            span can start at output position 0.
  3. ``on_force_complete``— the completion divert, ahead of pn114's own.

And one on the serving side:
  4. ``strip_prompt_seed``— pops ``pn_env_seed`` from chat_template_kwargs so
                            the prompt ends at ``<think>\\n``, and records the
                            exact seed text in ``vllm_xargs``.

MODES (GENESIS_PN114_SEED_MODE)
-------------------------------
  mirror (default) — force exactly the seed PN102 would have rendered. The
      observable sequence is identical to the prompt-rendered arm; this is
      what gate M2 asserts, and it is the mode to A/B against baseline to
      prove the mechanism is free.
  routed          — pick N from the ROUTED budget (the same
      ``max(3, round(budget / GENESIS_PN102_TOKENS_PER_STEP))`` PN102 uses),
      i.e. the actual escape from cap-only. Requires the route to have landed
      before the first token; a request whose route has not resolved by then
      is left alone (no seed) and counted, never guessed at.

ACCOUNTING (why the cap still binds identically)
------------------------------------------------
The template opens ``<think>`` in the PROMPT, so the holder runs in
``continue_thinking`` mode, where ``_update_think_state`` recomputes

    think_count = len(output_tok_ids) + (len(prompt) - (start_thinking + start_len))

from scratch every step. Prompt-rendered: the seed is in the prompt term.
Forced-span: the seed is in the output term. The sum is the same integer, so
``check_count_down``, the ``total_thinking_tokens > budget`` guillotine and
the absolute position at which ``</think>`` is forced are all unchanged. The
span therefore needs NO budget compensation — and must not be given any
(pn114's probe path adds ``consumed`` back because a probe is an artifact;
a seed is not).

Never raises into serving: every entry point is wrapped by its caller AND
internally, and any failure degrades to "no seed span", which is the
prompt-rendered status quo.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("vllm.genesis.pn114seed")

_TABLE_PATH = "/tmp/genesis_pn114_seed_ids.json"
_TABLE: dict[str, Any] | None = None
_TABLE_TRIED = False

# state keys (namespaced; the holder state dict is shared with pn108/112/114)
K_TEXT = "_pn114_seed_text"      # seed text the API stripped, from xargs
K_PHASE = "_pn114_seed_phase"    # "armed" while the span is in flight
K_DONE = "_pn114_seed_done"      # one arm per request, ever

XARG_KEY = "pn114_seed_text"

STATS = {
    "armed": 0,
    "completed": 0,
    "declined_no_table": 0,
    "declined_unknown_seed": 0,
    "declined_late": 0,
    "declined_provisional": 0,
    "declined_no_route_n": 0,
    "stripped": 0,
}

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    log.warning(msg, *args)


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


def enabled() -> bool:
    """Master flag. DEFAULT OFF — this rewrites what the model reads."""
    return _env_bool("GENESIS_ENABLE_PN114_SEED_SPAN", False)


def mode() -> str:
    m = os.environ.get("GENESIS_PN114_SEED_MODE", "mirror").strip().lower()
    return m if m in ("mirror", "routed") else "mirror"


def table() -> dict[str, Any] | None:
    """The boot-derived seed table, or None. Read once per process."""
    global _TABLE, _TABLE_TRIED
    if not _TABLE_TRIED:
        _TABLE_TRIED = True
        try:
            with open(_TABLE_PATH, encoding="utf-8") as f:
                t = json.load(f)
            if isinstance(t, dict) and t.get("by_text"):
                _TABLE = t
            else:
                _warn_once("badtable",
                           "PN114-SEED: %s has no seeds — inert", _TABLE_PATH)
        except Exception:
            _warn_once("notable",
                       "PN114-SEED: seed table %s missing/unreadable — inert "
                       "(run /fixes/pn114_seed_ids.py at boot)", _TABLE_PATH)
    return _TABLE


def seed_ids_for_text(text: str) -> list[int] | None:
    """Token ids for a seed string, ONLY if boot proved it splits exactly."""
    t = table()
    if not t:
        return None
    ids = t["by_text"].get(text)
    return list(ids) if ids else None


def steps_for_budget(budget: int) -> int:
    """PN102's own N formula (answer_rescue.py:381) — mirrored, not invented."""
    tps = max(50, _env_int("GENESIS_PN102_TOKENS_PER_STEP", 193))
    return max(3, round(budget / tps))


def routed_seed_text(budget: int, prior_text: str | None) -> str | None:
    """The seed text a ROUTED budget should announce.

    Keeps the label/tail family of whatever PN102 chose for this request (so
    the routed span differs from the mirror span in N and in nothing else);
    falls back to the v5 shape when the request carried no seed at all.
    """
    t = table()
    if not t:
        return None
    label, tail = "Budget", "plain"
    if prior_text:
        for key, txt in t.get("by_steps", {}).items():
            if txt == prior_text:
                parts = key.split("|")
                if len(parts) == 3:
                    label, tail = parts[0], parts[1]
                break
    n = steps_for_budget(budget)
    return t.get("by_steps", {}).get(f"{label}|{tail}|{n}")


# ───────────────────────────────────────────────────────────────────────────
# 4. serving side — pop the seed out of the prompt, carry it in vllm_xargs
# ───────────────────────────────────────────────────────────────────────────
def strip_prompt_seed(request: Any) -> bool:
    """True iff this request's seed moved from the prompt into vllm_xargs.

    Fail-CLOSED and symmetric with the engine: a seed the boot table does not
    know is left in the prompt, and then ``maybe_arm`` also declines it, so a
    request can never lose its seed entirely.
    """
    try:
        if not enabled():
            return False
        ctk = getattr(request, "chat_template_kwargs", None)
        if not isinstance(ctk, dict):
            return False
        seed = ctk.get("pn_env_seed")
        if not seed or not isinstance(seed, str):
            return False
        if ctk.get("enable_thinking") is False:
            return False
        if seed_ids_for_text(seed) is None:
            _warn_once("unknownseed",
                       "PN114-SEED: seed %r not in the boot table — left in "
                       "the prompt (the engine declines it too)", seed)
            return False
        ctk.pop("pn_env_seed", None)
        # ctk may be a pydantic field holding the same object; reassign so a
        # copy-on-read model still sees the removal.
        try:
            request.chat_template_kwargs = ctk
        except Exception:
            pass
        xargs = getattr(request, "vllm_xargs", None)
        if not isinstance(xargs, dict):
            xargs = {}
        xargs[XARG_KEY] = seed
        request.vllm_xargs = xargs
        STATS["stripped"] += 1
        return True
    except Exception:
        log.warning("PN114-SEED: strip_prompt_seed raised — prompt untouched",
                    exc_info=True)
        return False


# ───────────────────────────────────────────────────────────────────────────
# 1. sync_batch — stash the stripped seed on the holder state entry
# ───────────────────────────────────────────────────────────────────────────
def note_params(state: dict[str, Any], params: Any) -> None:
    try:
        if not enabled() or not isinstance(state, dict):
            return
        extra = getattr(params, "extra_args", None)
        if not isinstance(extra, dict):
            return
        text = extra.get(XARG_KEY)
        if isinstance(text, str) and text:
            state[K_TEXT] = text
    except Exception:
        log.warning("PN114-SEED: note_params raised — ignored", exc_info=True)


# ───────────────────────────────────────────────────────────────────────────
# 2. update_state — arm the span, exactly once, at output position 0
# ───────────────────────────────────────────────────────────────────────────
def maybe_arm(state: dict[str, Any], think_start_len: int,
              req_id: str | None = None) -> bool:
    """Arm the forced seed span. True iff a span was armed this call."""
    try:
        if not enabled() or not isinstance(state, dict):
            return False
        if state.get(K_DONE) or state.get(K_PHASE):
            return False
        prior = state.get(K_TEXT)
        if not prior:
            return False              # nothing was stripped for this request
        budget = state.get("thinking_token_budget", -1)
        if not isinstance(budget, int) or budget <= 0:
            return False
        if state.get("output_tok_ids"):
            # A token already landed: position 0 is gone, so a span here would
            # NOT be position-identical. Decline — never approximate.
            state[K_DONE] = True
            STATS["declined_late"] += 1
            _warn_once("late", "PN114-SEED: first arm chance already passed "
                              "(req=%s) — no seed span", req_id)
            return False
        if state.get("in_end") or not state.get("in_think", False):
            return False              # closing, or the prompt never opened
        if state.get("force_seq"):
            return False              # some other span owns the forcer
        if mode() == "routed":
            if state.get("_h119_provisional"):
                # H119 has not resolved this request's route yet (chunked
                # prefill). Wait; if it never resolves before token 0 the
                # `output_tok_ids` guard above closes the window honestly.
                STATS["declined_provisional"] += 1
                return False
            text = routed_seed_text(budget, prior)
            if not text:
                state[K_DONE] = True
                STATS["declined_no_route_n"] += 1
                _warn_once("nonrouted",
                           "PN114-SEED: no table entry for the routed N "
                           "(budget=%s) — no seed span", budget)
                return False
        else:
            text = prior
        ids = seed_ids_for_text(text)
        if not ids:
            state[K_DONE] = True
            STATS["declined_unknown_seed"] += 1
            return False
        _arm(state, ids)
        state[K_PHASE] = "armed"
        state[K_DONE] = True
        STATS["armed"] += 1
        log.info("PN114-SEED: armed %d-token seed (mode=%s budget=%s req=%s) "
                 "%r", len(ids), mode(), budget, req_id, text)
        return True
    except Exception:
        log.warning("PN114-SEED: maybe_arm raised — no span armed",
                    exc_info=True)
        try:
            state[K_DONE] = True
        except Exception:
            pass
        return False


def _arm(state: dict[str, Any], ids: list[int]) -> None:
    """Hand the span to PN114's generalized forcer.

    Deliberately NOT pn114._arm: that parks thinking_token_budget at a 10M
    sentinel and gives it back plus ``consumed`` on resume, which is right for
    a probe (an artifact the request never asked for) and wrong for a seed
    (which the prompt-rendered arm charges to the budget in full). The span
    also cannot hit the guillotine while it runs — _update_think_state takes
    the ``in_end`` branch, which neither decrements check_count_down nor tests
    the budget — so there is nothing to park.
    """
    state["force_seq"] = list(ids)
    state["force_seq_base"] = len(state.get("output_tok_ids", []))
    state["in_think"] = False
    state["in_end"] = True
    state["end_count"] = 0
    state["bonus_token_forced"] = False
    state["force_index"] = [0]
    # Pause PN108/PN112 for the duration: with a prompt-rendered seed those
    # detectors never saw the seed tokens (pn108._think_token_slice returns
    # OUTPUT only in continue_thinking mode), so letting them see the forced
    # copy would be a real asymmetry. pn114's phase flag is the existing
    # pause signal; we set it through pn114's own accessor and clear it on
    # completion, and pn114.on_force_complete is never reached for our span
    # because our completion handler runs first and claims it.
    try:
        from vllm._genesis.plateau import pn114 as _pn114
        _pn114._st(state)["phase"] = "pn114seed"
    except Exception:
        log.debug("PN114-SEED: could not set the pn114 pause phase",
                  exc_info=True)


# ───────────────────────────────────────────────────────────────────────────
# 3. completion divert — resume thinking, charging the span like prompt tokens
# ───────────────────────────────────────────────────────────────────────────
def _countdown_after_span(state: dict[str, Any]) -> int | None:
    """Charge the span to the budget exactly as prompt tokens would be.

    ``_update_think_state`` has TWO accountings and they are not equivalent:

      * the RECOMPUTE branch derives ``think_count`` from scratch
        (``len(output) + think-tokens-in-prompt``), which already counts a
        forced span correctly and needs no help; but
      * the fast path above it early-returns on ``check_count_down``, a
        RUNNING counter seeded at init from the prompt's think tokens and
        decremented by newly sampled tokens — and it is skipped entirely
        while ``in_end`` is set, i.e. for every step of the span.

    So a span that is not charged here buys the request len(span) extra
    thinking tokens before the guillotine, and the two arms diverge exactly
    len(span) tokens later. Measured, not assumed: gate M2 caught this.

    The prompt-rendered arm's counter at this same absolute position is
    ``budget - (think tokens in its prompt)``; the forced arm's equivalent is
    ``budget - (think tokens in ITS prompt + the span it just emitted)``. The
    init-style prompt count is recoverable from ``start_thinking``, which
    ``_init_state_entry`` sets to ``len(prompt) - think_count - 1``.
    """
    budget = state.get("thinking_token_budget", -1)
    if not isinstance(budget, int) or budget <= 0:
        return None
    prompt_len = len(state.get("prompt_tok_ids") or ())
    start = state.get("start_thinking", -1)
    if state.get("continue_thinking") and start >= 0:
        # `_init_state_entry` sets start_thinking = len(prompt) - think_count
        # - 1, so this recovers the INIT-style prompt think count — which is
        # the one the running counter was seeded with. (It differs by one from
        # the recompute branch's `len(prompt) - (start + start_len)`; that
        # off-by-one is the holder's, and mirroring it is the whole point.)
        in_prompt = max(0, prompt_len - start - 1)
    else:
        in_prompt = 0
    # Deliberately NOT clamped at 0: a non-positive countdown is how the
    # holder expresses "already out of budget", and the degenerate
    # budget <= len(seed) case must close at the same absolute position on
    # both arms.
    return budget - (in_prompt + len(state.get("output_tok_ids", ())))


def on_force_complete(state: dict[str, Any]) -> bool:
    """True iff this module owned the finished span (caller must NOT reset)."""
    try:
        if not isinstance(state, dict) or state.get(K_PHASE) != "armed":
            return False
        state[K_PHASE] = None
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        state["end_count"] = 0
        state["bonus_token_forced"] = False
        cd = _countdown_after_span(state)
        if cd is not None:
            state["check_count_down"] = cd
        if cd is not None and cd <= 0:
            # Degenerate case: the seed alone meets or exceeds the budget. The
            # prompt-rendered arm decides this at INIT (`token_exhausted <= 0`
            # in _init_state_entry) and forces </think> as its very first
            # sampled token. Resuming here instead would free ONE token first,
            # because the guillotine is only re-evaluated on the next step —
            # a one-position divergence gate M2 caught. Close now, through the
            # stock think-end path (force_seq cleared above).
            state["in_end"] = True
            state["in_think"] = True
            state["force_index"] = [0]
        else:
            # Straight back into the think block; the recompute branch of
            # _update_think_state re-derives think_count from scratch next
            # step and already counts the span (see ACCOUNTING).
            state["in_end"] = False
            state["in_think"] = True
            state["force_index"] = []
        try:
            from vllm._genesis.plateau import pn114 as _pn114
            st = state.get("_pn114")
            if isinstance(st, dict) and st.get("phase") == "pn114seed":
                st["phase"] = None
            del _pn114
        except Exception:
            pass
        STATS["completed"] += 1
        return True
    except Exception:
        log.warning("PN114-SEED: on_force_complete raised — letting the "
                    "stock completion run", exc_info=True)
        return False


def stats() -> dict[str, int]:
    return dict(STATS)
