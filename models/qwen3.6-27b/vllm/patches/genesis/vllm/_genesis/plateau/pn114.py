"""PN114 — forced-span probe machinery (2026-07-23).

Three env-gated mechanisms sharing one generalized forcing machine (the
holder's </think> end-forcer, made sequence-generic by the
patch_pn114_forced_span.py graft via per-seq state["force_seq"]):

  1. Fixed-depth probes (GENESIS_ENABLE_PN114_PROBE=1): at think depths
     PN114_DEPTHS, force ONE combined span "\\nMy current answer: (" +
     free_len HOLE rows (None = unforced; the letter) + ")\\n", then resume
     thinking. Position-anchored end to end — MTP cannot overrun. Stop rule
     (PN114_MODE=enforce): PN114_STABLE_K consecutive probes with the same
     first token AND conf >= PN114_CMIN -> close think (budget := spent +
     PN114_GRACE). shadow = probe + log only (the disturbance-screen config).
  2. R5 confirm-at-fire (GENESIS_PN112_CONFIRM=1): PN112's fire is deferred
     to a probe; confident letter -> commit the cut; weak -> cancel fire and
     cool down. Wired from pn112.observe_state.
  3. R1b wrap-up close (GENESIS_PN112_WRAPUP=1): any think-close goes through
     "\\n<wrapup sentence>\\n</think>" instead of a bare forced </think>.
     Exposed as arm_wrapup(); pn112 calls it at fire.

Token ids come from /tmp/genesis_pn114_ids.json written at boot by
fixes/pn114_boot_ids.py (tokenizer-derived; engine process has no tokenizer).
All logging vllm.* namespaced (genesis.* drops INFO — 07-22 lesson).
Design doc: ~/shared/PN114-DESIGN-probe-stability-20260723.md.
"""
from __future__ import annotations

import json
import os
from typing import Any

from vllm.logger import init_logger

log = init_logger("vllm.genesis.pn114")

_STATE_KEY = "_pn114"
_IDS_PATH = "/tmp/genesis_pn114_ids.json"
_IDS: dict[str, list[int]] | None = None
_STATS = {"probes": 0, "closes": 0, "confirm_ok": 0, "confirm_cancel": 0}
_OBSERVED_ONCE = False
_DEPTH_SNAP_ONCE = False


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


def probes_enabled() -> bool:
    return _env_bool("GENESIS_ENABLE_PN114_PROBE", False)


def wrapup_enabled() -> bool:
    return _env_bool("GENESIS_PN112_WRAPUP", False)


def confirm_enabled() -> bool:
    return _env_bool("GENESIS_PN112_CONFIRM", False)


def any_enabled() -> bool:
    return (probes_enabled() or wrapup_enabled() or confirm_enabled()
            or _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False))


# one-shot import-time diagnostic: shows the env truth in whichever process
# imports this module (graft G imports it before its any_enabled() gate, so
# this line appears even when everything below stays dark).
log.info("PN114: module import (probe=%s wrapup=%s confirm=%s at_cap=%s)",
         probes_enabled(), wrapup_enabled(), confirm_enabled(),
         _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False))


def _ids() -> dict[str, list[int]] | None:
    global _IDS
    if _IDS is None:
        try:
            with open(_IDS_PATH) as f:
                _IDS = json.load(f)
        except Exception:
            log.warning("PN114: ids file missing/unreadable (%s) — disabled",
                        _IDS_PATH)
            _IDS = {}
    return _IDS or None


def _cfg() -> dict[str, Any]:
    depths = []
    for p in (os.environ.get("GENESIS_PN114_DEPTHS", "1024,2048,3072,4096,6144")
              .split(",")):
        try:
            depths.append(int(p))
        except ValueError:
            pass
    return {
        "mode": os.environ.get("GENESIS_PN114_MODE", "shadow").strip().lower(),
        "depths": sorted(depths),
        "stable_k": _env_int("GENESIS_PN114_STABLE_K", 2),
        "cmin": _env_float("GENESIS_PN114_CMIN", 13.5),
        "grace": _env_int("GENESIS_PN114_GRACE", 384),
        # v2 constrained probe: 1 free token (the letter inside "( )")
        "free_len": _env_int("GENESIS_PN114_FREE_LEN", 1),
    }


def _st(state: dict[str, Any]) -> dict[str, Any]:
    st = state.get(_STATE_KEY)
    basis = state.get("start_thinking", -1)
    # ultra-review #13: reset per think block like pn108/pn112 — stale
    # done_depths/probes/cooldown silently disabled block 2+.
    if st is None or st.get("basis") != basis:
        st = {"basis": basis, "phase": None, "done_depths": [], "probes": [],
              "cfg": _cfg(), "saved_budget": None, "reason": None,
              "confirm_cooldown": 0, "free_start": 0, "free_confs": []}
        state[_STATE_KEY] = st
    return st


def _live_think_len(state: dict[str, Any], st: dict[str, Any]) -> int | None:
    """LIVE think depth via pn108's spec-aware slice (BUG-120: the holder's
    state["think_count"] is FROZEN near its init value while a request is
    under budget — the early-return in _update_think_state only walks
    check_count_down — so think_count must never be used for depth or
    accounting). None when not derivable (not mid-think / slice failed).
    think_start_len is stashed in st["tsl"] by observe_state each step."""
    try:
        from vllm._genesis.plateau import pn108 as _pn108
        toks = _pn108._think_token_slice(state, st.get("tsl", 1))
        if toks is not None:
            return len(toks)
    except Exception:
        pass
    return None


def _probe_span(ids: dict[str, list[int]],
                st: dict[str, Any]) -> list[int | None]:
    """The probe as ONE positionally-forced span with a free HOLE for the
    letter: probe text + free_len unforced rows (None = wildcard/free) +
    the ")\\n" close. The close is position-anchored by the holder's span
    walk, so MTP cannot overrun the free window (2026-07-23 canary: the
    old step-granularity free window let the drafter land extra tokens
    before the close armed, producing "(D) 3)")."""
    close = ids.get("close_paren") or ids.get("newline", [])
    return (list(ids.get("probe", [])) + [None] * st["cfg"]["free_len"]
            + list(close))


def phase_active(state: dict[str, Any]) -> bool:
    """True while a forced span / free window is in flight (PN108/PN112 must
    pause their windows — probe tokens would pollute novelty/C)."""
    st = state.get(_STATE_KEY)
    return bool(st and st.get("phase"))


def _arm(state: dict[str, Any], seq: list[int], phase: str,
         req_id: str | None) -> None:
    """Arm the generalized forcer with `seq` starting next step."""
    st = _st(state)
    st["phase"] = phase
    st["free_confs"] = []
    if st["saved_budget"] is None:
        st["saved_budget"] = state.get("thinking_token_budget", -1)
        # keep the budget trigger far away while the span runs
        state["thinking_token_budget"] = 10_000_000
    # Span soundness redesign (ultra-review #2): the holder's B-site walk
    # recomputes emitted progress positionally from force_seq_base against
    # the LANDED output each step — no draft credit, no pre-increment — so
    # index 0 forces intact and no sacrificial prepend is needed.
    state["force_seq"] = list(seq)
    state["force_seq_base"] = len(state.get("output_tok_ids", []))
    state["in_think"] = False
    state["in_end"] = True
    state["end_count"] = 0
    state["bonus_token_forced"] = False
    state["force_index"] = [0]
    log.info("PN114: arm %s (len=%d) req=%s", phase, len(seq), req_id)


def _resume_thinking(state: dict[str, Any], consumed: int) -> None:
    """Return to think mode after a probe; probe tokens don't charge the
    thinking budget (budget grows by what the span consumed)."""
    st = _st(state)
    saved = st.get("saved_budget")
    if saved is not None and saved > 0:
        # min() belt-and-braces (#8): if a detector shrank the budget while
        # the span ran (should be paused, but fail-safe), keep the smaller.
        cur = state.get("thinking_token_budget", saved)
        base = saved if cur >= 1_000_000 else min(saved, cur)
        state["thinking_token_budget"] = base + consumed
    st["saved_budget"] = None
    state["in_end"] = False
    state["in_think"] = True
    state["end_count"] = 0
    state["force_index"] = []
    state["bonus_token_forced"] = False
    # BUG-120: spent = LIVE slice depth (computed after the in_end flip so
    # the slice is valid); frozen think_count only as a last-resort fallback.
    _live = _live_think_len(state, st)
    _spent = _live if _live is not None else state.get("think_count", 0)
    state["check_count_down"] = max(
        1, state["thinking_token_budget"] - _spent
    )


def on_force_complete(state: dict[str, Any]) -> bool:
    """Called by the graft when a forced sequence finishes. Return True to
    SKIP the holder's answer-mode reset (probe phases stay inside think);
    False = let the original completion run (wrap-up / normal close)."""
    st = state.get(_STATE_KEY)
    if not st or not st.get("phase"):
        return False
    phase = st["phase"]
    if isinstance(phase, str) and phase.startswith("pn117"):
        # PN117 deep-band rescue injection: delegate completion (resume think,
        # span uncharged). PN117 owns the detector; pn114 owns the forcer.
        try:
            from vllm._genesis.plateau import pn117_rescue as _pn117
            return _pn117.on_force_complete(state, st)
        except Exception:
            log.warning("PN114: pn117 completion raised — bare resume",
                        exc_info=True)
            fs = state.get("force_seq") or []
            st["phase"] = None
            state["force_seq"] = None
            state.pop("force_seq_base", None)
            _resume_thinking(state, len(fs))
            return True
    if phase == "probe_force":
        # unified probe span (probe + hole + close) fully LANDED: finish.
        # The letter sits right after the probe text — position-derived
        # from force_seq_base, immune to same-step trailing frees.
        ids0 = _ids() or {}
        fs = state.get("force_seq") or []
        base = state.get("force_seq_base")
        probe_len = len(ids0.get("probe", []))
        st["free_start"] = (
            base + probe_len if base is not None
            else max(0, len(state.get("output_tok_ids", []))
                     - len(fs) + probe_len)
        )
        st["phase"] = None
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        _finish_probe(state, st, len(fs))
        return True
    if phase == "wrapup":
        st["phase"] = None
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        # restore the real budget before the answer-mode reset reads it —
        # else a later think block inherits the parked 10M sentinel.
        saved = st.get("saved_budget")
        if saved is not None and saved > 0:
            state["thinking_token_budget"] = saved
        st["saved_budget"] = None
        return False  # normal answer-mode reset proceeds
    return False


def _finish_probe(state: dict[str, Any], st: dict[str, Any],
                  consumed: int) -> None:
    out = state.get("output_tok_ids", [])
    letter = out[st["free_start"]] if st["free_start"] < len(out) else None
    conf = (sum(st["free_confs"]) / len(st["free_confs"])
            if st["free_confs"] else 0.0)
    st["probes"].append({"letter": letter, "conf": round(conf, 3),
                         "family": "confirm" if st.get("reason") == "confirm"
                         else "depth"})
    _STATS["probes"] += 1
    _resume_thinking(state, consumed)
    cfg = st["cfg"]
    reason = st.get("reason")
    st["reason"] = None
    req = st.get("req_id")
    log.info("PN114: probe done reason=%s letter=%s conf=%.2f n=%d req=%s",
             reason, letter, conf, len(st["probes"]), req)
    if reason == "confirm":
        # R5: confident articulated answer -> commit PN112's deferred cut
        if letter is not None and conf >= cfg["cmin"]:
            _STATS["confirm_ok"] += 1
            _close(state, st, "confirm",
                   grace=_env_int("GENESIS_PN112_GRACE", 64))
        else:
            _STATS["confirm_cancel"] += 1
            _live = _live_think_len(state, st)
            st["confirm_cooldown"] = (
                _live if _live is not None
                else state.get("think_count", 0)) + 512
            p112 = state.get("_pn112")
            if isinstance(p112, dict):
                p112["streak"] = 0
                p112["fired"] = False
            log.info("PN114: confirm CANCELLED (conf %.2f < %.2f) req=%s",
                     conf, cfg["cmin"], req)
        return
    # fixed-depth path: stability stop rule
    k = cfg["stable_k"]
    # ultra-review #12: stability judged on depth-family probes only —
    # confirm probes at fire points contaminated the window.
    ps = [p for p in st["probes"] if p.get("family") != "confirm"]
    if (len(ps) >= k
            and len({p["letter"] for p in ps[-k:]}) == 1
            and ps[-1]["letter"] is not None
            and min(p["conf"] for p in ps[-k:]) >= cfg["cmin"]):
        if cfg["mode"] == "enforce":
            _close(state, st, "stable")
        else:
            log.info("PN114: WOULD-CLOSE (shadow, stable x%d letter=%s) req=%s",
                     k, ps[-1]["letter"], req)


def _close(state: dict[str, Any], st: dict[str, Any], why: str,
           grace: int | None = None) -> None:
    _STATS["closes"] += 1
    # ultra-review #6: confirm-committed cuts belong to PN112's config —
    # its grace (64), not PN114's probe grace (384, the killed regression).
    if grace is None:
        grace = st["cfg"]["grace"]
    # BUG-120: close at the LIVE depth, not the frozen think_count (which
    # would set a near-zero budget and guillotine instantly).
    _live = _live_think_len(state, st)
    think = _live if _live is not None else state.get("think_count", 0)
    if not (wrapup_enabled() and arm_wrapup(state, st.get("req_id"))):
        # bare cut (also the fallback when an end-forcing is already active)
        state["thinking_token_budget"] = think + grace
        state["check_count_down"] = grace
    # review R7: a committed close must also re-latch PN112's fired flag —
    # its streak machinery must not re-fire (and re-probe) into the grace
    # window of a close that is already running.
    p112 = state.get("_pn112")
    if isinstance(p112, dict):
        p112["fired"] = True
    log.info("PN114: CLOSE (%s) at think=%d grace=%d req=%s",
             why, think, grace, st.get("req_id"))


def arm_wrapup(state: dict[str, Any], req_id: str | None = None) -> bool:
    """R1b: close the think block through the wrap-up sentence + </think>.
    Caller (pn112 fire / PN114 stop rule) uses this INSTEAD of a bare cut.
    Refuses while an end-forcing is already in flight (arming would mangle
    a half-emitted </think> sequence) — caller falls back to the bare cut."""
    ids = _ids()
    if not ids or not ids.get("wrapup_close"):
        return False
    if state.get("in_end"):
        return False
    _arm(state, ids["wrapup_close"], "wrapup", req_id)
    return True


def request_confirm(state: dict[str, Any], req_id: str | None = None) -> bool:
    """R5: pn112 calls this at fire instead of cutting. Returns True when the
    probe was armed (pn112 must NOT cut; PN114 finishes the decision)."""
    ids = _ids()
    if not ids or not ids.get("probe"):
        return False
    st = _st(state)
    if st.get("phase"):
        return True  # already probing
    # ultra-review #3: never arm onto an active/imminent close — the probe
    # would land AFTER </think>, in the visible answer.
    if state.get("in_end") or not state.get("in_think", True):
        return True  # close in flight; treat as handled (no cut, no probe)
    _live = _live_think_len(state, st)
    _depth = _live if _live is not None else state.get("think_count", 0)
    if _depth < st.get("confirm_cooldown", 0):
        return True  # cooling down; treat as handled (no cut)
    st["reason"] = "confirm"
    st["req_id"] = req_id
    _arm(state, _probe_span(ids, st), "probe_force", req_id)
    return True


def observe_state(state: dict[str, Any], think_start_len: int,
                  seq_idx: int = -1, conf: float | None = None,
                  req_id: str | None = None) -> None:
    """Per tracked request per step, after pn112.observe_state."""
    global _OBSERVED_ONCE
    if not _OBSERVED_ONCE:
        # one-shot liveness/diagnostic line per process (canary debugging:
        # proves the observe seat runs and shows the env truth in-engine)
        _OBSERVED_ONCE = True
        log.info(
            "PN114: observe seat alive (probe=%s wrapup=%s confirm=%s "
            "at_cap=%s ids=%s)",
            probes_enabled(), wrapup_enabled(), confirm_enabled(),
            _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False),
            sorted((_ids() or {}).keys()),
        )
    if not any_enabled():
        return
    if state.get("thinking_token_budget", -1) <= 0:
        return
    st = _st(state)
    st["req_id"] = req_id
    st["tsl"] = think_start_len  # stash for _live_think_len callees
    ids = _ids()
    if not ids:
        return
    phase = st.get("phase")
    if phase == "probe_force" and conf is not None:
        # hole-region conf capture: end_count is the span walk's LANDED
        # position; once it passes the probe text, the letter row is being
        # (or was just) sampled — approximates the old free-window confs.
        if (state.get("end_count", 0) >= len(ids.get("probe", []))
                and len(st["free_confs"]) < st["cfg"]["free_len"] + 2):
            st["free_confs"].append(float(conf))
    if phase:
        return  # force in flight; nothing to do at observe seat
    # Think depth = the spec-aware LIVE slice (pn108, BUG-107d): the holder's
    # state["think_count"] is FROZEN near 0 while a request is under budget
    # (the early-return in _update_think_state only walks check_count_down),
    # so depth-keyed logic on think_count can never fire mid-think — proven
    # by the 2026-07-23 canary (first observe with think>=400 arrived at
    # think=2997, in_end=True). No think_count fallback for the same reason.
    think = _live_think_len(state, st)
    if think is None:
        return  # not mid-think per the slice: nothing to arm onto
    # P-cap (2026-07-23, GENESIS_PN112_WRAPUP_AT_CAP=1): a deep STILL-WORKING
    # request about to hit the hard budget guillotine closes through the
    # wrap-up sentence instead (targets the class PN112 never touches; the
    # trace review says the answer body usually finishes the job — this
    # measures whether a soft landing cuts the relocation cost).
    if _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False):
        budget = state.get("thinking_token_budget", -1)
        # Guards (review R5): never on top of an active end-forcing, never on
        # a budget that a cut already shrank to think+grace (that IS a close
        # in progress — only pre-empt the ORIGINAL grant's guillotine).
        if (budget > 512 + 64 and think >= budget - 512
                and not state.get("in_end")
                and not st.get("wrapup_at_cap_done")):
            st["wrapup_at_cap_done"] = True
            if arm_wrapup(state, req_id):
                log.info("PN114: wrapup-at-cap armed at think=%d budget=%d "
                         "req=%s", think, budget, req_id)
                return
    if not probes_enabled():
        return
    global _DEPTH_SNAP_ONCE
    if (not _DEPTH_SNAP_ONCE and st["cfg"]["depths"]
            and think >= st["cfg"]["depths"][0]):
        # one-shot diagnostic: state truth at the first depth crossing
        _DEPTH_SNAP_ONCE = True
        log.info(
            "PN114: depth-gate snapshot think=%d budget=%s in_think=%s "
            "in_end=%s depths=%s done=%s phase=%s basis=%s",
            think, state.get("thinking_token_budget"),
            state.get("in_think"), state.get("in_end"),
            st["cfg"]["depths"], st["done_depths"], st.get("phase"),
            st.get("basis"),
        )
    # ultra-review #3: no depth-arm during/near a close (bench grants can
    # coincide exactly with depths; think_count freezes during in_end).
    budget = state.get("thinking_token_budget", -1)
    if state.get("in_end") or not state.get("in_think", True):
        return
    if 0 < budget <= think + 128:
        return
    for d in st["cfg"]["depths"]:
        if d in st["done_depths"]:
            continue
        if think >= d:
            st["done_depths"].append(d)
            st["reason"] = "depth%d" % d
            _arm(state, _probe_span(ids, st), "probe_force", req_id)
            return
