"""PN117 — deep-band rescue injection (2026-07-23, plan 2.3 / M9).

The deep band (steps-15/25 items) is chronically starved: the model grinds a
low-confidence descent and never commits. PN117 detects that state IN FLIGHT
from the sampling-confidence tap and injects a first-person own-voice
continuation ("plenty of room, work it properly" / "pull it together") as REAL
tokens at a clean sentence boundary — never a forced </think>, never a numeric
banner (untrained numeric uptake is a confirmed negative), never a bracketed
system note (treated as external and argued with). >85% in-trace compliance for
Qwen-family self-talk injections (s12-P17).

Two triggers (either fires at most once per think block):
  1. conf-band: c_mean over the think-token window [WIN_LO, WIN_HI] < CTHRESH
     (offline: corr -0.56 remaining, AUC 0.73 needs-deep) — the low-confidence
     grinder that hasn't settled. Injects GENESIS_PN117_ARM (1|2|3|5).
  2. converge cue (GENESIS_PN117_CONVERGE=1): fires at think_len >=
     CONV_FRAC * thinking_token_budget with no conf condition — the P17
     register applied to the descent ("time to pull it together"). Injects
     arm 5 (converge cue).

Mechanics: PN117 owns ONLY the detector state (_pn117, basis-reset per block).
The forced-span emit, budget-parking and resume are pn114's already-grafted
machinery — PN117 arms via pn114._arm(phase="pn117_inject") and pn114's
on_force_complete delegates the completion back here (resume thinking, span
uncharged). While a span is in flight pn114.phase_active() is True, so
PN108/PN112 pause exactly as they do for probes.

Conf arrives per-step from patch_pn112_conf_tap.py — so PN117 REQUIRES
GENESIS_ENABLE_PN112_SETTLED_STOP=1 (shadow is enough; it is on in the
champion config). Called from pn112.observe_state (pure-python, no new graft).
Arm-text / sentence-end token ids come from the pn114 boot ids file
(fixes/pn114_boot_ids.py, extended for PN117). All logging vllm.* namespaced.

Env knobs (master default OFF; rollback = unset the master env):
  GENESIS_ENABLE_PN117_RESCUE   master gate                    (default off)
  GENESIS_PN117_MODE            shadow | enforce               (default shadow)
  GENESIS_PN117_ARM             conf-band arm text 1|2|3|5     (default 1)
  GENESIS_PN117_CTHRESH         conf-band fire threshold       (default 13.0)
  GENESIS_PN117_WIN_LO          think-token window low         (default 600)
  GENESIS_PN117_WIN_HI          think-token window high        (default 800)
  GENESIS_PN117_MIN_SAMPLES     min confs in window to judge   (default 8)
  GENESIS_PN117_MIN_BUDGET      skip tiny-budget requests      (default 1024)
  GENESIS_PN117_CONVERGE        enable the converge trigger    (default off)
  GENESIS_PN117_CONV_FRAC       converge fraction of budget    (default 0.7)
"""
from __future__ import annotations

import logging
import os
from typing import Any

try:
    from vllm.logger import init_logger
    log = init_logger("vllm.genesis.pn117")
except Exception:  # pragma: no cover
    log = logging.getLogger("vllm.genesis.pn117")

_STATE_KEY = "_pn117"
_STATS = {"observed": 0, "fired_cband": 0, "fired_converge": 0,
          "armed": 0, "landed": 0}
_OBSERVED_ONCE = False


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


def is_enabled() -> bool:
    return _env_bool("GENESIS_ENABLE_PN117_RESCUE", False)


def get_stats() -> dict[str, int]:
    return dict(_STATS)


# sentence-end tokens the boundary check accepts (used only as a fallback set
# of literal chars if the boot ids file omits sentence_end — the real ids are
# tokenizer-derived in the boot writer).
_SENTENCE_CHARS = (".", "\n", "?", "!")


def _cfg() -> dict[str, Any]:
    return {
        "enforce": os.environ.get("GENESIS_PN117_MODE", "shadow")
        .strip().lower() == "enforce",
        "arm": _env_int("GENESIS_PN117_ARM", 1),
        "cthresh": _env_float("GENESIS_PN117_CTHRESH", 13.0),
        "win_lo": _env_int("GENESIS_PN117_WIN_LO", 600),
        "win_hi": _env_int("GENESIS_PN117_WIN_HI", 800),
        "min_samples": _env_int("GENESIS_PN117_MIN_SAMPLES", 8),
        "min_budget": _env_int("GENESIS_PN117_MIN_BUDGET", 1024),
        "converge": _env_bool("GENESIS_PN117_CONVERGE", False),
        "conv_frac": _env_float("GENESIS_PN117_CONV_FRAC", 0.7),
    }


def _st(state: dict[str, Any]) -> dict[str, Any]:
    """Per-seq detector state, reset per think block (like pn112/pn114)."""
    st = state.get(_STATE_KEY)
    basis = state.get("start_thinking", -1)
    if st is None or st.get("basis") != basis:
        st = {"basis": basis, "confs": [], "fired": False, "pending": None,
              "armed": False, "reason": None, "c_mean": None, "req_id": None,
              "cfg": _cfg()}
        state[_STATE_KEY] = st
    return st


def _arm_key(n: int) -> str | None:
    """Map the arm selector to the boot-ids key; 4 = silent-raise control."""
    if n in (1, 2, 3, 5):
        return "arm%d" % n
    return None  # arm 4 (silent raise) or invalid -> no injected text


def _ids():
    """Reuse pn114's boot-ids loader (same /tmp file, single source)."""
    try:
        from vllm._genesis.plateau import pn114 as _pn114
        return _pn114._ids()
    except Exception:
        return None


def _phase_active(state: dict[str, Any]) -> bool:
    try:
        from vllm._genesis.plateau import pn114 as _pn114
        return _pn114.phase_active(state)
    except Exception:
        return False


def _at_sentence_boundary(state: dict[str, Any],
                          ids: dict[str, list[int]]) -> bool:
    """True iff the previously landed token id is in the sentence-end set."""
    out = state.get("output_tok_ids") or []
    if not out:
        return False
    last = out[-1]
    send = ids.get("sentence_end")
    if send:
        return last in send
    return False


def _do_arm(state: dict[str, Any], st: dict[str, Any], key: str) -> None:
    """Arm pn114's forcer with the arm-text ids as ONE forced span (no holes,
    no </think>, no close-paren), phase 'pn117_inject'. pn114 parks the budget
    and its on_force_complete resumes thinking with the span uncharged."""
    ids = _ids() or {}
    seq = ids.get(key)
    if not seq:
        log.warning("PN117: arm text '%s' missing from ids file — skip req=%s",
                    key, st.get("req_id"))
        st["pending"] = None
        return
    # never arm onto an active/imminent close (would land in the answer) or
    # while another span is in flight (pn114.phase_active) — same rule as probes.
    if (_phase_active(state) or state.get("in_end")
            or not state.get("in_think", True)):
        st["pending"] = None  # window closing; drop this injection, safe
        return
    try:
        from vllm._genesis.plateau import pn114 as _pn114
        _pn114._arm(state, list(seq), "pn117_inject", st.get("req_id"))
    except Exception:
        log.warning("PN117: arm raised — skip", exc_info=True)
        st["pending"] = None
        return
    st["pending"] = None
    st["armed"] = True
    _STATS["armed"] += 1
    log.info("PN117: arm %s (len=%d) reason=%s c_mean=%s req=%s",
             key, len(seq), st.get("reason"),
             ("%.2f" % st["c_mean"]) if st.get("c_mean") is not None else "na",
             st.get("req_id"))


def on_force_complete(state: dict[str, Any], st114: dict[str, Any]) -> bool:
    """Called by pn114.on_force_complete for a 'pn117_inject' span. Resume
    thinking with the span uncharged (pn114's saved_budget mechanics). Return
    True so the holder SKIPS the answer-mode reset (we stay inside think)."""
    try:
        from vllm._genesis.plateau import pn114 as _pn114
        fs = state.get("force_seq") or []
        consumed = len(fs)
        st114["phase"] = None
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        _pn114._resume_thinking(state, consumed)
        _STATS["landed"] += 1
        st = state.get(_STATE_KEY) or {}
        log.info("PN117: land+resume consumed=%d budget=%s req=%s",
                 consumed, state.get("thinking_token_budget"),
                 st.get("req_id"))
        return True
    except Exception:
        log.warning("PN117: on_force_complete raised", exc_info=True)
        return False


def _fire(st: dict[str, Any], reason: str, arm_n: int, c_mean: float | None,
          think_len: int, budget: int,
          state: dict[str, Any] | None = None) -> None:
    """Record a fire. shadow = log the would-fire + would-inject arm + c_mean,
    change nothing. enforce = queue the injection for the next boundary."""
    st["fired"] = True
    st["reason"] = reason
    st["c_mean"] = c_mean
    key = _arm_key(arm_n)
    if reason == "converge":
        _STATS["fired_converge"] += 1
    else:
        _STATS["fired_cband"] += 1
    cm = ("%.2f" % c_mean) if c_mean is not None else "na"
    if not st["cfg"]["enforce"]:
        log.info(
            "PN117: WOULD-FIRE (shadow) reason=%s would-inject=%s c_mean=%s "
            "think=%d budget=%d req=%s",
            reason, key or "silent", cm, think_len, budget, st.get("req_id"),
        )
        return
    # [2026-07-23] RAISE_ON_FIRE: grow the enforced budget at fire so the
    # room-cue arms are HONEST (plan 2.3: deep-suspected items need the room
    # the cue promises; without a map the base budget is lean). Applied before
    # any span arms, so pn114 parks the RAISED budget. Arm 4 = raise only
    # (the silent-raise negative control). check_count_down grows by the same
    # delta (holder counts down toward the cap). Never raise on converge fires
    # (the converge cue asks for landing, not room).
    raise_amt = _env_int("GENESIS_PN117_RAISE_ON_FIRE", 0)
    if raise_amt > 0 and state is not None and reason != "converge":
        cur = state.get("thinking_token_budget", 0)
        if isinstance(cur, int) and 0 < cur < 1_000_000:  # not parked/sentinel
            state["thinking_token_budget"] = cur + raise_amt
            ccd = state.get("check_count_down")
            if isinstance(ccd, int):
                state["check_count_down"] = ccd + raise_amt
            log.info("PN117: budget raised %d -> %d (fire) req=%s",
                     cur, cur + raise_amt, st.get("req_id"))
    if key is None:
        # arm 4 (silent raise) enforced: negative control, no injection.
        log.info("PN117: FIRE reason=%s SILENT (arm 4) c_mean=%s think=%d "
                 "req=%s", reason, cm, think_len, st.get("req_id"))
        return
    st["pending"] = key
    log.info(
        "PN117: FIRE reason=%s inject=%s c_mean=%s think=%d budget=%d "
        "(await sentence boundary) req=%s",
        reason, key, cm, think_len, budget, st.get("req_id"),
    )


def observe(state: dict[str, Any], think_len: int, conf: float | None,
            req_id: str | None) -> None:
    """Per tracked request per step, called from pn112.observe_state after its
    think-slice is computed. `think_len` = the LIVE pn108 think-slice length;
    `conf` = this step's C (None = tap absent -> PN117 inert)."""
    global _OBSERVED_ONCE
    if not is_enabled():
        return
    if not _OBSERVED_ONCE:
        _OBSERVED_ONCE = True
        log.info("PN117: observe seat alive (mode=%s arm=%s cthresh=%.1f "
                 "win=[%d,%d] converge=%s ids=%s)",
                 os.environ.get("GENESIS_PN117_MODE", "shadow"),
                 _env_int("GENESIS_PN117_ARM", 1),
                 _env_float("GENESIS_PN117_CTHRESH", 13.0),
                 _env_int("GENESIS_PN117_WIN_LO", 600),
                 _env_int("GENESIS_PN117_WIN_HI", 800),
                 _env_bool("GENESIS_PN117_CONVERGE", False),
                 sorted((_ids() or {}).keys()))
    if state.get("thinking_token_budget", -1) <= 0:
        return
    # pause while any forced span / free window is in flight — our own arm sets
    # pn114's phase, so this also blocks re-entrancy during our injection.
    if _phase_active(state):
        return
    ids = _ids()
    if not ids:
        return
    st = _st(state)
    st["req_id"] = req_id
    cfg = st["cfg"]
    _STATS["observed"] += 1
    # accumulate conf inside/around the detector window (bounded history)
    if conf is not None:
        st["confs"].append((think_len, float(conf)))
        lo_keep = cfg["win_lo"] - 64
        if st["confs"] and st["confs"][0][0] < lo_keep:
            st["confs"] = [(t, c) for (t, c) in st["confs"] if t >= lo_keep]
    # a queued injection lands at the NEXT clean sentence boundary
    if st.get("pending"):
        if _at_sentence_boundary(state, ids):
            _do_arm(state, st, st["pending"])
        return
    if st["fired"]:
        return
    budget = state.get("thinking_token_budget", -1)
    if budget < cfg["min_budget"]:
        return  # tiny-budget request: skip (no deep band to rescue)
    # never fire onto an active/imminent close
    if state.get("in_end") or not state.get("in_think", True):
        return
    # Trigger 2 — converge cue: descent past CONV_FRAC of budget, no conf gate.
    if cfg["converge"] and budget > 0 and think_len >= cfg["conv_frac"] * budget:
        _fire(st, "converge", 5, None, think_len, budget, state)
        # a converge fire may also land this step if we're already on a boundary
        if st.get("pending") and _at_sentence_boundary(state, ids):
            _do_arm(state, st, st["pending"])
        return
    # Trigger 1 — conf-band: live think_len in [WIN_LO, WIN_HI] AND c_mean over
    # the window below CTHRESH.
    if cfg["win_lo"] <= think_len <= cfg["win_hi"]:
        w = [c for (t, c) in st["confs"]
             if cfg["win_lo"] <= t <= cfg["win_hi"]]
        if len(w) >= cfg["min_samples"]:
            c_mean = sum(w) / len(w)
            if c_mean < cfg["cthresh"]:
                _fire(st, "cband", cfg["arm"], c_mean, think_len, budget,
                      state)
                if st.get("pending") and _at_sentence_boundary(state, ids):
                    _do_arm(state, st, st["pending"])
