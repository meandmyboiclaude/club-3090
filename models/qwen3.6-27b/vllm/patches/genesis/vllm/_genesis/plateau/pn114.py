"""PN114 — forced-span probe machinery (2026-07-23).

Three env-gated mechanisms sharing one generalized forcing machine (the
holder's </think> end-forcer, made sequence-generic by the
patch_pn114_forced_span.py graft via per-seq state["force_seq"]):

  1. Fixed-depth probes (GENESIS_ENABLE_PN114_PROBE=1): at think depths
     PN114_DEPTHS, force "\\nMy current answer: ", free-sample 3 tokens
     (letter capture + conf), force "\\n", resume thinking. Stop rule
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
        "free_len": 3,
    }


def _st(state: dict[str, Any]) -> dict[str, Any]:
    st = state.get(_STATE_KEY)
    if st is None:
        st = {"phase": None, "done_depths": [], "probes": [], "cfg": _cfg(),
              "saved_budget": None, "reason": None, "confirm_cooldown": 0,
              "free_start": 0, "free_confs": []}
        state[_STATE_KEY] = st
    return st


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
    if st["saved_budget"] is None:
        st["saved_budget"] = state.get("thinking_token_budget", -1)
        # keep the budget trigger far away while the span runs
        state["thinking_token_budget"] = 10_000_000
    state["force_seq"] = list(seq)
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
        state["thinking_token_budget"] = saved + consumed
    st["saved_budget"] = None
    state["in_end"] = False
    state["in_think"] = True
    state["end_count"] = 0
    state["force_index"] = []
    state["bonus_token_forced"] = False
    state["check_count_down"] = max(
        1, state["thinking_token_budget"] - state.get("think_count", 0)
    )


def on_force_complete(state: dict[str, Any]) -> bool:
    """Called by the graft when a forced sequence finishes. Return True to
    SKIP the holder's answer-mode reset (probe phases stay inside think);
    False = let the original completion run (wrap-up / normal close)."""
    st = state.get(_STATE_KEY)
    if not st or not st.get("phase"):
        return False
    phase = st["phase"]
    if phase == "probe_force":
        st["phase"] = "probe_free"
        st["free_start"] = len(state.get("output_tok_ids", []))
        st["free_confs"] = []
        state["in_end"] = False
        state["in_think"] = True
        state["end_count"] = 0
        state["force_index"] = []
        state["force_seq"] = None
        return True
    if phase == "probe_nl":
        ids = _ids() or {}
        consumed = (len(ids.get("probe", [])) + st["cfg"]["free_len"]
                    + len(ids.get("newline", [1])))
        st["phase"] = None
        state["force_seq"] = None
        _finish_probe(state, st, consumed)
        return True
    if phase == "wrapup":
        st["phase"] = None
        state["force_seq"] = None
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
    st["probes"].append({"letter": letter, "conf": round(conf, 3)})
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
            _close(state, st, "confirm")
        else:
            _STATS["confirm_cancel"] += 1
            st["confirm_cooldown"] = state.get("think_count", 0) + 512
            p112 = state.get("_pn112")
            if isinstance(p112, dict):
                p112["streak"] = 0
                p112["fired"] = False
            log.info("PN114: confirm CANCELLED (conf %.2f < %.2f) req=%s",
                     conf, cfg["cmin"], req)
        return
    # fixed-depth path: stability stop rule
    k = cfg["stable_k"]
    ps = st["probes"]
    if (len(ps) >= k
            and len({p["letter"] for p in ps[-k:]}) == 1
            and ps[-1]["letter"] is not None
            and min(p["conf"] for p in ps[-k:]) >= cfg["cmin"]):
        if cfg["mode"] == "enforce":
            _close(state, st, "stable")
        else:
            log.info("PN114: WOULD-CLOSE (shadow, stable x%d letter=%s) req=%s",
                     k, ps[-1]["letter"], req)


def _close(state: dict[str, Any], st: dict[str, Any], why: str) -> None:
    _STATS["closes"] += 1
    grace = st["cfg"]["grace"]
    think = state.get("think_count", 0)
    if wrapup_enabled():
        arm_wrapup(state, st.get("req_id"))
    else:
        state["thinking_token_budget"] = think + grace
        state["check_count_down"] = grace
    log.info("PN114: CLOSE (%s) at think=%d grace=%d req=%s",
             why, think, grace, st.get("req_id"))


def arm_wrapup(state: dict[str, Any], req_id: str | None = None) -> bool:
    """R1b: close the think block through the wrap-up sentence + </think>.
    Caller (pn112 fire / PN114 stop rule) uses this INSTEAD of a bare cut."""
    ids = _ids()
    if not ids or not ids.get("wrapup_close"):
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
    if state.get("think_count", 0) < st.get("confirm_cooldown", 0):
        return True  # cooling down; treat as handled (no cut)
    st["reason"] = "confirm"
    st["req_id"] = req_id
    _arm(state, ids["probe"], "probe_force", req_id)
    return True


def observe_state(state: dict[str, Any], think_start_len: int,
                  seq_idx: int = -1, conf: float | None = None,
                  req_id: str | None = None) -> None:
    """Per tracked request per step, after pn112.observe_state."""
    if not any_enabled():
        return
    if state.get("thinking_token_budget", -1) <= 0:
        return
    st = _st(state)
    st["req_id"] = req_id
    ids = _ids()
    if not ids:
        return
    phase = st.get("phase")
    if phase == "probe_free":
        if conf is not None:
            st["free_confs"].append(float(conf))
        out_len = len(state.get("output_tok_ids", []))
        if out_len - st["free_start"] >= st["cfg"]["free_len"]:
            _arm(state, ids.get("newline", []), "probe_nl", req_id)
            st["phase"] = "probe_nl"
        return
    if phase:
        return  # force in flight; nothing to do at observe seat
    # P-cap (2026-07-23, GENESIS_PN112_WRAPUP_AT_CAP=1): a deep STILL-WORKING
    # request about to hit the hard budget guillotine closes through the
    # wrap-up sentence instead (targets the class PN112 never touches; the
    # trace review says the answer body usually finishes the job — this
    # measures whether a soft landing cuts the relocation cost).
    if _env_bool("GENESIS_PN112_WRAPUP_AT_CAP", False):
        think = state.get("think_count", 0)
        budget = state.get("thinking_token_budget", -1)
        if (budget > 0 and think >= budget - 512
                and not st.get("wrapup_at_cap_done")):
            st["wrapup_at_cap_done"] = True
            if arm_wrapup(state, req_id):
                log.info("PN114: wrapup-at-cap armed at think=%d budget=%d "
                         "req=%s", think, budget, req_id)
                return
    if not probes_enabled():
        return
    think = state.get("think_count", 0)
    for d in st["cfg"]["depths"]:
        if d in st["done_depths"]:
            continue
        if think >= d:
            st["done_depths"].append(d)
            st["reason"] = "depth%d" % d
            _arm(state, ids["probe"], "probe_force", req_id)
            return
