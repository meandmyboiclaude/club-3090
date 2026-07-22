"""PN112 — settled-stop: end thinking once the answer is SETTLED (2026-07-22).

Purpose (USER): "stop the time spent past the moment it already has completed."
Qwen doesn't track time and won't stop itself; the v5 banner asks it to stop
when settled, but the answer is typically computed at ~30% of the chain
(termination-circuit / report-9) and the model keeps verifying. PN112 detects
the settled state SERVER-SIDE from sampling confidence and closes the think
block through the holder's normal force machinery.

Signal (validated 2026-07-22 doom-separation study, 16 k260 traces):
windowed C = mean(logsumexp(logits) - top20_logits) — big C = peaked/confident.
RIGHT (productive) traces run HIGH-C once the answer is reached; doomed
grinders never rise (mean_win AUC 0.92 inverted @800 tok). Consequence: a
completion detector naturally NEVER cuts a still-working or grinding trace
(they don't settle) — it only trims the post-answer verification tail. Safe by
construction on easy prod traffic (they settle fast and stop on their own; the
floor keeps us out of their way).

Seat: called from the ThinkingBudgetStateHolder update loop (same graft family
as PN108, sample_tokens eager phase — outside cudagraph). Confidence is tapped
in apply_to_logits by fixes/patch_pn112_conf_tap.py (logits are in scope there;
topk+logsumexp on <=16 rows, no float() copies, one small .tolist() sync).

Modes: shadow (log would-fire + calibration checkpoints, change nothing) /
enforce (cap budget to spent+grace like PN108; holder forces </think>).
Fail-open everywhere; inert unless GENESIS_ENABLE_PN112_SETTLED_STOP=1.

Env knobs:
  GENESIS_ENABLE_PN112_SETTLED_STOP  master gate (default off)
  GENESIS_PN112_MODE                 shadow | enforce   (default shadow)
  GENESIS_PN112_TAU                  settled C threshold      (default 13.0)
  GENESIS_PN112_WIN                  window, think tokens     (default 256)
  GENESIS_PN112_K                    consecutive good checks  (default 3)
  GENESIS_PN112_EVAL_EVERY           tokens between checks    (default 64)
  GENESIS_PN112_FLOOR                min think tokens first   (default 600)
  GENESIS_PN112_GRACE                grace tokens after fire  (default 64)
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from vllm.logger import init_logger
    log = init_logger("vllm._genesis.pn112")
except Exception:  # pragma: no cover
    log = logging.getLogger("vllm._genesis.pn112")

_STATE_KEY = "_pn112"
_STATS = {"observed_requests": 0, "settled": 0, "enforced": 0}


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
    return _env_bool("GENESIS_ENABLE_PN112_SETTLED_STOP", False)


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def _cfg() -> dict[str, Any]:
    return {
        "enforce": os.environ.get("GENESIS_PN112_MODE", "shadow").strip().lower()
        == "enforce",
        "tau": _env_float("GENESIS_PN112_TAU", 13.0),
        "win": _env_int("GENESIS_PN112_WIN", 256),
        "k": _env_int("GENESIS_PN112_K", 3),
        "eval_every": _env_int("GENESIS_PN112_EVAL_EVERY", 64),
        "floor": _env_int("GENESIS_PN112_FLOOR", 600),
        "grace": _env_int("GENESIS_PN112_GRACE", 64),
    }


# Calibration checkpoints logged in BOTH modes (cheap, one line each): the
# shadow screen reads these to pick TAU without any offline replay.
_CHECKPOINTS = (512, 1024, 2048, 4096, 8192)


def observe_state(
    state: dict[str, Any],
    think_start_len: int,
    seq_idx: int = -1,
    conf: float | None = None,
    req_id: str | None = None,
) -> None:
    """Per tracked request per step, after PN108's observe, before
    _update_think_state. `conf` = this step's C for the seq (None = no tap)."""
    if not is_enabled():
        return
    if state.get("thinking_token_budget", -1) <= 0:
        return
    # think-token slice length — reuse PN108's spec-aware slicing.
    try:
        from vllm._genesis.plateau import pn108 as _pn108
        tokens = _pn108._think_token_slice(state, think_start_len)
    except Exception:
        return
    if tokens is None:
        return
    think_len = len(tokens)
    basis = state.get("start_thinking", -1)
    st = state.get(_STATE_KEY)
    if st is None or st.get("basis") != basis:
        st = {
            "basis": basis,
            "confs": [],          # list[(think_len, C)] — bounded by trimming
            "streak": 0,
            "fired": False,
            "last_eval": 0,
            "next_ckpt": 0,
            "cfg": _cfg(),
        }
        state[_STATE_KEY] = st
        _STATS["observed_requests"] += 1
        log.info(
            "PN112: observing think block (budget=%d, basis=%d, mode=%s, req=%s)",
            state.get("thinking_token_budget", -1), basis,
            "enforce" if st["cfg"]["enforce"] else "shadow", req_id,
        )
    if st["fired"]:
        return
    cfg = st["cfg"]
    if conf is not None:
        st["confs"].append((think_len, float(conf)))
        # trim to ~2 windows of history
        cut = think_len - 2 * cfg["win"]
        if cut > 0 and st["confs"] and st["confs"][0][0] < cut:
            st["confs"] = [(t, c) for (t, c) in st["confs"] if t >= cut]
    # calibration checkpoint logging (shadow AND enforce; one line per level)
    while st["next_ckpt"] < len(_CHECKPOINTS) and think_len >= _CHECKPOINTS[st["next_ckpt"]]:
        w = [c for (t, c) in st["confs"] if t > think_len - cfg["win"]]
        # tokstep = mean think-tokens per sampler step in the window = MTP
        # accepted+1 (2026-07-23): free acceptance-rate telemetry — candidate
        # settled/doom side-signal (predictable text = higher acceptance).
        ts = [t for (t, c) in st["confs"] if t > think_len - cfg["win"]]
        deltas = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        tokstep = (sum(deltas) / len(deltas)) if deltas else 0.0
        log.info(
            "PN112: ckpt req=%s think=%d wmean=%s n=%d streak=%d tokstep=%.2f",
            req_id, _CHECKPOINTS[st["next_ckpt"]],
            f"{sum(w)/len(w):.2f}" if w else "na", len(w), st["streak"],
            tokstep,
        )
        st["next_ckpt"] += 1
    if think_len < cfg["floor"]:
        return
    if think_len - st["last_eval"] < cfg["eval_every"]:
        return
    st["last_eval"] = think_len
    w = [c for (t, c) in st["confs"] if t > think_len - cfg["win"]]
    if len(w) < 8:  # not enough samples in window (spec accepts several/step)
        return
    wmean = sum(w) / len(w)
    if wmean >= cfg["tau"]:
        st["streak"] += 1
    else:
        st["streak"] = 0
        return
    if st["streak"] < cfg["k"]:
        return
    # Degeneracy AND-gate (2026-07-23, hardening §4.4): repetition loops fake
    # high C — a fire additionally requires the recent window to be non-
    # degenerate (type-token ratio > 0.2, LoopGuard default). Degenerate
    # window ⇒ reset the streak and let PN108's plateau logic own the case.
    try:
        _tail = tokens[-cfg["win"]:]
        if _tail and len(set(_tail)) <= max(8, int(0.2 * len(_tail))):
            st["streak"] = 0
            log.info(
                "PN112: fire suppressed (degenerate window, ttr<=0.2) "
                "req=%s think=%d", req_id, think_len,
            )
            return
    except Exception:
        pass
    st["fired"] = True
    _STATS["settled"] += 1
    budget = state.get("thinking_token_budget", -1)
    new_budget = think_len + cfg["grace"]
    if not cfg["enforce"]:
        log.info(
            "PN112: SETTLED (shadow) req=%s at think=%d wmean=%.2f budget=%d "
            "(would cap to %d, saving %d)",
            req_id, think_len, wmean, budget, new_budget,
            max(0, budget - new_budget),
        )
        return
    if new_budget < budget:
        # Mirror PN108's enforce mechanics EXACTLY: budget := observed+grace
        # AND countdown := grace (think_count is unreliable on the prompt-
        # opened-think path — without the countdown the cap binds late/never).
        state["thinking_token_budget"] = new_budget
        state["check_count_down"] = cfg["grace"]
        _STATS["enforced"] += 1
        log.info(
            "PN112: SETTLED — cap %d -> %d at think=%d wmean=%.2f req=%s",
            budget, new_budget, think_len, wmean, req_id,
        )
