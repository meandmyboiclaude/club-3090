"""PN108 — plateau-triggered dynamic thinking cap (house-original, 2026-07-19).

Observation-side successor to the v6a/v6b banner candidates, which were KILLED
−9/−10 net in the 2026-07-19 bench window: instruction-side stop-pushes pulled
items off reasoning they needed (deep-band accuracy 82% → 55/59%). PN108 never
asks the model anything — it watches the think-token stream server-side and,
when the stream has demonstrably STALLED (novelty collapse over consecutive
windows), lowers the request's own ``thinking_token_budget`` to what is already
spent (+grace). The existing ThinkingBudgetStateHolder forcing machinery
(+1e9 think-end bump, spec-aware force_index) then closes the segment exactly
as a static cap would — a proven, MTP-safe path.

Design constraints inherited from the bench data:
- HIGH PRECISION over recall: a fire on a progressing deep item is a v6-style
  regression. Defaults arm only after 2048 think tokens and require 3
  consecutive 256-token windows under a 20% new-trigram floor (≥768 tokens of
  sustained low novelty before any action).
- Fail-open: any exception leaves the request exactly as today.
- Pure CPU/python on the eager ``sample_tokens`` path (outside cudagraph
  capture); cost ≈ a few µs per seat per step.

Gate: ``GENESIS_ENABLE_PN108_PLATEAU_CAP`` (ship-dark). Knobs (env):
``GENESIS_PN108_ARM_AFTER_TOKENS`` (2048), ``GENESIS_PN108_WINDOW_TOKENS``
(256), ``GENESIS_PN108_NOVELTY_FLOOR`` (0.20), ``GENESIS_PN108_CONSEC_WINDOWS``
(3), ``GENESIS_PN108_GRACE_TOKENS`` (0).

Calibration: ~/shared/pn108/CALIBRATION-20260719.md (offline pass over the
v5r2/v6a/v6b run artifacts; thresholds here are the pre-calibration defaults).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.plateau.pn108")

_STATS = {"observed_requests": 0, "windows_scored": 0, "fires": 0}
_STATE_KEY = "pn108"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def is_enabled() -> bool:
    return _env_bool("GENESIS_ENABLE_PN108_PLATEAU_CAP", False)


def _config() -> dict[str, Any]:
    return {
        "arm_after": max(0, _env_int("GENESIS_PN108_ARM_AFTER_TOKENS", 2048)),
        "window": max(32, _env_int("GENESIS_PN108_WINDOW_TOKENS", 256)),
        "floor": min(1.0, max(0.0, _env_float("GENESIS_PN108_NOVELTY_FLOOR", 0.20))),
        "consec": max(1, _env_int("GENESIS_PN108_CONSEC_WINDOWS", 3)),
        "grace": max(0, _env_int("GENESIS_PN108_GRACE_TOKENS", 0)),
    }


class PlateauDetector:
    """Rolling new-trigram-fraction detector over a token-id stream.

    Feed monotonically-growing chunks via ``observe``; returns True once the
    plateau condition has been met (sticky). Pure python, no deps.
    """

    __slots__ = ("seen", "buf", "low_streak", "fired", "scored_tokens", "cfg")

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.seen: set[tuple[int, int, int]] = set()
        self.buf: list[int] = []
        self.low_streak = 0
        self.fired = False
        self.scored_tokens = 0
        self.cfg = cfg or _config()

    def observe(self, new_tokens: list[int]) -> bool:
        if self.fired:
            return True
        if new_tokens:
            self.buf.extend(new_tokens)
        window = self.cfg["window"]
        while len(self.buf) >= window and not self.fired:
            chunk = self.buf[:window]
            del self.buf[:window]
            fresh = 0
            total = 0
            for tri in zip(chunk, chunk[1:], chunk[2:]):
                total += 1
                if tri not in self.seen:
                    fresh += 1
                    self.seen.add(tri)
            self.scored_tokens += window
            _STATS["windows_scored"] += 1
            if self.scored_tokens <= self.cfg["arm_after"]:
                continue  # trigram memory builds, but no verdicts pre-arm
            novelty = (fresh / total) if total else 1.0
            if novelty < self.cfg["floor"]:
                self.low_streak += 1
            else:
                self.low_streak = 0
            if self.low_streak >= self.cfg["consec"]:
                self.fired = True
        return self.fired


def _think_token_slice(state: dict[str, Any], think_start_len: int) -> list[int] | None:
    """Active-think token ids, or None when the request isn't mid-think."""
    if state.get("in_end", False):
        return None
    if state.get("end_thinking", -1) != -1:
        return None
    start = state.get("start_thinking", -1)
    output = state.get("output_tok_ids") or []
    if start >= 0:
        begin = start + think_start_len
        if begin >= len(output):
            return None
        return output[begin:]
    # prompt-side think (continue_thinking): the whole output is think tokens
    if state.get("continue_thinking", False) and state.get("in_think", False):
        return output or None
    return None


def observe_state(state: dict[str, Any], think_start_len: int) -> None:
    """Hook called per tracked request per step, BEFORE _update_think_state.

    On plateau: budget := think_count + grace, countdown := grace — the
    holder's own transition then forces the think-end token(s).
    """
    if not is_enabled():
        return
    if state.get("thinking_token_budget", -1) <= 0:
        return  # untracked / unlimited / already neutralized
    tokens = _think_token_slice(state, think_start_len)
    if tokens is None:
        return
    det = state.get(_STATE_KEY)
    if det is None:
        det = PlateauDetector()
        state[_STATE_KEY] = det
        state[_STATE_KEY + "_fed"] = 0
        _STATS["observed_requests"] += 1
    fed = state.get(_STATE_KEY + "_fed", 0)
    if len(tokens) <= fed:
        return
    fired = det.observe(tokens[fed:])
    state[_STATE_KEY + "_fed"] = len(tokens)
    if fired and not state.get(_STATE_KEY + "_applied", False):
        grace = det.cfg["grace"]
        think_count = int(state.get("think_count", 0) or 0)
        state["thinking_token_budget"] = think_count + grace
        state["check_count_down"] = grace
        state[_STATE_KEY + "_applied"] = True
        _STATS["fires"] += 1
        log.info(
            "PN108: plateau fire — capping think at %d(+%d grace) after %d scored "
            "tokens (streak=%d, window=%d, floor=%.2f)",
            think_count, grace, det.scored_tokens, det.low_streak,
            det.cfg["window"], det.cfg["floor"],
        )


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0
