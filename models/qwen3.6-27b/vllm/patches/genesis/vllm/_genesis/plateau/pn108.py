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

OFFLINE CALIBRATION VERDICT (2026-07-19, ~/shared/pn108/CALIBRATION-20260719.md,
454/500 items via Phoenix spans): genuine deep reasoning does NOT plateau —
deep-correct tail novelty median 0.89, 0/102 under the floor; only 2/44 deep
problems trip any threshold, and one (gpqa-127, DNA transcription) is CORRECT
low-novelty content a pure-novelty detector would kill. Consequences baked in
here: (1) fires additionally require PERIODICITY — a repeated 8-gram within
the window (tight loops repeat phrases; transcription/copying does not);
(2) default mode is SHADOW — would-fires are logged and counted, nothing is
mutated; enforce only via env after a live-capture arm proves post-close
accuracy. Expected role: runaway guard (gpqa-131 class), NOT a latency lever
(safe configs save ~223 tok / 100 items in this data).

Known limits (documented, accepted): (1) loops with a period longer than one
window (~256 tokens) never repeat an 8-gram in-window and are not caught —
recall is secondary for a guard; (2) ENFORCE mode inherits the PN75/BUG-028
dependency — think-end forcing under structured-output + MTP requires the
PN75 clamp boot-applied (present in the 134-patch baseline; verify before
enabling enforce on a stripped-down boot).

Gate: ``GENESIS_ENABLE_PN108_PLATEAU_CAP`` (ship-dark). Knobs (env):
``GENESIS_PN108_MODE`` (shadow|enforce, default shadow),
``GENESIS_PN108_ARM_AFTER_TOKENS`` (2048), ``GENESIS_PN108_WINDOW_TOKENS``
(256), ``GENESIS_PN108_NOVELTY_FLOOR`` (0.20), ``GENESIS_PN108_CONSEC_WINDOWS``
(2), ``GENESIS_PN108_REPEAT_MIN`` (3 — max same-8-gram count per window),
``GENESIS_PN108_GRACE_TOKENS`` (0).
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:  # vllm's logger prints INFO in-server; plain root logger may not
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.plateau.pn108")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.plateau.pn108")

_STATS = {"observed_requests": 0, "windows_scored": 0, "fires": 0, "shadow_fires": 0}
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
        "consec": max(1, _env_int("GENESIS_PN108_CONSEC_WINDOWS", 2)),
        "repeat_min": max(2, _env_int("GENESIS_PN108_REPEAT_MIN", 3)),
        "grace": max(0, _env_int("GENESIS_PN108_GRACE_TOKENS", 0)),
        "enforce": (
            os.environ.get("GENESIS_PN108_MODE", "shadow").strip().lower()
            == "enforce"
        ),
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
            # Periodicity gate (calibration 07-19): tight loops repeat the
            # SAME 8-gram many times inside one window; low-novelty-but-
            # legitimate content (sequence transcription, tables) does not.
            gram_counts: dict[tuple[int, ...], int] = {}
            max_rep = 0
            for i in range(0, max(0, len(chunk) - 7)):
                g = tuple(chunk[i : i + 8])
                c = gram_counts.get(g, 0) + 1
                gram_counts[g] = c
                if c > max_rep:
                    max_rep = c
            self.scored_tokens += window
            _STATS["windows_scored"] += 1
            if self.scored_tokens <= self.cfg["arm_after"]:
                continue  # trigram memory builds, but no verdicts pre-arm
            novelty = (fresh / total) if total else 1.0
            if novelty < self.cfg["floor"] and max_rep >= self.cfg["repeat_min"]:
                self.low_streak += 1
            else:
                self.low_streak = 0
            if self.low_streak >= self.cfg["consec"]:
                self.fired = True
        return self.fired


def _think_token_slice(state: dict[str, Any], think_start_len: int) -> list[int] | None:
    """Active-think token ids, or None when the request isn't mid-think.

    BUG-107d (2026-07-19 shadow window): when the chat template opens
    ``<think>`` in the PROMPT (this deployment's template does, line 147),
    ``start_thinking`` is a PROMPT-space index while ``output_tok_ids`` is
    output-space — slicing output with it always overflowed and returned
    None, leaving the detector structurally inert. The continue_thinking
    branch must therefore be checked FIRST, not gated behind ``start < 0``:
    in that mode every output token is a think token until the end sequence.
    """
    if state.get("in_end", False):
        return None
    if state.get("end_thinking", -1) != -1:
        return None
    output = state.get("output_tok_ids") or []
    # prompt-side think: start_thinking is prompt-space — never slice output
    # with it. The whole output is think tokens while the block stays open.
    if state.get("continue_thinking", False):
        if not state.get("in_think", False):
            return None
        return output or None
    start = state.get("start_thinking", -1)
    if start >= 0:
        begin = start + think_start_len
        if begin >= len(output):
            return None
        return output[begin:]
    return None


def observe_state(
    state: dict[str, Any],
    think_start_len: int,
    seq_idx: int = -1,
    req_id: str | None = None,
) -> None:
    """Hook called per tracked request per step, BEFORE _update_think_state.

    On plateau: budget := OBSERVED think length + grace, countdown := grace —
    the holder's own transition then forces the think-end token(s).
    Enforce-blocker fixes (truemean window, 07-20): (1) the cap basis is
    len(slice) — the detector's own ground truth — NOT state["think_count"],
    which reads ~5 on the prompt-opened-think path (107d index-space family)
    and would have guillotined instantly; (2) fires carry seq_idx AND req_id
    (07-22: index->req_id map handed over at the sync_batch call site) so
    telemetry joins to per-request outcomes.
    """
    if not is_enabled():
        return
    if state.get("thinking_token_budget", -1) <= 0:
        return  # untracked / unlimited / already neutralized
    tokens = _think_token_slice(state, think_start_len)
    if tokens is None:
        return
    # Slice basis = the absolute think-start index this slice is cut from.
    # A request can close one think block and open another; the old fed
    # counter would then exceed the fresh (shorter) slice and silence the
    # detector for every later block. New basis -> fresh detector per block.
    basis = state.get("start_thinking", -1)
    det = state.get(_STATE_KEY)
    if det is None or state.get(_STATE_KEY + "_basis") != basis:
        det = PlateauDetector()
        state[_STATE_KEY] = det
        state[_STATE_KEY + "_fed"] = 0
        state[_STATE_KEY + "_basis"] = basis
        state.pop(_STATE_KEY + "_applied", None)  # one verdict per block
        _STATS["observed_requests"] += 1
        # Execution proof: exactly one line per observed think block. Without
        # this, "0 fires" is indistinguishable from "hook never ran" — the
        # periodicity gate makes normal reasoning UNABLE to fire at any env
        # setting (that's the precision property), so fires can't serve as
        # the liveness signal. Found by the 07-19 shadow window's canary.
        log.info(
            "PN108: observing think block (budget=%d, basis=%d, mode=%s, req=%s)",
            state.get("thinking_token_budget", -1), basis,
            "enforce" if det.cfg["enforce"] else "shadow", req_id,
        )
    fed = state.get(_STATE_KEY + "_fed", 0)
    if len(tokens) <= fed:
        return
    fired = det.observe(tokens[fed:])
    state[_STATE_KEY + "_fed"] = len(tokens)
    if fired and not state.get(_STATE_KEY + "_applied", False):
        grace = det.cfg["grace"]
        think_len = len(tokens)  # observed think length — valid on BOTH paths
        state[_STATE_KEY + "_applied"] = True
        if det.cfg["enforce"]:
            state["thinking_token_budget"] = think_len + grace
            state["check_count_down"] = grace
            _STATS["fires"] += 1
            log.info(
                "PN108: plateau ENFORCE fire — seq=%d req=%s budget=%s capping "
                "think at %d(+%d grace) after %d scored tokens (window=%d, "
                "floor=%.2f, repeat_min=%d)",
                seq_idx, req_id, state.get("thinking_token_budget"), think_len,
                grace, det.scored_tokens, det.cfg["window"], det.cfg["floor"],
                det.cfg["repeat_min"],
            )
        else:
            _STATS["shadow_fires"] += 1
            log.info(
                "PN108: plateau SHADOW fire (no action) — seq=%d req=%s "
                "budget=%s would cap at %d after %d scored tokens (window=%d, "
                "floor=%.2f, repeat_min=%d)",
                seq_idx, req_id, state.get("thinking_token_budget"), think_len,
                det.scored_tokens, det.cfg["window"], det.cfg["floor"],
                det.cfg["repeat_min"],
            )


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0
