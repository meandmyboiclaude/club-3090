"""PN112 export — engine→serving sampling-confidence bridge (2026-07-23).

PN112's per-step confidence C (logsumexp − top20 mean over the sampler logits)
is the ONLY signal measured to discriminate a PREMATURE </think> close from a
SETTLED one: on live PN118 fires, wrong items averaged c_mean 9.16 vs 11.37 for
right ones. But C is computed in the EngineCore PROCESS (observe_state runs
there); PN118's close-gate lives in the APIServer PROCESS. A module-level dict
cannot cross that boundary — so this exporter drops the per-request rolling C
mean into a small /tmp file that the serving-side gate reads back, following the
/tmp/genesis_pn114_ids.json precedent (that one is boot-time serving→engine;
this one is runtime engine→serving, so it rotates and stays bounded).

Seat: called from pn112.observe_state at the same pure-python spot pn117.observe
is fed (think_len + conf + req_id in scope, no new graft). Requires the conf tap
(GENESIS_ENABLE_PN112_SETTLED_STOP=1; shadow is enough — the champion config
runs it) so `conf` is non-None, exactly as PN117 does.

File: /tmp/genesis_pn112_conf.json — a single json object
  {req_id: {"c_last": rolling_mean, "n": samples, "ts": monotonic}}
rewritten whole on each update via tmp+os.replace (atomic; no torn reads), kept
to the most-recent GENESIS_PN112EXP_MAX requests by ts (<32 KB). `ts` is
time.monotonic(): on Linux that is CLOCK_MONOTONIC (system-wide, seconds since
boot) so the serving process in the SAME container can age it against its own
monotonic clock — the TTL check the gate uses.

Fail-open everywhere: any exception disables only that write, never anything
else. One INFO line per boot; failures WARNING with exc_info. Inert unless
GENESIS_ENABLE_PN112_EXPORT=1 (default off). Rollback = unset that env.

Env knobs:
  GENESIS_ENABLE_PN112_EXPORT   master gate                      (default off)
  GENESIS_PN112EXP_WIN          rolling conf window, samples     (default 256)
  GENESIS_PN112EXP_EVERY        write cadence, observe steps     (default 32)
  GENESIS_PN112EXP_MAX          max requests kept in file by ts  (default 128)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict, deque
from typing import Any

try:
    from vllm.logger import init_logger
    log = init_logger("vllm._genesis.pn112_export")
except Exception:  # pragma: no cover
    log = logging.getLogger("vllm._genesis.pn112_export")

_PATH = "/tmp/genesis_pn112_conf.json"
# ~64 active rolling windows (per-req deque + counters); evicted LRU beyond it.
_ACTIVE_CAP = 64
# per-req rolling state, LRU-ordered (most-recently-touched at the end)
_WINDOWS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
# serialized snapshot actually written to the file, pruned to MAX by ts
_LAST: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_STATS = {"tracked": 0, "writes": 0, "errors": 0}
_BOOT_LOGGED = False


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


def is_enabled() -> bool:
    return _env_bool("GENESIS_ENABLE_PN112_EXPORT", False)


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def _reset_for_tests() -> None:
    """Clear all module state — offline tests only."""
    _WINDOWS.clear()
    _LAST.clear()
    for k in list(_STATS):
        _STATS[k] = 0
    global _BOOT_LOGGED
    _BOOT_LOGGED = False


def observe(req_id: Any, conf: float | None) -> None:
    """Per tracked request per step, from pn112.observe_state (pn117.observe
    seat). Accumulates `conf` into this request's rolling window and, every
    GENESIS_PN112EXP_EVERY steps, rewrites the export file. Fail-open."""
    if not is_enabled():
        return
    global _BOOT_LOGGED
    if not _BOOT_LOGGED:
        _BOOT_LOGGED = True
        log.info(
            "PN112 export: enabled (win=%d every=%d max=%d path=%s)",
            _env_int("GENESIS_PN112EXP_WIN", 256),
            _env_int("GENESIS_PN112EXP_EVERY", 32),
            _env_int("GENESIS_PN112EXP_MAX", 128), _PATH,
        )
    if req_id is None or conf is None:
        return
    try:
        key = str(req_id)
        win_n = max(1, _env_int("GENESIS_PN112EXP_WIN", 256))
        every = max(1, _env_int("GENESIS_PN112EXP_EVERY", 32))
        d = _WINDOWS.get(key)
        if d is None:
            d = {"win": deque(maxlen=win_n), "step": 0}
            _WINDOWS[key] = d
            _STATS["tracked"] += 1
        else:
            _WINDOWS.move_to_end(key)  # LRU touch
        d["win"].append(float(conf))
        d["step"] += 1
        # bound active windows — drop the least-recently-touched request
        while len(_WINDOWS) > _ACTIVE_CAP:
            _WINDOWS.popitem(last=False)
        if d["step"] % every == 0:
            _write(key, d)
    except Exception:  # pragma: no cover - defensive fail-open
        _STATS["errors"] += 1
        log.warning("PN112 export: observe raised — ignored", exc_info=True)


def _write(key: str, d: dict[str, Any]) -> None:
    """Update this request's serialized entry and atomically rewrite the file,
    keeping only the most-recent GENESIS_PN112EXP_MAX requests by ts."""
    win = d["win"]
    if not win:
        return
    entry = {
        "c_last": round(sum(win) / len(win), 4),
        "n": len(win),
        "ts": round(time.monotonic(), 3),
    }
    _LAST[key] = entry
    _LAST.move_to_end(key)  # most-recently-written at the end
    max_reqs = max(1, _env_int("GENESIS_PN112EXP_MAX", 128))
    # _LAST is write-recency ordered and writes advance monotonically, so
    # newest-by-ts == newest-by-position — pruning the front drops the oldest
    # and is tie-safe when several writes share a rounded ts.
    while len(_LAST) > max_reqs:
        _LAST.popitem(last=False)
    tmp = "%s.tmp.%d" % (_PATH, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_LAST, f)
        os.replace(tmp, _PATH)
        _STATS["writes"] += 1
    except Exception:
        _STATS["errors"] += 1
        log.warning("PN112 export: write failed (%s) — ignored", _PATH,
                    exc_info=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass
