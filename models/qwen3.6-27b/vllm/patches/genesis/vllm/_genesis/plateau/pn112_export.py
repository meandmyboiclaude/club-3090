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
  {req_id: {"c_last": rolling_mean, "c_trace": whole_trace_mean,
            "c_slope": recent_half - older_half, "n": samples,
            "ts": monotonic, "final": true (flush-on-close records only)}}
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
            d = {"win": deque(maxlen=win_n), "step": 0,
                 "trace_sum": 0.0, "trace_n": 0}
            _WINDOWS[key] = d
            _STATS["tracked"] += 1
        else:
            _WINDOWS.move_to_end(key)  # LRU touch
        c = float(conf)
        d["win"].append(c)
        d["trace_sum"] += c          # whole-trace running sum (for c_trace)
        d["trace_n"] += 1
        d["step"] += 1
        # bound active windows — drop the least-recently-touched request
        while len(_WINDOWS) > _ACTIVE_CAP:
            _WINDOWS.popitem(last=False)
        if d["step"] % every == 0:
            _write(key, d)
    except Exception:  # pragma: no cover - defensive fail-open
        _STATS["errors"] += 1
        log.warning("PN112 export: observe raised — ignored", exc_info=True)


def _entry(d: dict[str, Any], final: bool = False) -> dict[str, Any] | None:
    """Build a request's serialized record. Fields the serving-side gate reads:
      c_last  — rolling-window mean (the calibrated premature-vs-settled signal)
      c_trace — whole-trace mean (a settled trace rises into its close; the
                level alone can't tell a descending 12→9.8 from a flat 9.8)
      c_slope — recent-half mean minus older-half mean of the window (trend sign)
      n       — samples in the rolling window
      final   — set only on a flush-at-</think> record (the last window is the
                most diagnostic one; lets the gate drop MINN for flushed entries)
    """
    win = d["win"]
    if not win:
        return None
    lst = list(win)
    n = len(lst)
    c_last = sum(lst) / n
    half = n // 2
    if half >= 1 and (n - half) >= 1:
        older = sum(lst[:half]) / half
        recent = sum(lst[half:]) / (n - half)
        c_slope = recent - older
    else:
        c_slope = 0.0
    trace_n = d.get("trace_n", 0) or n
    trace_sum = d.get("trace_sum", 0.0) or sum(lst)
    c_trace = trace_sum / trace_n if trace_n else c_last
    entry = {
        "c_last": round(c_last, 4),
        "c_trace": round(c_trace, 4),
        "c_slope": round(c_slope, 4),
        "n": n,
        "ts": round(time.monotonic(), 3),
    }
    if final:
        entry["final"] = True
    return entry


def flush(req_id: Any) -> None:
    """Flush-on-close hook (2026-07-23, Fable R1): the holder knows when
    </think> lands — call this THEN to write the final window unconditionally,
    regardless of the EVERY cadence. Without it the shortest (most premature)
    closes — those that end before one write cadence (EVERY=32) or under MINN —
    export nothing or a stale value, and the gate is blind exactly on its target
    class. Idempotent (drops the active window so a repeat call no-ops) and
    fail-open. Inert unless the exporter is enabled."""
    if not is_enabled() or req_id is None:
        return
    try:
        key = str(req_id)
        d = _WINDOWS.get(key)
        if d is None:
            return  # never tracked, or already flushed
        _write(key, d, final=True)
        _WINDOWS.pop(key, None)  # close: repeat flushes are inert
    except Exception:  # pragma: no cover - defensive fail-open
        _STATS["errors"] += 1
        log.warning("PN112 export: flush raised — ignored", exc_info=True)


def _write(key: str, d: dict[str, Any], final: bool = False) -> None:
    """Update this request's serialized entry and atomically rewrite the file,
    keeping only the most-recent GENESIS_PN112EXP_MAX requests by ts."""
    entry = _entry(d, final=final)
    if entry is None:
        return
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
