#!/usr/bin/env python3
"""Offline tests for PN112 export — engine→serving confidence bridge.

The cross-process file bridge itself (EngineCore writes, APIServer reads) needs
a running container, but everything that could waste a bench window is pure
Python and tested here:
  - rolling window mean + sample count (n)
  - write cadence (only every GENESIS_PN112EXP_EVERY observe steps)
  - atomic whole-file rewrite: valid json, keyed by req_id, right fields
  - rotation: file keeps only the most-recent GENESIS_PN112EXP_MAX by ts
  - active-window eviction (LRU) beyond the ~64 cap
  - disabled master → nothing written
  - fail-open: conf None / req_id None / unwritable path never raise

Run: python3 test_pn112_export_logic.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

MOD_DIR = (Path(__file__).resolve().parents[1]
           / "models" / "qwen3.6-27b" / "vllm" / "patches" / "genesis"
           / "vllm" / "_genesis" / "plateau")
sys.path.insert(0, str(MOD_DIR))

import pn112_export as ex  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def reset(tmpdir, **env):
    for k in list(os.environ):
        if k.startswith("GENESIS_PN112EXP") or k == "GENESIS_ENABLE_PN112_EXPORT":
            del os.environ[k]
    ex._reset_for_tests()
    path = os.path.join(tmpdir, "genesis_pn112_conf.json")
    ex._PATH = path
    os.environ["GENESIS_ENABLE_PN112_EXPORT"] = "1"
    for k, v in env.items():
        os.environ["GENESIS_PN112EXP_" + k] = str(v)
    return path


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── disabled master ─────────────────────────────────────────────────────────


def test_disabled():
    print("\ndisabled master → no window, no file")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "genesis_pn112_conf.json")
        ex._reset_for_tests()
        ex._PATH = path
        os.environ.pop("GENESIS_ENABLE_PN112_EXPORT", None)
        for i in range(200):
            ex.observe("chatcmpl-x", 11.0)
        check("no file written when disabled", not os.path.exists(path))
        check("no windows tracked", ex.get_stats()["tracked"] == 0)


# ─── rolling mean + n + cadence ──────────────────────────────────────────────


def test_rolling_mean_and_cadence():
    print("\nrolling mean + sample count + write cadence")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=32, MAX=128)
        rid = "chatcmpl-aaa"
        # first 31 observes: below cadence, no file yet
        for _ in range(31):
            ex.observe(rid, 10.0)
        check("no write before EVERY steps", not os.path.exists(path))
        ex.observe(rid, 10.0)  # 32nd → write
        check("write at 32nd step", os.path.exists(path))
        data = read_file(path)
        check("keyed by raw req_id", rid in data, str(list(data)))
        check("c_last = rolling mean (10.0)", abs(data[rid]["c_last"] - 10.0) < 1e-6)
        check("n = 32 samples", data[rid]["n"] == 32, str(data[rid]["n"]))
        check("ts present + monotonic-shaped", isinstance(data[rid]["ts"], (int, float)))
        check("write count == 1", ex.get_stats()["writes"] == 1)


def test_window_bounds_mean():
    print("\nwindow cap: mean over last WIN samples only")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=64, EVERY=64, MAX=128)
        rid = "chatcmpl-win"
        # 64 low, then 64 high — after 128 steps the window holds only the highs
        for _ in range(64):
            ex.observe(rid, 5.0)
        for _ in range(64):
            ex.observe(rid, 15.0)
        data = read_file(path)
        check("n capped at WIN=64", data[rid]["n"] == 64, str(data[rid]["n"]))
        check("mean is the last-64 window (15.0)", abs(data[rid]["c_last"] - 15.0) < 1e-6,
              str(data[rid]["c_last"]))


# ─── atomic rewrite: whole file, single object ───────────────────────────────


def test_atomic_single_object():
    print("\nfile is a single json object over all tracked requests")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=4, MAX=128)
        for rid in ("chatcmpl-a", "chatcmpl-b", "chatcmpl-c"):
            for _ in range(4):
                ex.observe(rid, 12.0)
        data = read_file(path)
        check("single dict with all reqs", set(data) == {"chatcmpl-a", "chatcmpl-b", "chatcmpl-c"},
              str(list(data)))
        check("no .tmp left behind",
              not any(p.name.startswith("genesis_pn112_conf.json.tmp")
                      for p in Path(td).iterdir()))


# ─── rotation: keep only MAX by ts ───────────────────────────────────────────


def test_rotation_keeps_recent_max():
    print("\nrotation: file keeps only MAX most-recent requests by ts")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=8, EVERY=8, MAX=4)
        # 10 distinct requests, each written once; MAX=4 → only last 4 kept.
        # active-window LRU (64) does not evict here (< 64 reqs); the FILE prune
        # is what bounds it. Each req writes once (8 observes @ EVERY=8).
        for i in range(10):
            for _ in range(8):
                ex.observe("chatcmpl-%02d" % i, float(i))
        data = read_file(path)
        check("file holds exactly MAX=4 entries", len(data) == 4, str(list(data)))
        check("keeps the 4 most-recent (06..09)",
              set(data) == {"chatcmpl-06", "chatcmpl-07", "chatcmpl-08", "chatcmpl-09"},
              str(sorted(data)))


# ─── active-window LRU eviction ──────────────────────────────────────────────


def test_active_window_lru_evicts():
    print("\nactive windows bounded at ~64 (LRU eviction)")
    with tempfile.TemporaryDirectory() as td:
        reset(td, WIN=256, EVERY=1000, MAX=128)  # EVERY high → no writes, just windows
        for i in range(200):
            ex.observe("chatcmpl-%03d" % i, 10.0)
        check("active windows capped at 64", len(ex._WINDOWS) == 64, str(len(ex._WINDOWS)))
        # the most-recent 64 keys survive
        survivors = set(ex._WINDOWS)
        check("survivors are the most-recent 64",
              survivors == {"chatcmpl-%03d" % i for i in range(136, 200)},
              str(sorted(survivors))[:80])


# ─── fail-open paths ─────────────────────────────────────────────────────────


def test_failopen_none_inputs():
    print("\nfail-open: None req_id / None conf never raise, never write")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=1, MAX=128)
        ex.observe(None, 10.0)
        ex.observe("chatcmpl-z", None)
        check("no file for None inputs", not os.path.exists(path))
        check("no errors counted", ex.get_stats()["errors"] == 0, str(ex.get_stats()))
        # a valid observe still works afterward
        ex.observe("chatcmpl-z", 10.0)
        check("valid observe writes", os.path.exists(path))


def test_failopen_unwritable_path():
    print("\nfail-open: unwritable path → error counted, no raise")
    with tempfile.TemporaryDirectory() as td:
        reset(td, WIN=256, EVERY=1, MAX=128)
        ex._PATH = os.path.join(td, "no_such_dir", "conf.json")  # parent missing
        try:
            for _ in range(3):
                ex.observe("chatcmpl-bad", 10.0)
            raised = False
        except Exception:
            raised = True
        check("observe never raises on bad path", not raised)
        check("error counted", ex.get_stats()["errors"] >= 1, str(ex.get_stats()))


def test_reused_id_new_ts():
    print("\nsame req_id observed again → ts advances, entry updated in place")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=1, MAX=128)
        ex.observe("chatcmpl-r", 9.0)
        ts1 = read_file(path)["chatcmpl-r"]["ts"]
        for _ in range(5):
            ex.observe("chatcmpl-r", 9.0)
        d = read_file(path)["chatcmpl-r"]
        check("single entry for reused id", len(read_file(path)) == 1)
        check("n grew to 6", d["n"] == 6, str(d["n"]))
        check("ts non-decreasing", d["ts"] >= ts1, f"{d['ts']} vs {ts1}")


# ─── new per-request fields (c_trace, c_slope) ───────────────────────────────


def test_new_fields_present():
    print("\nentry carries c_trace + c_slope + n alongside c_last")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=4, MAX=128)
        for _ in range(4):
            ex.observe("chatcmpl-f", 12.0)
        e = read_file(path)["chatcmpl-f"]
        check("c_last present", "c_last" in e)
        check("c_trace present", "c_trace" in e, str(e))
        check("c_slope present", "c_slope" in e, str(e))
        check("n present", e["n"] == 4)
        check("flat window → c_slope 0", abs(e["c_slope"]) < 1e-6, str(e["c_slope"]))
        check("cadence entry not marked final", "final" not in e)


def test_c_trace_vs_c_last_diverge():
    print("\nc_trace = whole-trace mean, c_last = rolling-window mean (diverge)")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=4, EVERY=8, MAX=128)
        rid = "chatcmpl-t"
        for _ in range(4):
            ex.observe(rid, 10.0)
        for _ in range(4):
            ex.observe(rid, 20.0)  # 8th → write; window holds last 4 = 20s
        e = read_file(path)[rid]
        check("c_last = last-window mean (20.0)", abs(e["c_last"] - 20.0) < 1e-6,
              str(e["c_last"]))
        check("c_trace = whole-trace mean (15.0)", abs(e["c_trace"] - 15.0) < 1e-6,
              str(e["c_trace"]))


def test_c_slope_sign():
    print("\nc_slope = recent-half mean minus older-half mean (rising → positive)")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=4, EVERY=4, MAX=128)
        rid = "chatcmpl-s"
        for v in (10.0, 10.0, 20.0, 20.0):  # window [10,10,20,20]
            ex.observe(rid, v)
        e = read_file(path)[rid]
        check("rising window → c_slope ~= +10", abs(e["c_slope"] - 10.0) < 1e-6,
              str(e["c_slope"]))


# ─── flush-on-close (Fable R1) ───────────────────────────────────────────────


def test_flush_writes_below_cadence():
    print("\nflush: writes final window immediately, even under EVERY cadence")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=32, MAX=128)
        rid = "chatcmpl-flush"
        for _ in range(10):  # 10 < EVERY=32 → no cadence write yet
            ex.observe(rid, 9.0)
        check("no cadence write for short close", not os.path.exists(path))
        ex.flush(rid)
        check("flush wrote the file", os.path.exists(path))
        e = read_file(path)[rid]
        check("flushed n = 10 (final window)", e["n"] == 10, str(e["n"]))
        check("flushed c_last = mean (9.0)", abs(e["c_last"] - 9.0) < 1e-6)
        check("flushed entry marked final", e.get("final") is True, str(e))


def test_flush_idempotent():
    print("\nflush: idempotent — a second flush is inert (window closed)")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=1000, MAX=128)
        rid = "chatcmpl-once"
        for _ in range(5):
            ex.observe(rid, 8.0)
        ex.flush(rid)
        writes_after_first = ex.get_stats()["writes"]
        ex.flush(rid)  # window already popped → no-op
        check("no extra write on 2nd flush",
              ex.get_stats()["writes"] == writes_after_first, str(ex.get_stats()))
        check("no errors", ex.get_stats()["errors"] == 0)


def test_flush_failopen():
    print("\nflush: unknown id / None / disabled never raise, never write")
    with tempfile.TemporaryDirectory() as td:
        path = reset(td, WIN=256, EVERY=1000, MAX=128)
        try:
            ex.flush(None)
            ex.flush("chatcmpl-never-tracked")
            raised = False
        except Exception:
            raised = True
        check("flush never raises on miss", not raised)
        check("no file for miss-only flushes", not os.path.exists(path))
        # disabled master → flush is inert
        ex.observe("chatcmpl-d", 9.0)
        os.environ.pop("GENESIS_ENABLE_PN112_EXPORT", None)
        ex.flush("chatcmpl-d")
        check("disabled flush writes nothing", not os.path.exists(path))


def main():
    for t in (test_disabled, test_rolling_mean_and_cadence, test_window_bounds_mean,
              test_atomic_single_object, test_rotation_keeps_recent_max,
              test_active_window_lru_evicts, test_failopen_none_inputs,
              test_failopen_unwritable_path, test_reused_id_new_ts,
              test_new_fields_present, test_c_trace_vs_c_last_diverge,
              test_c_slope_sign, test_flush_writes_below_cadence,
              test_flush_idempotent, test_flush_failopen):
        t()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL PN112-EXPORT TESTS PASSED")


if __name__ == "__main__":
    main()
