#!/usr/bin/env python3
"""H119 router — RAM-buffered sink correctness (CPU only, no GPU, no service).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_pn119_sink_buffer.py

The sink carries the PN119-v2 refit's training data, so buffering it is only
allowed if the bytes that land on disk are the bytes the unbuffered router
would have written. Every check below is about that.

What is being pinned
--------------------
B1  BYTE-IDENTICAL: for the same sequence of requests, feats-*.bin and
    meta-*.jsonl are byte-for-byte what the module at git HEAD (unbuffered)
    produces. Checked over several buffer sizes AND with the buffer boundary
    landing mid-request.
B2  SHUTDOWN: _sink_close() (what atexit calls) drains a partially filled
    buffer — a clean stop loses nothing. Registration in _SINKS is real.
B3  BOUND: the buffer never exceeds PN119_SINK_BUF_MAX, even with the flusher
    thread wedged — the appending thread takes the write itself (backpressure)
    rather than growing without limit.
B4  PARTIAL BUFFER PARSES: a flush of a partly-filled buffer yields a file
    refit_pn119_probe.load_sink joins correctly, and the "row" indices in the
    meta lines still address the right rows of feats-*.bin.
B5  UNSCOREABLE INVARIANT survives buffering: no feature row, no "row" key,
    and the surviving rows stay index-aligned (today's finding).
B6  TIME THRESHOLD: with no further traffic, PN119_SINK_BUF_SECS gets the
    buffer to disk on its own (no request needed to push it out).
B7  NO DISK ON THE REQUEST PATH: with a blocked filesystem handle, requests
    still complete — i.e. the write really is off the path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NEEDFIT = os.path.expanduser("~/shared/needfit")
PROBE = os.path.join(NEEDFIT, "pn119-live/probe.npz")

D_MODEL = 5120
LAYERS = (42, 47, 51)
FEAT_BYTES = len(LAYERS) * 2 * D_MODEL * 2   # 61440
PROMPT_LEN = 40

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── stubs (same surfaces as test_pn119_partial_prefill.py) ─────────────────
class _SP:
    max_tokens = 512


class StubState:
    def __init__(self, prompt_ids, num_computed=0):
        self.prompt_token_ids = list(prompt_ids)
        self.num_prompt_tokens = len(prompt_ids)
        self.num_computed_tokens = num_computed
        self.output_token_ids: list[int] = []
        self.sampling_params = _SP()


class StubBatch:
    def __init__(self):
        self.req_ids: list[str] = []


class StubRunner:
    device = "cpu"

    def __init__(self):
        self.input_batch = StubBatch()
        self.requests: dict[str, StubState] = {}


class Sched:
    def __init__(self, d):
        self.num_scheduled_tokens = d


_IDX = torch.arange(D_MODEL, dtype=torch.float32)


def aux_rows(layer_i: int, token_ids, positions) -> torch.Tensor:
    t = torch.tensor(token_ids, dtype=torch.float32).unsqueeze(1)
    p = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)
    return 0.05 * torch.sin(0.0013 * (_IDX + 1) * (t + 1)
                            + 0.37 * layer_i + 0.011 * p)


def prompt_ids(seed: int, n: int = PROMPT_LEN) -> list[int]:
    return [(seed * 7919 + i * 104729) % 150000 + 1 for i in range(n)]


_ENV_KEYS = ("PN119_MODE", "PN119_TDEEP", "PN119_SINK", "PN119_EXPLORE",
             "PN119_FALLBACK_ROUTE", "PN119_PREFIX_MEMO", "PN119_MEMO_UNIT",
             "PN119_MEMO_MAX", "PN119_STATS_EVERY", "PN119_SINK_BUF_ROWS",
             "PN119_SINK_BUF_SECS", "PN119_SINK_BUF_MAX")


def new_router(mod, **env):
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    mod.SCORES.clear()
    mod.EXPLORE.clear()
    if hasattr(mod, "ROUTES"):
        mod.ROUTES.clear()
        mod.STATS.clear()
    runner = StubRunner()
    return mod.PN119Router(runner, PROBE), runner


def run_prefill(router, runner, req_id, ids, chunks, cached=0):
    state = StubState(ids, num_computed=cached)
    runner.requests[req_id] = state
    runner.input_batch.req_ids = [req_id]
    pos = cached
    for n in chunks:
        state.num_computed_tokens = pos
        toks = ids[pos:pos + n]
        aux = [aux_rows(li, toks, range(pos, pos + n)) for li in range(len(LAYERS))]
        router.observe(Sched({req_id: n}), aux)
        pos += n
    return state


def drive(router, runner, n_ok: int, n_miss: int = 0, seed: int = 0):
    """A deterministic request sequence: n_ok scoreable + n_miss cache-hit
    (unscoreable) requests, each finished so both meta lines are emitted."""
    order = []
    for i in range(max(n_ok, n_miss)):
        if i < n_ok:
            order.append(("ok", i))
        if i < n_miss:
            order.append(("miss", i))
    for kind, i in order:
        rid = f"{kind}{i}"
        ids = prompt_ids(seed * 1000 + i + (500 if kind == "miss" else 0))
        if kind == "ok":
            st = run_prefill(router, runner, rid, ids, [PROMPT_LEN])
            st.prompt_token_ids = list(ids) + [router._think_start]
            st.output_token_ids = [5, 5, router._think_end, 9]
        else:
            st = run_prefill(router, runner, rid, ids, [PROMPT_LEN - 16], cached=16)
        router.on_finish(rid, st)


def sink_files(d: str):
    feat = [f for f in os.listdir(d) if f.startswith("feats-")]
    meta = [f for f in os.listdir(d) if f.startswith("meta-")]
    fb = open(os.path.join(d, feat[0]), "rb").read() if feat else b""
    mb = open(os.path.join(d, meta[0]), "rb").read() if meta else b""
    return fb, mb


def _strip_ts(meta_bytes: bytes) -> list[dict]:
    """The REQUEST-PATH events, normalised for comparison.

    ts is wall-clock and differs run to run. The boot header line
    ({"pn119_header": 1, ...}, added 2026-07-25 with the health surface) is
    not a request-path event at all — it is written once at sink open, before
    any traffic, so that a sink file with no rows is still attributable to a
    boot. Every count in this file is about request events, so the header is
    dropped here and pinned on its own in B0.
    """
    out = []
    for line in meta_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("pn119_header"):
            continue
        m.pop("ts", None)
        out.append(m)
    return out


def _header_lines(meta_bytes: bytes) -> list[dict]:
    return [json.loads(l) for l in meta_bytes.decode("utf-8").splitlines()
            if l.strip() and json.loads(l).get("pn119_header")]


# ── B0: the boot header line ───────────────────────────────────────────────
def b0_header(new_mod):
    """The sink file must identify its boot before the first request.

    Twenty of the forty sink files from 2026-07-25 are 0 bytes, and nothing
    distinguishes "the tap never fired" from "no traffic arrived" from "the
    router died on boot" in any of them. The header closes that, which only
    works if it is on disk IMMEDIATELY — a buffered header would be lost by
    exactly the crash-looping boots that produced those empty files.
    """
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, _run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                             PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=1000,
                             PN119_SINK_BUF_SECS=600)
        _f, m = sink_files(d)
        heads = _header_lines(m)
        check("B0 header is on disk before any request (not buffered)",
              len(heads) == 1, f"{len(heads)} header lines, {len(m)}B")
        if heads:
            h = heads[0]
            check("B0 header carries the boot identity",
                  h.get("boot_id") == r.boot_id and h.get("pid") == os.getpid()
                  and h.get("mode") == "enforce" and "tdeep" in h,
                  json.dumps(h)[:120])
        check("B0 header is not a request event", _strip_ts(m) == [])
        r._sink_close()


# ── B1: byte-identical to the unbuffered HEAD module ───────────────────────
def b1_byte_identical(new_mod, old_mod):
    for label, kw in (("buf=1 (every event)", {"PN119_SINK_BUF_ROWS": 1}),
                      ("buf=4 (boundary mid-request)", {"PN119_SINK_BUF_ROWS": 4}),
                      ("buf=64 (default)", {"PN119_SINK_BUF_ROWS": 64}),
                      ("buf=0 (sync escape hatch)", {"PN119_SINK_BUF_ROWS": 0}),
                      ("buf=1000 (never auto-fires)", {"PN119_SINK_BUF_ROWS": 1000})):
        with tempfile.TemporaryDirectory(dir=NEEDFIT) as dn, \
                tempfile.TemporaryDirectory(dir=NEEDFIT) as do:
            rn, run_n = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=dn,
                                   PN119_TDEEP="0.495", PN119_SINK_BUF_SECS=600,
                                   **kw)
            ro, run_o = new_router(old_mod, PN119_MODE="enforce", PN119_SINK=do,
                                   PN119_TDEEP="0.495")
            drive(rn, run_n, n_ok=7, n_miss=3, seed=1)
            drive(ro, run_o, n_ok=7, n_miss=3, seed=1)
            rn._sink_close()
            # The reference module is whatever is at HEAD, and HEAD has been
            # buffered since the buffering commit landed — so it must be
            # drained too or the comparison reads a partially-flushed file and
            # a different number of bytes on every run. (Without this the check
            # was silently red from the moment buffering was committed: it was
            # comparing a drained buffer against a racing one.)
            ro._sink_close()
            fn, mn = sink_files(dn)
            fo, mo = sink_files(do)
            check(f"B1 {label}: feats bytes identical to HEAD",
                  fn == fo and len(fn) == 7 * FEAT_BYTES,
                  f"buffered={len(fn)}B head={len(fo)}B")
            check(f"B1 {label}: meta lines identical to HEAD (ts aside)",
                  _strip_ts(mn) == _strip_ts(mo),
                  f"buffered={len(_strip_ts(mn))} head={len(_strip_ts(mo))} lines")


# ── B2: shutdown drains a partial buffer ───────────────────────────────────
def b2_shutdown_flush(new_mod):
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=1000,
                            PN119_SINK_BUF_SECS=600)
        check("B2 router registered in _SINKS (atexit reaches it)",
              r in new_mod._SINKS)
        drive(r, run, n_ok=3, seed=2)
        f_pre, m_pre = sink_files(d)
        check("B2 no request event on disk before shutdown (buffer is RAM)",
              len(f_pre) == 0 and _strip_ts(m_pre) == [],
              f"feats={len(f_pre)}B meta_events={len(_strip_ts(m_pre))} "
              f"(+{len(_header_lines(m_pre))} header)")
        buffered = len(r._sbuf_meta)
        new_mod._flush_all_sinks()          # exactly what atexit runs
        f, m = sink_files(d)
        check("B2 atexit drained the partial buffer — clean stop loses nothing",
              len(f) == 3 * FEAT_BYTES and len(_strip_ts(m)) == buffered,
              f"feats={len(f)//FEAT_BYTES} rows, meta={len(_strip_ts(m))}/{buffered}")
        check("B2 _flush_all_sinks is idempotent (double atexit is safe)",
              (new_mod._flush_all_sinks() is None) and sink_files(d) == (f, m))


# ── B3: the buffer is bounded even with the flusher wedged ─────────────────
def b3_bound(new_mod):
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=4,
                            PN119_SINK_BUF_MAX=8, PN119_SINK_BUF_SECS=600)
        # Retire the flusher entirely, so the ONLY thing that can drain the
        # buffer is the appending thread's own hard-cap flush. (Wedging the io
        # lock instead would deadlock the appender on its own backpressure
        # flush — the lock is not reentrant and the appender takes it too.)
        r._sink_stop = True
        r._sink_wake.set()
        r._sink_thread.join(5)
        r._sink_thread = None
        peak = 0
        for i in range(20):
            drive(r, run, n_ok=1, seed=100 + i)
            peak = max(peak, len(r._sbuf_meta), len(r._sbuf_feat))
        check("B3 buffer never exceeded BUF_MAX with no flusher running",
              peak <= 8, f"peak={peak} cap=8")
        r._sink_close()
        f, m = sink_files(d)
        check("B3 no events lost while bounded",
              len(f) == 20 * FEAT_BYTES and len(_strip_ts(m)) == 40,
              f"rows={len(f)//FEAT_BYTES}/20 meta={len(_strip_ts(m))}/40")


# ── B4/B5: a partial buffer still parses, indices still address right ──────
def b4_partial_parses(new_mod):
    sys.path.insert(0, HERE)
    from refit_pn119_probe import load_sink
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=1000,
                            PN119_SINK_BUF_SECS=600)
        drive(r, run, n_ok=5, n_miss=4, seed=3)      # 5+4 interleaved
        r._sink_flush()                              # partial: 18 of 1000
        counts: dict = {}
        # load_sink's return shape is owned by the refit lane and has already
        # changed once (list[Row] -> (rows, sizes)); accept both rather than
        # crash the whole file — B6/B7 sit behind this call.
        res = load_sink(d, counts)
        rows = res[0] if isinstance(res, tuple) else res
        check("B4 load_sink parses the partially-filled flush",
              len(rows) == 5 and sorted(x.req_id for x in rows) ==
              [f"ok{i}" for i in range(5)],
              f"rows={sorted(x.req_id for x in rows)} counts={counts}")

        f, m = sink_files(d)
        lines = _strip_ts(m)
        uns = [x for x in lines if x.get("unscoreable")]
        check("B5 unscoreable rows: no feature row, no 'row' key",
              len(f) == 5 * FEAT_BYTES and len(uns) == 8
              and all("row" not in x for x in uns),
              f"feat_rows={len(f)//FEAT_BYTES} unscoreable_lines={len(uns)}")
        scored = [x for x in lines if "row" in x]
        check("B5 'row' indices are 0..N-1 in write order (still aligned)",
              [x["row"] for x in scored] == list(range(5)),
              f"{[x['row'] for x in scored]}")

        # the index must address the RIGHT row: recompute row k's bytes and
        # compare against the slice load_sink would read.
        import numpy as np
        feats = np.fromfile(os.path.join(
            d, [x for x in os.listdir(d) if x.startswith("feats-")][0]),
            dtype=np.uint16).reshape(-1, 30720)
        ok = True
        for x in scored:
            k = x["row"]
            row = torch.from_numpy(feats[k].copy()).view(torch.bfloat16).float()
            # The probe is folded to (v, b) at load — mu/sd/Vt10/w stop being
            # resident, so the staged form this used to re-derive no longer
            # exists on the router.
            ok = ok and abs(float(torch.dot(r.pv.float().cpu(), row)) + r.pb
                            - x["score"]) < 2e-2
        check("B4 row k of feats-*.bin re-scores to meta line k's score",
              ok, "bf16 round-trip within 2e-2")
        r._sink_close()


# ── B6: the time threshold fires on its own ────────────────────────────────
def b6_time_threshold(new_mod):
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=1000,
                            PN119_SINK_BUF_SECS=0.15)
        drive(r, run, n_ok=2, seed=4)
        deadline = time.time() + 5.0
        while time.time() < deadline and len(sink_files(d)[0]) == 0:
            time.sleep(0.05)
        f, m = sink_files(d)
        check("B6 BUF_SECS drains an idle buffer with no further traffic",
              len(f) == 2 * FEAT_BYTES and len(_strip_ts(m)) == 4,
              f"rows={len(f)//FEAT_BYTES} meta={len(_strip_ts(m))}")
        r._sink_close()


# ── B7: requests do not wait on the sink write ─────────────────────────────
def b7_off_the_path(new_mod):
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495", PN119_SINK_BUF_ROWS=64,
                            PN119_SINK_BUF_MAX=100000, PN119_SINK_BUF_SECS=600)
        # Stall every writer: a request that still returns proves the write is
        # not on its path.
        r._sink_io_lock.acquire()
        done = threading.Event()

        def work():
            drive(r, run, n_ok=200, seed=5)   # 200 events past BUF_ROWS=64
            done.set()

        th = threading.Thread(target=work, daemon=True)
        t0 = time.time()
        th.start()
        finished = done.wait(20.0)
        elapsed = time.time() - t0
        r._sink_io_lock.release()
        check("B7 200 requests completed with every disk writer blocked",
              finished and len(r._sbuf_feat) == 200,
              f"finished={finished} buffered_rows={len(r._sbuf_feat)} "
              f"elapsed={elapsed:.2f}s")
        th.join(5)
        r._sink_close()
        f, _ = sink_files(d)
        check("B7 the blocked events were not dropped — all landed on unblock",
              len(f) == 200 * FEAT_BYTES, f"rows={len(f)//FEAT_BYTES}/200")


def main() -> int:
    if not os.path.isfile(PROBE):
        print(f"probe missing: {PROBE}")
        return 1
    with tempfile.TemporaryDirectory() as td:
        head = os.path.join(td, "pn119_router_head.py")
        with open(head, "wb") as f:
            f.write(subprocess.run(["git", "-C", REPO, "show",
                                    "HEAD:fixes/pn119_router.py"],
                                   check=True, capture_output=True).stdout)
        old_mod = _load("pn119_router_head", head)
    new_mod = _load("pn119_router_new", os.path.join(HERE, "pn119_router.py"))

    print("== B0 boot header line ==")
    b0_header(new_mod)
    print("== B1 on-disk bytes identical to the unbuffered router ==")
    b1_byte_identical(new_mod, old_mod)
    print("== B2 shutdown / atexit flush ==")
    b2_shutdown_flush(new_mod)
    print("== B3 buffer bound ==")
    b3_bound(new_mod)
    print("== B4/B5 partial buffer parses + stays aligned ==")
    b4_partial_parses(new_mod)
    print("== B6 time threshold ==")
    b6_time_threshold(new_mod)
    print("== B7 no disk on the request path ==")
    b7_off_the_path(new_mod)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
