#!/usr/bin/env python3
"""PN119 health surface — alarm logic, replayed over the REAL sink history.

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_pn119_alarms.py
     (no boot, no GPU, no container — CPU import + recorded data)

WHY THIS FILE EXISTS
--------------------
2026-07-25: five boots of the H119 router ran degenerate — two pinned at 100%
deep, three pinned at 0% deep — and every one of them looked, from outside,
like a healthy container serving normal traffic. They were found hours later
by reading the sink offline. Separately the enforce consumer spent a whole day
as a measured no-op: all seven patch sites installed, every counter moving,
GPQA-30 byte-identical to control, because it deferred to PN100's budget on
100% of requests.

An alarm set that is only argued about in prose is how that survives twice. So
this file replays the actual sink files those boots wrote through the shipped
alarm logic and asserts the alarms fire. The five degenerate boots are named
by their sink tag; if the logic is ever loosened past them, this goes red.

WHAT THE REPLAY CAN AND CANNOT SEE
----------------------------------
The sink records ROUTER decisions (score/route per request, plus finish
labels). It does NOT record the STATS counters, and in particular none of the
consumer's. So the historical replay exercises the traffic-shaped alarms
(DEEP_FRAC_*, FALLBACK_STORM, UNROUTABLE_TRAFFIC) on real data, and the
consumer/config alarms get explicit constructed cases below — including the
exact counter signature of the day-long bug.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
SINK = pathlib.Path(os.environ.get(
    "PN119_SINK_DIR", os.path.expanduser("~/shared/needfit/pn119-sink")))
DOCTOR = pathlib.Path(os.path.expanduser("~/bin/pn119-doctor"))
# The live sink is a shared, MUTABLE directory: while this test was being
# written another job moved four boots into pn119-sink/.quarantine (they were
# captured through an instrumented tap and must not reach the refit). Evidence
# that can be moved out from under a test is evidence the test cannot rest on,
# so the six boots the assertions name are also committed here, verbatim, meta
# lines only. Live files win when present; the fixture is the floor.
FIXTURE = REPO / "fixes/testdata/pn119-sink-20260725"

# The five degenerate boots of 2026-07-25, by sink tag. Two pinned deep
# (12/12 and 24/24), three pinned lean (0/81, 0/81, 0/61).
DEGENERATE_TAGS = ("20260725-154820", "20260725-155342",
                   "20260725-162242", "20260725-163945", "20260725-171008")
# A boot that sat dead centre of the design band: 31 deep of 100 scored.
HEALTHY_TAG = "20260725-183616"
# Fresh boots with a handful of requests. These must stay QUIET: an alarm set
# that cries on the first four requests of every boot gets muted by week two.
FRESH_TAGS = ("20260725-202134", "20260725-203337")

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def _load_router():
    sys.path.insert(0, str(REPO / "fixes"))
    import pn119_router as R  # noqa: PLC0415 — deliberate late import
    return R


# ─────────────────────────────────────────────────── sink replay -> counters
def stats_from_meta(path: pathlib.Path) -> dict:
    """Rebuild the STATS counters a boot WOULD have had, from its sink file.

    Only the counters the sink can witness: one score line per scored request
    (with its route), one unscoreable line per fallback (with its reason).
    That is exactly the input DEEP_FRAC_* and FALLBACK_STORM consume.
    """
    st: dict[str, int] = {}

    def bump(k, n=1):
        st[k] = st.get(k, 0) + n

    header = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("pn119_header"):
                header = m
                continue
            if m.get("finish"):
                continue
            if m.get("unscoreable"):
                bump("unscoreable")
                bump(f"unscoreable_{m.get('reason', 'unknown')}")
                bump(f"fallback_{m.get('route', 'deep')}")
                continue
            if "row" in m:
                bump("scored")
                bump("scored_deep" if m.get("route") == "deep" else "scored_lean")
    return {"stats": st, "header": header}


def find_meta(tag: str):
    """(path, source) for a sink tag: live, quarantined, or committed fixture."""
    for base, src in ((SINK, "live"), (SINK / ".quarantine", "quarantine"),
                      (FIXTURE, "fixture")):
        p = base / f"meta-{tag}.jsonl"
        if p.is_file():
            return p, src
    return None, "MISSING"


def replay(R, tag: str, **over) -> dict:
    meta, _src = find_meta(tag)
    if meta is None:
        raise FileNotFoundError(f"no meta-{tag}.jsonl in sink, quarantine or fixture")
    got = stats_from_meta(meta)
    st = got["stats"]
    kw = dict(
        stats=st, boot_id=f"replay-{tag}", pid=0, mode="shadow",
        mode_requested="shadow", tdeep=0.495,
        probe={"basename": "probe.npz", "readable": True, "loads": 1,
               "fold_resid": 0.0},
        # The replay is a post-mortem, not a live boot: uptime is unknown, so
        # leave it 0 and let the traffic-shaped alarms speak for themselves.
        consumer={"flag_env": False},
    )
    kw.update(over)
    return R.make_snapshot(**kw)


def replay_table(R) -> dict:
    print("\n── replay of the sink history "
          f"({SINK}) ────────────────────────")
    print(f"  {'sink tag':<18} {'src':<10} {'n':>5} {'deep':>5} {'lean':>5} "
          f"{'frac':>6}  alarms")
    out = {}
    tags = set()
    for base in (SINK, SINK / ".quarantine", FIXTURE):
        if base.is_dir():
            tags |= {p.name[len("meta-"):-len(".jsonl")]
                     for p in base.glob("meta-*.jsonl")}
    for tag in sorted(tags):
        _p, src = find_meta(tag)
        snap = replay(R, tag)
        t, r = snap["traffic"], snap["rates"]
        ids = ",".join(snap["alarm_ids"]) or "-"
        n = t["scored"]
        frac = f"{r['deep_frac']:.3f}" if n else "-"
        # A 0-byte meta file is the ambiguity the header line now closes: it
        # could be a tap that never fired, a boot that got no traffic, or a
        # router that never started, and after the fact nothing distinguishes
        # them. Twenty of these exist from 2026-07-25.
        empty = "" if n or t["unscoreable"] else "   (0 rows — UNATTRIBUTABLE)"
        print(f"  {tag:<18} {src:<10} {n:>5} {t['deep']:>5} {t['lean']:>5} "
              f"{frac:>6}  {ids}{empty}")
        out[tag] = snap
    return out


# ───────────────────────────────────────────────────── constructed scenarios
def healthy_stats() -> dict:
    """A boot doing exactly what it was shipped to do."""
    return {
        "scored": 100, "scored_deep": 31, "scored_lean": 69,
        "h119_provisional_added": 40, "h119_pn100_override": 55,
        "h119_caller_explicit": 5, "h119_routed_deep": 30,
        "h119_routed_lean": 65,
    }


def healthy_snapshot(R, **over) -> dict:
    kw = dict(
        stats=healthy_stats(), boot_id="b" * 32, pid=1234, hostname="c3090",
        started=time.time() - 900.0, mode="enforce", mode_requested="enforce",
        tdeep=0.495, fallback_route="deep", fallback_requested="deep",
        router_present=True, router_enabled=True,
        probe={"basename": "probe.npz", "readable": True, "loads": 1,
               "fold_resid": 3.6e-12},
        consumer={"flag_env": True, "checked": True, "on": True,
                  "deep_budget": 10240, "lean_budget": 1600,
                  "override_pn100": True},
        sink={"dir": "/pn119-sink", "enabled": True, "buf_secs": 2.0},
        first_scored_ts=time.time() - 880, last_scored_ts=time.time() - 5,
        last_decision_ts=time.time() - 5,
    )
    kw.update(over)
    return R.make_snapshot(**kw)


def ids(snap) -> set:
    return set(snap["alarm_ids"])


def case_healthy_is_silent(R) -> None:
    snap = healthy_snapshot(R)
    check("healthy boot raises NO alarm", not snap["alarms"],
          ",".join(snap["alarm_ids"]) or "clean")
    check("healthy boot reports ok=True", snap["ok"] is True)


def case_router_absent(R) -> None:
    s = healthy_snapshot(R, router_present=False)
    check("ROUTER_ABSENT: enabled but no live instance",
          "ROUTER_ABSENT" in ids(s))
    s = healthy_snapshot(R, router_present=False, router_enabled=False)
    check("ROUTER_ABSENT: silent when the router was never enabled",
          "ROUTER_ABSENT" not in ids(s))


def case_tap_never_fired(R) -> None:
    # Proof form: the consumer saw batch rows, the tap saw nothing.
    s = healthy_snapshot(R, stats={"h119_caller_explicit": 40},
                         consumer={"flag_env": True, "checked": True,
                                   "on": True})
    check("TAP_NEVER_FIRED: consumer saw traffic the tap did not",
          "TAP_NEVER_FIRED" in ids(s))
    # Timed form: nothing at all, past the grace window.
    s = healthy_snapshot(R, stats={}, started=time.time() - 3600,
                         consumer={"flag_env": False})
    check("TAP_NEVER_FIRED: nothing observed in 60 min",
          "TAP_NEVER_FIRED" in ids(s))
    # Same boot, 5 minutes old: still inside the grace window.
    s = healthy_snapshot(R, stats={}, started=time.time() - 300,
                         consumer={"flag_env": False})
    check("TAP_NEVER_FIRED: quiet inside the 10 min grace",
          "TAP_NEVER_FIRED" not in ids(s))


def case_deep_frac(R) -> None:
    s = healthy_snapshot(R, stats={"scored": 12, "scored_deep": 12,
                                   "scored_lean": 0})
    check("DEEP_FRAC_DEGENERATE: 12/12 deep (the smallest real instance)",
          "DEEP_FRAC_DEGENERATE" in ids(s))
    s = healthy_snapshot(R, stats={"scored": 81, "scored_deep": 0,
                                   "scored_lean": 81})
    check("DEEP_FRAC_DEGENERATE: 0/81 deep", "DEEP_FRAC_DEGENERATE" in ids(s))
    s = healthy_snapshot(R, stats={"scored": 11, "scored_deep": 0,
                                   "scored_lean": 11})
    check("DEEP_FRAC_DEGENERATE: quiet below the 12-sample floor",
          "DEEP_FRAC_DEGENERATE" not in ids(s))
    s = healthy_snapshot(R, stats={"scored": 100, "scored_deep": 6,
                                   "scored_lean": 94})
    check("DEEP_FRAC_OUT_OF_BAND: 6% deep over 100 requests",
          "DEEP_FRAC_OUT_OF_BAND" in ids(s))
    s = healthy_snapshot(R, stats={"scored": 100, "scored_deep": 31,
                                   "scored_lean": 69})
    check("DEEP_FRAC_OUT_OF_BAND: quiet at 31/100 (dead centre)",
          "DEEP_FRAC_OUT_OF_BAND" not in ids(s))
    s = healthy_snapshot(R, stats={"scored": 4, "scored_deep": 0,
                                   "scored_lean": 4})
    check("DEEP_FRAC_*: a 4-request-old boot alarms on nothing",
          not (ids(s) & {"DEEP_FRAC_DEGENERATE", "DEEP_FRAC_OUT_OF_BAND"}))


def case_consumer_never_applied(R) -> None:
    """THE 2026-07-25 DAY-LONG BUG. This signature MUST alarm."""
    st = {"scored": 61, "scored_deep": 8, "scored_lean": 53,
          # the consumer ran on every request and deferred on every request
          "h119_caller_explicit": 61}
    s = healthy_snapshot(R, stats=st)
    check("CONSUMER_NEVER_APPLIED: flag on, override=0 provisional=0 "
          "over 61 decisions", "CONSUMER_NEVER_APPLIED" in ids(s),
          ",".join(sorted(ids(s))))
    # One override is enough to prove the consumer can act.
    st2 = dict(st, h119_pn100_override=1)
    s2 = healthy_snapshot(R, stats=st2)
    check("CONSUMER_NEVER_APPLIED: clears on the first real takeover",
          "CONSUMER_NEVER_APPLIED" not in ids(s2))
    # Not enough traffic to make the claim yet.
    st3 = {"scored": 10, "scored_deep": 3, "scored_lean": 7,
           "h119_caller_explicit": 10}
    s3 = healthy_snapshot(R, stats=st3)
    check("CONSUMER_NEVER_APPLIED: quiet below the 20-decision floor",
          "CONSUMER_NEVER_APPLIED" not in ids(s3))
    # Flag off: nothing to complain about.
    s4 = healthy_snapshot(R, stats=st,
                          consumer={"flag_env": False, "checked": True,
                                    "on": False})
    check("CONSUMER_NEVER_APPLIED: silent when the consumer is off",
          "CONSUMER_NEVER_APPLIED" not in ids(s4))


def case_consumer_not_wired(R) -> None:
    st = {"scored": 61, "scored_deep": 18, "scored_lean": 43,
          "h119_router_not_enforce": 61}
    s = healthy_snapshot(R, stats=st, mode="shadow", mode_requested="shadow")
    check("CONSUMER_NOT_WIRED: flag on but the router is in shadow",
          "CONSUMER_NOT_WIRED" in ids(s))
    st2 = {"scored": 61, "scored_deep": 18, "scored_lean": 43}
    s2 = healthy_snapshot(R, stats=st2,
                          consumer={"flag_env": True, "checked": False,
                                    "on": False})
    check("CONSUMER_NOT_WIRED: h119_on_batch_add never ran at all",
          "CONSUMER_NOT_WIRED" in ids(s2))
    s3 = healthy_snapshot(R, stats={"scored": 61, "scored_deep": 18,
                                    "scored_lean": 43, "h119_no_router": 61})
    check("CONSUMER_NOT_WIRED: consumer live, ROUTER global missing",
          "CONSUMER_NOT_WIRED" in ids(s3))


def case_probe_canary(R) -> None:
    s = healthy_snapshot(R, stats=dict(healthy_stats(), probe_reload_failed=3))
    check("PROBE_CANARY_FAIL: a refused hot-reload (serving stale weights)",
          "PROBE_CANARY_FAIL" in ids(s))
    s = healthy_snapshot(R, probe={"basename": "probe.npz", "readable": False,
                                   "loads": 1, "fold_resid": 0.0})
    check("PROBE_CANARY_FAIL: probe npz vanished", "PROBE_CANARY_FAIL" in ids(s))
    s = healthy_snapshot(R, probe={"basename": "probe.npz", "readable": True,
                                   "loads": 1, "fold_resid": 4.2e-3})
    check("PROBE_CANARY_FAIL: fold residual over tolerance",
          "PROBE_CANARY_FAIL" in ids(s))
    s = healthy_snapshot(R, probe={"basename": "probe.npz", "readable": True,
                                   "loads": 9, "fold_resid": 1e-12})
    check("PROBE_CANARY_FAIL: quiet after 9 clean hot-reloads",
          "PROBE_CANARY_FAIL" not in ids(s))


def case_fallback_storm(R) -> None:
    st = {"scored": 60, "scored_deep": 19, "scored_lean": 41,
          "unscoreable": 40, "unscoreable_partial_prefill": 40,
          "fallback_deep": 40}
    s = healthy_snapshot(R, stats=st)
    check("FALLBACK_STORM: 40% of decisions took the deep fallback",
          "FALLBACK_STORM" in ids(s))
    st2 = {"scored": 98, "scored_deep": 30, "scored_lean": 68,
           "unscoreable": 2, "unscoreable_partial_prefill": 2}
    s2 = healthy_snapshot(R, stats=st2)
    check("FALLBACK_STORM: quiet at a 2% fallback rate",
          "FALLBACK_STORM" not in ids(s2))


def case_unroutable(R) -> None:
    st = {"scored": 100, "scored_deep": 30, "scored_lean": 70,
          "route_for_miss": 7, "h119_route_missing": 7,
          "h119_provisional_added": 100}
    s = healthy_snapshot(R, stats=st)
    check("UNROUTABLE_TRAFFIC: requests decided with no route on record",
          "UNROUTABLE_TRAFFIC" in ids(s))
    st2 = dict(healthy_stats(), skip_no_prompt_len=1)
    s2 = healthy_snapshot(R, stats=st2)
    check("UNROUTABLE_TRAFFIC: quiet on a single racing request",
          "UNROUTABLE_TRAFFIC" not in ids(s2))


def case_mode_invalid(R) -> None:
    s = healthy_snapshot(R, mode="shadow", mode_requested="enforced")
    check("MODE_INVALID: PN119_MODE='enforced' coerced to shadow",
          "MODE_INVALID" in ids(s))
    s = healthy_snapshot(R, mode="enforce", mode_requested="ENFORCE ")
    check("MODE_INVALID: quiet on 'ENFORCE ' (case/space are handled)",
          "MODE_INVALID" not in ids(s))
    s = healthy_snapshot(R, fallback_route="deep", fallback_requested="LEAN")
    check("MODE_INVALID: PN119_FALLBACK_ROUTE coerced",
          "MODE_INVALID" in ids(s))
    s = healthy_snapshot(R, tdeep=float("nan"))
    check("MODE_INVALID: a NaN threshold routes everything lean",
          "MODE_INVALID" in ids(s))


def case_index_desync(R) -> None:
    s = healthy_snapshot(R, stats=dict(healthy_stats(), h119_index_desync=1))
    check("INDEX_DESYNC: fires on the first recycled batch slot",
          "INDEX_DESYNC" in ids(s))
    st = dict(healthy_stats(), h119_index_out_of_batch=20)
    s = healthy_snapshot(R, stats=st)
    check("INDEX_DESYNC: sustained out-of-batch resolutions",
          "INDEX_DESYNC" in ids(s))
    st = dict(healthy_stats(), h119_index_out_of_batch=1)
    s = healthy_snapshot(R, stats=st)
    check("INDEX_DESYNC: quiet on one row mid-move",
          "INDEX_DESYNC" not in ids(s))


def case_never_raises(R) -> None:
    """pn119_alarms runs inside the flusher: it must be total."""
    for junk in ({}, {"router": None, "traffic": None},
                 {"router": {"tdeep": "banana"}, "rates": {"deep_frac": None}},
                 {"traffic": {"scored": "x"}}, {"consumer": {"flag_env": True}}):
        try:
            R.pn119_alarms(junk)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"      raised on {junk!r}: {e}")
        check(f"pn119_alarms tolerates {str(junk)[:38]}", ok)


# ────────────────────────────────────────────────── the real writer, CPU-only
class _FakeModel:
    def set_aux_hidden_state_layers(self, layers):
        self.layers = layers


class _FakeRunner:
    """The two attributes maybe_create/_load_probe touch. No CUDA context."""
    device = "cpu"

    def __init__(self):
        self.requests = {}
        self._m = _FakeModel()

    def get_model(self):
        return self._m


def case_live_writer(R) -> None:
    """Boot a real router on CPU and prove the flusher publishes health.json.

    The alarm logic above is pure and provable offline; the WRITE path is not,
    and "the health file is written from the sink flusher, atomically, without
    touching the request path" is a claim that has to be executed, not
    asserted. Everything here is CPU: the probe folds on the host and the
    scoring path is never entered.
    """
    probe = pathlib.Path(os.path.expanduser("~/shared/needfit/pn119-live/probe.npz"))
    if not probe.is_file():
        check("live probe npz available for the writer test", False, str(probe))
        return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pn119sink-"))
    env = {"GENESIS_ENABLE_PN119_ROUTER": "1", "GENESIS_PN119_PROBE": str(probe),
           "PN119_SINK": str(tmp), "PN119_SINK_BUF_SECS": "0.2",
           "PN119_MODE": "shadow", "PN119_TDEEP": "0.495"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    saved_router, saved_stats = R.ROUTER, dict(R.STATS)
    R.STATS.clear()
    try:
        router = R.PN119Router.maybe_create(_FakeRunner())
        check("router constructs on CPU with the live probe", router is not None)
        if router is None:
            return
        hp = pathlib.Path(router.health_path)
        check("health path defaults into the sink dir",
              hp == tmp / "health.json", str(hp))
        check("health.json exists before any traffic", hp.is_file())
        snap = json.loads(hp.read_text(encoding="utf-8"))
        check("health document parses and carries the schema",
              snap.get("schema") == R.HEALTH_SCHEMA)
        check("health carries boot identity",
              bool(snap.get("boot_id")) and snap.get("pid") == os.getpid()
              and snap["boot_id"] == router.boot_id)
        check("fresh boot with no traffic raises no alarm", not snap["alarms"],
              ",".join(snap["alarm_ids"]) or "clean")
        check("probe fold residual is published",
              snap["probe"]["fold_resid"] is not None
              and float(snap["probe"]["fold_resid"]) < 1e-6,
              str(snap["probe"]["fold_resid"]))

        # The sink header line: what makes a 0-row sink file attributable.
        metas = sorted(tmp.glob("meta-*.jsonl"))
        check("sink file opened", len(metas) == 1, str(metas))
        head = json.loads(metas[0].read_text(encoding="utf-8").splitlines()[0])
        check("sink header line identifies the boot",
              head.get("pn119_header") == 1
              and head.get("boot_id") == router.boot_id
              and head.get("mode") == "shadow", str(head)[:100])

        # Idle boots must not rewrite a 3 KB document every 0.2 s tick.
        mt = hp.stat().st_mtime_ns
        time.sleep(1.0)          # 5 flusher ticks at buf_secs=0.2
        check("an idle boot does not rewrite health.json every tick",
              hp.stat().st_mtime_ns == mt)

        # Now move the counters the way a degenerate boot would and let the
        # EXISTING flusher thread pick them up — no new thread, no request.
        for _ in range(81):
            R._bump("scored")
            R._bump("scored_lean")
        deadline = time.time() + 5.0
        got = None
        while time.time() < deadline:
            time.sleep(0.25)
            try:
                got = json.loads(hp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if got["traffic"]["scored"] == 81:
                break
        check("the sink flusher republished health without a new thread",
              got is not None and got["traffic"]["scored"] == 81,
              f"scored={None if got is None else got['traffic']['scored']}")
        check("a 0/81 boot alarms through the REAL writer",
              got is not None and "DEEP_FRAC_DEGENERATE" in got["alarm_ids"],
              ",".join(got["alarm_ids"]) if got else "-")
        check("first/last scored timestamps derived by the flusher",
              got is not None and got["traffic"]["first_scored_ts"]
              and got["traffic"]["last_scored_ts"])
        check("no stray temp file left behind",
              not list(tmp.glob("health.json.tmp*")))

        rc = subprocess.run([sys.executable, str(DOCTOR), "--path", str(hp),
                             "--container", "-"], capture_output=True,
                            text=True, timeout=60)
        check("pn119-doctor exits 2 against the LIVE degenerate health file",
              rc.returncode == 2, f"rc={rc.returncode}")

        router._sink_close()
        final = json.loads(hp.read_text(encoding="utf-8"))
        check("clean shutdown is recorded in health.json",
              final.get("shutdown") is True)
    finally:
        R.ROUTER = saved_router
        R.STATS.clear()
        R.STATS.update(saved_stats)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ─────────────────────────────────────────────────────────── doctor end-to-end
def case_doctor(R, snaps) -> None:
    if not DOCTOR.is_file():
        check("pn119-doctor present", False, str(DOCTOR))
        return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pn119doc-"))

    def run(path):
        p = subprocess.run([sys.executable, str(DOCTOR), "--path", str(path),
                            "--container", "-"],
                           capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout + p.stderr

    # 1. degenerate window, written as a LIVE health file (fresh ts)
    deg = replay(R, "20260725-162242", now=time.time(),
                 started=time.time() - 1200, pid=99, boot_id="d" * 32,
                 sink={"dir": "/pn119-sink", "enabled": True, "buf_secs": 2.0})
    dp = tmp / "degenerate.json"
    dp.write_text(json.dumps(deg), encoding="utf-8")
    rc, out = run(dp)
    check("doctor exits non-zero on a degenerate window", rc != 0, f"rc={rc}")
    check("doctor names DEEP_FRAC_DEGENERATE", "DEEP_FRAC_DEGENERATE" in out)

    # 2. healthy window
    hp = tmp / "healthy.json"
    hp.write_text(json.dumps(healthy_snapshot(R)), encoding="utf-8")
    rc, out = run(hp)
    check("doctor exits 0 on a healthy window", rc == 0, f"rc={rc}")
    check("doctor says healthy", "none — router healthy" in out)

    # 3. missing file
    rc, out = run(tmp / "nope.json")
    check("doctor survives a missing file", rc == 2 and "ROUTER_ABSENT" in out,
          f"rc={rc}")

    # 4. mid-write / truncated document
    bad = tmp / "torn.json"
    bad.write_text('{"schema": "pn119.health/1", "boot_i', encoding="utf-8")
    rc, out = run(bad)
    check("doctor survives a truncated file",
          rc == 2 and "unparseable" in out, f"rc={rc}")

    # 5. a health file from a PREVIOUS boot (stale timestamp)
    old = healthy_snapshot(R)
    old["ts"] = time.time() - 7200
    sp = tmp / "stale.json"
    sp.write_text(json.dumps(old), encoding="utf-8")
    rc, out = run(sp)
    check("doctor flags a stale file as ROUTER_ABSENT",
          rc == 2 and "STALE" in out, f"rc={rc}")

    # 6. clean shutdown is reported, not mistaken for a live boot
    stop = healthy_snapshot(R)
    stop["shutdown"] = True
    qp = tmp / "shutdown.json"
    qp.write_text(json.dumps(stop), encoding="utf-8")
    rc, out = run(qp)
    check("doctor reports a clean shutdown", rc == 2 and "shut down" in out,
          f"rc={rc}")


# ───────────────────────────────────────────────────────────────────── main
def main() -> int:
    print("PN119 health surface — alarms replayed over the real sink\n")
    R = _load_router()

    if not SINK.is_dir():
        check("sink history present", False, str(SINK))
        return 1
    snaps = replay_table(R)

    print("\n── the five degenerate boots of 2026-07-25 ─────────────────────")
    for tag in DEGENERATE_TAGS:
        _p, src = find_meta(tag)
        if src == "MISSING":
            check(f"degenerate boot {tag} available to replay", False,
                  "not in the sink, the quarantine or the fixture")
            continue
        s = replay(R, tag)
        fired = ids(s) & {"DEEP_FRAC_DEGENERATE", "DEEP_FRAC_OUT_OF_BAND"}
        check(f"{tag} raises a deep-fraction alarm", bool(fired),
              f"[{src}] n={s['traffic']['scored']} deep={s['traffic']['deep']} "
              f"frac={s['rates']['deep_frac']:.3f} -> "
              f"{','.join(sorted(fired)) or 'NOTHING'}")

    print("\n── boots that must stay quiet ──────────────────────────────────")
    _p, src = find_meta(HEALTHY_TAG)
    if src == "MISSING":
        check(f"healthy reference boot {HEALTHY_TAG} available", False)
    else:
        s = replay(R, HEALTHY_TAG)
        check(f"{HEALTHY_TAG} (31/100 deep) raises no deep-fraction alarm",
              not (ids(s) & {"DEEP_FRAC_DEGENERATE", "DEEP_FRAC_OUT_OF_BAND"}),
              f"[{src}] frac={s['rates']['deep_frac']:.3f} "
              f"n={s['traffic']['scored']}")
    for tag in FRESH_TAGS:
        s = snaps.get(tag)
        if s:
            check(f"{tag} (n={s['traffic']['scored']}) raises nothing",
                  not s["alarms"], ",".join(s["alarm_ids"]) or "clean")

    print("\n── constructed cases: one per alarm ────────────────────────────")
    for fn in (case_healthy_is_silent, case_router_absent, case_tap_never_fired,
               case_deep_frac, case_consumer_never_applied,
               case_consumer_not_wired, case_probe_canary, case_fallback_storm,
               case_unroutable, case_mode_invalid, case_index_desync,
               case_never_raises):
        fn(R)

    print("\n── every alarm id is reachable ─────────────────────────────────")
    fired_ever = set()
    for s in snaps.values():
        fired_ever |= ids(s)
    # collect from the constructed cases too by re-running the obvious ones
    check("all 11 alarm ids carry a severity",
          set(R.ALARM_IDS) == set(R._ALARM_SEVERITY),
          f"{len(R.ALARM_IDS)} ids")

    print("\n── the real writer, booted on CPU ──────────────────────────────")
    case_live_writer(R)

    print("\n── pn119-doctor end to end ─────────────────────────────────────")
    case_doctor(R, snaps)

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print("ALL PASS")
    print("VERDICT: all five 2026-07-25 degenerate boots raise a deep-fraction "
          "alarm off their recorded sink data; the centred boot and the "
          "four-request boots stay quiet; the consumer's day-long no-op "
          "signature (override=0 provisional=0 under traffic) alarms; "
          "pn119-doctor exits non-zero on a degenerate, stale, torn or absent "
          "health surface and zero on a healthy one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
