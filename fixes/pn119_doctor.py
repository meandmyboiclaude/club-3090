#!/usr/bin/env python3
"""pn119-doctor — read the H119 lens router's health surface and judge it.

    pn119-doctor                      # default path + default container
    pn119-doctor --path /pn119-sink/health.json
    pn119-doctor --container vllm-tcbench-8021
    pn119-doctor --json               # the raw document
    pn119-doctor --watch 5            # re-render every 5 s

Exit codes (non-zero whenever an alarm is live, which is the point):
    0  no alarms
    1  warnings only
    2  at least one critical alarm — including "there is no health surface"
    3  usage / IO error unrelated to the router's health

WHY A SEPARATE READER, AND WHY STDLIB ONLY
------------------------------------------
The router publishes health.json from inside the engine container, where it
knows the counters and nothing else. Three of the failures worth catching are
only visible from OUTSIDE that process, so they are computed here:

  * the file does not exist         -> the router never started, or was never
                                       enabled, or died before its first tick;
  * the file is STALE               -> the writer stopped. A health file that
                                       is never removed reads exactly like a
                                       live one, and a document describing a
                                       dead boot is worse than no document;
  * the file is from a PREVIOUS BOOT-> boot_id/started predate the running
                                       container. This is the specific trap
                                       the 2026-07-25 forensics fell into:
                                       forty sink files, no boot identity, and
                                       hours spent reading a dead boot's data.

Stdlib only, no venv, no imports from the router module: this has to run on a
box where the router's dependencies (torch) are absent, and it must never be
the reason a diagnosis cannot be made.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_PATH = os.environ.get(
    "PN119_HEALTH", os.path.expanduser("~/shared/needfit/pn119-sink/health.json"))
DEFAULT_CONTAINER = os.environ.get("PN119_CONTAINER", "vllm-tcbench-8021")

# A health file is written every PN119_SINK_BUF_SECS (2 s by default). Five
# missed ticks with a 30 s floor: long enough that a busy flusher or a slow
# filesystem never trips it, short enough that a dead worker is caught inside
# a minute.
STALE_TICKS = 5
STALE_FLOOR_S = 30.0

# Router-enable env names, in the order the compose resolves them.
ENABLE_ENV = ("GENESIS_ENABLE_PN119_ROUTER", "GENESIS_ENABLE_H119_LENS_ROUTER")

C = {"crit": "\033[31m", "warn": "\033[33m", "ok": "\033[32m",
     "dim": "\033[2m", "b": "\033[1m", "r": "\033[0m"}


def paint(s: str, key: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"{C.get(key, '')}{s}{C['r']}"


# ─────────────────────────────────────────────────────────── container probe
def _inspect(container: str, fmt: str):
    """Best-effort `docker inspect`. Returns stdout or None.

    Tries unprivileged first, then `sudo -n` (podman on this box runs the
    engine as root). A missing/unreachable docker is NOT an error: every
    container-derived check degrades to "unknown", never to a false verdict.
    """
    for cmd in (["docker", "inspect", "--format", fmt, container],
                ["sudo", "-n", "docker", "inspect", "--format", fmt, container]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


def _parse_started(raw: str):
    """Container StartedAt -> epoch seconds, or None."""
    if not raw:
        return None
    s = raw.strip()
    # docker: 2026-07-25T22:32:51.241766138+02:00 ; podman may add " CEST"
    s = re.sub(r"\s+[A-Z]{3,4}$", "", s).replace(" ", "T", 1)
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    s = re.sub(r"Z$", "+00:00", s)
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def container_facts(name: str) -> dict:
    out = {"name": name, "known": False}
    state = _inspect(name, "{{.State.Running}}")
    if state is None:
        return out
    out["known"] = True
    out["running"] = state.lower() == "true"
    out["started"] = _parse_started(_inspect(name, "{{.State.StartedAt}}") or "")
    out["hostname"] = _inspect(name, "{{.Config.Hostname}}") or ""
    out["cid"] = (_inspect(name, "{{.Id}}") or "")[:12]
    env = _inspect(name, "{{range .Config.Env}}{{println .}}{{end}}") or ""
    kv = dict(l.split("=", 1) for l in env.splitlines() if "=" in l)
    out["env"] = kv
    out["router_enabled"] = any(
        str(kv.get(k, "")).strip().lower() in ("1", "true", "yes", "on")
        for k in ENABLE_ENV)
    out["mode"] = kv.get("PN119_MODE", "")
    out["consumer_flag"] = str(
        kv.get("GENESIS_ENABLE_H119_ROUTE_BUDGET", "")).strip() in ("1", "true")
    return out


# ─────────────────────────────────────────────────────────────── file access
def read_health(path: str, attempts: int = 3):
    """Read health.json, tolerating a concurrent write.

    The router writes via temp+os.replace, so a torn read is impossible on a
    sane filesystem — but a truncated file from an OLD writer, an NFS mount or
    a half-copied artifact is not, and the doctor's whole job is to still say
    something useful. Retry briefly, then report the parse failure as a fact.
    """
    last = None
    for i in range(attempts):
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            return None, "missing"
        except OSError as e:
            return None, f"unreadable: {e}"
        try:
            return json.loads(raw), None
        except json.JSONDecodeError as e:
            last = f"unparseable ({e})"
            if i + 1 < attempts:
                time.sleep(0.25)
    return None, last


# ───────────────────────────────────────────────────────────── reader alarms
def reader_alarms(snap, err, path, cf, now) -> list[dict]:
    """Alarms only an outside reader can raise."""
    out = []
    if snap is None:
        if cf.get("known") and cf.get("router_enabled") is False:
            # Configured OFF: no health surface is the correct state, not a
            # failure. Say so plainly instead of crying ROUTER_ABSENT.
            return []
        detail = (f"no health surface at {path} ({err})")
        if cf.get("known") and cf.get("running"):
            detail += " while the container is running with the router enabled"
        out.append({"id": "ROUTER_ABSENT", "severity": "critical",
                    "detail": detail})
        return out
    age = now - float(snap.get("ts", 0) or 0)
    tick = float((snap.get("sink") or {}).get("buf_secs") or 2.0)
    limit = max(STALE_FLOOR_S, STALE_TICKS * tick)
    if snap.get("shutdown"):
        out.append({"id": "ROUTER_ABSENT", "severity": "critical",
                    "detail": f"the router shut down cleanly {age / 60:.1f} min "
                              "ago — this document describes a boot that is over"})
    elif age > limit:
        out.append({"id": "ROUTER_ABSENT", "severity": "critical",
                    "detail": f"health surface is STALE: last written {age:.0f}s "
                              f"ago (limit {limit:.0f}s) — the flusher thread, "
                              "the worker process, or the whole container is gone"})
    started = float(snap.get("started", 0) or 0)
    cstart = cf.get("started")
    if cstart and started and started < cstart - 5.0:
        out.append({"id": "ROUTER_ABSENT", "severity": "critical",
                    "detail": f"health boot_id={snap.get('boot_id', '')[:12]} "
                              f"started {(cstart - started) / 60:.1f} min BEFORE "
                              f"container {cf['name']} did — you are reading a "
                              "previous boot's file"})
    host = str(snap.get("hostname") or "")
    chost = str(cf.get("hostname") or "")
    if host and chost and host != chost and host != cf.get("cid"):
        out.append({"id": "ROUTER_ABSENT", "severity": "critical",
                    "detail": f"health was written by host {host!r}, container "
                              f"{cf['name']} is {chost!r} — different boot"})
    return out


# ────────────────────────────────────────────────────────────────── renderer
def fmt_age(v):
    if not v:
        return "-"
    d = time.time() - float(v)
    if d < 90:
        return f"{d:.0f}s ago"
    if d < 5400:
        return f"{d / 60:.1f}m ago"
    return f"{d / 3600:.1f}h ago"


def render(snap, err, path, cf, alarms) -> None:
    print(paint("PN119 / H119 lens router — health", "b"))
    print(f"  file       {path}")
    if cf.get("known"):
        state = "running" if cf.get("running") else "STOPPED"
        print(f"  container  {cf['name']} [{cf.get('cid', '?')}] {state}"
              f"  router_enabled={cf.get('router_enabled')}"
              f"  mode={cf.get('mode') or '?'}"
              f"  consumer={cf.get('consumer_flag')}")
    else:
        print(f"  container  {cf['name']} (not inspectable — "
              "container checks skipped)")
    if snap is None:
        print(paint(f"  status     NO HEALTH SURFACE ({err})", "crit"))
        if cf.get("known") and cf.get("router_enabled") is False:
            print("             the router is DISABLED by env — expected.")
        print()
    else:
        r = snap.get("router") or {}
        t = snap.get("traffic") or {}
        ra = snap.get("rates") or {}
        c = snap.get("consumer") or {}
        p = snap.get("probe") or {}
        s = snap.get("sink") or {}
        up = float(snap.get("uptime_s") or 0)
        print(f"  boot       {snap.get('boot_id', '?')[:12]} pid={snap.get('pid')}"
              f" host={snap.get('hostname')} up={up / 60:.1f}m"
              f" written={fmt_age(snap.get('ts'))}"
              + ("  [CLEAN SHUTDOWN]" if snap.get("shutdown") else ""))
        print(f"  router     present={r.get('present')} enabled={r.get('enabled')}"
              f" mode={r.get('mode')} tdeep={r.get('tdeep')}"
              f" fallback={r.get('fallback_route')} explore={r.get('explore_rate')}")
        print(f"  probe      {p.get('basename')} loads={p.get('loads')}"
              f" fold_resid={p.get('fold_resid')} readable={p.get('readable')}")
        print(f"  traffic    scored={t.get('scored')}"
              f" (deep={t.get('deep')} lean={t.get('lean')})"
              f" unscoreable={t.get('unscoreable')}"
              f" unroutable={t.get('unroutable')} batch_adds={t.get('batch_adds')}")
        # The contract population. `scored` counts ROUTABLE (thinking-ON) rows
        # only; a boot where most traffic is thinking-off is not broken, but
        # the saving it can produce is bounded by (1 - thinking_off_rate) and
        # that has to be readable, not inferred.
        print(f"             thinking_off={t.get('thinking_off')}"
              f" unknown_routable={t.get('scored_unknown')}"
              f" decisions={t.get('decisions')}")
        print(f"             first={fmt_age(t.get('first_scored_ts'))}"
              f" last={fmt_age(t.get('last_scored_ts'))}")
        print(f"  rates      deep_frac={ra.get('deep_frac', 0):.3f}"
              f" (n={ra.get('deep_frac_n')}, band 0.25-0.35)"
              f"  fallback={ra.get('fallback_rate', 0):.3f}"
              f"  unscoreable={ra.get('unscoreable_rate', 0):.3f}"
              f"  unroutable={ra.get('unroutable_rate', 0):.3f}")
        # BUG-139. A censored finish is a LOWER BOUND recorded as a spend. The
        # 2026-07-25 sink was 54% censored and nothing displayed it, which is
        # exactly why it capped every probe fit for a day without being named.
        print(f"             censored={ra.get('censored_rate', 0):.3f}"
              f" (n={ra.get('censored_rate_n')}, {t.get('censored')} rows)"
              f"  thinking_off={ra.get('thinking_off_rate', 0):.3f}")
        print(f"  consumer   flag={c.get('flag_env')} on={c.get('on')}"
              f" deep={c.get('deep_budget')} lean={c.get('lean_budget')}"
              f" override_pn100={c.get('override_pn100')}")
        print(f"             applied={c.get('applied')}"
              f" (pn100_override={c.get('pn100_override')}"
              f" provisional={c.get('provisional_added')})"
              f" caller_explicit={c.get('caller_explicit')}"
              f" tier0={c.get('tier0_respected')}")
        print(f"             routed deep={c.get('routed_deep')}"
              f" lean={c.get('routed_lean')}"
              f" desync={c.get('index_desync')}"
              f" out_of_batch={c.get('index_out_of_batch')}")
        print(f"  sink       {s.get('dir')} rows={s.get('rows')}"
              f" pending={s.get('pending')} thread={s.get('thread')}"
              f" buf={s.get('buf_rows')}/{s.get('buf_secs')}s")
        print()
    if not alarms:
        print(paint("  ALARMS     none — router healthy", "ok"))
    else:
        crit = [a for a in alarms if a.get("severity") == "critical"]
        print(paint(f"  ALARMS     {len(alarms)} live "
                    f"({len(crit)} critical)", "crit" if crit else "warn"))
        for a in alarms:
            key = "crit" if a.get("severity") == "critical" else "warn"
            tag = "CRIT" if key == "crit" else "WARN"
            print(paint(f"    [{tag}] {a['id']}", key))
            print(f"           {a.get('detail', '')}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pn119-doctor", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=DEFAULT_PATH,
                    help=f"health.json path (default {DEFAULT_PATH})")
    ap.add_argument("--container", default=DEFAULT_CONTAINER,
                    help="container to cross-check the boot against; "
                         "'-' to skip the container probe entirely")
    ap.add_argument("--json", action="store_true",
                    help="emit the health document plus reader alarms as JSON")
    ap.add_argument("--quiet", action="store_true",
                    help="alarms and verdict only")
    ap.add_argument("--watch", type=float, metavar="SECS",
                    help="re-render every SECS until interrupted")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:      # argparse already printed
        return 3 if e.code else 0

    while True:
        now = time.time()
        cf = ({"name": args.container, "known": False}
              if args.container == "-" else container_facts(args.container))
        snap, err = read_health(args.path)
        alarms = list(reader_alarms(snap, err, args.path, cf, now))
        seen = {a["id"] for a in alarms}
        if snap is not None:
            for a in snap.get("alarms") or []:
                # The router's own alarms are kept verbatim — the reader does
                # not re-derive them, because the document may have been
                # written by an older router than this doctor. Only
                # ROUTER_ABSENT can collide, and the reader's version (missing
                # / stale / previous boot) is the more informative one.
                if not (a.get("id") == "ROUTER_ABSENT" and "ROUTER_ABSENT" in seen):
                    alarms.append(a)
        if args.json:
            print(json.dumps({"path": args.path, "error": err,
                              "container": {k: v for k, v in cf.items()
                                            if k != "env"},
                              "health": snap, "alarms": alarms}, indent=1,
                             default=str))
        elif args.quiet:
            for a in alarms:
                print(f"[{a.get('severity', '?').upper()}] {a['id']}: "
                      f"{a.get('detail', '')}")
            print("PN119 OK" if not alarms else f"PN119 {len(alarms)} ALARM(S)")
        else:
            render(snap, err, args.path, cf, alarms)
        if not args.watch:
            break
        try:
            time.sleep(max(args.watch, 0.5))
        except KeyboardInterrupt:
            break
        print()

    if any(a.get("severity") == "critical" for a in alarms):
        return 2
    return 1 if alarms else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
