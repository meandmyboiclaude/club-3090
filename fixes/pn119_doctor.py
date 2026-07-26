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

THE ENABLE FLAG IS RESOLVED THE ROUTER'S WAY, NOT THE OBVIOUS WAY
-----------------------------------------------------------------
`router_enabled_from()` below reimplements `pn119_router._router_enabled()`
(canonical-first, first-SET-wins, empty == unset). It used to be a plain
any()/OR over both names, which made the doctor report ENABLED on the one case
that matters most — canonical `GENESIS_ENABLE_H119_LENS_ROUTER=0` plus a stale
`GENESIS_ENABLE_PN119_ROUTER=1` left behind by the compose entrypoint shim —
while the router itself was OFF. A diagnostic that disagrees with the thing it
diagnoses is worse than no diagnostic.

It is a REIMPLEMENTATION rather than an import for two independent reasons:
  * the router imports torch at module scope, and this tool exists precisely
    for hosts where torch is not installed;
  * the router's helper reads `os.environ` — the doctor resolves the flag from
    a CONTAINER's env, harvested via `docker inspect`, which is a different
    mapping entirely. Even with torch present, reuse would mean mutating
    os.environ around a call, in a `--watch` loop, to answer a question about
    another process.
So the copy is deliberate — and `--selftest` is the thing that stops it from
drifting: it replays the whole precedence table against BOTH implementations
(loading the sidecar by path behind a stub torch) and fails on the first
disagreement. Run it after touching either side:
    /usr/bin/python3 fixes/pn119_doctor.py --selftest
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

# Router-enable env names. Keep these spellings and this precedence identical
# to pn119_router.ENABLE_FLAG / ENABLE_FLAG_LEGACY / _router_enabled().
ENABLE_FLAG = "GENESIS_ENABLE_H119_LENS_ROUTER"          # canonical
ENABLE_FLAG_LEGACY = "GENESIS_ENABLE_PN119_ROUTER"       # back-compat alias
# Reported so the operator can see WHICH name decided, not just the verdict.
ENABLE_ENV = (ENABLE_FLAG, ENABLE_FLAG_LEGACY)

_TRUTHY = ("1", "true", "yes", "on")


def router_enabled_from(env) -> bool:
    """Resolve the router master switch out of a mapping of env vars.

    Mirrors pn119_router._router_enabled() exactly — canonical-first,
    first-SET-wins, empty == unset:
      1. GENESIS_ENABLE_H119_LENS_ROUTER, whenever it is set to a non-empty
         value, DECIDES — including when it contradicts the legacy alias.
         `H119_LENS_ROUTER=0` with a stale `PN119_ROUTER=1` is OFF.
      2. otherwise GENESIS_ENABLE_PN119_ROUTER.
      3. otherwise off.
    An OR over both names (what this function used to be) would make the
    canonical kill-switch inert, because a stale legacy export is exactly what
    the compose entrypoint shim leaves lying around on every boot.
    Empty counts as UNSET: `${FOO:-}` yields "" under docker-compose, and that
    must fall THROUGH to the legacy name, not read as an explicit "off".
    """
    for name in ENABLE_ENV:
        raw = str(env.get(name, "") or "").strip()
        if raw:
            return raw.lower() in _TRUTHY
    return False


def enable_flag_source(env):
    """Which env name actually decided, or None when neither is set."""
    for name in ENABLE_ENV:
        if str(env.get(name, "") or "").strip():
            return name
    return None

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
    out["router_enabled"] = router_enabled_from(kv)
    out["router_enabled_by"] = enable_flag_source(kv)
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
        by = cf.get("router_enabled_by")
        print(f"  container  {cf['name']} [{cf.get('cid', '?')}] {state}"
              f"  router_enabled={cf.get('router_enabled')}"
              f" (by {by or 'neither name set'})"
              f"  mode={cf.get('mode') or '?'}"
              f"  consumer={cf.get('consumer_flag')}")
        # The contested case. Both names set and disagreeing is not an error —
        # the canonical one wins by design — but it is the exact shape that
        # reads as "the flag is on" to every grep, so name it out loud.
        env = cf.get("env") or {}
        vals = {k: str(env.get(k, "") or "").strip() for k in ENABLE_ENV}
        if all(vals.values()) and len({v.lower() in _TRUTHY
                                       for v in vals.values()}) > 1:
            print(paint(f"             NOTE {ENABLE_FLAG}={vals[ENABLE_FLAG]!r} "
                        f"overrides stale "
                        f"{ENABLE_FLAG_LEGACY}={vals[ENABLE_FLAG_LEGACY]!r} "
                        f"— canonical-first, the router agrees", "warn"))
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


# ───────────────────────────────────────────────────────────────── self-test
# Pure stdlib, no pytest, no GPU, no container, no vllm/torch: runnable as
#     /usr/bin/python3 fixes/pn119_doctor.py --selftest
# on the bare host. It is the anti-drift gate for router_enabled_from(): the
# precedence rule exists in two places by necessity (see the module docstring),
# so it is replayed against BOTH and any disagreement is a failure.
_FLAG_VALUES = (None, "", "   ", "0", "1", "true", "TRUE", "True", "yes",
                "on", "off", "no", "false", "2", "banana")


def _load_router():
    """Import fixes/pn119_router.py by path behind a stub torch, or None.

    Same trick fixes/test_h119flag_precedence.py uses. Confined to --selftest:
    the doctor's normal path must never import the sidecar (torch-free hosts).
    """
    import contextlib
    import importlib.util
    import pathlib
    import types

    path = pathlib.Path(__file__).resolve().parent / "pn119_router.py"
    if not path.is_file():
        return None, f"sidecar not present at {path}"
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        torch.float32 = "float32"
        torch.no_grad = contextlib.nullcontext
        sys.modules["torch"] = torch
    try:
        spec = importlib.util.spec_from_file_location("_h119doctor_router", path)
        mod = importlib.util.module_from_spec(spec)
        # In sys.modules BEFORE exec: @dataclass resolves its owning module out
        # of sys.modules on py3.12+ and raises AttributeError if it is absent.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:                                     # noqa: BLE001
        return None, f"sidecar present but failed to import: {e!r}"
    return mod, None


def selftest() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # ── 1. the doctor's own precedence table ────────────────────────────────
    E = router_enabled_from
    check(E({ENABLE_FLAG: "1"}) is True, "canonical alone must enable")
    check(E({ENABLE_FLAG: "0"}) is False, "canonical alone must disable")
    check(E({ENABLE_FLAG_LEGACY: "1"}) is True,
          "legacy alone must still enable (back-compat)")
    check(E({ENABLE_FLAG_LEGACY: "0"}) is False, "legacy alone must disable")
    check(E({}) is False, "neither name set must be off")
    check(E({"GENESIS_ENABLE_PN119": "1"}) is False,
          "GENESIS_ENABLE_PN119 is lane-2's TurboQuant kernel, not this router")
    # THE CONTESTED CASE — the whole reason this function replaced any().
    check(E({ENABLE_FLAG: "0", ENABLE_FLAG_LEGACY: "1"}) is False,
          "canonical=0 with a stale legacy=1 must resolve OFF "
          "(an OR here made the kill-switch inert)")
    check(E({ENABLE_FLAG: "1", ENABLE_FLAG_LEGACY: "0"}) is True,
          "canonical=1 must win over legacy=0")
    # EMPTY == UNSET: `${FOO:-}` yields "" under docker-compose.
    check(E({ENABLE_FLAG: "", ENABLE_FLAG_LEGACY: "1"}) is True,
          "empty canonical must fall THROUGH to legacy, not read as off")
    check(E({ENABLE_FLAG: "   ", ENABLE_FLAG_LEGACY: "1"}) is True,
          "whitespace-only canonical must fall through to legacy")
    check(E({ENABLE_FLAG: "", ENABLE_FLAG_LEGACY: "0"}) is False,
          "empty canonical + legacy=0 is off")
    # And the reported source has to agree with the verdict's provenance.
    check(enable_flag_source({ENABLE_FLAG: "0", ENABLE_FLAG_LEGACY: "1"})
          == ENABLE_FLAG, "canonical must be reported as the deciding name")
    check(enable_flag_source({ENABLE_FLAG: "", ENABLE_FLAG_LEGACY: "1"})
          == ENABLE_FLAG_LEGACY, "an empty canonical does not decide")
    check(enable_flag_source({}) is None, "neither set has no deciding name")

    n_doctor = 14
    print(f"doctor precedence table   {n_doctor - len(fails)}/{n_doctor} ok")

    # ── 2. parity against the router's own helper ───────────────────────────
    mod, why = _load_router()
    if mod is None:
        print(f"router parity             SKIPPED — {why}")
        # A sidecar that is present but unimportable is a regression, not an
        # environment quirk: fail loudly rather than pass a half-run gate.
        if "failed to import" in (why or ""):
            fails.append(f"router parity leg could not run: {why}")
    else:
        check(getattr(mod, "ENABLE_FLAG", None) == ENABLE_FLAG,
              f"router ENABLE_FLAG={getattr(mod, 'ENABLE_FLAG', None)!r} "
              f"!= doctor {ENABLE_FLAG!r}")
        check(getattr(mod, "ENABLE_FLAG_LEGACY", None) == ENABLE_FLAG_LEGACY,
              f"router ENABLE_FLAG_LEGACY="
              f"{getattr(mod, 'ENABLE_FLAG_LEGACY', None)!r} "
              f"!= doctor {ENABLE_FLAG_LEGACY!r}")
        saved = {k: os.environ.get(k) for k in ENABLE_ENV}
        n = 0
        try:
            for canon in _FLAG_VALUES:
                for legacy in _FLAG_VALUES:
                    env = {}
                    for k in ENABLE_ENV:
                        os.environ.pop(k, None)
                    if canon is not None:
                        env[ENABLE_FLAG] = canon
                        os.environ[ENABLE_FLAG] = canon
                    if legacy is not None:
                        env[ENABLE_FLAG_LEGACY] = legacy
                        os.environ[ENABLE_FLAG_LEGACY] = legacy
                    n += 1
                    mine, theirs = router_enabled_from(env), mod._router_enabled()
                    check(mine == theirs,
                          f"DISAGREE {ENABLE_FLAG}={canon!r} "
                          f"{ENABLE_FLAG_LEGACY}={legacy!r}: "
                          f"doctor={mine} router={theirs}")
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        print(f"router parity             {n} env combinations, "
              f"{'MISMATCHES' if fails else 'all agree'}")

    for f in fails:
        print(f"  FAIL {f}")
    print("FAILED" if fails else "PASSED", f"({len(fails)} failure(s))")
    return 1 if fails else 0


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
    ap.add_argument("--selftest", action="store_true",
                    help="replay the enable-flag precedence table against the "
                         "router's own helper and exit (no container, no GPU)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:      # argparse already printed
        return 3 if e.code else 0

    if args.selftest:
        return selftest()

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
