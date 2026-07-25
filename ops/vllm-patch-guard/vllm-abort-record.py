#!/usr/bin/env python3
"""Count HTTP-200-invisible aborts per vLLM container per hour, into Postgres.

Why this exists — the failure class is invisible end to end (BUG-127/BUG-126):
a CUDA OOM inside the engine step makes oom_resilience v7 abort every running
request; the server returns HTTP 200 with finish_reason=abort and ZERO
completion tokens; the caller counts 0 errors. On 2026-07-24 a no-think prod
replay lost 82/200 requests this way and the runner reported a clean run.
This recorder makes the class a queryable hourly rate — the judge for the
util-0.935 permanence call, the BUG-127 fragmentation-trend test, and the
prod flip.

What is counted (per container, per hour):
  aborts      responses finished with finish_reason=abort. Server-side truth:
              every aborted request appears exactly once in an oom_resilience
              "Aborting N running request(s)" line (v7 sends the abort output
              to the client), so aborts = sum of N.
  oom_events  runtime [oom_resilience] log lines (OOM step events, streak-cap
              give-ups, notify/abort failures). Boot-time apply/skip lines are
              deliberately NOT counted — every boot logs "v7 applied".
  requests    completed /v1/chat/completions + /v1/completions POSTs with
              HTTP 200 — the rate denominator. Aborted requests ALSO return
              200, so they are inside this denominator.

Usage:
  vllm-abort-record.py scan [--match vllm]        # journald, running ctrs, prev+curr hour -> DB
  vllm-abort-record.py count --log FILE           # banked container log, no DB (validation)
  vllm-abort-record.py count-results FILE.jsonl   # bench results jsonl: finish_reason=abort rows
  vllm-abort-record.py history <container> [--limit 24]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

DSN = os.environ.get(
    "VLLMOPS_DSN", "host=127.0.0.1 dbname=vllmops user=phoenix password=phoenix_local_2026"
)

# Runtime emissions of patch_oom_resilience v7 (club-3090/fixes/patch_oom_resilience.py).
# Keep in sync with that file; boot-time lines ("v7 applied", "already applied",
# FATAL generation mismatch, anchor-drift) must stay excluded.
RX_OOM_RUNTIME = re.compile(
    r"\[oom_resilience\] (?:CUDA OOM during engine step"
    r"|\d+ consecutive OOM recoveries"
    r"|client abort-notify failed"
    r"|abort failed)"
)
RX_ABORTING = re.compile(r"\[oom_resilience\] CUDA OOM during engine step \(\w+\)\. "
                         r"Aborting (\d+) running request\(s\)")
RX_REQUEST = re.compile(r'"POST /v1/(?:chat/)?completions HTTP/1\.1" 200')
# vLLM log-line stamp: `ERROR 07-24 05:10:12 [core.py:1341] ...` (no year).
RX_STAMP = re.compile(r"\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) (\d\d-\d\d) (\d\d):\d\d:\d\d ")


def db():
    try:
        import psycopg
        return psycopg.connect(DSN)
    except ImportError:
        pass
    try:
        import psycopg2
        return psycopg2.connect(DSN)
    except ImportError:
        sys.exit("need psycopg or psycopg2 (pip install psycopg[binary])")


def ensure_schema(con):
    with con.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS abort_hourly (
              container   text        NOT NULL,
              hour        timestamptz NOT NULL,
              aborts      integer     NOT NULL DEFAULT 0,
              oom_events  integer     NOT NULL DEFAULT 0,
              requests    integer     NOT NULL DEFAULT 0,
              captured_at timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (container, hour)
            )""")
    con.commit()


# ── extraction: ONE implementation, shared by every command ──────────────────

def extract(text: str) -> dict:
    aborts = sum(int(n) for n in RX_ABORTING.findall(text))
    oom_events = sum(1 for line in text.splitlines() if RX_OOM_RUNTIME.search(line))
    requests = len(RX_REQUEST.findall(text))
    return {"aborts": aborts, "oom_events": oom_events, "requests": requests}


def per_hour_breakdown(text: str) -> dict:
    """File-mode helper: bucket the STAMPED abort lines by MM-DD HH. vLLM's
    uvicorn access lines carry no stamp, so requests can't be bucketed from a
    bare file — journald mode (scan) is the authoritative hourly path."""
    hours: dict[str, dict] = {}
    for line in text.splitlines():
        m_ab = RX_ABORTING.search(line)
        m_oom = RX_OOM_RUNTIME.search(line)
        if not (m_ab or m_oom):
            continue
        st = RX_STAMP.search(line)
        key = f"{st.group(1)} {st.group(2)}h" if st else "unstamped"
        h = hours.setdefault(key, {"aborts": 0, "oom_events": 0})
        if m_oom:
            h["oom_events"] += 1
        if m_ab:
            h["aborts"] += int(m_ab.group(1))
    return hours


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_count(a):
    files = sorted(f for pat in a.log for f in glob.glob(pat))
    if not files:
        sys.exit(f"no files match {a.log}")
    grand = {"aborts": 0, "oom_events": 0, "requests": 0}
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        d = extract(text)
        for k in grand:
            grand[k] += d[k]
        pct = 100.0 * d["aborts"] / d["requests"] if d["requests"] else 0.0
        print(f"{os.path.basename(f):<58} aborts={d['aborts']:>4} "
              f"oom_lines={d['oom_events']:>4} requests={d['requests']:>5} "
              f"abort_pct={pct:5.1f}%")
        if a.per_hour and d["aborts"]:
            for hr, h in sorted(per_hour_breakdown(text).items()):
                print(f"    {hr}: aborts={h['aborts']} oom_lines={h['oom_events']}")
    if len(files) > 1:
        pct = 100.0 * grand["aborts"] / grand["requests"] if grand["requests"] else 0.0
        print(f"{'TOTAL':<58} aborts={grand['aborts']:>4} "
              f"oom_lines={grand['oom_events']:>4} requests={grand['requests']:>5} "
              f"abort_pct={pct:5.1f}%")
    return 0


def cmd_count_results(a):
    total = aborts = abort_zero_ctok = 0
    for line in open(a.file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        total += 1
        if r.get("finish_reason") == "abort":
            aborts += 1
            if r.get("completion_tokens") == 0:
                abort_zero_ctok += 1
    pct = 100.0 * aborts / total if total else 0.0
    print(f"{os.path.basename(a.file)}: rows={total} finish_reason=abort={aborts} "
          f"(of which completion_tokens=0: {abort_zero_ctok}) abort_pct={pct:.1f}%")
    return 0


def running_containers(match: str) -> list[str]:
    # UNION of rootless and root podman: the vLLM units run root containers
    # (vllm-tcbench-8021 is invisible to an unprivileged `podman ps`), and a
    # successful-but-empty rootless list must not short-circuit the sudo view.
    names: set[str] = set()
    for pre in ([], ["sudo", "-n"]):
        try:
            out = subprocess.run(
                pre + ["podman", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                names |= {n for n in out.stdout.split() if match in n}
        except (OSError, subprocess.TimeoutExpired):
            continue
    return sorted(names)


def journal_window(ctr: str, since: datetime, until: datetime) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    return subprocess.run(
        ["journalctl", f"CONTAINER_NAME={ctr}",
         "--since", since.strftime(fmt), "--until", until.strftime(fmt),
         "--no-pager", "-o", "cat"],
        capture_output=True, text=True,
    ).stdout


def cmd_scan(a):
    ctrs = running_containers(a.match)
    if not ctrs:
        print(f"no running containers match '{a.match}' — nothing to scan")
        return 0
    now = datetime.now().astimezone()
    curr = now.replace(minute=0, second=0, microsecond=0)
    prev = curr - timedelta(hours=1)
    with db() as con:
        ensure_schema(con)
        with con.cursor() as cur:
            for ctr in ctrs:
                for hour in (prev, curr):
                    text = journal_window(ctr, hour, hour + timedelta(hours=1))
                    d = extract(text)
                    # Re-scans converge each hour row to its final value; an
                    # all-zero row is still written so "no data" and "no
                    # aborts" stay distinguishable.
                    cur.execute(
                        """INSERT INTO abort_hourly
                             (container, hour, aborts, oom_events, requests)
                           VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT (container, hour) DO UPDATE SET
                             aborts=EXCLUDED.aborts, oom_events=EXCLUDED.oom_events,
                             requests=EXCLUDED.requests, captured_at=now()""",
                        (ctr, hour, d["aborts"], d["oom_events"], d["requests"]),
                    )
                    pct = 100.0 * d["aborts"] / d["requests"] if d["requests"] else 0.0
                    print(f"{ctr} {hour:%Y-%m-%d %H}h: aborts={d['aborts']} "
                          f"oom_lines={d['oom_events']} requests={d['requests']} "
                          f"abort_pct={pct:.1f}%")
        con.commit()
    return 0


def cmd_history(a):
    with db() as con:
        ensure_schema(con)
        with con.cursor() as cur:
            cur.execute(
                """SELECT hour, aborts, oom_events, requests
                   FROM abort_hourly WHERE container=%s
                   ORDER BY hour DESC LIMIT %s""",
                (a.container, a.limit))
            rows = cur.fetchall()
    print(f"{'hour':<22}{'aborts':>8}{'oom':>6}{'reqs':>7}{'abort%':>8}")
    for hour, ab, oom, req in rows:
        pct = 100.0 * ab / req if req else 0.0
        print(f"{str(hour)[:21]:<22}{ab:>8}{oom:>6}{req:>7}{pct:>7.1f}%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--match", default=os.environ.get("VLLM_WATCH_MATCH", "vllm"))
    s = sub.add_parser("count")
    s.add_argument("--log", action="append", required=True,
                   help="container log file or glob (repeatable)")
    s.add_argument("--per-hour", action="store_true")
    s = sub.add_parser("count-results")
    s.add_argument("file")
    s = sub.add_parser("history")
    s.add_argument("container")
    s.add_argument("--limit", type=int, default=24)
    a = ap.parse_args()
    return {"scan": cmd_scan, "count": cmd_count,
            "count-results": cmd_count_results, "history": cmd_history}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
