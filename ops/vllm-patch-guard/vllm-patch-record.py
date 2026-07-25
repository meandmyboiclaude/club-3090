#!/usr/bin/env python3
"""Record every vLLM boot's applied-patch set into Postgres, and diff it.

Why this exists — three failures inside one week (2026-07-13..19):

  1. FORGETTING   the expected patch count lived in chat scrollback. Every
                  retelling gave a different answer (57 / 88 / 93 / 122 / 155
                  for a true 134). Now it is a row, and it is queried.
  2. BAD DATA     counts came from ad-hoc greps that silently measured the wrong
                  thing: `journalctl CONTAINER_NAME=` returns EVERY incarnation
                  of a name (3 recreates -> 3x inflation), `podman logs`
                  truncates under the journald driver, and "Genesis Results"
                  prints ONCE PER LANE, so quoting one line halves the total.
                  Extraction lives here once, and a log that fails the trust
                  gates is REFUSED rather than reported with a caveat.
  3. SILENT LOSS  touching vLLM (image bump, rebase, compose edit) drops patches
                  whose anchors drifted, and nothing noticed for weeks. Every
                  boot is now diffed against the previous one, by name.

Usage:
  vllm-patch-record.py record <container> [--log FILE]
  vllm-patch-record.py diff <container> [--against N]   # default: previous boot
  vllm-patch-record.py history <container> [--limit 15]
  vllm-patch-record.py missing <container>              # ever-seen minus current
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

DSN = os.environ.get(
    "VLLMOPS_DSN", "host=127.0.0.1 dbname=vllmops user=phoenix password=phoenix_local_2026"
)


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


# ── extraction: ONE implementation, shared by every command ──────────────────

def container_started(ctr: str) -> str:
    for pre in ([], ["sudo"]):
        try:
            out = subprocess.run(
                pre + ["podman", "inspect", ctr, "--format", "{{.State.StartedAt}}"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.TimeoutExpired):
            continue
    sys.exit(f"container '{ctr}' not found — cannot scope the log")


def boot_log(ctr: str) -> tuple[str, str]:
    """Container-scoped log. Anchoring on StartedAt is what stops concatenated
    incarnations from inflating every count by an integer multiple."""
    started = container_started(ctr)
    # podman: "2026-07-19 10:23:12.336855471 +0200 CEST" — GNU date rejects
    # fraction plus both offset and zone name.
    clean = re.sub(r"\.\d+", "", started)
    clean = re.sub(r" [A-Z]{3,5}$", "", clean)
    since = subprocess.run(["date", "-d", clean, "+%Y-%m-%d %H:%M:%S"],
                           capture_output=True, text=True).stdout.strip()
    if not since:
        sys.exit(f"unparseable StartedAt: {started}")
    log = subprocess.run(
        ["journalctl", f"CONTAINER_NAME={ctr}", "--since", since, "--no-pager", "-o", "cat"],
        capture_output=True, text=True,
    ).stdout
    return log, clean


def assert_trustworthy(log: str) -> int:
    """Refuse untrustworthy input outright. A number with an asterisk still gets
    quoted later without the asterisk."""
    if not log.strip():
        sys.exit("REFUSING: boot log is empty (journald retention is ~21h here)")
    lanes = len(re.findall(r"Genesis Results:", log))
    if lanes < 1:
        sys.exit("REFUSING: no 'Genesis Results' — log truncated or pre-dispatch")
    boots = len(re.findall(r"Initial free memory", log))
    if boots > 1:
        sys.exit(f"REFUSING: log spans {boots} boots — every count would be {boots}x inflated")
    return lanes


def extract(log: str) -> dict:
    def total(pat):
        return sum(int(x) for x in re.findall(pat, log)) or 0

    # Sum EVERY lane — legacy genesis.apply_all AND the sndr v12 registry.
    disp = total(r"Genesis Results: (\d+) applied")
    house_names = sorted({
        m.group(1) for line in log.splitlines()
        if (m := re.match(r"^\[([a-z0-9_-]+)\]", line))
        and re.search(r"applied|wired|patched", line, re.I)
    })
    disp_names = sorted(set(re.findall(r"APPLY ([A-Z]+\d+[a-zA-Z]*)", log)))
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    img = re.search(r"image[=: ]+(\S*vllm\S*)", log)
    # "APPLY X" is the dispatcher ANNOUNCING an attempt, not confirming success:
    # a patch can announce and then fail to anchor. Subtract the drifted set so
    # 'applied' rows mean applied. Without this the name count (106) contradicts
    # the dispatcher's own tally (103) and the table is self-inconsistent.
    drift_names = sorted(set(
        re.findall(r"DRIFT skipped:\s*([A-Za-z]+\d+[a-zA-Z]*)", log)
    ) | set(
        re.findall(r"⚠️\s+([A-Za-z]+\d+[a-zA-Z]*)\s+.*required anchor", log)
    ))
    disp_names = sorted(set(disp_names) - set(drift_names))
    # BUG-122 2026-07-25: lane-2 (sndr) announces "APPLY <id>" as a DECISION and
    # used to log no outcome, so a module whose own env gate disagreed with the
    # entry-level decision skipped silently and was banked here as applied
    # (SPN71/73/92 phantoms: DB said applied, targets had 0 markers). sndr_lane
    # now emits "[Genesis lane-2/sndr] RESULT <status>: <name>". Trust the
    # RESULT over the announcement whenever present; logs predating that fix
    # carry no RESULT lines and behave exactly as they used to.
    lane2_results = re.findall(
        r"\[Genesis lane-2/sndr\] RESULT (\w+): ([A-Za-z]+\d+[a-zA-Z]*)", log
    )
    lane2_not_applied = {
        name for status, name in lane2_results if status != "applied"
    }
    # 2026-07-25 (regression fix): the RESULT set is LANE-2-ONLY, but disp_names
    # is CROSS-LANE — both lanes log through "[Genesis Dispatcher] APPLY <id>".
    # sndr_lane.apply_policy() suppresses the ~118 SHARED ids precisely because
    # lane-1 owns and applies them, and now emits "RESULT skipped: <id>" for
    # each — so a naive intersection flagged the patches lane-1 SUCCEEDED on
    # (measured: 64 of them on the 18:02 boot). Split the log at lane-2's
    # policy banner and only consider ids lane-2 itself announced, never ones
    # lane-1 announced or lane-2 later reported applied.
    _lines = log.splitlines()
    _cut = next((k for k, l in enumerate(_lines)
                 if "lane-2/sndr] policy" in l), 0)
    lane1_announced = set(re.findall(
        r"APPLY ([A-Z]+\d+[a-zA-Z]*)", "\n".join(_lines[:_cut])))
    lane2_announced = set(re.findall(
        r"APPLY ([A-Z]+\d+[a-zA-Z]*)", "\n".join(_lines[_cut:])))
    lane2_applied = {name for status, name in lane2_results if status == "applied"}
    phantom_names = sorted(
        (set(disp_names) & lane2_not_applied & lane2_announced)
        - lane2_applied - lane1_announced
    )
    disp_names = sorted(set(disp_names) - set(phantom_names))
    # Do NOT adjust `disp`. It comes from "Genesis Results: N applied" ==
    # PatchStats.applied_count (sndr/apply/_state.py), which counts OUTCOMES:
    # a phantom already lands there as _skipped(...) and was never included.
    # Subtracting here double-counted and could drive the headline negative.
    return {
        "lanes": len(re.findall(r"Genesis Results:", log)),
        "dispatcher_applied": disp,
        "house_applied": len(house_names),
        "total_active": disp + len(house_names),
        "sub_patches": total(r"applied (\d+) sub-patches?"),
        "direct_hunks": total(r"applied (\d+) hunk\(s\)"),
        "drift_skipped": len(re.findall(r"DRIFT skipped", log)),
        "hard_failures": len(re.findall(r"FAILED to apply|patch failed", log)),
        "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "image": img.group(1) if img else None,
        "disp_names": disp_names,
        "house_names": house_names,
        "phantom_names": phantom_names,
        "drift_names": sorted(set(re.findall(r"DRIFT skipped: (\w+)", log))),
    }


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_record(a):
    log, started = boot_log(a.container)
    assert_trustworthy(log)
    d = extract(log)
    with db() as con, con.cursor() as cur:
        cur.execute(
            """INSERT INTO boots (container, started_at, image, kv_tokens, lanes,
                 dispatcher_applied, house_applied, total_active, sub_patches,
                 direct_hunks, drift_skipped, hard_failures, log_path)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (container, started_at) DO UPDATE SET
                 captured_at=now(), total_active=EXCLUDED.total_active,
                 dispatcher_applied=EXCLUDED.dispatcher_applied,
                 house_applied=EXCLUDED.house_applied,
                 sub_patches=EXCLUDED.sub_patches, drift_skipped=EXCLUDED.drift_skipped,
                 hard_failures=EXCLUDED.hard_failures
               RETURNING id""",
            (a.container, started, d["image"], d["kv_tokens"], d["lanes"],
             d["dispatcher_applied"], d["house_applied"], d["total_active"],
             d["sub_patches"], d["direct_hunks"], d["drift_skipped"],
             d["hard_failures"], a.log or ""),
        )
        boot_id = cur.fetchone()[0]
        cur.execute("DELETE FROM boot_patches WHERE boot_id=%s", (boot_id,))
        rows = ([(boot_id, p, "dispatcher", "applied") for p in d["disp_names"]]
                + [(boot_id, p, "house", "applied") for p in d["house_names"]]
                + [(boot_id, p, "dispatcher", "drift") for p in d["drift_names"]]
                + [(boot_id, p, "dispatcher", "phantom")
                   for p in d["phantom_names"]])
        cur.executemany(
            "INSERT INTO boot_patches (boot_id,patch,lane,status) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT DO NOTHING", rows)
        con.commit()
    print(f"boot {boot_id} recorded: total_active={d['total_active']} "
          f"(dispatcher {d['dispatcher_applied']} + house {d['house_applied']}) "
          f"lanes={d['lanes']} sub={d['sub_patches']} drift={d['drift_skipped']} "
          f"fail={d['hard_failures']} kv={d['kv_tokens']}")
    return 0


def cmd_diff(a):
    with db() as con, con.cursor() as cur:
        cur.execute("SELECT id,started_at,total_active,drift_skipped,hard_failures "
                    "FROM boots WHERE container=%s ORDER BY started_at DESC LIMIT 2",
                    (a.container,))
        rows = cur.fetchall()
        if len(rows) < 2:
            print("need at least two recorded boots to diff")
            return 0
        cur_b, prev_b = rows[0], rows[1]
        def names(bid):
            cur.execute("SELECT patch FROM boot_patches WHERE boot_id=%s AND status='applied'", (bid,))
            return {r[0] for r in cur.fetchall()}
        now_n, was_n = names(cur_b[0]), names(prev_b[0])
    lost, gained = sorted(was_n - now_n), sorted(now_n - was_n)
    print(f"current  boot {cur_b[0]} {cur_b[1]}  total={cur_b[2]} drift={cur_b[3]} fail={cur_b[4]}")
    print(f"previous boot {prev_b[0]} {prev_b[1]}  total={prev_b[2]} drift={prev_b[3]} fail={prev_b[4]}")
    if lost:
        print(f"\n!! LOST {len(lost)}: " + ", ".join(lost))
    if gained:
        print(f"\n++ GAINED {len(gained)}: " + ", ".join(gained))
    if not lost and not gained:
        print("\nno change in the applied patch set")
    return 1 if lost or cur_b[2] < prev_b[2] else 0


def cmd_history(a):
    with db() as con, con.cursor() as cur:
        cur.execute("""SELECT started_at, total_active, dispatcher_applied, house_applied,
                              sub_patches, drift_skipped, hard_failures, kv_tokens
                       FROM boots WHERE container=%s ORDER BY started_at DESC LIMIT %s""",
                    (a.container, a.limit))
        print(f"{'started':<26}{'total':>6}{'disp':>6}{'house':>6}{'sub':>6}{'drift':>6}{'fail':>5}{'kv':>9}")
        for r in cur.fetchall():
            print(f"{str(r[0])[:25]:<26}{r[1]:>6}{r[2]:>6}{r[3]:>6}{r[4]:>6}{r[5]:>6}{r[6]:>5}{(r[7] or 0):>9}")
    return 0


def cmd_missing(a):
    """Patches seen in ANY past boot but absent from the latest — the slow-drift
    detector. A patch lost three image bumps ago never shows in a pairwise diff."""
    with db() as con, con.cursor() as cur:
        cur.execute("SELECT id FROM boots WHERE container=%s ORDER BY started_at DESC LIMIT 1",
                    (a.container,))
        row = cur.fetchone()
        if not row:
            print("no boots recorded"); return 0
        cur.execute("""SELECT DISTINCT p.patch FROM boot_patches p
                       JOIN boots b ON b.id=p.boot_id
                       WHERE b.container=%s AND p.status='applied'
                         AND p.patch NOT IN (SELECT patch FROM boot_patches
                                             WHERE boot_id=%s AND status='applied')
                       ORDER BY 1""", (a.container, row[0]))
        gone = [r[0] for r in cur.fetchall()]
    print(f"never-again patches ({len(gone)}): " + (", ".join(gone) if gone else "none"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "diff", "history", "missing"):
        s = sub.add_parser(name)
        s.add_argument("container")
        if name == "record":
            s.add_argument("--log", default="")
        if name == "history":
            s.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()
    return {"record": cmd_record, "diff": cmd_diff,
            "history": cmd_history, "missing": cmd_missing}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
