#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify which vLLM capabilities and which of OUR COMMITS are ACTUALLY in effect.

    LIVE / MISSING / DEGRADED per capability, by EFFECT — never by a log line.

Why this exists
---------------
Five times now an upstream bump has silently dropped work we had already done,
and nobody noticed for weeks.  The boot log is not evidence: today alone
produced four patches that logged "applied" while doing nothing —

  BUG-122  SPN71/73/92 announced APPLY, the module gate disagreed, ZERO markers
           in the targets, and the record DB said "applied" for weeks.
  P89      inert on EVERY pin (three drifted anchors).  What looked like proof
           it worked -- populated `reasoning_tokens` in bench rows -- was
           computed CLIENT-side.
  P39a     `apply_all` runs in its own process, then the entrypoint does
           `exec vllm serve`.  `exec` REPLACES the process, so setattr and
           monkey-patch effects never reach the server.  Only TEXT patches
           (which write files) survive.  It logged "applied" every boot for
           months and did nothing.
  PN346B   landed one sub-patch, silently soft-skipped the other half.

So every check here is one of:
  marker   — the literal string a text patch writes INTO the installed file
  file     — a source module physically present at its container path
  symbol   — an identifier compiled into the wheel
  flag     — the gating env var as the container actually received it

Two independent things are checked
----------------------------------
  1. CAPABILITIES (vllm-capability-inventory.json) — is patch X in effect?
  2. COMMITS      (vllm-commit-ledger.json)        — is the DELTA we made to
     patch X still there?  A bump can legitimately restore X in its original
     form while dropping our three later fixes to it; a capability check
     passes and the work is gone.  The commit sweep is the one that catches it.

Modes
-----
  --container NAME   a RUNNING container: boot-time text patches HAVE run, so
                     markers are expected present.  This is the real answer.
  --image REF        an image, offline: the installed vllm is PRISTINE, so
                     boot-patch markers are expected ABSENT.  Use this to
                     verify WHEEL-level capabilities (cherry-picks) and that
                     the lane sources are baked/mountable.  Safe: short-lived
                     CPU container, `--network=none`, no GPU, no server.
  --tree             the host working tree only (no container needed).

Exit codes:  0 all good · 1 capabilities/commits missing · 2 setup error
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_ledger_lib import (  # noqa: E402
    CONTAINER_VLLM, ContainerInspector, ImageInspector, LEDGER_DIR, MOUNTS,
    REPO, TreeInspector, container_env, eprint, image_of, load_json, run,
)

# States
LIVE = "LIVE"
MISSING = "MISSING"
DEGRADED = "DEGRADED"
DARK = "DARK"          # present but its gating flag is off — intentional
INERT = "INERT"        # structurally cannot take effect (exec-discards-setattr)
NA = "N/A"             # not checkable in this mode (e.g. boot markers on an image)
UNVERIFIABLE = "UNVERIFIABLE"  # no effect handle exists at all — a blind spot

# Which model family this rig serves.  Patches belonging to another family are
# inert by design, not lost.  Override with VLLM_LEDGER_MODEL_FAMILY.
MODEL_FAMILY = os.environ.get("VLLM_LEDGER_MODEL_FAMILY", "qwen3.6")


# ------------------------------------------------------------ path aliases --
# Upstream relocates files; the patch modules cope at runtime via
# resolve_vllm_file(), but a static read of the source only sees the literal
# it was written with.  Without these aliases the verifier reports ~15 healthy
# FLA patches as "target path gone" after every relocation.
PATH_ALIASES = [
    ("/model_executor/layers/fla/ops/", "/third_party/flash_linear_attention/ops/"),
    ("/third_party/flash_linear_attention/ops/", "/model_executor/layers/fla/ops/"),
    ("/model_executor/layers/fla/", "/third_party/flash_linear_attention/"),
    ("/third_party/flash_linear_attention/", "/model_executor/layers/fla/"),
]


def expand_targets(targets: list[str], insp) -> list[str]:
    """Add known relocation aliases for any declared target that is absent."""
    out = list(targets)
    for t in targets:
        if insp.exists(t):
            continue
        for old, new in PATH_ALIASES:
            if old in t:
                alt = t.replace(old, new)
                if alt not in out:
                    out.append(alt)
    return out


# --------------------------------------------------------------- flag logic --

def _truthy(v: str | None) -> bool:
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")


def flag_state(cap: dict, env: dict[str, str]) -> tuple[str, str]:
    """Return (state, detail) for the capability's gating flags.

    'on'   at least one gating flag is truthy in the container env
    'off'  a gating flag is present and explicitly falsy
    'unset' no gating flag appears in the env at all.  NOTE: unset is NOT the
            same as off — lane-2 rows with default_on=True engage anyway
            because both composes set GENESIS_SNDR_TRUST_DEFAULT_ON=1.
    """
    flags = [f for f in cap.get("flags", []) if f.startswith(("GENESIS_", "SNDR_"))]
    reg = cap.get("registry") or {}
    for extra in (reg.get("env_flag"),):
        if extra and extra not in flags:
            flags.append(extra)
    for a in (reg.get("env_flag_aliases") or []):
        if a not in flags:
            flags.append(a)
    if not flags:
        return "n/a", ""
    on = [f for f in flags if _truthy(env.get(f))]
    if on:
        return "on", ",".join(on[:3])
    seen = [f for f in flags if f in env]
    if seen:
        return "off", ",".join(seen[:3])
    if reg.get("default_on") and _truthy(env.get("GENESIS_SNDR_TRUST_DEFAULT_ON")):
        return "on", "default_on+TRUST_DEFAULT_ON"
    return "unset", ""


# ------------------------------------------------------------ capabilities --

def plan_reads(caps: list[dict], mode: str) -> list[str]:
    """Every container path the sweep will need, so we prefetch in one call."""
    paths: list[str] = []
    for c in caps:
        paths += c.get("targets", [])
        if c.get("source_container"):
            paths.append(c["source_container"])
    return sorted(set(p for p in paths if p.startswith("/")))


def verify_capability(cap: dict, insp, env: dict[str, str], mode: str) -> dict:
    kind = cap.get("kind")
    markers = cap.get("markers") or []
    targets = cap.get("targets") or []
    fstate, fdetail = flag_state(cap, env)
    reg = cap.get("registry") or {}
    lifecycle = reg.get("lifecycle") or ""
    notes: list[str] = []

    # --- structural verdicts that do not need any file read -----------------
    if cap.get("unreachable_by_dispatcher"):
        return {"state": MISSING, "why": "module has apply() but NO registry row "
                "— the dispatcher can never reach it (orphaned work)",
                "flag": fstate}
    if kind == "registry_only":
        return {"state": UNVERIFIABLE, "why": "registry row with no module; "
                "nothing to check — it still announces an APPLY decision",
                "flag": fstate}
    if lifecycle == "retired" or cap.get("status", "").startswith("retired") \
            or reg.get("deprecated") or cap.get("peer_lifecycle") == "retired":
        notes.append("retired")

    # Model-family gating: a Gemma-4 patch on a Qwen rig is inert BY DESIGN.
    # Reporting 60 of them as lost every run would bury the real signal.
    if cap.get("model_family") == "gemma4" and MODEL_FAMILY != "gemma4":
        return {"state": NA,
                "why": f"gemma4-family patch; this rig serves {MODEL_FAMILY}",
                "flag": fstate}

    # /fixes sidecars only run if the compose entrypoint actually calls them.
    # A commented-out line is a deliberate park (PN73/PN73T superseded by PN76,
    # PN79 retired on BUG-119), not a loss.
    if cap["lane"] == "fixes" and "compose_invoked" in cap:
        if not cap["compose_invoked"]:
            return {"state": DARK,
                    "why": "not invoked by either compose entrypoint"
                           + (" (commented out)" if cap.get("compose_commented") else ""),
                    "flag": fstate}

    # exec-discards-setattr: a lane-1 monkey-patch applied by apply_all in its
    # own process cannot reach the exec'd server.  P39a burned months on this.
    if kind == "monkey_patch" and cap["lane"].startswith("lane1"):
        return {"state": INERT,
                "why": "setattr-only in the apply_all process; `exec vllm serve` "
                       "replaces it before a token is served (P39a class). "
                       "Needs a self-install hook or a text patch.",
                "flag": fstate}

    # --- marker sweep: the primary EFFECT check -----------------------------
    if markers and targets:
        if mode == "image":
            return {"state": NA,
                    "why": "boot-time text patch; an image holds pristine vllm",
                    "flag": fstate}
        targets = expand_targets(targets, insp)
        hits, miss = [], []
        for m in markers:
            found = any(insp.contains(t, m) for t in targets)
            (hits if found else miss).append(m)
        present_targets = [t for t in targets if insp.exists(t)]
        if not present_targets:
            # The target path itself is gone.  Usually upstream moved or
            # deleted the file (fla/ops -> third_party/flash_linear_attention,
            # reasoning/qwen3_reasoning_parser.py removed).  This is a
            # RE-ANCHOR item, distinct from "applied but had no effect".
            state = NA if "retired" in notes else MISSING
            return {"state": state,
                    "why": f"target path no longer exists upstream "
                           f"({os.path.basename(targets[0])}) — needs re-anchor "
                           f"or formal retirement",
                    "flag": fstate}
        if hits and not miss:
            return {"state": LIVE, "why": f"marker present in {len(present_targets)} target(s)",
                    "flag": fstate}
        if hits and miss:
            return {"state": DEGRADED,
                    "why": f"PARTIAL: {len(hits)}/{len(markers)} markers present "
                           f"— missing {miss[:2]} (PN346B class: one sub-patch "
                           f"landed, the other soft-skipped)",
                    "flag": fstate}
        # Zero markers.  Now the whole question is whether it was SUPPOSED to
        # be on.  Never-enabled is not a loss; enabled-and-inert is the alarm.
        if fstate == "off":
            return {"state": DARK, "why": "flag explicitly off; 0 markers as expected",
                    "flag": fstate}
        if cap.get("lane2_suppressed_by_lane1"):
            return {"state": DARK,
                    "why": "lane-2 copy of a lane-1-owned shared id; sndr_lane "
                           "sets GENESIS_DISABLE_* by design",
                    "flag": fstate}
        if "retired" in notes:
            return {"state": NA, "why": "retired; 0 markers as expected",
                    "flag": fstate}
        if fstate == "unset" and not (reg.get("default_on")):
            return {"state": DARK,
                    "why": "no gating flag set anywhere and default_on is false "
                           "— never enabled, nothing lost",
                    "flag": fstate}
        return {"state": MISSING,
                "why": f"0/{len(markers)} markers in an existing target while the "
                       f"gate says it should be on ({fstate}) — "
                       f"announced-but-inert (BUG-122 class)",
                "flag": fstate}

    if markers and not targets:
        # target could not be derived statically (built by a helper function).
        # Fall back to a whole-tree marker grep; see grep_markers().
        return {"state": "_NEEDS_TREE_GREP", "why": "", "flag": fstate,
                "markers": markers}

    # --- module-presence check ---------------------------------------------
    src = cap.get("source_container")
    if src:
        # In image mode there are no bind mounts, so a lane source module is
        # absent by construction — it is delivered at run time from the host
        # tree.  Only a module baked INTO the image is checkable offline.
        if mode == "image" and any(src.startswith(c + "/") for c in MOUNTS.values()):
            return {"state": NA,
                    "why": "lane source arrives by bind mount at run time; "
                           "not present in a bare image",
                    "flag": fstate}
        if insp.exists(src):
            if fstate == "off":
                return {"state": DARK, "why": "module present, flag off", "flag": fstate}
            return {"state": LIVE, "why": "module present at its container path",
                    "flag": fstate}
        return {"state": MISSING, "why": f"module absent: {src}", "flag": fstate}

    return {"state": UNVERIFIABLE, "why": "no marker, no target, no module path",
            "flag": fstate}


def grep_markers(insp, markers: list[str], roots: list[str]) -> set[str]:
    """One Aho-Corasick pass for markers whose target we could not derive.

    GNU grep -F with a pattern file is O(corpus) regardless of pattern count,
    so this stays cheap even with several hundred markers.
    """
    # 5 chars is enough for the house convention ("# PN93:", "# H119:"); an
    # 8-char floor silently dropped those two and reported them MISSING when
    # they were plainly in the file.  Short-but-punctuated markers are fine.
    usable = sorted({m.split("\n")[0].strip() for m in markers
                     if len(m.split("\n")[0].strip()) >= 5})
    if not usable:
        return set()
    if insp.kind == "tree":
        found = set()
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
            fh.write("\n".join(usable) + "\n")
            pat = fh.name
        try:
            targets = [insp.root + r for r in roots] if insp.root not in ("", "/") else roots
            p = subprocess.run(["grep", "-rohF", "--binary-files=without-match",
                                "--exclude-dir=__pycache__", "-f", pat, *targets],
                               capture_output=True, text=True, errors="replace",
                               timeout=900)
            found = {ln.strip() for ln in p.stdout.splitlines()} & set(usable)
        finally:
            os.unlink(pat)
        return found

    script = (
        "import sys,os\n"
        f"pats = {usable!r}\n"
        f"roots = {roots!r}\n"
        "found=set()\n"
        "for root in roots:\n"
        "    for dp,dn,fn in os.walk(root):\n"
        "        dn[:] = [d for d in dn if d != '__pycache__']\n"
        "        for f in fn:\n"
        "            if not f.endswith(('.py','.pyi')): continue\n"
        "            try: b=open(os.path.join(dp,f),encoding='utf-8',errors='replace').read()\n"
        "            except OSError: continue\n"
        "            for p in pats:\n"
        "                if p not in found and p in b: found.add(p)\n"
        "        if len(found)==len(pats): break\n"
        "print('\\x00'.join(sorted(found)))\n"
    )
    rc, out, err = insp._run_python(script)
    if rc != 0:
        eprint(f"  !! marker sweep failed: {err[:200]}")
        return set()
    return {s for s in out.strip().split("\x00") if s}


# ---------------------------------------------------------------- commits ---

def verify_commits(ledger: dict, insp, mode: str) -> list[dict]:
    """Re-probe every commit fingerprint against the TARGET, not just the tree.

    The tree and the running image can disagree — the lanes are mounted ro from
    the tree, but the installed vllm inside the image is a different artifact.
    A fingerprint that lives in a lane source file is checked at its container
    path; one that lives in a patched vllm file is checked by the marker sweep.
    """
    commits = ledger.get("commits", [])
    tokens: set[str] = set()
    for c in commits:
        for fp in c.get("fingerprints", []):
            if fp["kind"] != "file":
                tokens.add(fp["token"])
    roots = ([CONTAINER_VLLM, "/usr/local/lib/python3.12/dist-packages/sndr", "/fixes"]
             if mode != "tree" else list(MOUNTS.keys()))
    found = grep_markers(insp, sorted(tokens), roots) if tokens else set()

    # file-kind fingerprints: map the host repo path onto its container path
    file_fps = {fp["token"] for c in commits for fp in c.get("fingerprints", [])
                if fp["kind"] == "file"}
    file_present: dict[str, bool] = {}
    for t in file_fps:
        host = os.path.join(REPO, t)
        cpath = None
        for h, cont in MOUNTS.items():
            if host.startswith(h + "/"):
                cpath = cont + host[len(h):]
                break
        if mode == "tree" or cpath is None:
            file_present[t] = os.path.exists(host)
        else:
            file_present[t] = insp.exists(cpath)

    # Which host paths are even visible inside a container?  Only the three
    # read-only lane mounts.  A commit that only touched compose YAML, ops
    # tooling or a boot guard has NOTHING inside the container by construction
    # — calling those "no longer in effect" would bury the real signal under
    # ~30 false alarms every run.
    def _reaches_container(c: dict) -> bool:
        if mode == "tree":
            return True
        for f in c.get("runtime_sample", []) or c.get("files_sample", []):
            host = os.path.join(REPO, f)
            if any(host.startswith(h + "/") for h in MOUNTS):
                return True
        return False

    out = []
    for c in commits:
        fps = c.get("fingerprints", [])
        if not fps:
            continue
        hits = [fp for fp in fps
                if (file_present.get(fp["token"], False) if fp["kind"] == "file"
                    else fp["token"] in found)]
        if hits:
            state = LIVE
        elif c["verdict"] in ("doc-only", "config-drift", "removal",
                              "superseded", "accepted-loss"):
            state = NA
        elif not _reaches_container(c):
            state = NA
        else:
            state = MISSING
        out.append({**c, "target_state": state,
                    "target_hits": [h["token"][:60] for h in hits[:2]]})
    return out


# ------------------------------------------------------------------ report --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--container", help="running container name (the real answer)")
    g.add_argument("--image", help="image ref (wheel-level check, offline)")
    g.add_argument("--tree", action="store_true", help="host working tree only")
    ap.add_argument("--inventory",
                    default=os.path.join(LEDGER_DIR, "vllm-capability-inventory.json"))
    ap.add_argument("--commits",
                    default=os.path.join(LEDGER_DIR, "vllm-commit-ledger.json"))
    ap.add_argument("--json", help="write the full machine-readable result here")
    ap.add_argument("--baseline", help="compare against a previous --json result "
                                       "and FAIL on any regression (the bump gate)")
    ap.add_argument("--no-commits", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.container:
        insp, mode = ContainerInspector(args.container), "container"
    elif args.image:
        insp, mode = ImageInspector(args.image), "image"
    else:
        insp, mode = TreeInspector("/"), "tree"

    try:
        inv = load_json(args.inventory)
    except OSError as e:
        eprint(f"FATAL: inventory unreadable: {e}")
        return 2
    caps = inv["capabilities"]

    env = container_env(args.container) if args.container else {}
    if args.container:
        img = image_of(args.container)
        print(f"# target   : container {args.container}")
        print(f"# image    : {img}")
    elif args.image:
        print(f"# target   : image {args.image} (PRISTINE vllm — wheel-level check)")
    else:
        print(f"# target   : host tree {REPO}")
    print(f"# inventory: {len(caps)} capabilities")

    if mode != "tree":
        insp.prefetch(plan_reads(caps, mode))

    results = []
    need_grep: list[dict] = []
    for c in caps:
        r = verify_capability(c, insp, env, mode)
        if r["state"] == "_NEEDS_TREE_GREP":
            need_grep.append({"cap": c, "res": r})
        results.append({"key": c["key"], "id": c["id"], "lane": c["lane"],
                        "kind": c["kind"], "summary": c.get("summary", "")[:100],
                        **r})

    if need_grep:
        allm: list[str] = []
        for n in need_grep:
            allm += n["res"]["markers"]
        roots = ([CONTAINER_VLLM] if mode != "tree" else [list(MOUNTS.keys())[0]])
        found = grep_markers(insp, allm, roots) if mode != "tree" else set()
        for n, r in zip(need_grep, [x for x in results if x["state"] == "_NEEDS_TREE_GREP"]):
            ms = [m.split("\n")[0].strip() for m in n["res"]["markers"]]
            hit = [m for m in ms if m in found]
            if mode == "image":
                r["state"], r["why"] = NA, "boot marker; image is pristine"
            elif hit:
                r["state"], r["why"] = LIVE, "marker found by tree sweep (target undeclared)"
            elif r["flag"] == "off":
                r["state"], r["why"] = DARK, "flag off; marker absent as expected"
            else:
                r["state"], r["why"] = MISSING, "marker absent anywhere in installed vllm"
            r.pop("markers", None)

    tally: dict[str, int] = {}
    for r in results:
        tally[r["state"]] = tally.get(r["state"], 0) + 1

    print("\n== CAPABILITIES ==")
    for k in (LIVE, DARK, NA, INERT, UNVERIFIABLE, DEGRADED, MISSING):
        if tally.get(k):
            print(f"  {k:12s} {tally[k]}")

    bad = [r for r in results if r["state"] in (MISSING, DEGRADED)]
    if bad and not args.quiet:
        print(f"\n-- ACT ON THESE ({len(bad)}) --")
        for r in sorted(bad, key=lambda x: (x["state"], x["lane"], x["key"]))[:80]:
            print(f"  {r['state']:9s} {r['lane']:15s} {r['key'][:38]:38s} {r['why'][:88]}")
        if len(bad) > 80:
            print(f"  ... and {len(bad)-80} more (use --json)")

    commit_results = []
    if not args.no_commits and os.path.exists(args.commits):
        led = load_json(args.commits)
        commit_results = verify_commits(led, insp, mode)
        ct: dict[str, int] = {}
        for c in commit_results:
            ct[c["target_state"]] = ct.get(c["target_state"], 0) + 1
        print(f"\n== COMMITS ({len(commit_results)} fingerprinted) ==")
        for k in (LIVE, NA, MISSING):
            if ct.get(k):
                print(f"  {k:12s} {ct[k]}")
        gone = [c for c in commit_results if c["target_state"] == MISSING]
        if gone:
            print(f"\n-- COMMITS NO LONGER IN EFFECT ({len(gone)}) --")
            for c in sorted(gone, key=lambda x: x["date"]):
                print(f"  {c['repo']:11s} {c['sha']} {c['date']} {c['subject'][:74]}")

    out = {"mode": mode, "target": insp.label, "tally": tally,
           "capabilities": results, "commits": commit_results}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.json}")

    rc = 0
    if args.baseline and os.path.exists(args.baseline):
        base = load_json(args.baseline)
        bmap = {c["key"]: c["state"] for c in base.get("capabilities", [])}
        regressed = [r for r in results
                     if bmap.get(r["key"]) in (LIVE, DARK)
                     and r["state"] in (MISSING, DEGRADED)]
        bc = {c["sha"]: c["target_state"] for c in base.get("commits", [])}
        creg = [c for c in commit_results
                if bc.get(c["sha"]) == LIVE and c["target_state"] == MISSING]
        print("\n== BUMP GATE (vs baseline) ==")
        print(f"  capabilities regressed: {len(regressed)}")
        print(f"  commits dropped       : {len(creg)}")
        for r in regressed[:40]:
            print(f"    CAP  {r['key'][:44]:44s} {bmap[r['key']]} -> {r['state']}")
        for c in creg[:40]:
            print(f"    CMT  {c['sha']} {c['date']} {c['subject'][:66]}")
        if regressed or creg:
            print("\n  GATE FAILED — the bump dropped work that was in effect before.")
            rc = 1
        else:
            print("  GATE PASSED — nothing that was in effect went missing.")
    elif bad:
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
