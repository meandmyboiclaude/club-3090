#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the COMMIT LEDGER: a content fingerprint for every commit we own.

Why commits and not patches
---------------------------
Asking "is patch P101 applied?" is not enough.  A bump can legitimately restore
P101 in its ORIGINAL form while silently discarding the three later commits that
fixed it — and a presence check passes.  Every commit is a delta we own; the
unit of tracking has to be the commit.

Why not commit hashes
---------------------
Hashes do not survive rebases, cherry-pick storms or squashes, and we do all
three.  Today's KVQ squash on club-dev1474-cherry-max is exactly the mechanism
that broke P101, P89, PN119-TQ and P18B.  So each commit is reduced to CONTENT
FINGERPRINTS: distinctive tokens the commit ADDED, which must still be findable
in the current tree and (where applicable) in the installed code inside the
image.

Fingerprint selection
---------------------
From `git show -U0` we harvest candidates that are durable and distinctive:
  * marker/log strings  ("Genesis P101 ...", "# PN96:", "[pn100-auto-...]")
  * env flag names      (GENESIS_ENABLE_*, SNDR_*, VLLM_*)
  * new identifiers     (_genesis_*, def <name>, class <name>, CONST = )
  * added file paths    (a whole new module IS a fingerprint)
Candidates are ranked by rarity in the current tree (a token that occurs once is
worth far more than one that occurs 200 times) and length.  We keep up to
`--keep` (default 5) INDEPENDENT fingerprints per commit so one incidental
deletion cannot raise a false alarm.

Verdicts
--------
  intact        >=1 fingerprint still present in the tree
  LOST          a RUNTIME-CODE commit whose every distinctive token is gone
                <-- the list that matters; each entry needs a human verdict of
                    "really lost" vs "superseded with evidence"
  PARTIAL-LOSS  a whole FILE the commit added is missing, even though some of
                its incidental tokens survive.  Decisive: without this rule
                KVQ-2 f785a3a5f read "intact" off two generic Triton kernel
                params while its entire new sink.py was absent.
  config-drift  only compose/yaml/json touched and the values were later
                retuned — expected, not a loss (KV pins, ctx sizes, ports)
  doc-only      only docs/benchmarks/READMEs touched; no runtime fingerprint
  removal       the commit only deleted code (its "fingerprint" is an absence)
  unfingerprintable  nothing distinctive could be derived (needs hand review)

Usage:
    python3 vllm-commit-fingerprint.py                 # all repos
    python3 vllm-commit-fingerprint.py --repo club     # one repo
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_ledger_lib import LEDGER_DIR, REPO, dump_json, eprint, git, run  # noqa: E402

VLLM_BUILD = os.environ.get("VLLM_BUILD", "/home/user/engines/vllm-build")
KVQ2SINK = os.environ.get("VLLM_KVQ2SINK", "/home/user/engines/vllm-kvq2sink")

# Paths in club-3090 that carry vLLM capability work.
CLUB_LANES = [
    "models/qwen3.6-27b/vllm",
    "fixes",
    "ops/vllm-patch-guard",
    "docs/UPSTREAM.md",
    "docs/PATCH_POLICY.md",
]

# The branches in ~/engines/vllm-build that carry OUR work.  Everything else
# in that repo (the `pr*` branches) is an upstream PR head kept as a
# cherry-pick source — 2265 commits of other people's code.
DOC_EXT = {".md", ".txt", ".rst", ".png", ".jpg", ".svg", ".csv"}

OUR_BUILD_BRANCHES = [
    "club-dev1474-cherry-max",   # wheel v2 — the shipped tree
    "club-dev1474-cherry",       # wheel v1
    "club-dev1060-cherry",       # the 07-13 line
    "kvq-dev",                   # squashed into cherry-max; local-only
    "mtpq-dev",                  # squashed into cherry-max; local-only
]
CONFIG_EXT = {".yml", ".yaml", ".json", ".toml", ".ini", ".env", ".jinja"}
# Extensions whose loss is a real capability loss (runtime behaviour).
RUNTIME_EXT = {".py", ".pyx", ".pyi", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h",
               ".hpp", ".sh", ".so", ".map"}
DOCISH = re.compile(r"(^|/)(docs?|benchmarks?|CHANGELOG|README|tests?)(/|$|\.)", re.I)

# ------------------------------------------------------------- candidates ---

RX_QUOTED = re.compile(r"""["']([^"'\n]{14,160})["']""")
RX_FLAG = re.compile(r"\b((?:GENESIS|SNDR)_[A-Z0-9_]{6,})\b")
RX_VLLMFLAG = re.compile(r"\b(VLLM_[A-Z0-9_]{6,})\b")
RX_DEF = re.compile(r"^\+\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]{7,})")
RX_CLASS = re.compile(r"^\+\s*class\s+([A-Za-z_][A-Za-z0-9_]{7,})")
RX_CONST = re.compile(r"^\+([A-Z_][A-Z0-9_]{9,})\s*=")
RX_GENID = re.compile(r"\b(_genesis_[a-z0-9_]{6,}|_sndr_[a-z0-9_]{6,})\b")

# Tokens that are structurally useless as fingerprints.
BAD_TOKEN = re.compile(
    r"^(?:https?://|/usr/local/|utf-8$|__main__|SPDX|Apache-2\.0)|"
    r"^[-=#*_ ]+$|^\s*$"
)
NOISE_WORDS = {
    "Co-Authored-By", "Author: Sandermage", "from __future__ import annotations",
}


def _clean(tok: str) -> str | None:
    tok = tok.strip()
    if len(tok) < 10 or len(tok) > 200:
        return None
    if "\n" in tok or "\t" in tok:
        return None
    if BAD_TOKEN.search(tok):
        return None
    if tok in NOISE_WORDS:
        return None
    if not any(c.isalnum() for c in tok):
        return None
    return tok


def candidates_for(repo: str, sha: str, paths: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """Return [(kind, token)] plus commit metadata."""
    meta_raw = git(repo, "show", "-s", "--format=%H%x1f%ad%x1f%an%x1f%s", "--date=short", sha)
    parts = meta_raw.strip().split("\x1f")
    meta = {
        "sha": parts[0][:12] if parts else sha[:12],
        "date": parts[1] if len(parts) > 1 else "",
        "author": parts[2] if len(parts) > 2 else "",
        "subject": parts[3] if len(parts) > 3 else "",
    }

    files = [f for f in git(repo, "show", "--pretty=", "--name-only", sha,
                            "--", *paths).splitlines() if f.strip()]
    added = [f for f in git(repo, "show", "--pretty=", "--name-only",
                            "--diff-filter=A", sha, "--", *paths).splitlines() if f.strip()]
    meta["files"] = len(files)
    meta["files_sample"] = files[:6]
    meta["added_files"] = added

    code_files = [f for f in files
                  if os.path.splitext(f)[1] not in DOC_EXT and not DOCISH.search(f)]
    runtime_files = [f for f in files if os.path.splitext(f)[1] in RUNTIME_EXT
                     and not DOCISH.search(f)]
    config_files = [f for f in files if os.path.splitext(f)[1] in CONFIG_EXT]
    meta["code_files"] = len(code_files)
    meta["runtime_files"] = len(runtime_files)
    meta["runtime_sample"] = runtime_files[:6]
    meta["config_files"] = len(config_files)

    diff = git(repo, "show", "-U0", "--pretty=", sha, "--", *paths)
    cands: list[tuple[str, str]] = []

    for f in added:
        if os.path.splitext(f)[1] in RUNTIME_EXT:
            cands.append(("file", f))

    plus_lines = 0
    minus_lines = 0
    cur_runtime = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            # "diff --git a/<path> b/<path>" — trust the b-side
            tail = line.split(" b/", 1)
            cur = tail[1] if len(tail) > 1 else ""
            cur_runtime = (os.path.splitext(cur)[1] in RUNTIME_EXT
                           and not DOCISH.search(cur))
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("-"):
            minus_lines += 1
            continue
        if not line.startswith("+"):
            continue
        plus_lines += 1
        if not cur_runtime:
            continue
        body = line[1:]
        for m in RX_DEF.finditer(line):
            t = _clean(m.group(1))
            if t:
                cands.append(("symbol", t))
        for m in RX_CLASS.finditer(line):
            t = _clean(m.group(1))
            if t:
                cands.append(("symbol", t))
        for m in RX_CONST.finditer(line):
            t = _clean(m.group(1))
            if t:
                cands.append(("symbol", t))
        for m in RX_GENID.finditer(body):
            t = _clean(m.group(1))
            if t:
                cands.append(("symbol", t))
        for m in RX_FLAG.finditer(body):
            t = _clean(m.group(1))
            if t:
                cands.append(("flag", t))
        for m in RX_VLLMFLAG.finditer(body):
            t = _clean(m.group(1))
            if t:
                cands.append(("flag", t))
        for m in RX_QUOTED.finditer(body):
            t = _clean(m.group(1))
            if t:
                cands.append(("string", t))
        # a distinctive comment line is often the ONLY thing a small fix adds
        s = body.strip()
        if s.startswith("#") and 24 <= len(s) <= 160:
            t = _clean(s.lstrip("# ").strip())
            if t:
                cands.append(("comment", t))

    meta["plus"] = plus_lines
    meta["minus"] = minus_lines
    # de-dupe, keep first-seen order
    out, seen = [], set()
    for k, t in cands:
        if t in seen:
            continue
        seen.add(t)
        out.append((k, t))
    return out, meta


# ------------------------------------------------------------ tree lookup ---

def bulk_present(root: str, tokens: list[str], subdirs: list[str] | None = None) -> dict[str, int]:
    """Count occurrences of each token in `root` with ONE grep pass.

    GNU grep -F with many patterns uses Aho-Corasick, so this is O(corpus)
    regardless of pattern count.  Returns {token: hit_count} (0 = absent).
    """
    counts = {t: 0 for t in tokens}
    if not tokens:
        return counts
    with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False,
                                     encoding="utf-8") as fh:
        for t in tokens:
            fh.write(t + "\n")
        patfile = fh.name
    try:
        targets = [os.path.join(root, s) for s in (subdirs or ["."])]
        targets = [t for t in targets if os.path.exists(t)]
        if not targets:
            return counts
        cmd = ["grep", "-rohF", "--binary-files=without-match",
               "--exclude-dir=.git", "--exclude-dir=__pycache__",
               "--exclude-dir=node_modules", "-f", patfile, *targets]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace", timeout=1800)
        for line in p.stdout.splitlines():
            line = line.strip()
            if line in counts:
                counts[line] += 1
        # grep -o only reports the matched substring; longer patterns whose
        # match got shadowed by a shorter overlapping one need a second pass.
        missed = [t for t in tokens if counts[t] == 0]
        if missed and len(missed) < len(tokens):
            with tempfile.NamedTemporaryFile("w", suffix=".pat2", delete=False,
                                             encoding="utf-8") as fh2:
                for t in missed:
                    fh2.write(t + "\n")
                pat2 = fh2.name
            try:
                cmd2 = ["grep", "-rlF", "--binary-files=without-match",
                        "--exclude-dir=.git", "--exclude-dir=__pycache__",
                        "-f", pat2, *targets]
                subprocess.run(cmd2, capture_output=True, text=True, timeout=1800)
                # per-token confirmation only for the residual (bounded, rare)
                for t in missed:
                    q = subprocess.run(
                        ["grep", "-rlF", "--binary-files=without-match",
                         "--exclude-dir=.git", "--exclude-dir=__pycache__",
                         "-e", t, *targets],
                        capture_output=True, text=True, timeout=120)
                    if q.stdout.strip():
                        counts[t] = len(q.stdout.strip().splitlines())
            finally:
                os.unlink(pat2)
    finally:
        os.unlink(patfile)
    return counts


# ----------------------------------------------------------------- driver ---

def collect_repo(name: str, repo: str, paths: list[str], rev_args: list[str],
                 keep: int, tree_root: str, tree_subdirs: list[str]) -> list[dict]:
    if not os.path.isdir(os.path.join(repo, ".git")) and \
       not os.path.isfile(os.path.join(repo, ".git")):
        eprint(f"[{name}] no git repo at {repo} — skipped")
        return []
    shas = [s for s in git(repo, "rev-list", *rev_args, "--", *paths).split() if s]
    eprint(f"[{name}] {len(shas)} commits touching the tracked paths")

    per_commit: list[tuple[dict, list[tuple[str, str]]]] = []
    all_tokens: set[str] = set()
    for i, sha in enumerate(shas):
        cands, meta = candidates_for(repo, sha, paths)
        meta["repo"] = name
        per_commit.append((meta, cands))
        for _k, t in cands:
            all_tokens.add(t)
        if (i + 1) % 50 == 0:
            eprint(f"  ..{i+1}/{len(shas)}")

    file_tokens = {t for _m, c in per_commit for k, t in c if k == "file"}
    relocated_files: set[str] = set()
    text_tokens = sorted(all_tokens - file_tokens)
    eprint(f"[{name}] probing {len(text_tokens)} distinct tokens against {tree_root}")
    counts = bulk_present(tree_root, text_tokens, tree_subdirs)
    # A file fingerprint survives a MOVE.  fixes/patch_pn119_lens_router.py
    # became patch_h119_lens_router.py (an id-collision rename) and three
    # sidecars moved to fixes/_archive/ — all deliberate, none lost.  Match the
    # exact path first, then fall back to the basename anywhere in the tree.
    basenames: dict[str, str] | None = None
    for t in file_tokens:
        if os.path.exists(os.path.join(tree_root, t)):
            counts[t] = 1
            continue
        if basenames is None:
            basenames = {}
            for dp, dn, fn in os.walk(tree_root):
                dn[:] = [d for d in dn if d not in (".git", "__pycache__",
                                                    "node_modules")]
                for f in fn:
                    basenames.setdefault(f, os.path.join(dp, f))
        counts[t] = 1 if os.path.basename(t) in basenames else 0
        if counts[t]:
            relocated_files.add(t)

    records = []
    for meta, cands in per_commit:
        scored = []
        for k, t in cands:
            n = counts.get(t, 0)
            if n <= 0:
                scored.append((0, k, t, 0))
                continue
            # rarity beats length: a token seen once is a perfect fingerprint.
            # KIND weight comes first though — a comment is the weakest handle
            # there is (it can live in a shell boot-guard that never enters the
            # container, which is how 2dc5938a884e came up MISSING against the
            # live container while being perfectly intact).
            rarity = 1000 // max(1, n)
            kind_w = {"file": 500, "symbol": 400, "flag": 400,
                      "string": 200, "comment": 0}.get(k, 100)
            scored.append((kind_w + rarity + min(len(t), 80), k, t, n))
        present = sorted([s for s in scored if s[3] > 0], key=lambda s: -s[0])
        absent = [s for s in scored if s[3] == 0]

        # Verdict order matters: a commit is only LOST if it changed RUNTIME
        # code.  Compose/yaml value tuning (KV pins, ctx sizes, ports) is meant
        # to be overwritten later and must never raise a loss alarm.
        # A file the commit ADDED is a decisive fingerprint: if that module is
        # gone, the commit is not intact no matter how many of its incidental
        # call-site tokens survive.  Without this rule KVQ-2 f785a3a5f read
        # "intact" off two generic Triton kernel params (stride_slot,
        # stride_head) while its whole new sink.py was absent.
        lost_files = [t for k, t, in [(k, t) for _s, k, t, _n in absent] if k == "file"]
        if lost_files and meta["runtime_files"]:
            verdict = "PARTIAL-LOSS"
        elif present:
            verdict = "intact"
        elif meta["runtime_files"] == 0 and meta["config_files"] > 0:
            verdict = "config-drift"
        elif meta["code_files"] == 0:
            verdict = "doc-only"
        elif not cands:
            verdict = "removal" if meta["minus"] and not meta["plus"] else "unfingerprintable"
        else:
            verdict = "LOST"

        records.append({
            "repo": meta["repo"],
            "sha": meta["sha"],
            "date": meta["date"],
            "subject": meta["subject"],
            "files": meta["files"],
            "code_files": meta["code_files"],
            "runtime_files": meta["runtime_files"],
            "runtime_sample": meta["runtime_sample"],
            "config_files": meta["config_files"],
            "plus": meta["plus"],
            "minus": meta["minus"],
            "verdict": verdict,
            "fingerprints": [
                {"kind": k, "token": t, "tree_hits": n} for _s, k, t, n in present[:keep]
            ],
            "dead_candidates": [
                {"kind": k, "token": t} for _s, k, t, _n in absent[:6]
            ] if verdict in ("LOST", "PARTIAL-LOSS") else [],
            "lost_files": lost_files,
            "relocated_files": sorted(
                t for _s, k, t, _n in scored
                if k == "file" and t in relocated_files),
            "candidate_count": len(cands),
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--repo", choices=["club", "vllm-build", "kvq2sink", "all"],
                    default="all")
    ap.add_argument("-o", "--out",
                    default=os.path.join(LEDGER_DIR, "vllm-commit-ledger.json"))
    ap.add_argument("--overrides",
                    default=os.path.join(LEDGER_DIR, "commit-verdict-overrides.json"),
                    help="hand verdicts keyed by short sha; survive regeneration")
    args = ap.parse_args()

    records: list[dict] = []

    if args.repo in ("club", "all"):
        records += collect_repo(
            "club-3090", REPO, CLUB_LANES, ["HEAD"], args.keep,
            REPO, ["models/qwen3.6-27b/vllm", "fixes", "ops", "docs"])

    if args.repo in ("vllm-build", "all") and os.path.isdir(VLLM_BUILD):
        # Only OUR branches.  `--all` here is a trap: the repo carries ~28
        # `pr*` branches that are UPSTREAM PR heads used as cherry-pick
        # sources, so --all ^origin/main yields 2265 commits of other people's
        # work instead of our 22.
        ours = [b for b in OUR_BUILD_BRANCHES
                if git(VLLM_BUILD, "rev-parse", "--verify", b).strip()]
        excl = []
        for r in git(VLLM_BUILD, "remote").split():
            for b in ("main", "master"):
                if git(VLLM_BUILD, "rev-parse", "--verify", f"{r}/{b}").strip():
                    excl.append(f"^{r}/{b}")
        eprint(f"[vllm-build] our branches: {ours}")
        rev = ours + ["--no-merges"] + excl
        records += collect_repo("vllm-build", VLLM_BUILD, ["."], rev, args.keep,
                                VLLM_BUILD, ["vllm", "csrc", "cmake", "setup.py"])

    if args.repo in ("kvq2sink", "all") and os.path.isdir(KVQ2SINK):
        remotes = git(KVQ2SINK, "remote").split()
        excl = []
        for r in remotes:
            for b in ("main", "master"):
                if git(KVQ2SINK, "rev-parse", "--verify", f"{r}/{b}").strip():
                    excl.append(f"^{r}/{b}")
        rev = ["HEAD", "--no-merges"] + excl
        # Probe against VLLM-BUILD, not against the worktree itself.  Probing a
        # branch against its own checkout can only ever say "intact" and tells
        # you nothing.  What we actually need to know is whether this work
        # reached the line that gets BUILT — and for kvq2-sink-runtime the
        # answer is no: it is CPU-tested, never GPU-run, never pushed to the
        # fork, and in no wheel.  It exists in exactly one place on disk.
        records += collect_repo("kvq2sink", KVQ2SINK, ["."], rev, args.keep,
                                VLLM_BUILD, ["vllm", "csrc"])

    # Hand verdicts win over the heuristic and survive every regeneration.
    overrides = {}
    if os.path.exists(args.overrides):
        try:
            import json as _json
            overrides = _json.load(open(args.overrides, encoding="utf-8"))
        except Exception as e:
            eprint(f"!! overrides unreadable: {e}")
    for r in records:
        ov = overrides.get(r["sha"]) or overrides.get(r["sha"][:12])
        if ov:
            r["auto_verdict"] = r["verdict"]
            r["verdict"] = ov.get("verdict", r["verdict"])
            r["human_note"] = ov.get("note", "")

    tally: dict[str, int] = {}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1

    dump_json(args.out, {
        "schema": "vllm-commit-ledger/1",
        "note": ("Commit hashes do not survive rebase/squash — every commit is "
                 "reduced to CONTENT fingerprints.  verdict=LOST means every "
                 "distinctive token the commit added is gone from the tree."),
        "tally": tally,
        "commits": records,
    })
    print(f"wrote {args.out}: {len(records)} commits")
    print("  verdicts:", dict(sorted(tally.items())))
    lost = [r for r in records if r["verdict"] in ("LOST", "PARTIAL-LOSS")]
    if lost:
        print(f"\n  LOST ({len(lost)}):")
        for r in lost:
            print(f"    {r['verdict']:12s} {r['repo']:11s} {r['sha']} {r['date']} {r['subject'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
