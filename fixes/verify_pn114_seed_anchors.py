#!/usr/bin/env python3
"""Count PN114-SEED's anchors as the BOOT will see them, on every pinned image.

WHY THIS EXISTS
---------------
Every anchor this patch uses is text that ANOTHER patch wrote earlier in the
same entrypoint:

    S1a  <- patch_h119_lens_router.py  (F site, sync_batch)
    S1b  <- upstream, but the H119 F site moved the code above it
    S2   <- patch_pn114_forced_span.py (graft G)
    S3   <- patch_pn114_forced_span.py (graft E)
    S4   <- patch_pn101_answer_rescue.py (hint site)

Counting them against the pristine image would therefore measure nothing.
On 2026-07-25 a patch shipped as a silent no-op for exactly this reason, and
`fixes/verify_h119_consumer_anchors.py` / `fixes/verify_pr48361_anchors.py`
exist because replaying the siblings first is what made the last two patches
work on the first boot.

This harness goes one step further than either: it replays the entrypoint
prefix INSIDE a throwaway container from each pinned image — genesis
`apply_all` included — so the bytes the resolver reads are the bytes the boot
produces, not a host reconstruction of them. It also builds the boot-time seed
table with the real tokenizer and re-checks the split-equivalence invariant
the whole mechanism rests on.

    python3 fixes/verify_pn114_seed_anchors.py [--pin TAG]

No GPU, no serving container touched, nothing written outside the throwaway.
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
HFCACHE = pathlib.Path.home() / ".cache/huggingface"
DIST = "/usr/local/lib/python3.12/dist-packages"

# The boot pin first. dev1060cherry-20260713 is deliberately absent: it will
# never be booted again (operator ruling 2026-07-25).
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
)

# Entrypoint order, restricted to the patches that write one of our two
# targets (thinking_budget_state.py / chat_completion/serving.py). Anything
# that FATALs for an unrelated reason is tolerated and reported, exactly as a
# `|| true` boot would behave for us: what matters is the resulting bytes.
PREFIX = (
    "patch_pn108_plateau_cap.py",
    "patch_pn112_conf_tap.py",
    "patch_pr44812_tool_guard.py",
    "patch_holder_syncbatch_fix.py",
    "pn114_boot_ids.py",
    "patch_pn114_forced_span.py",
    "patch_h119_lens_router.py",
    "patch_pn74_fix_p107_serving_attr.py",
    "patch_pn100_auto_thinking_budget.py",
    "patch_pn101_answer_rescue.py",
)

PROBE = r'''
import importlib.util, pathlib, sys, json

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

P = load("pn114_seed_patch", "/fixes/patch_pn114_seed_span.py")
tbs = pathlib.Path(P.TBS).read_text(encoding="utf-8")
srv = pathlib.Path(P.SRV).read_text(encoding="utf-8")
print("  post-sibling thinking_budget_state.py: %d lines" % len(tbs.splitlines()))
print("  post-sibling serving.py:               %d lines" % len(srv.splitlines()))
print("  counts: S1a=%d S1b=%d S1legacy=%d S2=%d S3=%d S4=%d" % (
    tbs.count(P.S1A_OLD), tbs.count(P.S1B_OLD), tbs.count(P.S1L_OLD),
    tbs.count(P.S2_OLD), tbs.count(P.S3_OLD), srv.count(P.S4_OLD)))
bad = 0
tsites, tprob = P.resolve_tbs_sites(tbs)
ssites, sprob = P.resolve_srv_sites(srv)
if tprob or sprob:
    bad = 1
    print("  BAD  %s" % "; ".join(p for p in (tprob, sprob) if p))
else:
    print("  OK   resolved %s" % ([n for n, _, _ in tsites]
                                  + [n for n, _, _ in ssites]))
    for text, sites, label in ((tbs, tsites, "thinking_budget_state.py"),
                               (srv, ssites, "serving.py")):
        out = text
        for _n, old, new in sites:
            out = out.replace(old, new, 1)
        try:
            compile(out, label, "exec")
            print("  OK   patched %s byte-compiles" % label)
        except SyntaxError as e:
            bad = 1
            print("  BAD  patched %s does not compile: %s" % (label, e))

# The mechanism's load-bearing invariant, re-measured on this pin's tokenizer:
# encode(BASE + seed) must equal encode(BASE) + encode(seed), or a forced span
# would land different ids at the same positions.
try:
    ids = load("pn114_seed_ids", "/fixes/pn114_seed_ids.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ids.MODEL, trust_remote_code=True,
                                        local_files_only=True)
    base = tok.encode(ids.BASE, add_special_tokens=False)
    cands = ids.candidates(64)
    split_ok = sum(1 for t, _l, _s, _t2 in cands
                   if tok.encode(ids.BASE + t, add_special_tokens=False)
                   == base + tok.encode(t, add_special_tokens=False))
    print("  split-equivalence: %d/%d seeds exact (base=%d tok)"
          % (split_ok, len(cands), len(base)))
    if split_ok != len(cands):
        print("  NOTE non-exact seeds are REJECTED by pn114_seed_ids.py and "
              "keep their prompt-rendered seed — not a failure, but the "
              "routed-N table is smaller than it looks")
except Exception as e:
    print("  WARN tokenizer check skipped: %s: %s" % (type(e).__name__, e))

sys.exit(bad)
'''


def run_pin(pin: str) -> bool:
    steps = [f"python3 -m vllm._genesis.patches.apply_all >/tmp/aa.log 2>&1 "
             f"|| echo '  NOTE apply_all rc!=0 (see /tmp/aa.log)'"]
    for name in PREFIX:
        steps.append(f"python3 /fixes/{name} >/tmp/p.log 2>&1 "
                     f"|| echo '  NOTE {name} rc!=0: '$(tail -1 /tmp/p.log)")
    steps.append("python3 /tmp/probe.py")
    # base64 in the script body, not an env var: `sudo` resets the environment,
    # so a `-e VAR` hand-off silently ships an EMPTY probe (and a green run
    # that measured nothing — the exact failure mode this harness exists for).
    blob = base64.b64encode(PROBE.encode("utf-8")).decode("ascii")
    script = f"echo {blob} | base64 -d > /tmp/probe.py\n" + "\n".join(steps)
    cmd = [
        "sudo", "podman", "run", "--rm", "--network", "none",
        "-e", "GENESIS_ENABLE_PN114_SEED_SPAN=1",
        "-v", f"{HERE}:/fixes:ro",
        "-v", f"{GENESIS}:{DIST}/vllm/_genesis:ro",
    ]
    if HFCACHE.is_dir():
        cmd += ["-v", f"{HFCACHE}:/root/.cache/huggingface:ro"]
    cmd += ["--entrypoint", "/bin/bash", pin, "-c", script]
    r = subprocess.run(cmd, text=True, timeout=1800,
                       capture_output=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="append",
                    help="override the pin list (repeatable)")
    args = ap.parse_args()
    pins = tuple(args.pin) if args.pin else PINS
    bad = 0
    for pin in pins:
        print(f"\n=== {pin}")
        if not run_pin(pin):
            bad += 1
    print()
    print("RESULT: all anchors resolve on every pin" if not bad
          else f"RESULT: {bad} pin(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
