#!/usr/bin/env python3
"""Count PR#48361's anchors as the BOOT will see them, on every pinned image.

Five sibling /fixes patches rewrite sched/scheduler.py before this one
(pn75 -> pn96 -> pn103 -> pn105 -> pn83). Counting anchors against the pristine
image is how a patch shipped as a silent no-op here on 2026-07-25, so this
replays the siblings first and only then asks the patcher's own resolver.

    python3 fixes/verify_pr48361_anchors.py

No GPU, no serving container touched — one throwaway container per pin.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_pr48361_mamba_align_split as P  # noqa: E402

# The boot pin first. Unlike verify_pn71/pn100/pn101/pn114_seed — which drop
# dev1060cherry-20260713 on the "never booted again" ruling — this script KEEPS
# it, and removing it is a regression, not cleanup:
#
#   dev1060cherry-20260713 is the only pin carrying the ORIGINAL full #48361
#   pick, i.e. the re-floor branch this patch restores (see the patcher's header
#   and the #48361 row in docs/UPSTREAM.md). It is therefore the POSITIVE
#   CONTROL: it is the one pin that must take the ALREADY_FIXED path and report
#   zero anchors, which is what proves the no-op detection below still works.
#   Drop it and a broken ALREADY_FIXED check reads as a clean run.
#
# Booting it is not required and not implied — replay() only ever runs a
# throwaway `sh -c` container (no GPU, --network none), so the never-booted-again
# ruling does not reach this list. Verified 2026-07-26: wheel v2 and wheel v1
# resolve both anchors count==1; dev1060cherry takes the ALREADY_FIXED branch.
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",  # the boot pin
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",     # wheel v1
    "localhost/vllm-qwen36-endgame:dev1060cherry-20260713",          # positive control
)
TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"

# In entrypoint order. Each rewrites scheduler.py; the boot sees their output.
SIBLINGS = (
    "patch_pn75_embedding_neg_index_guard.py",
    "patch_pn96_44993_structured_output_marker_step_fsm.py",
    "patch_pn103_spec_entry_schedule_reconcile.py",
    "patch_pn105_nan_logits_abort.py",
    "patch_pn83_rerank_micro_slots.py",
)


def replay(pin: str) -> str | None:
    """Run the siblings inside a throwaway container, return the resulting file."""
    script = " && ".join(
        [f"python3 /fixes/{s} >/dev/null 2>&1 || true" for s in SIBLINGS]
        + [f"cat {TARGET}"]
    )
    r = subprocess.run(
        ["sudo", "podman", "run", "--rm", "--network", "none",
         "-v", f"{HERE}:/fixes:ro", "--entrypoint", "/bin/sh", pin, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        print(f"  container failed rc={r.returncode}: {r.stderr.strip()[:200]}")
        return None
    return r.stdout


def main() -> int:
    bad = 0
    for pin in PINS:
        print(f"\n=== {pin}")
        src = replay(pin)
        if src is None:
            bad += 1
            continue
        print(f"  post-sibling scheduler.py: {len(src.splitlines())} lines")
        if P.ALREADY_FIXED in src:
            print("  OK   pin already carries upstream #48361's branch — "
                  "no anchors expected, patch no-ops here")
            continue
        ok, problems = P.resolve(src)
        for name, old, _new in P.HUNKS:
            print(f"    {name:<12} count={src.count(old)}")
        if problems:
            bad += 1
            for p in problems:
                print(f"  FAIL {p}")
        else:
            print(f"  OK   resolved {[n for n, _, _ in ok]}")
            patched = src
            for _n, old, new in ok:
                patched = patched.replace(old, new, 1)
            try:
                compile(patched, "scheduler.py", "exec")
                print("  OK   patched file byte-compiles")
            except SyntaxError as e:
                bad += 1
                print(f"  FAIL patched file does not compile: {e}")
    print()
    print("RESULT: all anchors unique post-patch" if not bad
          else f"RESULT: {bad} pin(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
