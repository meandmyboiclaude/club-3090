#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Count P7 / P7b's anchors as the BOOT will see them, on the boot pin.

Both patches target the GDN attention module, which P28, PN11, PN50 and PN350
also rewrite, so counting against the pristine image answers the wrong
question. It runs INSIDE the replay container, after the replay, so it can import the
patch modules and ask each one's own resolver — `pick_anchor` /
`pick_anchors` — which variant it would select, and report the count of EVERY
known variant. It also proves the rename alias is what makes the file
reachable at all: it resolves the OLD path literal the patches ask for.

Counted, not merely present: the bare in_proj call pair occurs TWICE in the
post-rename file (forward_cuda and forward_cpu), and a patch that took the
first hit would put a CUDA-stream dispatcher on the CPU forward path.

    python3 ops/vllm-capability-ledger/replay-boot-patches.py --then \
      "python3 /usr/local/lib/python3.12/dist-packages/vllm/_genesis/tools/verify_p7_p7b_anchors.py"

No GPU, no serving container touched.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GENESIS = os.path.dirname(HERE)

# The path the patches ASK for. resolve_vllm_file() aliases it to
# mamba/gdn/qwen_gdn_linear_attn.py on this pin; the alias is half the fix, so
# dump the resolved name and prove the alias is what makes it reachable.
OLD_REL = "model_executor/layers/mamba/gdn_linear_attn.py"
NEW_REL = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


def post_boot_source() -> str | None:
    """The target file as it stands AFTER this container's replay."""
    from vllm._genesis.guards import resolve_vllm_file
    target = resolve_vllm_file(OLD_REL)
    if target is None:
        print(f"  FAIL resolve_vllm_file({OLD_REL!r}) is None — the rename "
              f"alias in guards._PATH_ALIASES is not doing its job")
        return None
    print(f"resolved via alias -> {target}")
    return open(str(target), encoding="utf-8").read()


def main() -> int:
    src = post_boot_source()
    if src is None:
        return 1
    print(f"post-boot source: {len(src.splitlines())} lines")

    bad = 0
    from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
    print("\n=== P7")
    for label, count in p7.anchor_report(src):
        print(f"    {label:<24} count={count}")
    picked = p7.pick_anchor(src)
    if picked is None:
        bad += 1
        print("  FAIL no P7 variant is uniquely present")
    else:
        anchor, replacement = picked
        print("  OK   resolved a unique P7 anchor")
        patched = src.replace(anchor, replacement, 1)
        try:
            compile(patched, NEW_REL, "exec")
            print("  OK   patched file byte-compiles")
        except SyntaxError as e:
            bad += 1
            print(f"  FAIL patched file does not compile: {e}")

    from vllm._genesis.wiring.legacy import (
        patch_7b_gdn_dual_stream_customop as p7b)
    print("\n=== P7b")
    for name, counts in p7b.anchor_report(src).items():
        print(f"    {name:<28} counts={counts}  (newest variant first)")
    picked_b = p7b.pick_anchors(src)
    if picked_b is None:
        bad += 1
        print("  FAIL at least one P7b sub-patch has no unique anchor "
              "(both are required — a half-apply NameErrors at the call site)")
    else:
        print(f"  OK   resolved all {len(picked_b)} P7b sub-patch anchors")
        patched = src
        for _n, (a, r) in picked_b.items():
            patched = patched.replace(a, r, 1)
        try:
            compile(patched, NEW_REL, "exec")
            print("  OK   patched file byte-compiles")
        except SyntaxError as e:
            bad += 1
            print(f"  FAIL patched file does not compile: {e}")

    print()
    print("RESULT: OK" if not bad else f"RESULT: {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
