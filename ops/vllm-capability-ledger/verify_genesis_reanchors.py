#!/usr/bin/env python3
"""Count the 2026-07-26 genesis re-anchors as the BOOT will see them.

House pattern, same shape as `fixes/verify_pr48361_anchors.py`: one throwaway
container per pin, genesis bind-mounted exactly as the compose mounts it, the
module asked to build its own patcher so the anchor selection under test is
the one that will run.

    python3 ops/vllm-capability-ledger/verify_genesis_reanchors.py
    python3 ops/vllm-capability-ledger/verify_genesis_reanchors.py --live

`--live` additionally pulls the POST-BOOT copy of every target out of the
serving container (one single-file `podman cp` each — never a recursive scan
inside it; a recursive grep in the serving container OOM-killed the engine on
2026-07-25) and counts against those instead. Counting against pristine image
bytes alone is how a patch shipped as a silent no-op on 2026-07-25.

No GPU, no server touched, `--network none`.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
DIST = "/usr/local/lib/python3.12/dist-packages"

PINS = ("localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",)
SERVING_CONTAINER = "vllm-tcbench-8021"

# Every file the probe counts against, container-absolute.
LIVE_TARGETS = (
    f"{DIST}/vllm/v1/attention/backends/turboquant_attn.py",
    f"{DIST}/vllm/v1/core/sched/scheduler.py",
    f"{DIST}/vllm/v1/sample/rejection_sampler.py",
    f"{DIST}/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py",
    f"{DIST}/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py",
    f"{DIST}/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py",
)


def pull_live(dest: pathlib.Path) -> int:
    """One `podman cp` per file. Cheap, bounded, and never recursive."""
    got = 0
    for path in LIVE_TARGETS:
        r = subprocess.run(
            ["sudo", "podman", "cp", f"{SERVING_CONTAINER}:{path}",
             str(dest / pathlib.PurePosixPath(path).name)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            got += 1
        else:
            print(f"  (live copy skipped: {path} — {r.stderr.strip()[:120]})")
    subprocess.run(["sudo", "chown", "-R", f"{__import__('os').getuid()}",
                    str(dest)], capture_output=True)
    return got


def run(pin: str, live_dir: pathlib.Path | None) -> int:
    mounts = [
        "-v", f"{GENESIS}:{DIST}/vllm/_genesis:ro",
        "-v", f"{GENESIS / 'sndr'}:{DIST}/sndr:ro",
        "-v", f"{HERE}:/led:ro",
    ]
    if live_dir is not None:
        mounts += ["-v", f"{live_dir}:/led-live:ro"]
    cmd = (["sudo", "podman", "run", "--rm", "--network", "none"] + mounts
           + ["--entrypoint", "python3", pin, "/led/_reanchor_probe.py"])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(r.stdout)
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr[-4000:], file=sys.stderr)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also count against the serving container's "
                         "post-boot files (single-file copies only)")
    args = ap.parse_args()

    bad = 0
    with tempfile.TemporaryDirectory(prefix="led-live-") as td:
        live_dir = None
        if args.live:
            live_dir = pathlib.Path(td)
            n = pull_live(live_dir)
            print(f"pulled {n}/{len(LIVE_TARGETS)} post-boot files from "
                  f"{SERVING_CONTAINER}\n")
        for pin in PINS:
            print(f"=== {pin}")
            bad += 1 if run(pin, live_dir) else 0
    print("RESULT: all pins OK" if not bad else f"RESULT: {bad} pin(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
