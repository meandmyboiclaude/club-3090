#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a whole boot's patch pass into a THROWAWAY container and read the result.

Why this exists
---------------
Counting a patch's anchor against the pristine image is how six patches shipped
as silent no-ops in a single day: by the time a patch runs, its sibling patches
have already rewritten the same file, so the bytes the boot sees are not the
bytes in the image.  `fixes/verify_pr48361_anchors.py` solved that for ONE file
by hand-listing its five siblings.  This generalises it: it replays the ENTIRE
boot patch pass — `apply_all` (lane-1 + lane-2) followed by every uncommented
`python3 /fixes/...` line of the compose entrypoint, in compose order, with the
real environment — and then lets you grep or dump the resulting installed vllm.

No GPU, no serving container touched.  The compose entrypoint's gpu-guard is
deliberately NOT replayed: it exists to stop a GPU-blind SERVE, and every text
patch we verify here is a byte rewrite that does not read the device.  Patches
that gate on a real device will report skipped; those are visible in --log and
must not be read as losses.

Usage
-----
    # markers, as the boot sees them
    python3 replay-boot-patches.py --grep "Genesis P83 DEBUG instrumentation v7.53.6"

    # the post-boot bytes of one file, for anchor counting
    python3 replay-boot-patches.py --dump v1/core/kv_cache_manager.py > /tmp/f.py

    # what each stage actually said
    python3 replay-boot-patches.py --log --only-genesis

Env comes from the live container by default (`--env-from`), which is the only
copy of the compose's `${VAR:-default}` interpolation we do not have to reimplement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_ledger_lib as L  # noqa: E402

BOOT_PIN = "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725"
COMPOSE = os.path.join(
    L.VLLM_DIR, "compose/single/tcbench8021.yml")
CVLLM = L.CONTAINER_VLLM

# Env keys that must NOT be carried into a throwaway CPU container.
ENV_DROP = {"PATH", "HOSTNAME", "HOME", "TERM", "container", "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
            "CUDA_VERSION", "NV_LIBCUBLAS_VERSION"}


def entrypoint_script_lines(compose: str) -> list[str]:
    """Every uncommented `python3 <script>` line of the compose entrypoint, in order."""
    text = open(compose, encoding="utf-8").read()
    out: list[str] = []
    in_ep = False
    for raw in text.splitlines():
        if re.match(r"^\s*entrypoint:\s*$", raw):
            in_ep = True
            continue
        if in_ep and re.match(r"^\s{4}\w[\w-]*:", raw):  # next service key
            break
        if not in_ep:
            continue
        m = re.match(r"^\s*python3\s+(/\S+\.py)\s*$", raw)
        if m:
            out.append(m.group(1))
    return out


def build_script(scripts: list[str], stages: str, tail: str) -> str:
    parts = ["set +e"]
    if stages in ("all", "genesis"):
        parts.append(
            'echo "=== apply_all"; python3 -m vllm._genesis.patches.apply_all '
            '2>&1 | sed "s/^/  /"; echo "  rc=$?"')
    if stages in ("all", "fixes"):
        for s in scripts:
            parts.append(
                f'echo "=== {s}"; python3 {shlex.quote(s)} 2>&1 | sed "s/^/  /"; '
                f'echo "  rc=$?"')
    parts.append(tail)
    return "\n".join(parts)


def run(pin: str, env: dict[str, str], script: str, timeout: int) -> tuple[int, str]:
    cmd = ["sudo", "podman", "run", "--rm", "--network", "none"]
    for k, v in sorted(env.items()):
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v", f"{L.GENESIS}:{CVLLM}/_genesis:ro",
        "-v", f"{L.SNDR}:/usr/local/lib/python3.12/dist-packages/sndr:ro",
        "-v", f"{L.FIXES}:/fixes:ro",
        "-v", f"{os.path.join(L.VLLM_DIR, 'patches/local')}:/patches:ro",
        "--entrypoint", "/bin/bash", pin, "-c", script,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       errors="replace")
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=BOOT_PIN)
    ap.add_argument("--env-from", default="vllm-tcbench-8021",
                    help="live container to read the boot env from, or a json file")
    ap.add_argument("--stages", choices=("all", "genesis", "fixes", "none"),
                    default="all")
    ap.add_argument("--grep", nargs="*", default=[],
                    help="literal markers to locate in the patched install")
    ap.add_argument("--dump", default=None,
                    help="vllm-relative path to print after the replay")
    ap.add_argument("--then", default=None,
                    help="shell command to run INSIDE the throwaway container "
                         "after the replay. This is how a checker that has to "
                         "`import vllm._genesis` runs against post-boot bytes; "
                         "the lane mounts put the genesis tree at "
                         f"{CVLLM}/_genesis, so its own tools are already there.")
    ap.add_argument("--log", action="store_true", help="print the replay log")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    if os.path.isfile(a.env_from):
        env = json.load(open(a.env_from))
    else:
        env = L.container_env(a.env_from)
    env = {k: v for k, v in env.items() if k not in ENV_DROP}

    tail_parts = ['echo "=== REPLAY-DONE"']
    if a.grep:
        # Search the INSTALLED vllm only.  _genesis/ and sndr/ are mounted
        # inside it and carry every marker in their own source (README trap 8).
        tail_parts.append(
            "python3 - <<'PYEOF'\n"
            "import os, json\n"
            f"NEEDLES = {json.dumps(a.grep)}\n"
            f"ROOT = {json.dumps(CVLLM)}\n"
            "hits = {n: [] for n in NEEDLES}\n"
            "for dp, dn, fn in os.walk(ROOT):\n"
            "    dn[:] = [d for d in dn if d not in ('_genesis', 'sndr', '__pycache__')]\n"
            "    for f in fn:\n"
            "        if not f.endswith(('.py', '.pyi')):\n"
            "            continue\n"
            "        p = os.path.join(dp, f)\n"
            "        try:\n"
            "            t = open(p, encoding='utf-8', errors='replace').read()\n"
            "        except OSError:\n"
            "            continue\n"
            "        for n in NEEDLES:\n"
            "            c = t.count(n)\n"
            "            if c:\n"
            "                hits[n].append((p[len(ROOT) + 1:], c))\n"
            "print('=== GREP ' + json.dumps(hits))\n"
            "PYEOF")
    if a.dump:
        tail_parts.append(f'echo "=== DUMP"; cat {CVLLM}/{a.dump}')
    if a.then:
        tail_parts.append(f'echo "=== THEN"; {a.then}; echo "=== THEN-RC=$?"')

    scripts = entrypoint_script_lines(COMPOSE)
    script = build_script(scripts, a.stages, "\n".join(tail_parts))
    rc, out = run(a.pin, env, script, a.timeout)

    if a.log or rc != 0:
        head, _, rest = out.partition("=== REPLAY-DONE")
        sys.stderr.write(head)
        out = rest
    for line in out.splitlines():
        if line.startswith("=== GREP "):
            hits = json.loads(line[len("=== GREP "):])
            for n, where in hits.items():
                mark = "PRESENT" if where else "ABSENT "
                print(f"{mark}  {n!r}")
                for p, c in where:
                    print(f"           {c}x {p}")
    if a.dump:
        body = out.partition("=== DUMP\n")[2]
        sys.stdout.write(body.partition("=== THEN")[0] if a.then else body)
    if a.then:
        body = out.partition("=== THEN\n")[2]
        body, _, tail = body.partition("=== THEN-RC=")
        sys.stdout.write(body)
        try:
            return int(tail.strip().splitlines()[0])
        except (ValueError, IndexError):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
