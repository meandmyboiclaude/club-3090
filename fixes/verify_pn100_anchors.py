#!/usr/bin/env python3
"""Count PN100's anchors as the BOOT will see them, on every live pin.

PN100's anchors do not live in the image. `apply_all` runs first and writes
genesis PN16 / P89 / PN288 / P107 into
`entrypoints/openai/chat_completion/serving.py`, and PN16 — the block the old
single anchor keyed on — is OPT-IN ENV
(`GENESIS_ENABLE_PN16_LAZY_REASONER=1`, set in the boot compose). Counting
against pristine image content is how a patch shipped as a silent no-op here on
2026-07-25, so this replays the real thing: the boot compose's environment plus
its entrypoint script truncated immediately before the PN100 line, then asks the
patcher's own resolver what it sees.

Two arms per pin:

  boot-env  — the compose's environment verbatim. This is the live boot.
  pn16-off  — the same, with GENESIS_ENABLE_PN16_LAZY_REASONER=0. One env flip,
              legitimate as an A/B, and the shape that used to end the boot:
              apply_all skips PN16, the sole anchor vanishes, the patcher
              returned 1 and `set -e` killed the entrypoint.

    python3 fixes/verify_pn100_anchors.py [--pin TAG] [--keep DIR]

No GPU, no serving container touched — one throwaway container per arm. The
gpu-guard line is stripped from the replayed prefix (it is the one entrypoint
step that cannot run CPU-only); everything else runs exactly as the boot runs
it, with `set -e` relaxed so one CPU-only casualty does not truncate the replay.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import patch_pn100_auto_thinking_budget as P  # noqa: E402

# The compose whose entrypoint + environment IS the boot. tcbench8021 carries
# the boot pin; endgame8020 runs the byte-identical PN100 stanza.
COMPOSE = REPO / "models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml"

# dev1060cherry-20260713 is deliberately absent: it will never be booted again.
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",  # the boot pin
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
)
TARGET = ("/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
          "chat_completion/serving.py")
STOP_AT = "patch_pn100_auto_thinking_budget"

MOUNTS = (
    (REPO / "fixes", "/fixes", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis",
     "/usr/local/lib/python3.12/dist-packages/vllm/_genesis", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/sndr",
     "/usr/local/lib/python3.12/dist-packages/sndr", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/local/"
            "qwen3coder_tool_parser_deferred_commit.py",
     "/patches/qwen3coder_tool_parser_deferred_commit.py", "ro"),
)


def _expand(value: str) -> str:
    """Resolve compose interpolation the way `docker compose` would with no .env."""
    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"\$\{([A-Za-z_]\w*):-([^{}]*)\}", r"\2", value)
        value = re.sub(r"\$\{([A-Za-z_]\w*)\}", "", value)
    return value


def compose_environment() -> list[str]:
    txt = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"\n    environment:\n(.*?)\n    [a-z_]+:", txt, re.S)
    if not m:
        raise SystemExit(f"no environment: block found in {COMPOSE}")
    out = []
    for line in m.group(1).split("\n"):
        s = line.strip()
        if not s.startswith("- "):
            continue
        kv = s[2:]
        if kv[:1] in "\"'":
            kv = kv[1:-1]
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        out.append(f"{key}={_expand(val)}")
    return out


def entrypoint_prefix() -> list[str]:
    """The boot's entrypoint script, truncated just before the PN100 line."""
    txt = COMPOSE.read_text(encoding="utf-8")
    m = re.search(
        r"\n    entrypoint:\n      - /bin/bash\n      - -c\n      - \|\n(.*?)\n      - --\n",
        txt, re.S)
    if not m:
        raise SystemExit(f"no block-scalar entrypoint found in {COMPOSE}")
    lines = [ln[8:] if ln.startswith(" " * 8) else ln
             for ln in m.group(1).split("\n")]
    out = []
    for ln in lines:
        if STOP_AT in ln:
            break
        # The gpu-guard cannot run CPU-only and would abort the replay.
        if "torch.cuda.is_available" in ln or "[gpu-guard]" in ln:
            continue
        # `set -e` would truncate the replay at the first CPU-only casualty;
        # the boot keeps it, which is exactly the hazard PN100 must not join.
        out.append("set +e" if ln.strip() == "set -e" else ln)
    if not any(STOP_AT in ln for ln in lines):
        raise SystemExit(f"{STOP_AT} not found in the entrypoint of {COMPOSE}")
    return out


def replay(pin: str, env: list[str], workdir: pathlib.Path) -> str | None:
    """Run apply_all + the entrypoint prefix in a throwaway container."""
    script = workdir / "prefix.sh"
    script.write_text("\n".join(entrypoint_prefix()) + f"\ncat {TARGET}\n",
                      encoding="utf-8")
    envfile = workdir / "env.list"
    envfile.write_text("\n".join(env) + "\n", encoding="utf-8")
    cmd = ["sudo", "podman", "run", "--rm", "--network", "none",
           "--env-file", str(envfile)]
    for src, dst, mode in MOUNTS:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += ["-v", f"{workdir}:/work:ro", "--entrypoint", "/bin/bash", pin,
            "/work/prefix.sh"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"    container failed rc={r.returncode}: {r.stderr.strip()[-400:]}")
        return None
    # The patch log is interleaved on stdout ahead of the file; the file starts
    # at the first genesis wiring marker or the upstream module docstring.
    return r.stdout


def check(pin: str, arm: str, env: list[str], workdir: pathlib.Path) -> bool:
    print(f"\n=== {pin}  [{arm}]")
    out = replay(pin, env, workdir)
    if out is None:
        return False
    # `cat` is the last command; find where the python file begins.
    idx = out.find("# [Genesis wiring marker")
    if idx == -1:
        idx = out.find("# SPDX-License-Identifier")
    if idx == -1:
        print("    could not locate the start of serving.py in the replay output")
        return False
    src = out[idx:]
    pn16 = "applied" if P.PN16_CALL in src else "NOT applied"
    print(f"    serving.py: {len(src.splitlines())} lines · genesis PN16 {pn16}")

    if P.MARKER in src:
        print("    FAIL PN100 marker already in the pre-PN100 file — "
              "another step is writing it")
        return False

    name, off, counts, problems = P.resolve(src)
    for cname, c in counts:
        print(f"      {cname:<16} count={c}")
    if name is None:
        for p in problems:
            print(f"      {p}")
        print("    FAIL no anchor variant resolved — PN100 would SKIP "
              "(loud, boot survives) and the router would have no call site")
        return False

    patched = src[:off] + P.BLOCK + src[off:]
    try:
        compile(patched, "serving.py", "exec")
    except SyntaxError as e:
        print(f"    FAIL patched file does not compile: {e}")
        return False
    ordered = "n/a (PN16 absent)"
    at = src.find(P.PN16_CALL)
    if at != -1:
        ordered = "OK (before PN16)" if off < at else "WRONG (after PN16)"
    print(f"    OK   resolved via '{name}' · compiles · ordering: {ordered}")
    return at == -1 or off < at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="append", help="override the pin list")
    ap.add_argument("--keep", help="write the replayed scripts to this dir")
    args = ap.parse_args()
    pins = args.pin or list(PINS)

    base_env = compose_environment()
    off_env = [e for e in base_env
               if not e.startswith("GENESIS_ENABLE_PN16_LAZY_REASONER=")]
    off_env.append("GENESIS_ENABLE_PN16_LAZY_REASONER=0")
    arms = (("boot-env", base_env), ("pn16-off", off_env))

    print(f"compose: {COMPOSE}")
    print(f"prefix:  {len(entrypoint_prefix())} entrypoint lines replayed "
          f"(up to, not including, {STOP_AT})")
    print(f"env:     {len(base_env)} variables from the compose")

    bad = 0
    ctx = (pathlib.Path(args.keep) if args.keep else None)
    if ctx:
        ctx.mkdir(parents=True, exist_ok=True)
    for pin in pins:
        for arm, env in arms:
            if ctx:
                d = ctx / f"{pin.rsplit(':', 1)[-1]}-{arm}"
                d.mkdir(exist_ok=True)
                if not check(pin, arm, env, d):
                    bad += 1
            else:
                with tempfile.TemporaryDirectory() as td:
                    if not check(pin, arm, env, pathlib.Path(td)):
                        bad += 1
    print()
    print("RESULT: every arm resolves an anchor ahead of PN16"
          if not bad else f"RESULT: {bad} arm(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
