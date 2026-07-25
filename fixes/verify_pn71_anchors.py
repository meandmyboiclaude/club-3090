#!/usr/bin/env python3
"""Verify the PN71 family's anchors against BOOT-TIME content, on every live pin.

WHY THIS EXISTS
---------------
PN71 has been re-anchored repeatedly, and BUG-122 caught PN71-family flags going
phantom — a patch announcing APPLY while the module re-gated on an unset flag and
silently skipped. Counting anchors against the file as ``podman run --entrypoint cat``
hands it over is not enough either: at boot, `apply_all` and several sibling /fixes
patches rewrite these files first, and a patch that shipped as a silent no-op on
2026-07-25 did so for exactly that reason.

So the replay here is, per pin, in a throwaway container: the compose's expanded
environment, ``python3 -m vllm._genesis.patches.apply_all``, then the sibling /fixes
writers in entrypoint order, then the two target files come back out base64-encoded
and this script runs each PATCHER'S OWN resolver against them.

  vllm/entrypoints/openai/chat_completion/protocol.py   <- PN71 (A)/(B)
      Sibling writers: none before PN71 (h119_route_api runs after it).
  vllm/entrypoints/openai/chat_completion/serving.py    <- PN71T (F)/(S)
      Sibling writers, in entrypoint order:
          pn74 -> pn100 -> pn101 -> h119_route_api

It is CPU-only and touches nothing that is running:
  * ``podman run --rm`` from a pinned image writes only to its own ephemeral overlay
    — the image, the host and the live vllm-tcbench-8021 container are untouched.
  * NO GPU device is injected and none is needed. Do not add
    ``--device nvidia.com/gpu=all`` "to be faithful": the live server holds the card
    and a second CUDA context is a real OOM risk (BUG-126).

THE ENV EXPANSION IS LOAD-BEARING — the compose writes ``${VAR:-default}`` values that
docker-compose expands before the container sees them; passing the literal ``${...}``
through aborts the process inside torch's allocator-config tokenizer, apply_all never
runs, and the replay silently degrades to "pristine".

Usage:  python3 fixes/verify_pn71_anchors.py [--keep] [--pin TAG]
Needs:  rootful podman (the images live in the root store).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DIST = "/usr/local/lib/python3.12/dist-packages"
PROTOCOL_REL = "vllm/entrypoints/openai/chat_completion/protocol.py"
SERVING_REL = "vllm/entrypoints/openai/chat_completion/serving.py"
TARGETS = (PROTOCOL_REL, SERVING_REL)

COMPOSE = REPO / "models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml"
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"

# dev1060cherry-20260713 is deliberately absent: it will never be booted again.
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",  # the boot pin
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
)

# /fixes patches that WRITE a target file BEFORE the PN71 family, in entrypoint order.
# PN71 itself is the first protocol.py writer; PN71T is wired after PN101.
REPLAY = (
    "patch_pn74_fix_p107_serving_attr.py",
    "patch_pn100_auto_thinking_budget.py",
    "patch_pn101_answer_rescue.py",
    # NOT wired in tcbench8021.yml today (PN114 ships dark), but it writes
    # serving.py and anchors on PN101's hint block. Replayed in the strict
    # position — before ours — so our anchors are counted against the worst case
    # rather than only against the compose as it happens to be wired this week.
    "patch_pn114_seed_span.py",
)
# Runs AFTER both PN71 patches — replayed last, only to prove it does not eat our
# anchors or get eaten by ours.
REPLAY_AFTER = ("patch_h119_route_api.py",)

PATCHERS = (
    ("PN71  protocol.py", "patch_pn71_reasoning_alias.py", PROTOCOL_REL),
    ("PN71T serving.py", "patch_pn71t_truncation_signal.py", SERVING_REL),
)


def _sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, **kw)


def compose_env() -> list[str]:
    """The compose's `environment:` list, with ${VAR:-default} expanded."""
    text = COMPOSE.read_text(encoding="utf-8")
    out: list[str] = []

    def expand(v: str) -> str:
        def sub(m):
            name, dflt = m.group(1), m.group(2)
            return os.environ.get(name) or (dflt or "")
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", sub, v)

    for line in text.splitlines():
        m = re.match(r"^      - ([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            out.append(f"{m.group(1)}={expand(m.group(2))}")
    return out


def _audit_replay_set() -> list[str]:
    """Re-derive REPLAY from the tree; a drift here is how a no-op ships."""
    problems: list[str] = []
    ours = {name for _l, name, _t in PATCHERS}
    referencing: set[str] = set()
    for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("cliff2b/*.py")):
        if p.name.startswith(("verify_", "test_")):
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        if "write_text" not in src:
            continue
        # Match on (basename + the package dir) rather than the full relative
        # path: several patchers build TARGET from ADJACENT STRING LITERALS
        # ("...dist-packages/vllm/entrypoints/openai/" "chat_completion/serving.py"),
        # so the path never appears contiguously and a path-substring audit
        # reports them as non-writers — under-inclusion, which is precisely the
        # silent-no-op failure this audit exists to catch.
        for rel in TARGETS:
            if rel.rsplit("/", 1)[-1] in src and "chat_completion" in src:
                referencing.add(p.name)
    known = set(REPLAY) | set(REPLAY_AFTER) | ours
    extra = referencing - known
    if extra:
        problems.append(
            f"/fixes patches write a target file but are NOT replayed: {sorted(extra)}"
            " — add them to REPLAY in entrypoint order")
    stale = (set(REPLAY) | set(REPLAY_AFTER)) - referencing
    if stale:
        problems.append(f"REPLAY names patches that no longer write a target: {sorted(stale)}")
    return problems


REPLAY_SCRIPT = """set +e
mkdir -p /out
for f in {targets}; do
  b=$(echo "$f" | tr '/' '_')
  base64 -w0 "{dist}/$f" > "/out/{tag}.pristine.$b" 2>/dev/null
done
python3 -m vllm._genesis.patches.apply_all > /out/{tag}.applyall.log 2>&1
echo "apply_all rc=$?" > /out/{tag}.rc
for p in {replay}; do
  python3 /fixes/$p >> /out/{tag}.replay.log 2>&1
  echo "$p rc=$?" >> /out/{tag}.rc
done
for f in {targets}; do
  b=$(echo "$f" | tr '/' '_')
  base64 -w0 "{dist}/$f" > "/out/{tag}.post.$b" 2>/dev/null
done
# Second stage: apply OUR patches, then the after-siblings, and prove the whole
# stack still byte-compiles together.
for p in {ours}; do
  python3 /fixes/$p >> /out/{tag}.ours.log 2>&1
  echo "$p rc=$?" >> /out/{tag}.rc
done
for p in {after}; do
  python3 /fixes/$p >> /out/{tag}.after.log 2>&1
  echo "$p rc=$?" >> /out/{tag}.rc
done
for f in {targets}; do
  b=$(echo "$f" | tr '/' '_')
  python3 -c "import py_compile,sys; py_compile.compile('{dist}/$f', doraise=True)" \
    >> /out/{tag}.compile.log 2>&1
  echo "compile $f rc=$?" >> /out/{tag}.rc
done
echo done
"""


def replay_in_container(image: str, out: pathlib.Path, env: list[str], tag: str) -> None:
    args = ["sudo", "podman", "run", "--rm"]
    for e in env:
        args += ["--env", e]
    args += [
        "-v", f"{GENESIS}:{DIST}/vllm/_genesis:ro",
        "-v", f"{GENESIS / 'sndr'}:{DIST}/sndr:ro",
        "-v", f"{REPO / 'fixes'}:/fixes:ro",
        "-v", f"{out}:/out:rw",
        "--entrypoint", "bash", image, "-c",
        REPLAY_SCRIPT.format(
            dist=DIST, targets=" ".join(TARGETS), tag=tag,
            replay=" ".join(REPLAY),
            ours=" ".join(name for _l, name, _t in PATCHERS),
            after=" ".join(REPLAY_AFTER),
        ),
    ]
    cp = _sh(args)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr.decode("utf-8", "replace")[-2000:] + "\n")
        raise SystemExit(f"podman replay failed for {image}")


def _read_b64(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes().strip()
    if not raw:
        return None
    return base64.b64decode(raw).decode("utf-8")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def patcher_ns(filename: str) -> dict:
    """The live patcher's namespace, minus its `sys.exit(main())` / `apply()` tail."""
    src = (HERE / filename).read_text(encoding="utf-8")
    src = re.sub(r"\nif __name__ == \"__main__\":\n(?:    .*\n)+", "\n", src)
    g: dict = {"__name__": "pn71_probe"}
    exec(compile(src, filename, "exec"), g)  # noqa: S102
    return g


def check(label: str, filename: str, text: str) -> int:
    g = patcher_ns(filename)
    print(f"  -- {label}: lines={len(text.splitlines())} md5={md5(text)}")
    for name, n in g["counts"](text).items():
        flag = "OK  " if n == 1 else "BAD "
        print(f"     {flag} anchor ({name}) count={n}")
    hunks, problems = g["resolve"](text)
    if problems:
        for p in problems:
            print(f"     BAD  {p}")
        return 1
    patched = text
    for _n, old, new in hunks:
        patched = patched.replace(old, new, 1)
    try:
        compile(patched, filename, "exec")
    except SyntaxError as e:
        print(f"     BAD  patched file does not byte-compile: {e}")
        return 1
    print("     OK   resolved + patched file byte-compiles")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the replay output dir")
    ap.add_argument("--pin", help="verify a single pin")
    args = ap.parse_args()

    bad = 0
    for p in _audit_replay_set():
        print(f"REPLAY-SET  BAD  {p}")
        bad += 1

    pins = (args.pin,) if args.pin else PINS
    env = compose_env()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pn71-anchors-"))
    try:
        for pin in pins:
            tag = pin.rsplit(":", 1)[-1]
            print(f"\n=== {pin}")
            replay_in_container(pin, tmp, env, tag)
            rc = (tmp / f"{tag}.rc").read_text(encoding="utf-8") if (tmp / f"{tag}.rc").exists() else ""
            print("  replay rc: " + " | ".join(rc.split("\n")).strip(" |"))
            for label, filename, rel in PATCHERS:
                b = rel.replace("/", "_")
                post = _read_b64(tmp / f"{tag}.post.{b}")
                if post is None:
                    print(f"  -- {label}: BAD  target not readable after replay")
                    bad += 1
                    continue
                bad += check(label, filename, post)
            for line in rc.splitlines():
                if line.startswith("compile ") and not line.endswith("rc=0"):
                    print(f"  BAD  {line}")
                    bad += 1
    finally:
        if args.keep:
            print(f"\nreplay output kept at {tmp}")
        else:
            subprocess.run(["rm", "-rf", str(tmp)], check=False)

    print()
    print("RESULT: all PN71-family anchors unique on every pin" if not bad
          else f"RESULT: {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
