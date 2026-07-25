#!/usr/bin/env python3
"""Verify H119's E/F/G consumer anchors against BOOT-TIME content, not pristine.

WHY THIS EXISTS
---------------
The first cut of sites E/F/G was validated against the file as it ships in the
image (`podman run --rm --entrypoint cat ... thinking_budget_state.py`), where
every anchor counted exactly 1. It still soft-skipped at boot:

    [h119-lens-router] soft-skip E-G: anchor(s) ['F-add'] not unique on this pin

A GPQA-30 with the consumer "on" came back byte-identical to the run with it
off. Two independent reasons, both of which "validate against pristine" hides:

  1. FIVE other genesis patches rewrite thinking_budget_state.py earlier in the
     same entrypoint, so pristine content is not what site F sees.
  2. The pin the compose actually runs (dev1474cherrymax-1757-20260725) was
     never in the validated set at all — only the two older pins were. Upstream
     added `relaxed_thinking` there and restructured sync_batch's add loop,
     which is what actually zeroed the F anchor.

This harness replays those five patches, in entrypoint order, onto the file
extracted from EVERY pinned image — so the anchor counts it reports are the
counts the boot actually sees, on the pin that actually boots.

Entrypoint order (models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml):
    patch_pn108_plateau_cap.py
    patch_pn112_conf_tap.py
    patch_pr44812_tool_guard.py
    patch_holder_syncbatch_fix.py
    pn114_boot_ids.py          (no-op without a PN114-family flag)
    patch_pn114_forced_span.py
    patch_h119_lens_router.py  <- us
`python3 -m vllm._genesis.patches.apply_all` runs before all of them but does
not touch this file (checked: no genesis patch names thinking_budget_state.py).

Usage:  python3 fixes/verify_h119_consumer_anchors.py [--keep]
Needs:  rootful podman (the images live in the root store).
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import py_compile
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DIST = "/usr/local/lib/python3.12/dist-packages"
TBS_REL = "vllm/v1/sample/thinking_budget_state.py"

# Every pin the consumer claims to support.
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
    "localhost/vllm-qwen36-endgame:dev1060cherry-20260713",
)

# Files the replayed patches read/write. Anything missing is extracted as
# absent and the patch is expected to soft-skip that site, exactly as at boot.
NEEDED = (
    TBS_REL,
    "vllm/v1/worker/gpu_input_batch.py",   # pn108 TARGET2
    "vllm/parser/qwen3.py",                # pr44812 site A
    "vllm/config/reasoning.py",            # pr44812 site B
)

# In entrypoint order. pn114_boot_ids.py is skipped: it is flag-gated off in
# the shipped compose and writes no bytes to the holder when skipped.
REPLAY = (
    "patch_pn108_plateau_cap.py",
    "patch_pn112_conf_tap.py",
    "patch_pr44812_tool_guard.py",
    "patch_holder_syncbatch_fix.py",
    "patch_pn114_forced_span.py",
)


def _podman(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "podman", *args], capture_output=True)


def extract(image: str, root: pathlib.Path) -> dict[str, bool]:
    """Pull NEEDED out of the image into `root`. Returns rel -> present."""
    present: dict[str, bool] = {}
    for rel in NEEDED:
        cp = _podman(["run", "--rm", "--entrypoint", "cat", image, f"{DIST}/{rel}"])
        ok = cp.returncode == 0 and bool(cp.stdout)
        present[rel] = ok
        if ok:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(cp.stdout)
    return present


def replay(root: pathlib.Path, verbose: bool = True) -> None:
    """Run the sibling patches against `root` instead of the real dist-packages."""
    for name in REPLAY:
        src = (HERE / name).read_text(encoding="utf-8")
        # The patches hard-code the container path; re-point them at the sandbox.
        src = src.replace(DIST, str(root))
        g = {"__name__": "__main__", "__file__": str(HERE / name)}
        try:
            exec(compile(src, name, "exec"), g)  # noqa: S102 - deliberate
        except SystemExit as e:
            if e.code not in (0, None):
                raise SystemExit(f"replay {name} failed rc={e.code}")


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def patcher_ns() -> dict:
    """The live patcher's namespace, minus its bare `sys.exit(main())` tail."""
    src = (HERE / "patch_h119_lens_router.py").read_text(encoding="utf-8")
    src = src.replace("\nsys.exit(main())", "\n")
    g: dict = {"__name__": "h119_probe"}
    exec(compile(src, "patch_h119_lens_router.py", "exec"), g)  # noqa: S102
    return g


def anchor_report(text: str, g: dict) -> tuple[list[str] | None, str]:
    """Run the patcher's OWN site resolver against `text`.

    Returns (site names applied, diagnostic line). Counting here would only
    re-implement the thing under test; this exercises the real decision.
    """
    counts = [f"E-shims={text.count(g['E_OLD'])}"]
    for vname, pairs in g["F_VARIANTS"]:
        counts.append(f"{vname}={[text.count(o) for o, _ in pairs]}")
    counts.append(f"G-resolve={text.count(g['G_OLD'])}")
    sites, problem = g["_resolve_consumer_sites"](text)
    if problem:
        return None, f"{problem}  [{'  '.join(counts)}]"
    return [n for n, _, _ in sites], "  ".join(counts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the sandbox trees")
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="h119-anchors-"))
    bad = 0
    g = patcher_ns()
    try:
        for image in PINS:
            root = tmp / image.rsplit(":", 1)[-1]
            root.mkdir(parents=True)
            print(f"\n=== {image}")
            present = extract(image, root)
            if not present[TBS_REL]:
                print("  thinking_budget_state.py ABSENT — E-G soft-skip is correct")
                continue
            tbs = root / TBS_REL
            print(f"  pristine  md5={md5(tbs)}  lines={len(tbs.read_text().splitlines())}")
            replay(root)
            post = tbs.read_text(encoding="utf-8")
            print(f"  post-patch md5={md5(tbs)}  lines={len(post.splitlines())}")
            sites, detail = anchor_report(post, g)
            print(f"  counts: {detail}")
            if sites is None:
                bad += 1
                print("  BAD no consumer site set fits this pin's boot content")
                continue
            print(f"  OK  resolved sites: {sites}")
            # And the patched result must byte-compile.
            patched = post
            for name, old, new in g["_resolve_consumer_sites"](post)[0]:
                patched = patched.replace(old, new, 1)
            out = root / "patched_tbs.py"
            out.write_text(patched, encoding="utf-8")
            try:
                py_compile.compile(str(out), doraise=True,
                                   cfile=str(out) + "c")
                print("  OK  patched file byte-compiles")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"  BAD patched file does not compile: {e}")
    finally:
        if args.keep:
            print(f"\nsandbox kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "all anchors unique post-patch" if not bad
          else f"{bad} anchor(s) NOT unique post-patch")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
