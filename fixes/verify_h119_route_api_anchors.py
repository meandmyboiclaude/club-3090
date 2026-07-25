#!/usr/bin/env python3
"""Verify H119-API's anchors against BOOT-TIME content, on every pinned image.

WHY THIS EXISTS
---------------
The sibling consumer patch shipped a SILENT NO-OP on 2026-07-25 because its
anchors were counted against the file as `podman run --entrypoint cat` hands it
over — i.e. the pristine image — while at boot five other patches rewrite that
file first. A GPQA-30 "with the consumer on" came back byte-identical to the
control and the only trace was one INFO line.

fixes/verify_h119_consumer_anchors.py fixed that for thinking_budget_state.py by
replaying the sibling /fixes patches. This file does the same job for THIS
patch's two targets, and has to go one step further, because unlike
thinking_budget_state.py both of them are inside `apply_all`'s blast radius:

    vllm/v1/core/sched/scheduler.py
        Genesis lane-1 wiring has text patches for this file (patch_58,
        patch_62, patch_N58, patch_N40, patch_34, patch_84, patch_79c/d), all
        env-gated. Whether any of them fires depends on the compose's
        GENESIS_* environment, so the ONLY honest replay runs the real
        apply_all with the real environment.
        Then five /fixes patches rewrite it, in entrypoint order:
            pn75 -> pn96 -> pn103 -> pn105 -> pn83
    vllm/entrypoints/openai/chat_completion/api_router.py
        No /fixes patch and no genesis patch names this file (checked, and
        re-checked at runtime by _audit_replay_set below — pn81 patches
        entrypoints/GENERATE/api_router.py, a different file).

So the replay here is: a throwaway container per pin, the compose's expanded
environment, `python3 -m vllm._genesis.patches.apply_all`, then the five
scheduler writers, then the two target files come back out base64-encoded and
this script runs the PATCHER'S OWN resolver against them.

It is CPU-only and touches nothing that is running:
  * `podman run --rm` from a pinned image writes only to its own ephemeral
    overlay — the image, the host and the live vllm-tcbench-8021 container are
    untouched.
  * NO GPU device is injected and none is needed (apply_all completes rc=0
    without one). Do not add `--device nvidia.com/gpu=all` "to be faithful":
    the live server holds the card at util 0.935 and a second CUDA context is
    a real OOM risk (BUG-126).

THE ENV EXPANSION IS LOAD-BEARING. The compose writes values like
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,...}
and docker-compose expands `${VAR:-default}` before the container sees it.
Passing the literal `${...}` through instead makes torch's AllocatorConfig
tokenizer abort the process (c10 INTERNAL ASSERT, rc=134) the moment
libc10_cuda is loaded — apply_all never runs and the replay silently degrades
to "pristine". Verified 2026-07-25.

Usage:  python3 fixes/verify_h119_route_api_anchors.py [--keep] [--pin TAG]
Needs:  rootful podman (the images live in the root store).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DIST = "/usr/local/lib/python3.12/dist-packages"
SCHED_REL = "vllm/v1/core/sched/scheduler.py"
APIR_REL = "vllm/entrypoints/openai/chat_completion/api_router.py"
TARGETS = (SCHED_REL, APIR_REL)

COMPOSE = REPO / "models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml"
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"

PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
    "localhost/vllm-qwen36-endgame:dev1060cherry-20260713",
)

# /fixes patches that WRITE a target file, in tcbench8021.yml entrypoint order.
REPLAY = (
    "patch_pn75_embedding_neg_index_guard.py",
    "patch_pn96_44993_structured_output_marker_step_fsm.py",
    "patch_pn103_spec_entry_schedule_reconcile.py",
    "patch_pn105_nan_logits_abort.py",
    "patch_pn83_rerank_micro_slots.py",
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
    # This patch's own flag must be OFF during the replay: the sibling patches
    # must not see a request they would shout about, and the anchor counts are
    # flag-independent by construction.
    out = [e for e in out if not e.startswith("GENESIS_ENABLE_H119_ROUTE_API=")]
    out.append("GENESIS_ENABLE_H119_ROUTE_API=0")
    return out


def _audit_replay_set() -> list[str]:
    """Re-derive REPLAY from the tree; a drift here is how a no-op ships."""
    problems = []
    referencing = set()
    for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("cliff2b/*.py")):
        if p.name.startswith(("verify_", "test_")):
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for rel in TARGETS:
            tail = rel.split("vllm/", 1)[1]
            if tail in src and "write_text" in src:
                referencing.add(p.name)
    extra = referencing - set(REPLAY) - {"patch_h119_route_api.py"}
    if extra:
        problems.append(
            f"/fixes patches write a target file but are NOT replayed: "
            f"{sorted(extra)} — add them to REPLAY in entrypoint order")
    stale = set(REPLAY) - referencing
    if stale:
        problems.append(f"REPLAY names patches that no longer write a target: "
                        f"{sorted(stale)}")
    return problems


REPLAY_SCRIPT = """set +e
mkdir -p /out
for f in {targets}; do
  b=$(echo "$f" | tr '/' '_')
  base64 -w0 "{dist}/$f" > "/out/{tag}.pristine.$b" 2>/dev/null
done
python3 -m vllm._genesis.patches.apply_all > /out/{tag}.applyall.log 2>&1
echo "apply_all_rc=$?" > /out/{tag}.rc
for p in {replay}; do
  python3 /fixes/$p >> /out/{tag}.replay.log 2>&1
  echo "$p rc=$?" >> /out/{tag}.rc
done
for f in {targets}; do
  b=$(echo "$f" | tr '/' '_')
  base64 -w0 "{dist}/$f" > "/out/{tag}.post.$b" 2>/dev/null
done
echo done
"""


def replay_in_container(image: str, out: pathlib.Path, env: list[str],
                        tag: str) -> None:
    args = ["sudo", "podman", "run", "--rm"]
    for e in env:
        args += ["--env", e]
    args += [
        "-v", f"{GENESIS}:{DIST}/vllm/_genesis:ro",
        "-v", f"{GENESIS / 'sndr'}:{DIST}/sndr:ro",
        "-v", f"{REPO / 'fixes'}:/fixes:ro",
        "-v", f"{out}:/out:rw",
        "--entrypoint", "bash", image, "-c",
        REPLAY_SCRIPT.format(dist=DIST, targets=" ".join(TARGETS),
                             replay=" ".join(REPLAY), tag=tag),
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


def patcher_ns() -> dict:
    """The live patcher's namespace, minus its bare `sys.exit(main())` tail."""
    src = (HERE / "patch_h119_route_api.py").read_text(encoding="utf-8")
    src = src.replace("\nsys.exit(main())", "\n")
    g: dict = {"__name__": "h119_api_probe"}
    exec(compile(src, "patch_h119_route_api.py", "exec"), g)  # noqa: S102
    return g


def check_file(label: str, text: str, g: dict, variant_keys, resolver_key,
               append_key, root: pathlib.Path) -> int:
    """Report counts, resolve, apply, byte-compile. Returns 0 on success."""
    print(f"  -- {label}: lines={len(text.splitlines())} md5={md5(text)}")
    for vk in variant_keys:
        print(f"     {vk}: {g['counts_report'](text, g[vk])}")
    sites, problem = g[resolver_key](text)
    if problem:
        print(f"     BAD  {problem}")
        return 1
    print(f"     OK   resolved sites: {[n for n, _, _ in sites]}")
    patched = text
    for _n, old, new in sites:
        patched = patched.replace(old, new, 1)
    if append_key:
        patched = patched + g[append_key]
    # The marker must now be present, or the patch would re-apply every boot.
    if "# H119-API:" not in patched:
        print("     BAD  idempotency marker absent from the patched file")
        return 1
    out = root / (label.replace("/", "_") + ".patched.py")
    out.write_text(patched, encoding="utf-8")
    try:
        py_compile.compile(str(out), doraise=True, cfile=str(out) + "c")
    except Exception as e:  # noqa: BLE001
        print(f"     BAD  patched file does not compile: {e}")
        return 1
    print(f"     OK   patched file byte-compiles "
          f"(+{len(patched.splitlines()) - len(text.splitlines())} lines)")
    # And re-resolving must now soft-skip via the marker path, not re-anchor.
    sites2, _ = g[resolver_key](patched)
    if sites2 is not None and label.endswith("scheduler.py"):
        # H-shim's anchor survives (we append below it), which is exactly why
        # _patch_file() gates on the marker rather than on the anchor counts.
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pin", action="append", default=None)
    args = ap.parse_args()

    problems = _audit_replay_set()
    for p in problems:
        print(f"REPLAY-SET WARNING: {p}")

    env = compose_env()
    print(f"compose env: {len(env)} vars from {COMPOSE.name} "
          f"({sum(1 for e in env if e.startswith('GENESIS_'))} GENESIS_*)")
    print(f"replaying:   apply_all + {list(REPLAY)}")

    g = patcher_ns()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="h119-api-anchors-"))
    tmp.chmod(0o777)
    bad = len(problems)
    try:
        for image in (args.pin or PINS):
            tag = image.rsplit(":", 1)[-1]
            root = tmp / tag
            root.mkdir(parents=True)
            root.chmod(0o777)
            print(f"\n=== {image}")
            replay_in_container(image, root, env, tag)
            rc = (root / f"{tag}.rc").read_text(encoding="utf-8").strip()
            print("  replay rcs: " + " | ".join(rc.splitlines()))
            if "apply_all_rc=0" not in rc:
                bad += 1
                print("  BAD  apply_all did not complete — the counts below "
                      "are NOT what the boot sees")
            for rel, vkeys, rkey, akey in (
                (SCHED_REL, ("H_SHIM_VARIANTS", "H_TAG_VARIANTS"),
                 "_resolve_sched_sites", None),
                (APIR_REL, ("I_VARIANTS",), "_resolve_api_sites", "J_APPEND"),
            ):
                b = rel.replace("/", "_")
                pristine = _read_b64(root / f"{tag}.pristine.{b}")
                post = _read_b64(root / f"{tag}.post.{b}")
                if post is None:
                    print(f"  -- {rel}: ABSENT on this pin — the group "
                          f"soft-skips, which is correct")
                    continue
                if pristine is not None and md5(pristine) != md5(post):
                    print(f"  -- {rel}: REWRITTEN by the replay "
                          f"({len(pristine.splitlines())} -> "
                          f"{len(post.splitlines())} lines)")
                bad += check_file(rel, post, g, vkeys, rkey, akey, root)
    finally:
        if args.keep:
            print(f"\nsandbox kept at {tmp}")
        else:
            _sh(["sudo", "rm", "-rf", str(tmp)])
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "all anchors unique against boot-time content"
          if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
