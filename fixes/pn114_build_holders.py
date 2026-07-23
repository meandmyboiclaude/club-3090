#!/usr/bin/env python3
"""Build /tmp/pn114/ holder variants for the PN114 span-soundness tests.

Runs INSIDE the vllm-tcbench-8021 container:
  1. holder_pre.py      = the LIVE patched holder (old grafts) — T-A baseline
  2. holder_prepatch.py = live text with the OLD pn114 grafts REVERSED
                          (round-trip re-apply must reproduce live exactly)
  3. thinking_budget_state.py = prepatch + the reworked graft script
                          (/fixes/patch_pn114_forced_span.py) — what the next
                          boot will produce

Needs /tmp/pn114/patch_old.py = the pre-redesign graft script
(`git show HEAD:fixes/patch_pn114_forced_span.py`), podman-cp'd in first.
"""
import importlib.util
import pathlib
import py_compile
import sys

LIVE = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
    "thinking_budget_state.py"
)
WORK = pathlib.Path("/tmp/pn114")
OLD = WORK / "patch_old.py"
NEW = pathlib.Path("/fixes/patch_pn114_forced_span.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reverse(txt, grafts, tag):
    for marker, anchor, repl, what in reversed(grafts):
        n = txt.count(repl)
        if n != 1:
            raise RuntimeError(
                f"reverse[{tag}] of {what!r}: replacement occurs {n}x")
        txt = txt.replace(repl, anchor)
    return txt


def apply(txt, grafts, tag):
    for marker, anchor, repl, what in grafts:
        n = txt.count(anchor)
        if n != 1:
            raise RuntimeError(f"apply[{tag}] {what!r}: anchor occurs {n}x")
        if marker not in repl:
            raise RuntimeError(f"apply[{tag}] {what!r}: marker missing")
        txt = txt.replace(anchor, repl, 1)
    return txt


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    old = load("pn114_patch_old", OLD)
    new = load("pn114_patch_new", NEW)
    live = LIVE.read_text(encoding="utf-8")

    # The live container may carry either graft generation — or a MIX (a
    # boot mid-iteration). Reverse site by site, trying both generations'
    # replacements, until no pn114 markers remain.
    prepatch = live
    candidates = ([(a, r, "new:" + w) for _m, a, r, w in new.GRAFTS]
                  + [(a, r, "old:" + w) for _m, a, r, w in old.GRAFTS])
    progress = True
    while progress:
        progress = False
        for anchor, repl, tag in candidates:
            if prepatch.count(repl) == 1:
                prepatch = prepatch.replace(repl, anchor)
                print(f"reversed {tag}")
                progress = True
    leftovers = [m for m in ("# PN114", "# P-pen:") if m in prepatch]
    if leftovers:
        print(f"FATAL: markers remain after reversal: {leftovers}")
        return 1

    (WORK / "holder_prepatch.py").write_text(prepatch, encoding="utf-8")
    pre = apply(prepatch, old.GRAFTS, "old->pre")
    nt = apply(prepatch, new.GRAFTS, "new")
    (WORK / "holder_pre.py").write_text(pre, encoding="utf-8")
    out = WORK / "thinking_budget_state.py"
    out.write_text(nt, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    py_compile.compile(str(WORK / "holder_pre.py"), doraise=True)
    print("BUILD OK: holder_pre.py + thinking_budget_state.py (new) ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
