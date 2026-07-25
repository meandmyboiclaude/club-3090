#!/usr/bin/env python3
"""In-container half of `verify_genesis_reanchors.py` — never run by hand.

Runs inside a throwaway container built from a pinned image, with the genesis
tree bind-mounted exactly the way the compose mounts it. For each re-anchored
capability it asks the MODULE ITSELF to build its patcher (so the anchor
selection under test is the one the boot will use), counts every anchor in the
real target file, applies to a scratch copy and byte-compiles the result.

Counting against pristine image bytes without the sibling patches is how a
patch shipped as a silent no-op on 2026-07-25, so the caller may point this at
the LIVE post-boot files instead via /led-live (read-only copies made with a
single-file `podman cp` — never a recursive scan inside the serving container).

Exit code 0 = every case met its expectation.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
LIVE = pathlib.Path("/led-live")  # optional; post-boot copies keyed by basename

failures: list[str] = []
lines: list[str] = []


def say(s: str) -> None:
    lines.append(s)
    print(s, flush=True)


def target_for(patcher) -> pathlib.Path:
    """Prefer a post-boot copy of the patcher's target when one was supplied."""
    p = pathlib.Path(str(patcher.target_file))
    live = LIVE / p.name
    if live.is_file():
        return live
    return p


def check(label: str, patcher, expect: dict[str, int]) -> None:
    """expect: {sub_patch_name: expected_anchor_count}.

    On a post-boot file the anchor is legitimately count == 0 because the
    patch already consumed it. That is a PASS, not a drift — but only when the
    patch's own marker is in the file. "Anchor gone and marker absent" is the
    real failure and stays a failure.
    """
    if patcher is None:
        failures.append(f"{label}: patcher is None (target unresolved)")
        say(f"  FAIL {label}: patcher is None")
        return
    src = target_for(patcher)
    say(f"  {label}: target {src}")
    text = src.read_text(encoding="utf-8")
    if str(src).startswith(str(LIVE)) and patcher.marker in text:
        say(f"    post-boot: marker present -> ALREADY APPLIED this boot "
            f"(anchors legitimately consumed)")
        return
    ok = True
    for sp in patcher.sub_patches:
        n = text.count(sp.anchor)
        want = expect.get(sp.name)
        verdict = "ok" if want is None or n == want else "MISMATCH"
        if want is not None and n != want:
            ok = False
            failures.append(
                f"{label}/{sp.name}: anchor count {n}, expected {want}"
            )
        say(f"    {sp.name:<44} count={n} expect={want} {verdict}")
    if not ok:
        return
    # Apply for real on a scratch copy and byte-compile the result.
    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td) / src.name
        shutil.copy2(src, scratch)
        patched = scratch.read_text(encoding="utf-8")
        applied = []
        for sp in patcher.sub_patches:
            if patched.count(sp.anchor) == 1:
                patched = patched.replace(sp.anchor, sp.replacement, 1)
                applied.append(sp.name)
        try:
            compile(patched, src.name, "exec")
            say(f"    applied {applied} -> byte-compiles OK")
        except SyntaxError as e:
            failures.append(f"{label}: patched file does not compile: {e}")
            say(f"    FAIL patched file does not compile: {e}")


def absent(label: str, path: pathlib.Path, needle: str) -> None:
    p = LIVE / path.name if (LIVE / path.name).is_file() else path
    n = p.read_text(encoding="utf-8").count(needle)
    say(f"  {label}: {needle!r} count={n} (expect 0)")
    if n:
        failures.append(f"{label}: {needle!r} present {n}×, expected absent")


def present(label: str, path: pathlib.Path, needle: str, want: int = 1) -> None:
    p = LIVE / path.name if (LIVE / path.name).is_file() else path
    n = p.read_text(encoding="utf-8").count(needle)
    say(f"  {label}: {needle!r} count={n} (expect {want})")
    if n != want:
        failures.append(f"{label}: {needle!r} count {n}, expected {want}")


def main() -> int:
    sys.path.insert(0, str(VLLM.parent))
    say(f"pin python sees vllm at {VLLM}")
    say(f"live post-boot copies: {sorted(p.name for p in LIVE.glob('*.py'))}"
        if LIVE.is_dir() else "live post-boot copies: (none supplied)")

    # ---- P26: partial absorption ------------------------------------------
    say("\n== P26 TQ prefill output prealloc (partial absorption)")
    from vllm._genesis.wiring.legacy import patch_26_prefill_output as p26
    tq = VLLM / "v1/attention/backends/turboquant_attn.py"
    present("P26 cu_2 absorbed-by-upstream marker", tq, p26.CU2_ABSORBED_UPSTREAM_FORM, 1)
    tq_live = (LIVE / tq.name).is_file()
    if not tq_live:
        # Only meaningful pre-boot: post-boot the pool IS in the file because
        # P26 put it there.
        for m in p26.UPSTREAM_DRIFT_MARKERS:
            absent("P26 patch-level drift marker", tq, m)
    else:
        present("P26 pool landed this boot", tq, "acquire_prefill_output", 1)
    check("P26", p26._make_patcher(),
          {"p26_output_alloc": 1, "p26_cu_2_alloc": 0})

    # ---- P34: absorbed on the build line ----------------------------------
    say("\n== P34 Mamba zero-collapse deadlock guard (absorbed)")
    from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
    sched = VLLM / "v1/core/sched/scheduler.py"
    live_sched = (LIVE / sched.name).is_file()
    sched_text = (LIVE / sched.name if live_sched else sched).read_text(
        encoding="utf-8")
    hits = [m for m in p34.UPSTREAM_DRIFT_MARKERS if m in sched_text]
    say(f"  P34 drift markers present: {hits}")
    if live_sched:
        # POST-BOOT the guard is gone on purpose: /fixes pr48361 removes it
        # and floors unconditionally. apply_all runs BEFORE /fixes, so P34
        # still self-retired on the marker at its own dispatch time.
        n = sched_text.count("PR48361: floor UNCONDITIONALLY")
        say(f"  P34 post-boot: pr48361 unconditional floor count={n} "
            f"(it deliberately removes the guard P34 wanted)")
        if n != 1:
            failures.append(
                "P34 post-boot: pr48361's unconditional floor not found — "
                "the supersession chain recorded on this row is stale")
    elif not hits:
        failures.append("P34: no drift marker present — absorption claim is stale")

    # ---- P82: re-anchored --------------------------------------------------
    say("\n== P82 SGLang threshold_single OR-clause (re-anchored)")
    from vllm._genesis.wiring.spec_decode import (
        patch_82_sglang_acceptance_threshold as p82,
    )
    rs = VLLM / "v1/sample/rejection_sampler.py"
    rs_text = (LIVE / rs.name if (LIVE / rs.name).is_file() else rs).read_text(
        encoding="utf-8")
    for anchor, _v, lbl in p82.P82_ANCHOR_FORMS:
        say(f"  P82 anchor {lbl:<24} count={rs_text.count(anchor)}")
    p82_applied = p82.GENESIS_P82_MARKER_PREFIX in rs_text
    picked = p82._select_anchor(rs_text)
    if picked is None and not p82_applied:
        failures.append("P82: no anchor generation matched")
        say("  FAIL P82: no anchor generation matched")
    elif p82_applied:
        say("  P82 already applied in this file (marker present) — anchors "
            "legitimately consumed")
    else:
        say(f"  P82 selected generation: {picked[2]}")
    absent("P82 removed false drift marker", rs,
           "sample_recovered_tokens_kernel_MUST_NOT_BE_A_DRIFT_MARKER")
    present("P82 the symbol that used to retire it", rs,
            "sample_recovered_tokens_kernel", 2)
    if not p82_applied:
        check("P82", p82._make_patcher(0.3, 0), {"p82_threshold_or_clause": 1})

    # ---- P91B: relocated inc target ---------------------------------------
    say("\n== P91B AutoRound row-group cdiv (inc package split)")
    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    from sndr.engines.vllm.patches.quantization import (
        p91b_autoround_row_group_cdiv_multi_scheme as p91b,
    )
    resolved = p91b._resolve_inc_target()
    say(f"  P91B inc target resolved: {resolved}")
    if resolved is None:
        failures.append("P91B: inc target unresolved")
    check("P91B inc/dev338", p91b._make_inc_dev338_patcher(),
          {"p91b_inc_dev338_floor_partition_to_cdiv": 1})
    check("P91B wNa16", p91b._make_wna16_patcher(),
          {"p91b_ct_wNa16_floor_input_size_to_cdiv": 1})
    check("P91B w4a8_fp8", p91b._make_w4a8_fp8_patcher(),
          {"p91b_ct_w4a8_fp8_floor_input_size_to_cdiv": 1})

    say("")
    if failures:
        say(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            say(f"  - {f}")
        return 1
    say("RESULT: all re-anchors counted and unique")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("GENESIS_ENABLE_P82", "1")
    sys.exit(main())
