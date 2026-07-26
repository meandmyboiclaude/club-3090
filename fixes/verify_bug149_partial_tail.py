#!/usr/bin/env python3
"""BUG-149 — prove the Hunk-B tail-boundary exemption, without a boot.

Runs the REAL `Scheduler._mamba_block_aligned_split` (the function as the boot
sees it, after every sibling /fixes patch has rewritten scheduler.py) against a
copy of itself carrying only Hunk B's `end != tail_boundary` clause, and drives
a full chunked prefill through both at the `cache-override-apc.yaml` geometry.

Three claims, all falsifiable here:

  1. UNWEDGE   under APC (block 4128 / prefix_match_unit 258) the shipped
               function never completes the prefill for 93% of prompt lengths;
               the fixed one completes every length.
  2. INERT     with APC off (hash_block_size == block_size) the two functions
               return byte-identical chunk sequences for every length, so the
               exemption cannot affect the configs we actually ship.
  3. CONTRACT  every chunk end the fixed function emits is either on the block
               grid or is exactly the mandated `tail_boundary` — the BUG-140
               corruption (arbitrary mid-block ends) does not come back.

Run in-container against the live serving source:

    python3 fixes/verify_bug149_partial_tail.py

Exit 0 = all three hold.  This is not a substitute for a boot; it is what makes
a boot worth spending, since the override previously hung the box on request 1.
"""

from __future__ import annotations

import inspect
import sys
import types

BLOCK = 4128  # mamba_block_size, forced by BUG-131 at turboquant_3bit_nc
HASH = 258  # prefix_match_unit from cache-override-apc.yaml
BUDGET = 4128  # max_num_batched_tokens
MAXLEN = 20000

OLD_LINE = "if end < prefill_end and end % block_size != 0:"
NEW_LINE = "if end < prefill_end and end % block_size != 0 and end != tail_boundary:"


def _load_pair():
    """Return (shipped_fn, fixed_fn) built from the live scheduler source."""
    from vllm.v1.core.sched.scheduler import Scheduler

    shipped = Scheduler._mamba_block_aligned_split
    src = inspect.getsource(shipped)

    # Dedent to module level so it can be exec'd standalone.
    lines = src.splitlines(keepends=True)
    pad = len(lines[0]) - len(lines[0].lstrip())
    src = "".join(line[pad:] if line.strip() else line for line in lines)

    if NEW_LINE in src:
        # The patched image already carries the exemption: derive the SHIPPED
        # (pre-fix) variant instead, so the comparison still has two arms.
        fixed_src = src
        shipped_src = src.replace(NEW_LINE, OLD_LINE)
        if shipped_src == fixed_src:
            sys.exit("FAIL: could not derive the pre-fix variant")
    elif OLD_LINE in src:
        shipped_src = src
        fixed_src = src.replace(OLD_LINE, NEW_LINE)
    else:
        sys.exit(
            "FAIL: neither the shipped nor the fixed re-floor line is present in "
            "_mamba_block_aligned_split — anchors have drifted, fix this script "
            "before trusting any result from it."
        )

    ns_a: dict = {}
    ns_b: dict = {}
    exec(compile(shipped_src, "<shipped>", "exec"), ns_a)
    exec(compile(fixed_src, "<fixed>", "exec"), ns_b)
    return ns_a["_mamba_block_aligned_split"], ns_b["_mamba_block_aligned_split"]


def _sched(partial_hits: bool):
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=BLOCK, mamba_cache_mode="align"),
        use_eagle=True,  # MTP is on in every shipped compose
        hash_block_size=HASH if partial_hits else BLOCK,
        mamba_partial_cache_hit=partial_hits,
    )


def prefill(fn, nprompt: int, partial_hits: bool):
    """Drive the whole chunked prefill. -> (chunk_ends, wedged_at | None)."""
    self_ = _sched(partial_hits)
    computed = 0
    ends: list[int] = []
    # A prefill needs at most ceil(n/1) steps; the cap is a runaway guard.
    for _ in range(MAXLEN // 1 + 8):
        if computed >= nprompt:
            return ends, None
        req = types.SimpleNamespace(
            num_computed_tokens=computed,
            num_prompt_tokens=nprompt,
            num_tokens=nprompt,
            shared_prefix_boundary=0,
        )
        n = fn(self_, req, min(BUDGET, nprompt - computed))
        if n == 0:
            return ends, computed
        computed += n
        ends.append(computed)
    return ends, "loop-limit"


def main() -> int:
    shipped, fixed = _load_pair()
    ok = True

    # ── 1. UNWEDGE ────────────────────────────────────────────────────────
    wedged_shipped = [n for n in range(1, MAXLEN + 1) if prefill(shipped, n, True)[1] is not None]
    wedged_fixed = [n for n in range(1, MAXLEN + 1) if prefill(fixed, n, True)[1] is not None]
    pct = 100.0 * len(wedged_shipped) / MAXLEN
    print(f"1. UNWEDGE   APC on, prompts 1..{MAXLEN}")
    print(f"     shipped: {len(wedged_shipped):>6} wedge ({pct:.1f}%)")
    print(f"     fixed:   {len(wedged_fixed):>6} wedge")
    if not wedged_shipped:
        print("     FAIL: the shipped function did not wedge — nothing to fix?")
        ok = False
    if wedged_fixed:
        print(f"     FAIL: still wedges at {wedged_fixed[:10]}")
        ok = False

    # ── 2. INERT ──────────────────────────────────────────────────────────
    diffs = []
    for n in range(1, MAXLEN + 1):
        if prefill(shipped, n, False) != prefill(fixed, n, False):
            diffs.append(n)
    print(f"2. INERT     APC off, prompts 1..{MAXLEN}: {len(diffs)} differing")
    if diffs:
        print(f"     FAIL: diverges at {diffs[:10]}")
        ok = False

    # ── 3. CONTRACT ───────────────────────────────────────────────────────
    illegal = []
    for n in range(1, MAXLEN + 1):
        ends, wedge = prefill(fixed, n, True)
        if wedge is not None:
            continue
        tail = n // HASH * HASH
        prefill_end = max(n, n - 1)
        for e in ends:
            # Ends at or past the prefill end are unconstrained; before it, an
            # end must be block-aligned or exactly the mandated tail boundary.
            if e < prefill_end and e % BLOCK != 0 and e != tail:
                illegal.append((n, e))
    print(f"3. CONTRACT  off-grid ends that are not the mandated tail: {len(illegal)}")
    if illegal:
        print(f"     FAIL: {illegal[:10]}")
        ok = False

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
