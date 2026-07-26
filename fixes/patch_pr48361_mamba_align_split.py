#!/usr/bin/env python3
"""PR #48361 — stop `mamba_cache_mode=align` writing prefix-cache entries whose
hash lies about the recurrent state they hold.  (BUG-140, P1.)

THE DEFECT
----------
The align path hashes mamba block-table slot `p` as "the SSM state at exactly
(p+1)*block_size tokens".  `_mamba_block_aligned_split` is what must guarantee
that, by making every non-final prefill chunk end on a block boundary.  It does
not: a chunk can end mid-block, the kernel snapshots a DIFFERENT offset into
that slot, and the hash then advertises a state the block does not contain.
A later request that hits that block resumes from the wrong recurrent state and
silently emits corrupted output — no exception, no log, no counter.

THIS IS THE COMMON PATH, NOT A RACE.  The compose runs
`--max-num-batched-tokens 4128` and the attention block size is forced to 4128
(BUG-131), so ANY prefill sharing a step with a decode (4 tokens under MTP-3)
takes a 4124-token first chunk — mid-block, hashed as state@4128.  Measured at
the live geometry, 5160-token prompts:

    1 cached mamba pos 0 as state@4128 but block holds state@3096
    2 cached mamba pos 0 as state@4128 but block holds state@2060
    3 cached mamba pos 0 as state@4128 but block holds state@1020
    4 cached mamba pos 0 as state@4128 but block holds state@4104

WHY IT HAS NOT BITTEN, AND WHY THAT IS TEMPORARY
------------------------------------------------
BUG-131 is accidentally protecting us: real traffic is under 4129 tokens, so it
produces zero block hashes and never READS the poisoned entries (102,596
queries / 0 hits).  The entries ARE being written today for any prompt >= 4129.
Enabling `CACHE_OVERRIDE=cache-override-apc.yaml` (prefix_match_unit: 258)
starts reading them.  Do not enable that override without this patch.

TWO INDEPENDENT CAUSES, and both must go
----------------------------------------
Isolated by three-way bisect against the restored upstream tests
(115 checks, run in-container against the live serving source):

  fork HEAD                                     20 failed / 95 passed
  minus our #40757 floor guard                  11 failed / 104 passed
  ... plus PR #48361's boundary re-floor        115 passed / 0 failed

  A) OUR #40757 guard (`if aligned > start: end = aligned`) keeps a sub-block
     end whenever flooring would empty the chunk.  It was added to stop a
     scheduler spin, but it is a local workaround for a SYMPTOM: it buys
     progress by writing a chunk boundary the hash contract forbids.
  B) UPSTREAM is missing the re-floor after the mandatory-stop clamp: `stops`
     can pull `end` back off the grid, and nothing puts it back.  PR #48361
     fixes this and is STILL UNMERGED upstream, so `origin/main` carries the
     same defect and ships no test for it.

The guard in (A) is safe to drop ONLY because (B) lands with it — upstream's
own `test_deferred_fragment_progresses_with_block_budget` is the anti-spin
assertion, it FAILS on fork HEAD, and it PASSES in the fixed configuration.
That test is the evidence the starvation (A) was written to prevent does not
return.  Never apply one hunk without the other.

THE HUNK-B EXEMPTION (BUG-149) — why the re-floor is not unconditional
---------------------------------------------------------------------
Hunk B as first written re-floored EVERY off-grid `end`.  That is correct for
three of the four mandatory stops, which are block-aligned by construction, and
catastrophic for the fourth: `tail_boundary` deliberately sits on the HASH grid
whenever `Scheduler.mamba_partial_cache_hit` is on (it is the position the
prompt's partial-tail entry can only be registered at).  Flooring it drives
`end` back to `start`, `_mamba_block_aligned_split` returns 0, and both call
sites treat 0 as "cannot schedule" (`continue` at :641, `break` at :1009) — so
the request sits in the waiting queue forever at 0% GPU.

That is BUG-149, and it is why `cache-override-apc.yaml` hangs the box rather
than merely under-performing.  Measured against the real function at the
override's geometry (block 4128 / prefix_match_unit 258), 93.2% of prompt
lengths in 1..20000 never complete their prefill; the survivors are the exact
multiples of 258.  With the override off, `tail_boundary` is 0 and the
exemption cannot fire, so this changes nothing on the shipped configs — proven
by an all-lengths equivalence sweep in `verify_bug149_partial_tail.py`.

The alignment contract is not weakened: chunk ends before `last_cache_position`
are still block-floored by the earlier clause, so a budget-driven mid-block end
(the BUG-140 corruption) is still impossible.  Only the one position the
partial-hit feature itself mandates is allowed through, and
`KVCacheCoordinator.cache_blocks` caches exactly that position unaligned when
`enable_partial_hash_hits` is set.

DELIVERY
--------
TEXT patch.  `apply_all` runs standalone and the entrypoint then does
`exec vllm serve`, which replaces the process — so a setattr/monkeypatch would
be discarded before a token is served and only file writes survive.

Anchors are counted, and counted against the file AS THE BOOT SEES IT — five
sibling /fixes patches rewrite this same file first (pn75 -> pn96 -> pn103 ->
pn105 -> pn83).  Counting against pristine image bytes is exactly how a patch
shipped as a silent no-op here on 2026-07-25.  Verify without a boot:

    python3 fixes/verify_pr48361_anchors.py

DEFAULT ON.  This is a correctness fix for silent output corruption; a
correctness fix that ships dark is a correctness fix nobody runs, and this tree
has lost work that way five times.  Kill switch: GENESIS_DISABLE_PR48361=1.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

LOG = "[pr48361-mamba-align]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED = VLLM / "v1/core/sched/scheduler.py"
MARKER = "# PR48361:"

# ── Hunk A — drop the #40757 sub-block escape ──────────────────────────────
# Anchored on code only (no comment text), so rewording the comment above it
# cannot silently invalidate the patch.
A_OLD = (
    "            aligned = end // block_size * block_size\n"
    "            if aligned > start:\n"
    "                end = aligned\n"
)
A_NEW = (
    "            # PR48361: floor UNCONDITIONALLY. The old escape kept a\n"
    "            # sub-block end when flooring would empty the chunk, which\n"
    "            # buys progress by writing a boundary the mamba hash contract\n"
    "            # forbids. Hunk B restores progress the correct way.\n"
    "            end = end // block_size * block_size\n"
)

# ── Hunk B — PR #48361's re-floor after the mandatory-stop clamp ───────────
B_OLD = (
    "        end = min((s for s in stops if start < s < end), default=end)\n"
    "        return max(end - start, 0)\n"
)
B_NEW = (
    "        end = min((s for s in stops if start < s < end), default=end)\n"
    "        # PR48361: a mandatory stop can pull `end` back off the block\n"
    "        # grid. Until the prefill's last cacheable position, re-floor it,\n"
    "        # clamped at `start` so the caller still sees an empty chunk and\n"
    "        # skips rather than spins.\n"
    "        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)\n"
    "        # BUG-149: `tail_boundary` is the one mandatory stop that is NOT on\n"
    "        # the block grid by construction — under `mamba_partial_cache_hit`\n"
    "        # it sits on the hash grid, and the coordinator caches it verbatim\n"
    "        # (`cache_blocks` skips its own alignment when partial hits are on).\n"
    "        # Re-flooring it clamps `end` back to `start` and the request is\n"
    "        # skipped every step forever. It is 0 whenever partial hits are off,\n"
    "        # so this exemption is inert on the block-aligned configs.\n"
    "        if end < prefill_end and end % block_size != 0 and end != tail_boundary:\n"
    "            end = max(end // block_size * block_size, start)\n"
    "        return max(end - start, 0)\n"
)

HUNKS = (("A-floor", A_OLD, A_NEW), ("B-refloor", B_OLD, B_NEW))

# Some pins are ALREADY CORRECT and must not be reported as drift.
# `dev1060cherry-20260713` carries the original full #48361 pick (commit
# 310495614) — its `_mamba_block_aligned_split` still has upstream's
# `elif num_computed_tokens_after_sched < prefill_end:` branch and the
# unconditional floor. The 2026-07-25 replay (25e7a397b) re-picked only the
# slimmed upstream head and silently dropped both, which is what created
# BUG-140 on the newer pins. So zero anchor matches there means "nothing to
# do", not "anchors drifted" — and shouting about it would train the next
# reader to ignore the shout that matters.
ALREADY_FIXED = "elif num_computed_tokens_after_sched < prefill_end:"


def _shout(lines: list[str]) -> None:
    """A soft-skip here is a silent correctness hole. Make it impossible to miss."""
    bar = "=" * 72
    for stream in (sys.stderr,):
        print(bar, file=stream)
        for ln in lines:
            print(ln, file=stream)
        print(bar, file=stream)
    logging.getLogger("vllm.pr48361").error(" | ".join(lines))


def resolve(src: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Return (applicable hunks, problems). Counts, never guesses."""
    ok, bad = [], []
    for name, old, new in HUNKS:
        n = src.count(old)
        if n == 1:
            ok.append((name, old, new))
        else:
            bad.append(f"{name}: anchor count {n} (need exactly 1)")
    return ok, bad


def main() -> int:
    if os.environ.get("GENESIS_DISABLE_PR48361", "").strip() in ("1", "true", "yes", "on"):
        print(f"{LOG} disabled by GENESIS_DISABLE_PR48361 — mamba align split "
              f"left as-is (prefix-cache entries may be poisoned; do NOT enable "
              f"cache-override-apc.yaml)")
        return 0
    if not SCHED.is_file():
        print(f"{LOG} {SCHED} absent on this pin — skip")
        return 0

    src = SCHED.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied (marker present) — skip")
        return 0
    if ALREADY_FIXED in src:
        print(f"{LOG} pin already carries upstream #48361's boundary branch — "
              f"nothing to do (this is the pre-replay shape, not drift)")
        return 0

    ok, bad = resolve(src)
    if bad or len(ok) != len(HUNKS):
        _shout([
            f"{LOG} ERROR: NOT APPLIED — BUG-140 remains live on this boot.",
            *[f"  {b}" for b in bad],
            "  mamba_cache_mode=align can write prefix-cache entries whose hash",
            "  advertises a recurrent state the block does not hold. Corruption",
            "  is silent. DO NOT enable CACHE_OVERRIDE=cache-override-apc.yaml.",
            "  Re-anchor with: python3 fixes/verify_pr48361_anchors.py",
        ])
        return 0  # never take a boot down over a patch

    # Both hunks or neither: hunk A removes the starvation guard and hunk B is
    # what makes that safe. Half of this is worse than none of it.
    for _name, old, new in ok:
        src = src.replace(old, new, 1)
    SCHED.write_text(src, encoding="utf-8")
    print(f"{LOG} applied {len(ok)}/{len(HUNKS)} hunks "
          f"({', '.join(n for n, _, _ in ok)}) — mamba chunk ends now land on "
          f"the block grid, so cached SSM state matches its hash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
