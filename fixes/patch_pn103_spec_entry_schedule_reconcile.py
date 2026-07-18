#!/usr/bin/env python3
"""PN103 — reconcile spec-decode entries with the final per-request schedule (BUG-076).

Root cause (aibox 2026-07-18, twice crash-dump-confirmed): in Scheduler.schedule()
the spec window for a request is sized BEFORE late reducers run. On hybrid
GDN/mamba models `_mamba_block_aligned_split` (merged vllm#45477) — and, in the
waiting-queue path, the encoder budget — can shrink `num_scheduled_tokens[req]`
AFTER `scheduled_spec_decode_tokens[req]` was written with the full window
(`pad_spec_decode` writes `[-1] * num_spec_tokens` unconditionally; the async
placeholder path can hit the same skew). The emitted SchedulerOutput then
carries e.g. `num_scheduled_tokens=1` with spec entry `[-1, -1, -1]`.

Every consumer trusts `1 + len(entry)` == scheduled row count, so one such step
desyncs: grammar-bitmask rows shift onto the wrong logits rows (fully-masked
row -> argmax -> token id 0 -> "!" completions -> `Failed to advance FSM ...
for tokens 0` -> HTTP 500), spec-rejection accounting subtracts phantom drafts
from `num_computed_tokens` (self-propagating position/slot skew), and GDN
spec-state metadata mis-indexes (ScatterGatherKernel.cu:163 device assert ->
EngineCore death). PIECEWISE-only arm reproduced the bangs, so this is not a
cudagraph defect; PN358's shape-mismatch warnings are a downstream witness.

Fix: enforce the invariant at the single emission choke point — for every
request with a spec entry, `len(entry) == num_scheduled_tokens[req] - 1` —
by trimming (or dropping) oversized entries just before SchedulerOutput
construction. Healthy steps are a no-op; each trim logs a WARNING with the
request id so occurrences stay countable in the journal.

Anchor: the scheduling-constraints check block (pristine dev1060). Runs before
Genesis import-time patches; region is untouched by P58/PN524 anchors.
Self-check: idempotent via MARKER; fail-loud on drift/ambiguity.
"""
import pathlib
import sys

LOG = "[pn103-spec-entry-reconcile]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "# PN103:"

SITE_OLD = (
    "        # Check if the scheduling constraints are satisfied.\n"
    "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
    "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n"
    "\n"
    "        assert token_budget >= 0\n"
)

SITE_NEW = (
    "        # Check if the scheduling constraints are satisfied.\n"
    "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
    "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n"
    "\n"
    "        # PN103: reconcile spec-decode entries with the final schedule.\n"
    "        # Late reducers (mamba block-aligned split on hybrid models,\n"
    "        # encoder budget) can shrink num_scheduled_tokens AFTER the spec\n"
    "        # window was sized; consumers trust 1 + len(entry) == scheduled\n"
    "        # rows, and one oversized entry desyncs grammar-bitmask rows,\n"
    "        # spec-rejection accounting and GDN spec metadata (BUG-076:\n"
    '        # token-0 "!" completions, FSM-advance failures, scatter-gather\n'
    "        # device asserts). Trim to the invariant; drop when no spec\n"
    "        # position is actually scheduled.\n"
    "        for _pn103_rid in list(scheduled_spec_decode_tokens):\n"
    "            _pn103_want = num_scheduled_tokens.get(_pn103_rid, 0) - 1\n"
    "            _pn103_spec = scheduled_spec_decode_tokens[_pn103_rid]\n"
    "            if len(_pn103_spec) > _pn103_want:\n"
    "                if _pn103_want <= 0:\n"
    "                    del scheduled_spec_decode_tokens[_pn103_rid]\n"
    "                else:\n"
    "                    scheduled_spec_decode_tokens[_pn103_rid] = _pn103_spec[\n"
    "                        :_pn103_want\n"
    "                    ]\n"
    "                logger.warning(\n"
    '                    "PN103: trimmed spec entry for %s: %d -> %d '
    '(scheduled %d)",\n'
    "                    _pn103_rid,\n"
    "                    len(_pn103_spec),\n"
    "                    max(_pn103_want, 0),\n"
    "                    num_scheduled_tokens.get(_pn103_rid, 0),\n"
    "                )\n"
    "\n"
    "        assert token_budget >= 0\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skip")
        return 0
    n = src.count(SITE_OLD)
    if n == 0:
        print(
            f"{LOG} FATAL: anchor-not-found — upstream drifted; "
            "re-derive against the live file",
            file=sys.stderr,
        )
        return 1
    if n > 1:
        print(f"{LOG} FATAL: ambiguous anchor ({n} hits)", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(SITE_OLD, SITE_NEW, 1), encoding="utf-8")
    print(f"{LOG} applied: spec entries reconciled with final schedule at emission")
    return 0


sys.exit(main())
