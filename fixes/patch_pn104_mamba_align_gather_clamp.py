#!/usr/bin/env python3
"""PN104 — clamp align-mode mamba state-slot gather indices (BUG-076 tier-1).

In mamba_cache_mode="align", mamba_get_block_table_tensor selects each
request's state slots as block_table[req, (seq_lens-1)//block_size + k],
k in 0..num_speculative_blocks. Under async scheduling, seq_lens includes
in-flight/placeholder tokens; a skewed or boundary-crossing length pushes
(start + k) past the table's last column -> torch.gather index OOB ->
`ScatterGatherKernel.cu:163 Assertion idx_dim < index_size` device assert ->
EngineCore death, killing every in-flight request (BUG-076 tier-1, twice
crash-dump-confirmed on prod + bench).

Fix: clamp the gather indices to the last valid column. A clamped read
returns the request's highest cached state slot — wrong-but-valid for the
one poisoned step (PN105 aborts the request cleanly if the forward NaNs);
the engine and all co-batched requests survive. No-op for in-range indices.
"""
import pathlib
import sys

LOG = "[pn104-mamba-align-gather-clamp]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/utils.py"
)
MARKER = "# PN104:"

SITE_OLD = (
    "        indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)\n"
    "        return torch.gather(block_table, 1, indices_to_gather)\n"
)

SITE_NEW = (
    "        indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)\n"
    "        # PN104: async-skewed seq_lens can push the state-slot index past\n"
    "        # the table's last column; an OOB gather is a device-side assert\n"
    "        # that kills the whole engine (BUG-076 tier-1). Clamp to the last\n"
    "        # valid column — a wrong-but-valid slot for one poisoned step is\n"
    "        # recoverable per-request (PN105 aborts on NaN); a dead engine is\n"
    "        # not. No-op for in-range indices. Silent by design: a GPU-side\n"
    "        # engagement check would device-sync every metadata build (TPOT\n"
    "        # tax); the PN106D nan-event dump carries the audit trail instead.\n"
    "        indices_to_gather = torch.clamp(\n"
    "            indices_to_gather, max=block_table.shape[1] - 1\n"
    "        )\n"
    "        return torch.gather(block_table, 1, indices_to_gather)\n"
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
    if n != 1:
        print(f"{LOG} FATAL: anchor hits={n} — upstream drifted", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(SITE_OLD, SITE_NEW, 1), encoding="utf-8")
    print(f"{LOG} applied: align-mode state-slot gather is OOB-safe")
    return 0


sys.exit(main())
