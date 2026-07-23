#!/usr/bin/env python3
"""Holder sync_batch move/swap state-leak fix (2026-07-23, ultra-review #1).

The pinned holder's moved-loop uses `.get` on SWAP and never clears a stale
destination on unidirectional moves. A tracked<->untracked SWAP (routine:
reorder_batch_to_split_decodes_and_prefills runs every step, and PN100
classify calls are untracked rows churning among tracked ones) leaves the
SAME state dict registered at BOTH rows -> forced </think>/span tokens can
hit an unbudgeted request and the shared output_tok_ids corrupts the tracked
one. Stock vLLM's process_dict_updates pops both sides; this restores that
semantics. Prod-relevant (PN100 runs on both composes).

Apply anywhere in the boot chain (disjoint anchor; before/after other holder
grafts is irrelevant). Idempotent by marker; anchor drift FATAL.
"""
import pathlib
import sys

LOG = "[patch_holder_syncbatch_fix]"
HOLDER = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
    "thinking_budget_state.py"
)

MARKER = "# syncbatch-fix:"
ANCHOR = """        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                state1 = self._state.get(i1)
                state2 = self._state.get(i2)
                if state1 is not None:
                    self._state[i2] = state1
                if state2 is not None:
                    self._state[i1] = state2
            else:
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state
"""
REPLACEMENT = """        for i1, i2, direction in batch_update.moved:
            # syncbatch-fix: pop BOTH sides (stock process_dict_updates
            # semantics) — .get-based SWAP left one dict registered at two
            # rows on tracked<->untracked swaps; unidirectional moves never
            # cleared a stale destination. Ultra-review 2026-07-23 #1.
            if direction == MoveDirectionality.SWAP:
                state1 = self._state.pop(i1, None)
                state2 = self._state.pop(i2, None)
                if state1 is not None:
                    self._state[i2] = state1
                if state2 is not None:
                    self._state[i1] = state2
            else:
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state
                else:
                    self._state.pop(i2, None)
"""


def main() -> int:
    if not HOLDER.exists():
        print(f"{LOG} FATAL: holder missing: {HOLDER}", flush=True)
        return 1
    src = HOLDER.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} SKIP (already applied)", flush=True)
        return 0
    count = src.count(ANCHOR)
    if count != 1:
        print(f"{LOG} FATAL: anchor occurs {count}x (need 1)", flush=True)
        return 1
    HOLDER.write_text(src.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    cache = HOLDER.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(HOLDER.stem + ".*.pyc"):
            try:
                pyc.unlink()
                print(f"{LOG} dropped stale pyc {pyc.name}", flush=True)
            except OSError as exc:
                print(f"{LOG} WARN: {exc}", flush=True)
    print(f"{LOG} applied", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
