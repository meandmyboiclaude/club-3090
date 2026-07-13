#!/usr/bin/env python3
"""PN87 — Mamba/GDN prefix-cache hit must stop before the boundary under MTP.

Backport of vllm#43650 (merged upstream AFTER the dev1060 pin 9e57de71) to
vllm/v1/core/single_type_kv_cache_manager.py::MambaManager.find_longest_cache_hit
(+6/-0 upstream).

Bug: with EAGLE/MTP + prefix caching, full-attention managers match one extra
block and then drop the final matched block (it must be recomputed to produce
the hidden states the draft head needs). Mamba/GDN state blocks are laid out
[null, ..., state] — the ONLY real block is the final one holding the SSM
state snapshot, so applying the same "drop the last matched block" treatment
removes the state itself. Upstream fix: when the eagle drop is requested,
the Mamba finder simply does not SEARCH the final block
(`max_num_blocks -= 1`), guaranteeing the tail is recomputed without ever
popping a state block.

In our pin the drop is signalled per-call via the `drop_eagle_block`
parameter (upstream PR base still names it `use_eagle`); MambaManager
accepts it and silently IGNORES it. Exposure in the pin:
  - UnitaryKVCacheCoordinator (pure mamba/GDN model + MTP) passes
    drop_eagle_block=True (kv_cache_coordinator.py ~496) — hit lands on the
    boundary, the tail block is never recomputed, draft head sees stale
    hidden state → accuracy loss.
  - find_longest_cache_hit_per_group (~826) — same mechanism.
(The hybrid coordinator masks the bug for full-attn+mamba models by bounding
mamba's max_length to full attention's post-drop length.)

Anchor drift vs PR: parameter renamed use_eagle→drop_eagle_block, and the pin
adds a fine-grained partial-hash search loop before the coarse block loop
(MambaManager.supports_fine_grained_hash_lookup=True). Semantics preserved by
guarding BOTH search loops: the coarse loop exactly as upstream, the
fine-grained loop with the analogous `max_num_partial_units -= 1` (its drop
unit is one hash block, mirroring the coordinator's eagle_margin logic).

Retire when the pin advances past vllm#43650: self-retires when
MambaManager.find_longest_cache_hit already references drop_eagle_block (or
use_eagle) in its body.
"""
import pathlib
import sys

LOG = "[pn87-mamba-eagle-cache-hit-boundary]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py"
)
MARKER = "# PN87:"

# Coarse block-granularity loop (upstream hunk, adapted param name).
COARSE_OLD = (
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
)
COARSE_NEW = (
    "        max_num_blocks = max_length // block_size\n"
    "        if drop_eagle_block and max_num_blocks > 0:\n"
    "            # PN87: vllm#43650 backport — full-attention cache-hit lookup\n"
    "            # matches one extra block and then drops that final block as\n"
    "            # it's only partially accepted. Mamba/GDN state blocks are\n"
    "            # [null, ..., state] so popping after a match removes the state.\n"
    "            # Instead, we can only search up to the boundary (not include\n"
    "            # final), forcing the tail to be recomputed for the draft head.\n"
    "            max_num_blocks -= 1\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
)

# Fine-grained partial-hash loop (pin-only; not in the PR base).
FINE_OLD = (
    "            max_num_partial_units = min(\n"
    "                max_length // hash_block_size, len(block_hashes)\n"
    "            )\n"
    "            for fine_idx in range(max_num_partial_units - 1, -1, -1):\n"
)
FINE_NEW = (
    "            max_num_partial_units = min(\n"
    "                max_length // hash_block_size, len(block_hashes)\n"
    "            )\n"
    "            if drop_eagle_block and max_num_partial_units > 0:\n"
    "                # PN87: vllm#43650 backport (fine-grained analogue) — the\n"
    "                # drop unit for partial-hash managers is one hash block;\n"
    "                # exclude the final unit instead of popping the state block.\n"
    "                max_num_partial_units -= 1\n"
    "            for fine_idx in range(max_num_partial_units - 1, -1, -1):\n"
)

CLASS_SPLIT = "class MambaManager(SingleTypeKVCacheManager):"


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0

    if text.count(CLASS_SPLIT) != 1:
        print(f"{LOG} FATAL: MambaManager class anchor not found or ambiguous "
              f"— upstream refactor; re-derive", file=sys.stderr)
        return 1
    head, tail = text.split(CLASS_SPLIT, 1)

    # Upstream-merged drift: finder body already consults the eagle flag.
    finder_body = tail.split("def find_longest_cache_hit", 1)
    if len(finder_body) == 2:
        body = finder_body[1]
        # Cut at the next method to scope the check to this finder.
        body = body.split("\n    def ", 1)[0]
        sig_end = body.find("-> tuple")
        rest = body[sig_end:] if sig_end != -1 else body
        if "drop_eagle_block" in rest or "use_eagle" in rest:
            print(f"{LOG} upstream drift: MambaManager finder already handles "
                  f"the eagle drop — self-retire (no-op)")
            return 0

    for name, old in (("coarse", COARSE_OLD), ("fine-grained", FINE_OLD)):
        n_tail = tail.count(old)
        if n_tail != 1:
            print(f"{LOG} FATAL: {name}-loop anchor "
                  f"{'not found' if n_tail == 0 else 'ambiguous'} in "
                  f"MambaManager — upstream refactor; re-derive (MTP + prefix "
                  f"caching on mamba/GDN models loses accuracy without this)",
                  file=sys.stderr)
            return 1
        if head.count(old) or text.count(old) != 1:
            print(f"{LOG} FATAL: {name}-loop anchor also matches outside "
                  f"MambaManager — refusing ambiguous patch", file=sys.stderr)
            return 1

    tail = tail.replace(COARSE_OLD, COARSE_NEW, 1).replace(FINE_OLD, FINE_NEW, 1)
    TARGET.write_text(head + CLASS_SPLIT + tail)
    print(f"{LOG} applied: MambaManager.find_longest_cache_hit now excludes "
          f"the final block/hash-unit when drop_eagle_block is set "
          f"(vllm#43650 backport, both search loops)")
    return 0


sys.exit(main())
