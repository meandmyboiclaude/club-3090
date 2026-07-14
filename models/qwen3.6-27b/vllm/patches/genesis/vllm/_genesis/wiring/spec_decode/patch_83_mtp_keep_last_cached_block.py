# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 83 — MTP keep-last-cached-block (vllm#38182 mitigation).

Roots cause: `vllm/v1/core/single_type_kv_cache_manager.py:559-562`
(FullAttentionManager) and `:673-684` (SlidingWindowManager) force-pop the
last matched cached block when the per-call `drop_eagle_block` flag is True.
This is intentional for true Eagle/Eagle3 drafters which need the last
block's hidden states re-materialised. **However**, vLLM treats MTP as
"eagle" via `vllm/config/speculative.py:1070-1071`:

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash")

and the coordinator turns that into the per-call drop flag at
`vllm/v1/core/kv_cache_coordinator.py:643`:

    drop_eagle_block = use_eagle and idx not in eagle_verified

so MTP still drives `drop_eagle_block=True` on full-attention layers.
For MTP this pop() is OVERLY CONSERVATIVE because MTP has its own drafter
LAYER that consumes KV directly (not pre-materialised hidden states from
the prefill). The pop costs ~1024 recomputed tokens per cache-hit on
hybrid models like Qwen3.6-MoE where P5 (KV cache page size unification)
LCM-pads block size up to Mamba layer requirement.

Empirical evidence (this rig, Qwen3.6-35B-A3B-FP8 + MTP K=3, prod 2× A5000):
- prefix-cache ON  + default v7.47 launch: ~164 tok/s mean, multi-turn
  TTFT ~430ms on 2.5K shared context (cache effectively useless due to
  pop + scheduler overhead)
- prefix-cache OFF + same launch (v7.48): ~213 tok/s mean (+30%),
  multi-turn TTFT same ~430ms (no cache benefit at all)
- prefix-cache ON  + --block-size 16 (v7.50): ~163 tok/s mean (block-size
  silently overridden by P5 LCM-pad on hybrid Qwen3.6 → no improvement)

P83 skips the pop() when `GENESIS_ENABLE_P83=1` is set, allowing MTP
to actually benefit from prefix-cache hits. **Targets MTP only; do NOT
enable for true Eagle/Eagle3** — those drafters genuinely need the
re-materialised hidden states.

================================================================
SAFETY MODEL
================================================================

- Default OFF. Opt-in via env `GENESIS_ENABLE_P83=1`.
- Wraps both pop sites in a guard checked at the outer-method scope.
  Drafter quality unaffected when env is unset.
- If MTP drafter reads stale hidden_states (the case the original pop
  guards against), expected effect is slightly lower acceptance rate
  on the FIRST K tokens after a cache hit — not a correctness bug,
  just a small efficiency regression on first-burst.
- Long-running streams (turns 2+) get FULL prefix-cache benefit,
  which dwarfs any first-burst acceptance loss.

================================================================
EXPECTED OUTCOMES
================================================================

Best case: prefix-cache ON + P83 ON → match v748 throughput (~213 tok/s
mean) AND restore multi-turn TTFT to ~150ms on 2.5K shared context.
This is the "have your cake and eat it" outcome.

Worst case: drafter accepts fewer K tokens on first turn after cache
hit → 5-10% per-turn TPS regression on first burst, then catches up.
Multi-turn workloads net win.

Status: opt-in via `GENESIS_ENABLE_P83=1`. Default OFF.

Tunable knobs
-------------
- `GENESIS_ENABLE_P83` (default unset/0): master switch

Compatibility
-------------
- MTP: TARGETED — this is the case P83 fixes
- Eagle / Eagle3: do NOT enable — they need the dropped hidden states
- DFlash: unknown; do not enable until tested
- ngram / suffix: pop() never fires anyway (`use_eagle()` returns False)

================================================================
UPSTREAM REBASE NOTE (2026-06-07, vllm-new @ 9c7f7741)
================================================================

Re-anchored for nightly. Upstream PR #44082 (e9e08c49b, 2026-06-02,
"[Bugfix] Cache the EAGLE/MTP lookahead block in the SWA prefix-cache
mask") **renamed** the `find_longest_cache_hit` parameter `use_eagle`
→ `drop_eagle_block` at BOTH pop sites. Anchors updated accordingly
(`if use_eagle and computed_blocks[0]:` → `if drop_eagle_block and
computed_blocks[0]:`). The pop bodies are otherwise byte-identical to
the pre-rebase source.

#44082 does NOT absorb vllm#38182. Its actual change was twofold:
  1. The hybrid coordinator now dedups the eagle drop across the
     cache-hit convergence loop via an `eagle_verified` set
     (`kv_cache_coordinator.py:625,643,662`, ref issue #32802) — the
     drop still fires once, just not repeatedly per candidate length.
  2. The SWA path was taught to *cache* (not evict) the lookahead block
     inside the sliding-window prefix-cache mask.
Neither path stops the FullAttention pop for `method='mtp'`:
`config/speculative.py:1070-1071` still returns True for "mtp", the
coordinator still sets `drop_eagle_block=True` for MTP groups, and the
FullAttentionManager still pops the last cached block. Verified
line-level against vllm-new. vllm#38182 remains OPEN upstream — P83 is
NOT obsolete and is still required to recover MTP prefix-cache hits.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment.
"""
from __future__ import annotations

import logging

# Audit A-19 (2026-05-05): tightly coupled subpatches — both apply
# or both stay un-applied. Shared marker is acceptable here because the
# subpatches together form one logical fix; partial application is not
# desired anyway. _AUDIT_A19_EXEMPT documents this intentional design.
_AUDIT_A19_EXEMPT = True  # tightly coupled subpatches
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p83_mtp_keep_last_cached_block")


GENESIS_P83_MARKER = "Genesis P83 MTP keep-last-cached-block (vllm#38182 mitigation) v7.53.1_debug"


# ─── Site 1: FullAttentionManager.find_longest_cache_hit ───────────────────
#
# REBASE 2026-07-13 (nightly-9e57de71, dev1060): upstream 481e481b (hybrid
# partial prefix-cache) rewrote FullAttentionManager.find_longest_cache_hit —
# the eagle drop is no longer a `computed.pop()` loop but a length decrement
# (`hit_length -= min(alignment_tokens, block_size)`; the block trim happens
# afterwards via `del computed[num_blocks:]`). vllm#38182 is still NOT
# absorbed: the drop still fires unconditionally for method='mtp'. Guard
# re-anchored to gate the decrement instead of the pop. NOT dev799-compatible
# (framework supports a single anchor form per sub-patch). Site 2
# (SlidingWindowManager) is byte-identical on dev1060 — anchor unchanged.

P83_SITE1_OLD = (
    "        # Eagle needs the tokens right before the generation point recomputed:\n"
    "        # drop one hash unit when fine-grained (the tail block's KV is\n"
    "        # append-only, so it still covers the reduced length), else one cache\n"
    "        # block.\n"
    "        if drop_eagle_block and hit_length > 0:\n"
    "            hit_length -= min(alignment_tokens, block_size)\n"
)

P83_SITE1_NEW = (
    "        # Eagle needs the tokens right before the generation point recomputed:\n"
    "        # drop one hash unit when fine-grained (the tail block's KV is\n"
    "        # append-only, so it still covers the reduced length), else one cache\n"
    "        # block.\n"
    "        if drop_eagle_block and hit_length > 0:\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            # [Genesis P83 vllm#38182 mitigation] Skip the eagle drop when\n"
    "            # GENESIS_ENABLE_P83=1. MTP drafter reads KV directly (not\n"
    "            # pre-materialised hidden states), so the drop is overly\n"
    "            # conservative for method='mtp'. Do NOT enable for true\n"
    "            # Eagle/Eagle3 — those drafters genuinely need the drop.\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            import os as _genesis_p83_os\n"
    "            _genesis_p83_skip = _genesis_p83_os.environ.get(\n"
    "                'GENESIS_ENABLE_P83', '').strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            if not _genesis_p83_skip:\n"
    "                hit_length -= min(alignment_tokens, block_size)\n"
)


# ─── Site 2: SlidingWindowManager.find_longest_cache_hit ───────────────────

P83_SITE2_OLD = (
    "        if drop_eagle_block and computed_blocks[0]:\n"
    "            for computed in computed_blocks:\n"
    "                computed.pop()\n"
    "            # Re-align after eagle pop: the pop may break the alignment\n"
    "            # when block_size != alignment_tokens (hybrid models with\n"
    "            # different page sizes, e.g. Gemma4).\n"
    "            while (\n"
    "                block_size != alignment_tokens\n"
    "                and len(computed_blocks[0]) * block_size % alignment_tokens != 0\n"
    "            ):\n"
    "                for computed in computed_blocks:\n"
    "                    computed.pop()\n"
)

P83_SITE2_NEW = (
    "        if drop_eagle_block and computed_blocks[0]:\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            # [Genesis P83 vllm#38182 mitigation] Skip pop() when GENESIS_ENABLE_P83=1\n"
    "            # See P83 wiring docstring. MTP-only safe.\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            import os as _genesis_p83_os\n"
    "            _genesis_p83_skip = _genesis_p83_os.environ.get(\n"
    "                'GENESIS_ENABLE_P83', '').strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            if not _genesis_p83_skip:\n"
    "                for computed in computed_blocks:\n"
    "                    computed.pop()\n"
    "                # Re-align after eagle pop: the pop may break the alignment\n"
    "                # when block_size != alignment_tokens (hybrid models with\n"
    "                # different page sizes, e.g. Gemma4).\n"
    "                while (\n"
    "                    block_size != alignment_tokens\n"
    "                    and len(computed_blocks[0]) * block_size % alignment_tokens != 0\n"
    "                ):\n"
    "                    for computed in computed_blocks:\n"
    "                        computed.pop()\n"
)


def _make_patcher_kv_cache_manager() -> TextPatcher | None:
    """Optional debug-only sub-patcher for kv_cache_manager.py — instruments
    get_computed_blocks() to log entry path + early-exit decision + block_hashes count."""
    target = resolve_vllm_file("v1/core/kv_cache_manager.py")
    if target is None:
        return None

    debug_anchor = "        # We skip finding the prefix cache hit when prefix caching is\n"
    debug_replacement = (
        "        # ════════════════════════════════════════════════════════════════\n"
        "        # [Genesis P83 DEBUG instrumentation] entry diagnostics\n"
        "        # ════════════════════════════════════════════════════════════════\n"
        "        import os as _g83_os; import sys as _g83_s\n"
        "        if _g83_os.environ.get('GENESIS_P83_DEBUG', '') == '1':\n"
        "            _g83_s.stderr.write(\n"
        "                '[GENESIS_P83_DEBUG_GCB] req=' + request.request_id[:8]\n"
        "                + ' coord=' + type(self.coordinator).__name__\n"
        "                + ' enable_caching=' + str(self.enable_caching)\n"
        "                + ' skip=' + str(request.skip_reading_prefix_cache)\n"
        "                + ' num_tokens=' + str(request.num_tokens)\n"
        "                + ' num_hashes=' + str(len(request.block_hashes))\n"
        "                + '\\n')\n"
        "            _g83_s.stderr.flush()\n"
        "        # We skip finding the prefix cache hit when prefix caching is\n"
    )

    # Also instrument the AFTER-call to log how many tokens hit
    after_anchor = (
        "        computed_blocks, num_new_computed_tokens = (\n"
        "            self.coordinator.find_longest_cache_hit(\n"
        "                request.block_hashes, max_cache_hit_length\n"
        "            )\n"
        "        )\n"
    )
    after_replacement = (
        "        computed_blocks, num_new_computed_tokens = (\n"
        "            self.coordinator.find_longest_cache_hit(\n"
        "                request.block_hashes, max_cache_hit_length\n"
        "            )\n"
        "        )\n"
        "        if _g83_os.environ.get('GENESIS_P83_DEBUG', '') == '1':\n"
        "            _g83_s.stderr.write(\n"
        "                '[GENESIS_P83_DEBUG_HITS] req=' + request.request_id[:8]\n"
        "                + ' max_len=' + str(max_cache_hit_length)\n"
        "                + ' num_hashes_in=' + str(len(request.block_hashes))\n"
        "                + ' hit_tokens=' + str(num_new_computed_tokens)\n"
        "                + '\\n')\n"
        "            _g83_s.stderr.flush()\n"
    )

    # Instrument cache_blocks() — store side. Critical: confirms whether
    # cache POPULATE happens at all and what num_tokens / enable_caching values flow in.
    cache_blocks_anchor = (
        "    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:\n"
        "        \"\"\"Cache the blocks for the request, if enabled.\n"
    )
    cache_blocks_replacement = (
        "    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:\n"
        "        import os as _g83cb_os; import sys as _g83cb_s\n"
        "        if _g83cb_os.environ.get('GENESIS_P83_DEBUG', '') == '1':\n"
        "            _g83cb_s.stderr.write(\n"
        "                '[GENESIS_P83_DEBUG_STORE] req=' + request.request_id[:8]\n"
        "                + ' enable_caching=' + str(self.enable_caching)\n"
        "                + ' num_computed_tokens=' + str(num_computed_tokens)\n"
        "                + ' num_hashes=' + str(len(request.block_hashes))\n"
        "                + '\\n')\n"
        "            _g83cb_s.stderr.flush()\n"
        "        \"\"\"Cache the blocks for the request, if enabled.\n"
    )

    return TextPatcher(
        patch_name="P83 DEBUG instrumentation kv_cache_manager.py",
        target_file=str(target),
        marker="Genesis P83 DEBUG instrumentation v7.53.6",
        sub_patches=[
            TextPatch(
                name="p83_debug_get_computed_blocks_entry",
                anchor=debug_anchor,
                replacement=debug_replacement,
                required=True,
            ),
            TextPatch(
                name="p83_debug_hit_count",
                anchor=after_anchor,
                replacement=after_replacement,
                required=False,
            ),
            TextPatch(
                name="p83_debug_cache_blocks_store",
                anchor=cache_blocks_anchor,
                replacement=cache_blocks_replacement,
                required=False,
            ),
        ],
        upstream_drift_markers=["[Genesis P83 DEBUG", "GENESIS_P83_DEBUG_GCB", "GENESIS_P83_DEBUG_HITS", "GENESIS_P83_DEBUG_STORE"],
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P83 v1/core/single_type_kv_cache_manager.py — MTP keep-last-cached-block "
            "(vllm#38182 mitigation)"
        ),
        target_file=str(target),
        marker=GENESIS_P83_MARKER,
        sub_patches=[
            TextPatch(
                name="p83_full_attention_skip_pop",
                anchor=P83_SITE1_OLD,
                replacement=P83_SITE1_NEW,
                required=True,
            ),
            TextPatch(
                name="p83_sliding_window_skip_pop",
                anchor=P83_SITE2_OLD,
                replacement=P83_SITE2_NEW,
                required=False,  # SlidingWindow is rarer; soft skip if anchor moved
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P83",
            "_genesis_p83_skip",
            "GENESIS_ENABLE_P83",
            # Upstream-side markers: if vLLM ever merges its own MTP-aware pop
            # guard for vllm#38182, the drop site will gain an MTP exclusion.
            # NOTE 2026-06-07: PR #44082 renamed use_eagle→drop_eagle_block but
            # did NOT add such a guard (verified: MTP still pops — see rebase
            # note in module docstring). These remain HYPOTHETICAL future forms;
            # all are confirmed absent from vllm-new so they cannot false-skip.
            "skip_eagle_pop_for_mtp",        # hypothetical upstream helper name
            "drop_eagle_block and not is_mtp",  # hypothetical guarded drop
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P83 — MTP keep-last-cached-block."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P83")
    log_decision("P83", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/single_type_kv_cache_manager.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P83] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P83" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed an MTP-aware pop guard",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    # Optional debug-only sub-patch on kv_cache_manager.py
    if os.environ.get("GENESIS_P83_DEBUG", "") == "1":
        debug_patcher = _make_patcher_kv_cache_manager()
        if debug_patcher is not None and os.path.isfile(debug_patcher.target_file):
            with open(debug_patcher.target_file) as f:
                debug_content = f.read()
            if debug_patcher.marker not in debug_content:
                debug_result, debug_failure = debug_patcher.apply()
                if debug_result == TextPatchResult.FAILED:
                    log.warning(
                        "[P83-debug] failed to inject get_computed_blocks "
                        "instrumentation: %s",
                        debug_failure.reason if debug_failure else "unknown",
                    )

    return "applied", (
        "P83 applied: MTP keep-last-cached-block guard installed at both "
        "FullAttentionManager and SlidingWindowManager pop sites. "
        "Activates ONLY when GENESIS_ENABLE_P83=1 is set in env. "
        "MTP-targeted; do NOT enable for true Eagle/Eagle3."
    )
