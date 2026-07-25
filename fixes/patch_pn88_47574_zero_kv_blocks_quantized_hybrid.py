#!/usr/bin/env python3
"""PN88 — zero newly allocated KV blocks for quantized + hybrid caches.

Backport of vllm#47574 (merged upstream AFTER the dev1060 pin 9e57de71) to
vllm/v1/kv_cache_interface.py::KVCacheConfig (+33/-1 upstream).

Bug: KVCacheConfig.needs_kv_cache_zeroing only returns True for mamba layers.
In hybrid configs that mix a block-dropping attention type (sliding-window /
chunked-local) with a QUANTIZED KV cache, all groups share the same tensors:
a page freed mid-request by a block-dropping group can be reallocated and
read before every slot is written in that step (partial-block tail,
alignment padding). Stale quantized bytes decode to NaN/Inf and poison
attention. Zeroing new blocks makes such slots read as 0; uniform or
unquantized configs are unaffected (property stays False for them, so no
extra memset cost).

Upstream adds two helper properties (has_block_dropping_layers,
has_quantized_kv_cache) and widens needs_kv_cache_zeroing.

PN88 adaptation vs the PR: TQFullAttentionSpec encodes TurboQuant in the
spec subclass and keeps kv_quant_mode == KVQuantMode.NONE (the same trap
PN80 documents for the runner dtype-select), so upstream's
`kv_quant_mode != NONE` test would miss TQ layers. has_quantized_kv_cache
counts TQFullAttentionSpec explicitly. No behavior change for our
tools-text deployment (full-attention-only → has_block_dropping_layers is
False); the TQ clause only matters for TQ + SWA/chunked-local hybrids.

Anchor drift vs PR: none otherwise — the property pair at the tail of
KVCacheConfig matches the PR base exactly (verified against the extracted
image file). All referenced spec classes are defined earlier in the same
module; no new imports needed.

Retire when the pin advances past vllm#47574: self-retires when
has_quantized_kv_cache / has_block_dropping_layers already exist upstream.
"""
import pathlib
import sys

LOG = "[pn88-zero-kv-blocks-quantized-hybrid]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py"
)
MARKER = "# PN88:"

OLD = (
    "    @property\n"
    "    def has_mamba_layers(self) -> bool:\n"
    "        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)\n"
    "\n"
    "    @property\n"
    "    def needs_kv_cache_zeroing(self) -> bool:\n"
    "        return self.has_mamba_layers\n"
)

NEW = (
    "    @property\n"
    "    def has_mamba_layers(self) -> bool:\n"
    "        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)\n"
    "\n"
    "    # PN88: vllm#47574 backport — zero new KV blocks for quantized + hybrid caches.\n"
    "    @property\n"
    "    def has_block_dropping_layers(self) -> bool:\n"
    "        \"\"\"Any group uses an attention type that frees KV blocks mid-request\n"
    "        (sliding-window or chunked-local).\"\"\"\n"
    "        return any(\n"
    "            isinstance(g.kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec))\n"
    "            for g in self.kv_cache_groups\n"
    "        )\n"
    "\n"
    "    @property\n"
    "    def has_quantized_kv_cache(self) -> bool:\n"
    "        \"\"\"Any group stores its KV cache in a quantized dtype (FP8/NVFP4/TQ).\n"
    "\n"
    "        PN88 adaptation: TQFullAttentionSpec encodes TurboQuant in the spec\n"
    "        subclass and keeps kv_quant_mode == NONE, so it is counted explicitly.\n"
    "        \"\"\"\n"
    "        return any(\n"
    "            (\n"
    "                isinstance(g.kv_cache_spec, AttentionSpec)\n"
    "                and g.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE\n"
    "            )\n"
    "            or isinstance(g.kv_cache_spec, TQFullAttentionSpec)\n"
    "            for g in self.kv_cache_groups\n"
    "        )\n"
    "\n"
    "    @property\n"
    "    def needs_kv_cache_zeroing(self) -> bool:\n"
    "        \"\"\"Whether newly allocated KV cache blocks must be zeroed before use.\n"
    "\n"
    "        Required in two cases:\n"
    "        - Mamba layers, whose state is read before being fully written.\n"
    "        - Hybrid configs mixing a block-dropping attention type with a quantized\n"
    "          KV cache. Groups share the same tensors, so a page freed mid-request by\n"
    "          a block-dropping group can be reallocated and read before every slot is\n"
    "          written this step (partial-block tail, alignment padding). Stale\n"
    "          quantized bytes decode to NaN/Inf and poison attention; zeroing makes\n"
    "          such slots read as 0. Uniform/unquantized configs are unaffected.\n"
    "        \"\"\"\n"
    "        needs_quantized_reuse_zeroing = (\n"
    "            self.has_block_dropping_layers and self.has_quantized_kv_cache\n"
    "        )\n"
    "        return self.has_mamba_layers or needs_quantized_reuse_zeroing\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # Upstream-merged drift: helper properties already exist.
    # [2026-07-25] vllm#47574 merged upstream (8ce53a616e) with a DIFFERENT
    # formulation than the PR head this backport mirrors: a single
    # `has_mixed_precision_kv_cache` property now feeds needs_kv_cache_zeroing
    # (covers the quantized-hybrid stale-bytes case this patch targets).
    if (
        "has_quantized_kv_cache" in text
        or "has_block_dropping_layers" in text
        or "has_mixed_precision_kv_cache" in text
    ):
        print(f"{LOG} upstream drift: quantized/hybrid zeroing properties "
              f"already present — self-retire (no-op)")
        return 0
    if OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found (KVCacheConfig zeroing "
              f"properties) — upstream refactor; re-derive (quantized+hybrid "
              f"configs can read stale quantized bytes as NaN/Inf without "
              f"this fix)", file=sys.stderr)
        return 1
    if text.count(OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor", file=sys.stderr)
        return 1
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"{LOG} applied: needs_kv_cache_zeroing now covers block-dropping + "
          f"quantized (incl. TQ) hybrid caches (vllm#47574 backport)")
    return 0


sys.exit(main())
