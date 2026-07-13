#!/usr/bin/env python3
"""PN95 — TQ continuation-prefill dequant buffers off the locked workspace
(vllm#43357).

Target: vllm/v1/attention/backends/turboquant_attn.py
Pin:    nightly-9e57de7197f234f9d9187715d96e07e007048c0f (dev1060)

UPSTREAM ISSUE (#43357, still OPEN): TQ continuation-prefill crashes the
engine on any prompt > max_num_batched_tokens —
    AssertionError: Workspace is locked but allocation from
    'turboquant_attn.py:...:_continuation_prefill' requires 12.00 MB,
    current size is 3.06 MB. Workspace growth is not allowed after locking.
Reported on dev169, where `_reserve_workspace` pre-reserved ONLY the decode
scratch (their 3.06 MB); the continuation-prefill dequant buffers were never
reserved, so the first post-lock chunked-prefill continuation hit the
WorkspaceManager growth assertion.

WHAT THE CODE ACTUALLY SIZES BY (important — the issue's proposed formula is
wrong): `_continuation_prefill` requests
    2 x (1, num_kv_heads, round_up(cached_len, block_size), head_dim) fp16
i.e. the demand scales with the CACHED PREFIX length (up to ~max_model_len),
NOT with the chunk size / max_num_batched_tokens. Any reservation derived
from the chunk size alone under-reserves and still crashes at long context.

STATE ON OUR PIN (dev1060): upstream added a continuation reservation to
`_reserve_workspace` sized for the worst case
    max_cached_len = max_model_len - 1  ->  ~2*max_model_len*Hk*D*2 bytes
(~300 MB at our 75K max-model-len, 4 KV heads, head_dim 256). That closes the
crash for OUR live flags, but:
  1. it permanently pins ~300 MB of VRAM for a rare path, allocated at
     builder-init AFTER the memory profiler ran (so it silently eats the
     gpu-memory-utilization headroom instead of being budgeted);
  2. the gate `enable_chunked_prefill and max_num_batched_tokens > 128`
     leaves #43357 fully reproducible when prefix caching produces
     continuation chunks with chunked prefill disabled (gate skips the
     reservation; `_continuation_prefill` is still reachable) — the exact
     locked-workspace assertion, just via a different scheduler config;
  3. under DBO (num_ubatches=2) the reservation lands on ubatch 0 only.

FIX: take the continuation-prefill dequant buffers out of the (lockable)
WorkspaceManager entirely and serve them from a dedicated module-level
grow-only buffer pair (same pattern as PN79's decode scratch):
  - sized from the ACTUAL per-call demand (round_up(cached_len, block)) and
    grown monotonically — first long request pays a one-time alloc;
  - STATIC UPPER BOUND (documented, fail-loud): alloc_len can never exceed
    round_up(max_model_len - 1, block_size); the bound is recorded at builder
    init from live config and enforced with a RuntimeError, so a logic bug
    upstream cannot silently grow the pool without limit;
  - the ~300 MB max_model_len workspace reservation is dropped (reclaimed as
    real headroom), and the locked-workspace assertion becomes unreachable
    from this path regardless of gate/scheduler config.

CUDAGRAPH SAFETY (verified in code, do NOT regress PN79):
  - TQ declares `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`;
    only uniform-decode-shaped batches are captured (q_len 1 or 1+K for MTP,
    i.e. <= 4 on our stack). `_continuation_prefill` is only reachable for
    q_len > _CONTINUATION_DECODE_THRESHOLD (128), so it can never execute
    inside a capture — a growing (address-moving) buffer here cannot be
    baked into any graph. The prefill path is replay-irrelevant.
  - PN79's invariant is untouched: the DECODE scratch reservation in
    `_reserve_workspace` and PN79's fixed decode scratch are not modified;
    captured decode graphs still never touch a moving allocation. PN95 only
    REMOVES a workspace grow trigger (the big prefill buffers no longer run
    through the workspace at all), which strengthens PN79's assumption.

Style: standalone idempotent commit-patch (see patch_pn80_*): exact-string
anchors with uniqueness checks, MARKER idempotence, self-retire on upstream
drift, fail-loud otherwise. Order-independent with PN79/PN86/tq_buffer_pool
(disjoint anchor regions; verified both orders against the extracted image
file). Retire when the pin advances past an upstream fix for #43357 that
removes the `current_workspace_manager().get_simultaneous` call from
`_continuation_prefill`.
"""
import pathlib
import sys

LOG = "[pn95-tq-prefill-workspace-sizing]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py"
)
MARKER = "# PN95:"

# --- 1. module-level dedicated grow-only dequant buffer pool ----------------
# Anchor: the _build_hadamard def — present verbatim in stock dev1060 and
# untouched by PN79 (which anchors on _CONTINUATION_DECODE_THRESHOLD above)
# and by PN86/tq_buffer_pool (which patch further down).
HELPERS_OLD = "\ndef _build_hadamard(d: int, device_str: str) -> torch.Tensor:\n"
HELPERS_NEW = (
    "\n"
    "# PN95: vllm#43357 — continuation-prefill K/V dequant buffers live in a\n"
    "# dedicated grow-only pool, NOT in the lockable WorkspaceManager. Demand\n"
    "# scales with the cached prefix length (round_up(cached_len, block)), so a\n"
    "# post-lock long continuation must never be a workspace growth request.\n"
    "# The prefill path is never cudagraph-captured (TQ cg support is\n"
    "# UNIFORM_BATCH; _continuation_prefill requires q_len > 128), so a moving\n"
    "# address on growth is safe. Static upper bound: alloc_len is capped at\n"
    "# round_up(max_model_len - 1, block_size), recorded at builder init\n"
    "# (_PN95_MAX_ALLOC_LEN) and enforced fail-loud below.\n"
    "_PN95_MAX_ALLOC_LEN: int = 0\n"
    "_PN95_PREFILL_DEQUANT_BUFS: dict = {}\n"
    "\n"
    "\n"
    "def _pn95_prefill_dequant_bufs(num_kv_heads, alloc_len, head_dim, device):\n"
    "    \"\"\"Grow-only (k_buf, v_buf) pair, each (1, Hk, >=alloc_len, D) fp16.\n"
    "\n"
    "    Shared across layers/requests (they execute sequentially, same as the\n"
    "    previous WorkspaceManager buffers). Never shrinks; never exceeds the\n"
    "    documented static bound.\"\"\"\n"
    "    if _PN95_MAX_ALLOC_LEN and alloc_len > _PN95_MAX_ALLOC_LEN:\n"
    "        raise RuntimeError(\n"
    "            f\"PN95: continuation-prefill dequant request alloc_len=\"\n"
    "            f\"{alloc_len} exceeds the static bound {_PN95_MAX_ALLOC_LEN} \"\n"
    "            f\"(= round_up(max_model_len - 1, block_size)); cached_len must \"\n"
    "            f\"be < max_model_len — upstream sizing logic changed?\"\n"
    "        )\n"
    "    key = (num_kv_heads, head_dim, str(device))\n"
    "    bufs = _PN95_PREFILL_DEQUANT_BUFS.get(key)\n"
    "    if bufs is None or bufs[0].shape[2] < alloc_len:\n"
    "        _PN95_PREFILL_DEQUANT_BUFS.pop(key, None)\n"
    "        del bufs\n"
    "        bufs = (\n"
    "            torch.empty(1, num_kv_heads, alloc_len, head_dim,\n"
    "                        dtype=torch.float16, device=device),\n"
    "            torch.empty(1, num_kv_heads, alloc_len, head_dim,\n"
    "                        dtype=torch.float16, device=device),\n"
    "        )\n"
    "        _PN95_PREFILL_DEQUANT_BUFS[key] = bufs\n"
    "    return bufs\n"
    "\n"
    "\n"
    "def _build_hadamard(d: int, device_str: str) -> torch.Tensor:\n"
)

# --- 2. _reserve_workspace: drop the ~max_model_len workspace reservation ---
# (record the pool's static bound instead; keep the decode reservation above
# this block untouched — PN79 depends on it as its non-captured fallback).
RESERVE_OLD = (
    "        reserve_continuation_prefill = (\n"
    "            scheduler_config.enable_chunked_prefill\n"
    "            and scheduler_config.max_num_batched_tokens > _CONTINUATION_DECODE_THRESHOLD\n"
    "        )\n"
    "        if not reserve_continuation_prefill:\n"
    "            return\n"
    "\n"
    "        max_cached_len = max(0, model_config.max_model_len - 1)\n"
    "        alloc_len = round_up(max_cached_len, self.kv_cache_spec.block_size)\n"
    "        cache_buf_shape = (1, num_kv_heads, alloc_len, head_size)\n"
    "        current_workspace_manager().get_simultaneous(\n"
    "            (cache_buf_shape, torch.float16),\n"
    "            (cache_buf_shape, torch.float16),\n"
    "        )\n"
)
RESERVE_NEW = (
    "        # PN95: vllm#43357 — the continuation-prefill dequant buffers no\n"
    "        # longer come from the (lockable) WorkspaceManager, so the worst-case\n"
    "        # max_model_len-sized reservation (~2*max_model_len*Hk*D*2B, ~300 MB\n"
    "        # at 75K ctx / 4 KV heads / D=256) is dropped and reclaimed. Record\n"
    "        # the pool's static upper bound from live config instead. Note: set\n"
    "        # unconditionally (no chunked-prefill/mnbt gate) — prefix-caching\n"
    "        # continuations reach _continuation_prefill even when chunked\n"
    "        # prefill is off, which is exactly the residual #43357 hole.\n"
    "        global _PN95_MAX_ALLOC_LEN\n"
    "        max_cached_len = max(0, model_config.max_model_len - 1)\n"
    "        _PN95_MAX_ALLOC_LEN = max(\n"
    "            _PN95_MAX_ALLOC_LEN,\n"
    "            round_up(max_cached_len, self.kv_cache_spec.block_size),\n"
    "        )\n"
)

# --- 3. _continuation_prefill: acquire from the pool, not the workspace -----
CONT_OLD = (
    "        # Use WorkspaceManager for dequant buffers.\n"
    "        # Shared across all layers — saves 60× memory at long context.\n"
    "        # Required for CUDA Graph capture (per-layer growth incompatible with CG).\n"
    "        k_buf, v_buf = current_workspace_manager().get_simultaneous(\n"
    "            (buf_shape, torch.float16),\n"
    "            (buf_shape, torch.float16),\n"
    "        )\n"
)
CONT_NEW = (
    "        # PN95: vllm#43357 — dedicated grow-only pool instead of the lockable\n"
    "        # WorkspaceManager (post-lock growth here was the engine-killing\n"
    "        # assertion). Shared across all layers, exactly like the previous\n"
    "        # workspace buffers. This path is never cudagraph-captured\n"
    "        # (q_len > _CONTINUATION_DECODE_THRESHOLD can't be a uniform-decode\n"
    "        # capture), so buffer growth/address moves are replay-safe.\n"
    "        k_buf, v_buf = _pn95_prefill_dequant_bufs(Hk, alloc_len, D, device)\n"
)

REPLACEMENTS = (
    ("dequant-pool helpers", HELPERS_OLD, HELPERS_NEW),
    ("_reserve_workspace continuation reservation", RESERVE_OLD, RESERVE_NEW),
    ("_continuation_prefill buffer acquisition", CONT_OLD, CONT_NEW),
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # Upstream-merged drift: retire once _continuation_prefill no longer draws
    # its dequant buffers from the WorkspaceManager.
    if "def _continuation_prefill" in text:
        body = text.split("def _continuation_prefill", 1)[1]
        body = body.split("\n    def ", 1)[0]
        if "current_workspace_manager().get_simultaneous" not in body:
            print(f"{LOG} upstream drift: _continuation_prefill no longer uses "
                  f"the WorkspaceManager — self-retire (no-op)")
            return 0
    for name, old, _new in REPLACEMENTS:
        if old not in text:
            print(f"{LOG} FATAL: anchor-not-found ({name}) — upstream refactor "
                  f"of turboquant_attn.py; re-derive before boot (any prompt "
                  f"longer than max_num_batched_tokens can crash the engine "
                  f"via the locked-workspace assertion, vllm#43357)",
                  file=sys.stderr)
            return 1
        if text.count(old) != 1:
            print(f"{LOG} FATAL: ambiguous anchor ({name})", file=sys.stderr)
            return 1
    for _name, old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: TQ continuation-prefill dequant buffers moved to a "
          f"dedicated grow-only pool (static bound = round_up(max_model_len-1, "
          f"block)); ~max_model_len-sized workspace reservation reclaimed "
          f"(vllm#43357)")
    return 0


sys.exit(main())
