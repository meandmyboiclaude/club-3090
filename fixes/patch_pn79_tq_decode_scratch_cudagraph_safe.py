"""PN79 — cudagraph-safe TurboQuant decode scratch (backport of vLLM PR #46067).

ROOT CAUSE (issue #45670 / our 2026-06-26 cudagraph crash, same device-side-assert
class as BUG-028): `TurboQuantImpl._decode_attention` draws its three decode scratch
buffers (mid_o_buf / output_buf / lse_buf) from the *growable* shared WorkspaceManager
(`current_workspace_manager().get_simultaneous(...)`). The TQ attention backend declares
`_cudagraph_support = UNIFORM_BATCH`, so uniform-decode batches (including MTP K+1,
q_len=4) ARE cudagraph-captured — even under cudagraph_mode=PIECEWISE — which BAKES the
scratch buffer addresses into the captured graphs. The WorkspaceManager FREES + REALLOCS
its buffer (calling empty_cache, which unmaps the old address) whenever any TQ buffer
must grow — e.g. a big continuation-prefill / a larger capture size. Once it moves, every
captured decode graph still points at the freed address -> use-after-free on the next
replay -> `CUDA error: device-side assert triggered`, surfaced asynchronously at the next
sync (so the traceback shows prepare_inputs/replay, not the attention kernel). On our
stack `_reserve_workspace` only pre-reserves for max_num_seqs(=5) while captures run up to
max_cudagraph_capture_size(=40), so the grow is guaranteed.

FIX (mirrors #46067): give TQ decode a DEDICATED, fixed-size scratch allocated once at the
max cudagraph capture batch and cached module-level. It never goes through WorkspaceManager
and never resizes, so its address is stable for the lifetime of the captured graphs. Each
call slices [:B]. Eager batches (cudagraphs off) or batches beyond the captured max fall
back to the original WorkspaceManager path (no graph to dangle). Numerically identical —
same shapes/dtypes, just a non-moving allocation.

Target: vllm/v1/attention/backends/turboquant_attn.py.
Style: standalone idempotent commit-patch; runs in the compose entrypoint after apply_all.
Never exits non-zero (set -e safe); graceful no-op if an anchor is gone (vLLM bumped).
Intended to be retired once #46067 (or equivalent) merges upstream.
"""
import sys
import pathlib

LOG = "[pn79-tq-decode-scratch]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py"
)
MARKER = "# PN79:"

# --- 1. module-level fixed-scratch helpers (after _CONTINUATION_DECODE_THRESHOLD) ---
HELPERS_OLD = "_CONTINUATION_DECODE_THRESHOLD = 128\n"
HELPERS_NEW = (
    "_CONTINUATION_DECODE_THRESHOLD = 128\n"
    "\n"
    "# PN79: vllm#46067 backport — fixed-size, never-moving TurboQuant decode scratch.\n"
    "# The shared WorkspaceManager scratch is freed+realloc'd (empty_cache) when any TQ\n"
    "# buffer grows; addresses baked into captured uniform-decode graphs then dangle ->\n"
    "# device-side assert under load (#45670). A dedicated scratch sized to the max\n"
    "# cudagraph capture batch never moves, so captured pointers stay valid.\n"
    "_PN79_DECODE_SCRATCH: dict = {}\n"
    "\n"
    "\n"
    "@functools.lru_cache(maxsize=1)\n"
    "def _pn79_max_decode_cudagraph_batch() -> int:\n"
    "    try:\n"
    "        v = get_current_vllm_config().compilation_config.max_cudagraph_capture_size\n"
    "        return int(v) if v else 0\n"
    "    except Exception:\n"
    "        return 0\n"
    "\n"
    "\n"
    "def _pn79_get_decode_scratch(max_batch, num_heads, num_kv_splits, head_size, dtype, device):\n"
    "    \"\"\"Fixed (mid_o, output, lse) scratch sized for the largest captured batch.\n"
    "    Shared across all TQ layers and captured graphs; addresses never move.\"\"\"\n"
    "    key = (max_batch, num_heads, num_kv_splits, head_size, dtype, device)\n"
    "    bufs = _PN79_DECODE_SCRATCH.get(key)\n"
    "    if bufs is None:\n"
    "        bufs = (\n"
    "            torch.empty(max_batch, num_heads, num_kv_splits, head_size + 1,\n"
    "                        dtype=torch.float32, device=device),\n"
    "            torch.empty(max_batch, num_heads, head_size, dtype=dtype, device=device),\n"
    "            torch.empty(max_batch, num_heads, dtype=torch.float32, device=device),\n"
    "        )\n"
    "        _PN79_DECODE_SCRATCH[key] = bufs\n"
    "    return bufs\n"
)

# --- 2. _decode_attention: use the fixed scratch when this batch is cudagraph-captured ---
DECODE_OLD = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        if is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            mid_o_buf, output_buf, lse_buf = (\n"
    "                current_workspace_manager().get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "            )\n"
)
DECODE_NEW = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        # PN79 (vllm#46067): if this decode is cudagraph-captured (TQ declares\n"
    "        # UNIFORM_BATCH cg support), the scratch address is baked into the graph; the\n"
    "        # growable WorkspaceManager would move it on the next grow -> dangling ptr ->\n"
    "        # device-side assert. Use a fixed scratch (never moves), sliced [:B].\n"
    "        _pn79_max_b = _pn79_max_decode_cudagraph_batch()\n"
    "        if _pn79_max_b and _pn79_max_b >= B:\n"
    "            _pn79_mid, _pn79_out, _pn79_lse = _pn79_get_decode_scratch(\n"
    "                _pn79_max_b, Hq, S, D, query.dtype, query.device\n"
    "            )\n"
    "            mid_o_buf, output_buf, lse_buf = _pn79_mid[:B], _pn79_out[:B], _pn79_lse[:B]\n"
    "        elif is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            mid_o_buf, output_buf, lse_buf = (\n"
    "                current_workspace_manager().get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "            )\n"
)

REPLACEMENTS = [
    ("decode-scratch helpers", HELPERS_OLD, HELPERS_NEW),
    ("_decode_attention fixed scratch", DECODE_OLD, DECODE_NEW),
]


def main():
    # [2026-07-23 ultra-review #15] this is a CRASH-CLASS protection (compose
    # marks it Required for cudagraph stability on TQ3+MTP; PN353B was dropped
    # BECAUSE pn79 covers it). Anchor drift must be LOUD like pn86/pn95 —
    # a silent skip boots green and dies hours later with a device-side assert.
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present.", file=sys.stderr)
        sys.exit(1)
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied")
        return
    for name, old, _new in REPLACEMENTS:
        if old not in text:
            print(
                f"{LOG} FATAL: anchor '{name}' not found in {TARGET.name} — TQ backend "
                f"shape changed (vLLM bumped?); re-anchor before booting: this patch "
                f"guards a cudagraph crash class.",
                file=sys.stderr,
            )
            sys.exit(1)
    for _name, old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: TQ decode scratch now fixed-size/cudagraph-safe (#46067)")


main()
