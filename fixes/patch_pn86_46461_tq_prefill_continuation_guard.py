#!/usr/bin/env python3
"""PN86 — TurboQuant prefill fast-path continuation guard.

Backport of vllm#46461 (merged upstream AFTER the dev1060 pin 9e57de71) to
vllm/v1/attention/backends/turboquant_attn.py::_prefill_attention (+28/-3).

Bug: the flash-attn fast path gates only on
    max_query_len == max_seq_len
Both are batch-level maxima, so a MIXED batch — one long first-chunk prefill
(q_len == seq_len, and it owns both maxima) alongside shorter CONTINUATION
requests (q_len < seq_len, prefix-cache hit) — still satisfies the gate. The
fast path then calls flash_attn_varlen with cu_seqlens_k=query_start_loc,
i.e. K/V = current-chunk raw K/V only, silently DROPPING the cached prefix
K/V of every continuation request. Under prefix caching this corrupts those
requests' attention output (wrong logits, no crash). Directly exercised by
our stack: TQ3 KV + prefix caching on :8020.

Fix (upstream-faithful): compute a vectorized has-continuation flag from the
CPU-resident query_start_loc_cpu/seq_lens_cpu (no GPU sync) and require
`not _has_continuation` on the fast path. Continuations then take the
existing per-request path that reads cached K/V from the TQ cache.

Retire when the pin advances past vllm#46461 (upstream merge): the patcher
self-retires when `_has_continuation` already appears in
_prefill_attention.

Anchor drift vs PR: none — the pin (dev1060) matches the PR base text
exactly in this hunk (verified against the extracted image file).
"""
import pathlib
import sys

LOG = "[pn86-tq-prefill-continuation-guard]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py"
)
MARKER = "# PN86:"

OLD = (
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # max_query_len == max_seq_len means no request has prior cached KV.\n"
    "        # Both are Python ints — no GPU sync.\n"
    "        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:\n"
    "            return self._flash_attn_varlen(\n"
    "                q=query,\n"
    "                k=key,\n"
    "                v=value,\n"
    "                cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "                cu_seqlens_k=attn_metadata.query_start_loc,\n"
    "                max_seqlen_q=attn_metadata.max_query_len,\n"
    "                max_seqlen_k=attn_metadata.max_query_len,\n"
    "            )\n"
)

NEW = (
    "        # PN86: vllm#46461 backport — guard the fast path: it is only valid\n"
    "        # when NO continuation requests exist in the batch.  max_query_len ==\n"
    "        # max_seq_len alone is NOT sufficient — a long first-chunk prefill\n"
    "        # (q_len == seq_len) can coexist with shorter continuation requests\n"
    "        # (q_len < seq_len, prefix cache hit), causing flash_attn_varlen with\n"
    "        # cu_seqlens_k=query_start_loc to LOSE the cached prefix K/V for\n"
    "        # continuations.\n"
    "        #\n"
    "        # Vectorized check on CPU tensors: diff() computes all query lengths,\n"
    "        # ne() + any() short-circuits on the first mismatch.  .item() on a\n"
    "        # CPU scalar tensor is a host read — no GPU sync.\n"
    "        _has_continuation = False\n"
    "        if (\n"
    "            attn_metadata.query_start_loc_cpu is not None\n"
    "            and attn_metadata.seq_lens_cpu is not None\n"
    "        ):\n"
    "            _qsl = attn_metadata.query_start_loc_cpu\n"
    "            _sl = attn_metadata.seq_lens_cpu\n"
    "            _has_continuation = (\n"
    "                ((_qsl[1 : len(_sl) + 1] - _qsl[: len(_sl)]) != _sl).any().item()\n"
    "            )\n"
    "\n"
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # Guarded: only fires when NO continuation requests exist in the batch.\n"
    "        # Both max values are Python ints — no GPU sync.\n"
    "        if (\n"
    "            _HAS_FLASH_ATTN\n"
    "            and attn_metadata.max_query_len == attn_metadata.max_seq_len\n"
    "            and not _has_continuation\n"
    "        ):\n"
    "            return self._flash_attn_varlen(\n"
    "                q=query,\n"
    "                k=key,\n"
    "                v=value,\n"
    "                cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "                cu_seqlens_k=attn_metadata.query_start_loc,\n"
    "                max_seqlen_q=attn_metadata.max_query_len,\n"
    "                max_seqlen_k=attn_metadata.max_query_len,\n"
    "            )\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # Upstream-merged drift: the continuation guard already exists.
    if "def _prefill_attention" in text:
        body = text.split("def _prefill_attention", 1)[1][:8000]
        if "_has_continuation" in body:
            print(f"{LOG} upstream drift: continuation guard already present "
                  f"— self-retire (no-op)")
            return 0
    if OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found (prefill fast path) — upstream "
              f"refactor of _prefill_attention; re-derive before boot (mixed "
              f"first-chunk+continuation batches SILENTLY corrupt output "
              f"under prefix caching without this fix)", file=sys.stderr)
        return 1
    if text.count(OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor (prefill fast path)", file=sys.stderr)
        return 1
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"{LOG} applied: TQ prefill flash-attn fast path now requires a "
          f"continuation-free batch (vllm#46461 backport)")
    return 0


sys.exit(main())
