#!/usr/bin/env python3
"""Skip Qwen3_5MTP drafter's transient lm_head allocation.

_maybe_share_lm_head unconditionally replaces drafter.lm_head with the
target model's lm_head for MTP drafters. Allocating then freeing creates a
2.37 GiB transient peak that OOMs Genesis at util>=0.95 on 24 GB.
This patch swaps the allocation for a PPMissingLayer placeholder.

REBASE 2026-06-07 (vllm-new 9c7f7741): two upstream drifts handled:
  1. _maybe_share_lm_head moved out of v1/spec_decode/eagle.py into
     v1/spec_decode/llm_base_proposer.py (EAGLE refactor #44078/#44338).
     The MTP branch is unchanged: for MTP drafters share_lm_head is always
     True and the proposer does `del self.model.lm_head; self.model.lm_head
     = target.lm_head` (llm_base_proposer.py:1389-1400) plus rebinds each
     layer's shared_head.head (1411-1418). self.model there IS the Qwen3_5MTP
     module, so its self.lm_head (our patch target) is overwritten regardless.
     Replacing the initial ParallelLMHead with PPMissingLayer() is still safe.
  2. The ParallelLMHead(...) call now passes quant_config=self.quant_config;
     OLD anchor updated to match (qwen3_5_mtp.py:381-386).
"""
import sys
from pathlib import Path

TARGET_PATHS = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py",
    "/home/user/engines/vllm/vllm/model_executor/models/qwen3_5_mtp.py",
]

OLD = '''            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )'''

NEW = '''            else:
                # NOEDEL-OVERNIGHT: skip transient ~2.37 GiB ParallelLMHead alloc.
                # eagle.py:_maybe_share_lm_head unconditionally replaces it via
                # MTP weight-sharing with the target model. Original alloc would
                # OOM Genesis at util>=0.95 on 24 GB cards. Use placeholder.
                self.lm_head = PPMissingLayer()'''

MARKER = "# NOEDEL-OVERNIGHT: skip transient"

def patch(path):
    p = Path(path)
    if not p.exists():
        return f"missing: {path}"
    text = p.read_text()
    if MARKER in text:
        return f"already-applied: {path}"
    if OLD not in text:
        return f"anchor-not-found: {path}"
    p.write_text(text.replace(OLD, NEW, 1))
    return f"patched: {path}"

found = 0
for path in TARGET_PATHS:
    result = patch(path)
    print(f"[drafter_skip_fix] {result}", flush=True)
    if result.startswith(("patched", "already-applied")):
        found += 1

if found == 0:
    print("[drafter_skip_fix] WARNING: no targets patched", file=sys.stderr)
    sys.exit(0)  # non-fatal
