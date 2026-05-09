#!/usr/bin/env python3
"""Skip Qwen3_5MTP drafter's transient lm_head allocation.

eagle.py:_maybe_share_lm_head unconditionally replaces drafter.lm_head
with the target model's lm_head for MTP drafters. Allocating then freeing
creates a 2.37 GiB transient peak that OOMs Genesis at util>=0.95 on 24 GB.
This patch swaps the allocation for a PPMissingLayer placeholder.
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
