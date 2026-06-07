"""Pool TurboQuant decode scratch buffers to prevent VRAM fragmentation.

Replaces per-step torch.zeros/torch.empty calls in turboquant_attn.py with
reusable pre-allocated buffers. This is the manual equivalent of Genesis P36
(drifted on v0.21) — targets the same allocation sites.

Hot allocations that fragment VRAM under concurrent MTP decode:
- forward() output buffer: torch.zeros(num_tokens, H*D) every call
- mixed-batch attn_out: torch.empty(N, H, D) every mixed decode
- prefill output: torch.zeros(N, Hq, D) every prefill

Strategy: add a module-level buffer cache keyed by (shape, dtype, device).
Buffers are reused if shape matches; reallocated only when shape grows.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_tq_buffer_pool")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: tq_buffer_pool"
TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py")

POOL_CODE = '''
# PATCH: tq_buffer_pool — reuse decode scratch buffers to prevent VRAM fragmentation
import torch as _bp_torch

class _TQBufferPool:
    """Reusable GPU buffer pool for TQ attention. Buffers grow but never shrink."""
    def __init__(self):
        self._buffers = {}

    def get_zeros(self, key, shape, dtype, device):
        buf = self._buffers.get(key)
        numel = 1
        for s in shape:
            numel *= s
        if buf is not None and buf.dtype == dtype and buf.device == device and buf.numel() >= numel:
            out = buf[:numel].view(shape)
            out.zero_()
            return out
        new_buf = _bp_torch.zeros(numel, dtype=dtype, device=device)
        self._buffers[key] = new_buf
        return new_buf.view(shape)

    def get_empty(self, key, shape, dtype, device):
        buf = self._buffers.get(key)
        numel = 1
        for s in shape:
            numel *= s
        if buf is not None and buf.dtype == dtype and buf.device == device and buf.numel() >= numel:
            return buf[:numel].view(shape)
        new_buf = _bp_torch.empty(numel, dtype=dtype, device=device)
        self._buffers[key] = new_buf
        return new_buf.view(shape)

_tq_pool = _TQBufferPool()
'''

# Replacement patterns: replace torch.zeros/empty with pool calls
REPLACEMENTS = [
    # forward() output buffer (line ~424)
    {
        "old": "            output = torch.zeros(\n                num_tokens,\n                self.num_heads * self.head_size,\n                dtype=query.dtype,\n                device=query.device,\n            )",
        "new": "            output = _tq_pool.get_zeros('fwd_out', (num_tokens, self.num_heads * self.head_size), query.dtype, query.device)",
    },
    # Mixed-batch attn_out (line ~671)
    {
        "old": "            attn_out = torch.empty(\n                N, self.num_heads, self.head_size, device=device, dtype=q.dtype\n            )",
        "new": "            attn_out = _tq_pool.get_empty('mixed_out', (N, self.num_heads, self.head_size), q.dtype, device)",
    },
    # Prefill output (line ~981)
    {
        "old": "        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)",
        "new": "        output = _tq_pool.get_zeros('prefill_out', (N, Hq, D), query.dtype, query.device)",
    },
]


def apply():
    if not TARGET.exists():
        log.warning("[tq_buffer_pool] turboquant_attn.py not found")
        return

    text = TARGET.read_text()
    if MARKER in text:
        log.info("[tq_buffer_pool] already applied")
        return

    # Inject pool class before the first class definition
    class_markers = ["class TurboQuantAttentionBackend", "class TurboQuant"]
    inject_point = -1
    for cm in class_markers:
        idx = text.find(cm)
        if idx >= 0:
            inject_point = idx
            break

    if inject_point < 0:
        log.warning("[tq_buffer_pool] no TurboQuant class found")
        return

    text = text[:inject_point] + POOL_CODE + "\n" + text[inject_point:]

    applied = 0
    for r in REPLACEMENTS:
        if r["old"] in text:
            text = text.replace(r["old"], r["new"], 1)
            applied += 1

    if applied == 0:
        log.warning("[tq_buffer_pool] no allocation sites matched — anchors may have drifted")
        TARGET.write_text(text)  # still write pool code for future use
        return

    TARGET.write_text(text)
    log.info(f"[tq_buffer_pool] applied: {applied}/{len(REPLACEMENTS)} allocation sites pooled")


apply()
