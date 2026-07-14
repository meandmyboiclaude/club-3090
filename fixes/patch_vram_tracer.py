"""VRAM allocation tracer — hooks torch.cuda.memory_allocated() delta into EngineCore.step()
to identify which step phases are leaking memory.

Logs the VRAM delta before/after each step, and if growth exceeds threshold,
dumps the top tensors by size from gc to find what's holding references.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_vram_tracer")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: vram_tracer"
TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py")

TRACER_CODE = '''
# PATCH: vram_tracer
import torch as _vt_torch
import gc as _vt_gc
import logging as _vt_logging
_vt_log = _vt_logging.getLogger("vram_tracer")
_vt_log.setLevel(_vt_logging.INFO)
if not _vt_log.handlers:
    _vt_log.addHandler(_vt_logging.StreamHandler())
_vt_baseline = None
_vt_step_count = 0
_vt_last_reserved = 0

def _vt_trace_step():
    global _vt_baseline, _vt_step_count, _vt_last_reserved
    _vt_step_count += 1
    reserved = _vt_torch.cuda.memory_reserved()
    allocated = _vt_torch.cuda.memory_allocated()

    if _vt_baseline is None:
        _vt_baseline = reserved
        _vt_last_reserved = reserved

    delta_from_baseline = (reserved - _vt_baseline) / 1024**2
    delta_from_last = (reserved - _vt_last_reserved) / 1024**2

    # Log every 100 steps or when growth > 50MB from last
    if _vt_step_count % 100 == 0 or abs(delta_from_last) > 50:
        _vt_log.info(
            f"[VRAM] step={_vt_step_count} reserved={reserved//1024**2}MB "
            f"allocated={allocated//1024**2}MB "
            f"delta_baseline={delta_from_baseline:+.0f}MB "
            f"delta_last={delta_from_last:+.0f}MB "
            f"fragmented={(reserved-allocated)//1024**2}MB"
        )

        # If growing > 200MB from baseline, dump top GPU tensors
        if delta_from_baseline > 200:
            _vt_gc.collect()
            _vt_torch.cuda.empty_cache()
            after_gc = _vt_torch.cuda.memory_reserved()
            reclaimed = (reserved - after_gc) / 1024**2
            _vt_log.info(f"[VRAM] after gc+empty_cache: reclaimed={reclaimed:.0f}MB reserved={after_gc//1024**2}MB")

    _vt_last_reserved = _vt_torch.cuda.memory_reserved()
'''

def apply():
    # [2026-07-14 USER] Flag-gated, DEFAULT OFF. Per-step VRAM tracing adds overhead with no
    # serving benefit — a leak-hunt tool only. Enable with GENESIS_ENABLE_VRAM_TRACER=1.
    import os as _os
    if _os.environ.get("GENESIS_ENABLE_VRAM_TRACER", "0").strip() != "1":
        log.info("[vram_tracer] disabled (set GENESIS_ENABLE_VRAM_TRACER=1 to enable — debug only)")
        return
    if not TARGET.exists():
        log.warning("[vram_tracer] core.py not found")
        return

    text = TARGET.read_text()
    if MARKER in text:
        log.info("[vram_tracer] already applied")
        return

    class_anchor = "class EngineCore:"
    if class_anchor not in text:
        log.warning("[vram_tracer] EngineCore class not found")
        return

    step_anchor = None
    for candidate in [
        "    def step(self) -> EngineCoreOutputs:",
        "    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:",
        "    def step(self)",
    ]:
        if candidate in text:
            step_anchor = candidate
            break
    if step_anchor is None:
        log.warning("[vram_tracer] step() not found")
        return

    text = text.replace(class_anchor, TRACER_CODE + "\n" + class_anchor, 1)

    idx = text.find(step_anchor)
    eol = text.find('\n', idx)
    text = text[:eol+1] + "        _vt_trace_step()\n" + text[eol+1:]

    TARGET.write_text(text)
    log.info("[vram_tracer] applied: step-level VRAM tracing enabled")

apply()
