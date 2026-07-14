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
            # [2026-07-14] attribute the live growth: dump the Genesis
            # prealloc registry (namespace -> bytes) so pool-sizing bugs
            # (BUG-072 class) are visible without a bisect.
            try:
                from sndr.runtime.prealloc import GenesisPreallocBuffer as _vt_GPB
                _vt_info = _vt_GPB.get_registry_info()
                _vt_log.info(f"[VRAM] GPB total={_vt_info['total_human']} buffers={_vt_info['total_buffers']}")
                for _vt_e in sorted(_vt_info["entries"], key=lambda x: -x["bytes"])[:10]:
                    if _vt_e["bytes"] > 16 * 1024 * 1024:
                        _vt_log.info(f"[VRAM]   GPB {_vt_e['size_human']:>10}  {_vt_e['namespace']}  {_vt_e['shape']}")
            except Exception as _vt_ex:
                _vt_log.info(f"[VRAM] GPB dump failed: {_vt_ex}")
            # [2026-07-14] the promised gc tensor dump — top CUDA tensors by
            # size; shapes identify the holder (KV pool / weights excluded by
            # the >64MB-and-not-seen-at-baseline filter being impractical, so
            # just print top-15 and diff by eye across dumps).
            try:
                _vt_seen = {}
                for _vt_o in _vt_gc.get_objects():
                    try:
                        if _vt_torch.is_tensor(_vt_o) and _vt_o.is_cuda:
                            _vt_nb = _vt_o.element_size() * _vt_o.numel()
                            if _vt_nb > 64 * 1024 * 1024:
                                _vt_k = (tuple(_vt_o.shape), str(_vt_o.dtype))
                                _vt_c, _vt_b = _vt_seen.get(_vt_k, (0, 0))
                                _vt_seen[_vt_k] = (_vt_c + 1, _vt_b + _vt_nb)
                    except Exception:
                        pass
                for _vt_k, (_vt_c, _vt_b) in sorted(_vt_seen.items(), key=lambda x: -x[1][1])[:15]:
                    _vt_log.info(f"[VRAM]   TENSOR {_vt_b//1024**2:>6}MB x{_vt_c}  shape={list(_vt_k[0])} {_vt_k[1]}")
                # [2026-07-14] name the OWNER of the biggest repeated tensor:
                # walk referrers of one instance (dict -> owning object attr).
                if _vt_seen:
                    _vt_top = max(_vt_seen.items(), key=lambda x: x[1][1])[0]
                    for _vt_o in _vt_gc.get_objects():
                        try:
                            if (_vt_torch.is_tensor(_vt_o) and _vt_o.is_cuda
                                    and tuple(_vt_o.shape) == _vt_top[0]
                                    and str(_vt_o.dtype) == _vt_top[1]):
                                for _vt_r in _vt_gc.get_referrers(_vt_o):
                                    if isinstance(_vt_r, dict):
                                        _vt_keys = [k for k, v in _vt_r.items() if v is _vt_o]
                                        for _vt_r2 in _vt_gc.get_referrers(_vt_r):
                                            _vt_log.info(f"[VRAM]   OWNER keys={_vt_keys} holder={type(_vt_r2).__name__} {getattr(_vt_r2, '__qualname__', '')}"[:220])
                                    elif isinstance(_vt_r, (list, tuple)):
                                        for _vt_r2 in _vt_gc.get_referrers(_vt_r):
                                            _vt_log.info(f"[VRAM]   OWNER via {type(_vt_r).__name__} holder={type(_vt_r2).__name__}"[:220])
                                break
                        except Exception:
                            pass
            except Exception as _vt_ex:
                _vt_log.info(f"[VRAM] tensor dump failed: {_vt_ex}")

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

    # [2026-07-14] hook BOTH step() and step_with_batch_queue() — the endgame
    # engine (async scheduling) runs step_with_batch_queue; hooking only
    # step() produced zero output on the exact config being hunted.
    step_anchors = []
    for candidate in [
        "    def step(self) -> EngineCoreOutputs:",
        "    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:",
        "    def step(self)",
    ]:
        if candidate in text:
            step_anchors.append(candidate)
            break
    for candidate in [
        "    def step_with_batch_queue(",
    ]:
        if candidate in text:
            step_anchors.append(candidate)
    if not step_anchors:
        log.warning("[vram_tracer] no step function found")
        return

    text = text.replace(class_anchor, TRACER_CODE + "\n" + class_anchor, 1)

    for step_anchor in step_anchors:
        idx = text.find(step_anchor)
        # skip past the (possibly multi-line) signature to the colon line end
        sig_end = text.find(':', idx)
        eol = text.find('\n', sig_end)
        text = text[:eol+1] + "        _vt_trace_step()\n" + text[eol+1:]

    TARGET.write_text(text)
    log.info("[vram_tracer] applied: step-level VRAM tracing enabled")

apply()
