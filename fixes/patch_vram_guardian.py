"""VRAM Guardian — proactive memory pressure management for vLLM.

Runs a background thread in the EngineCore subprocess that monitors actual
GPU memory usage (via torch.cuda.mem_get_info) and triggers gc + empty_cache
when usage exceeds a configurable threshold. Unlike periodic clearing, this
only fires when memory is actually under pressure, minimizing performance impact.

The guardian also sets torch.cuda.set_per_process_memory_fraction() as a hard
backstop — PyTorch's allocator will OOM-retry (gc + empty_cache internally)
when it hits this wall, preventing the process from touching the last few %
of VRAM.

Environment variables:
  VRAM_GUARDIAN_SOFT_PCT  - trigger gc+empty_cache at this % (default: 90)
  VRAM_GUARDIAN_HARD_PCT  - set_per_process_memory_fraction cap (default: 93)
  VRAM_GUARDIAN_POLL_MS   - check interval in ms (default: 500)
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_vram_guardian")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: vram_guardian"
TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")

GUARDIAN_CODE = '''
# PATCH: vram_guardian
import threading as _vg_threading
import time as _vg_time
import os as _vg_os
import gc as _vg_gc
import logging as _vg_logging

_vg_log = _vg_logging.getLogger("vram_guardian")
_vg_log.setLevel(_vg_logging.INFO)
if not _vg_log.handlers:
    _vg_log.addHandler(_vg_logging.StreamHandler())

def _vg_start_guardian():
    import torch

    if not torch.cuda.is_available():
        return

    soft_pct = int(_vg_os.environ.get("VRAM_GUARDIAN_SOFT_PCT", "90"))
    hard_pct = int(_vg_os.environ.get("VRAM_GUARDIAN_HARD_PCT", "93"))
    poll_ms = int(_vg_os.environ.get("VRAM_GUARDIAN_POLL_MS", "500"))

    total_memory = torch.cuda.get_device_properties(0).total_memory
    soft_bytes = int(total_memory * soft_pct / 100)
    hard_fraction = hard_pct / 100.0

    # Hard cap — PyTorch auto-retries with gc+empty_cache when hitting this
    torch.cuda.set_per_process_memory_fraction(hard_fraction)
    _vg_log.info(
        f"[vram_guardian] started: soft={soft_pct}% hard={hard_pct}% "
        f"poll={poll_ms}ms total={total_memory//1024**2}MB "
        f"soft_trigger={soft_bytes//1024**2}MB "
        f"hard_cap={int(total_memory*hard_fraction)//1024**2}MB"
    )

    cleared_count = 0

    # Track allocated growth to diagnose leak vs fragmentation
    _diag = {"baseline_alloc": None, "baseline_reserved": None, "tick": 0}

    def _guardian_loop():
        nonlocal cleared_count
        while True:
            _vg_time.sleep(poll_ms / 1000.0)
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                alloc = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                _diag["tick"] += 1

                if _diag["baseline_alloc"] is None:
                    _diag["baseline_alloc"] = alloc
                    _diag["baseline_reserved"] = reserved

                # Log diagnostics every 200 ticks (~100s)
                if _diag["tick"] % 200 == 0:
                    da = (alloc - _diag["baseline_alloc"]) / 1024**2
                    dr = (reserved - _diag["baseline_reserved"]) / 1024**2
                    frag = (reserved - alloc) / 1024**2
                    _vg_log.info(
                        f"[vram_diag] tick={_diag['tick']} "
                        f"alloc={alloc//1024**2}MB(+{da:.0f}) "
                        f"reserved={reserved//1024**2}MB(+{dr:.0f}) "
                        f"frag={frag:.0f}MB "
                        f"nvidia={used//1024**2}MB clears={cleared_count}"
                    )

                # Periodic defrag: clear cache every 300 ticks (~60s at 200ms poll)
                # regardless of pressure — prevents fragmentation buildup
                frag_mb = (reserved - alloc) / 1024**2
                if _diag["tick"] % 300 == 0 and frag_mb > 50:
                    _vg_gc.collect()
                    torch.cuda.empty_cache()
                    cleared_count += 1

                if used > soft_bytes:
                    pre_alloc = alloc
                    pre_reserved = reserved
                    _vg_gc.collect()
                    torch.cuda.empty_cache()
                    free2, _ = torch.cuda.mem_get_info()
                    post_alloc = torch.cuda.memory_allocated()
                    post_reserved = torch.cuda.memory_reserved()
                    reclaimed = (free2 - free) / 1024**2
                    cleared_count += 1
                    if cleared_count <= 20 or cleared_count % 50 == 0:
                        _vg_log.info(
                            f"[vram_guardian] cleared #{cleared_count}: "
                            f"nvidia: {used//1024**2}->{(total-free2)//1024**2}MB "
                            f"alloc: {pre_alloc//1024**2}->{post_alloc//1024**2}MB "
                            f"reserved: {pre_reserved//1024**2}->{post_reserved//1024**2}MB "
                            f"reclaimed={reclaimed:.0f}MB"
                        )
            except Exception:
                pass

    t = _vg_threading.Thread(target=_guardian_loop, daemon=True, name="vram-guardian")
    t.start()
'''

def apply():
    if not TARGET.exists():
        log.warning("[vram_guardian] gpu_worker.py not found")
        return

    text = TARGET.read_text()
    if MARKER in text:
        log.info("[vram_guardian] already applied")
        return

    anchor = "def init_device(self)"
    if anchor not in text:
        anchor = "def init_device("
        if anchor not in text:
            log.warning("[vram_guardian] init_device anchor not found")
            return

    idx = text.find(anchor)
    eol = text.find('\n', idx)

    # Inject guardian start call at end of init_device.
    # Find the next method definition to know where init_device ends.
    next_def = text.find('\n    def ', eol + 1)
    if next_def < 0:
        log.warning("[vram_guardian] can't find end of init_device")
        return

    # REBASE 2026-06-07: a 4-space-indented comment block (the
    # "# FIXME(youkaichao & ywang96)" decorator-style comment in vllm-new
    # gpu_worker.py:347-348) sits between init_device's last statement and
    # the `def load_model` that follows. Inserting an 8-space body call right
    # before that `\n    def` would land it AFTER the 4-space comment ->
    # IndentationError at import. Walk the insertion point backwards over any
    # trailing comment/blank lines so the call lands after the method's real
    # last code line, still inside init_device.
    insert_at = next_def
    while True:
        prev_nl = text.rfind('\n', 0, insert_at)
        if prev_nl < 0:
            break
        line = text[prev_nl + 1:insert_at]
        stripped = line.strip()
        # Stop at the first real code line; skip blanks and comment lines.
        if stripped and not stripped.startswith('#'):
            break
        insert_at = prev_nl

    # Insert guardian start at the method's true end (after last code line).
    inject_call = "\n        _vg_start_guardian()"
    text = text[:insert_at] + inject_call + text[insert_at:]

    # Insert guardian code before the class
    class_anchor = "class GpuWorker"
    if class_anchor not in text:
        class_anchor = "class Worker"
        if class_anchor not in text:
            log.warning("[vram_guardian] Worker class not found")
            return

    cidx = text.find(class_anchor)
    text = text[:cidx] + GUARDIAN_CODE + "\n" + text[cidx:]

    TARGET.write_text(text)
    log.info("[vram_guardian] applied")

apply()
