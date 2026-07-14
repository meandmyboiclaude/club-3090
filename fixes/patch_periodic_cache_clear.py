"""Periodic torch.cuda.empty_cache() via vLLM EngineCore.step().

PyTorch's caching allocator holds freed GPU blocks. Under MTP spec-decode,
transient draft buffers fragment the pool, causing nvidia-smi reported VRAM
to creep upward under sustained concurrent load.

This patch injects an empty_cache() call every N engine steps into
EngineCore.step() in core.py — a small, stable file unlikely to drift.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_periodic_cache_clear")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

CLEAR_INTERVAL = 1
MARKER = "# PATCH: periodic_cache_clear"
TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py")

def apply():
    if not TARGET.exists():
        log.warning("[cache_clear] core.py not found")
        return

    text = TARGET.read_text()
    if MARKER in text:
        log.info("[cache_clear] already applied")
        return

    # Strategy: add a module-level counter and function, then patch step()
    # to call it. We find "def step(" and inject after the def line.

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
        log.warning("[cache_clear] step() anchor not found")
        return

    # Add module-level code right before the class definition
    class_anchor = "class EngineCore:"
    if class_anchor not in text:
        log.warning("[cache_clear] EngineCore class not found")
        return

    module_inject = """
{marker}
import torch as _pcc_torch
_pcc_counter = 0

def _pcc_maybe_clear():
    global _pcc_counter
    _pcc_counter += 1
    if _pcc_counter >= {interval}:
        _pcc_counter = 0
        _pcc_torch.cuda.empty_cache()

""".format(marker=MARKER, interval=CLEAR_INTERVAL)

    # Insert before class definition
    text = text.replace(class_anchor, module_inject + class_anchor, 1)

    # Insert call at start of step() body
    step_idx = text.find(step_anchor)
    eol = text.find('\n', step_idx)
    # The next line after def step(...): is the body
    text = text[:eol+1] + "        _pcc_maybe_clear()\n" + text[eol+1:]

    TARGET.write_text(text)
    log.info(f"[cache_clear] applied: empty_cache() every {CLEAR_INTERVAL} steps")

apply()
