"""VRAM budget cap — hard-limits PyTorch's CUDA memory pool.

Uses torch.cuda.set_per_process_memory_fraction() to cap PyTorch's
allocator at a fixed fraction of total VRAM. When the allocator hits
the cap, it triggers garbage collection and cache clearing automatically
(PyTorch does this internally when malloc fails and retries).

This prevents the gradual VRAM creep from MTP draft buffers, TQ scratch
tensors, and Triton kernel workspace — they all compete within the capped
pool and get evicted when it fills, instead of growing into free VRAM
indefinitely.

Applied to the EngineCore subprocess via monkey-patching the model loading
phase in gpu_worker.py.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_vram_budget")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: vram_budget"

# We cap at 0.95 of total VRAM. The model + KV at 0.85 util uses ~85%.
# The remaining 10% (0.85 to 0.95) is the scratch budget for MTP/TQ transients.
# Anything beyond 0.95 is reserved for CUDA driver, nvidia-smi overhead, etc.
MEMORY_FRACTION = 0.88

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")

def apply():
    if not TARGET.exists():
        log.warning("[vram_budget] gpu_worker.py not found")
        return

    text = TARGET.read_text()
    if MARKER in text:
        log.info("[vram_budget] already applied")
        return

    # Find the init_device or load_model method where CUDA is initialized
    anchor = "def init_device(self)"
    if anchor not in text:
        anchor = "def init_device("
        if anchor not in text:
            log.warning("[vram_budget] init_device anchor not found")
            return

    idx = text.find(anchor)
    eol = text.find('\n', idx)

    inject = """
        {marker}
        import torch as _vb_torch
        if _vb_torch.cuda.is_available():
            _vb_torch.cuda.set_per_process_memory_fraction({fraction})
            _vb_total = _vb_torch.cuda.get_device_properties(0).total_mem
            _vb_cap = int(_vb_total * {fraction}) // 1024**2
            import logging as _vb_log
            _vb_log.getLogger("vram_budget").info(
                f"[vram_budget] CUDA memory capped at {{_vb_cap}}MB "
                f"({{int({fraction}*100)}}% of {{_vb_total//1024**2}}MB)")
""".format(marker=MARKER, fraction=MEMORY_FRACTION)

    text = text[:eol+1] + inject + text[eol+1:]
    TARGET.write_text(text)
    log.info(f"[vram_budget] applied: memory_fraction={MEMORY_FRACTION}")

apply()
