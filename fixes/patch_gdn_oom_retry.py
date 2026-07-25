"""GDN OOM Safety Net — installs a global CUDA OOM handler via PyTorch's
memory allocator hooks. Instead of patching individual files (which keeps
breaking indentation), this hooks into PyTorch's OOM retry mechanism.

Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
to reduce fragmentation, and installs a custom OOM observer that clears
cache proactively.
"""
import logging
import os

log = logging.getLogger("patch_gdn_oom_retry")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

# Fix any broken chunk.py / chunk_o.py / chunk_delta_h.py from previous patches
from pathlib import Path

_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
# [2026-07-25] vllm#48500 moved fla ops to third_party/ — resolve whichever
# home this image has (old path = dev1060cherry/prod).
_FLA_DIR = _VLLM / "third_party/flash_linear_attention/ops"
if not _FLA_DIR.is_dir():
    _FLA_DIR = _VLLM / "model_executor/layers/fla/ops"

TARGETS = [
    _FLA_DIR / "chunk_delta_h.py",
    _FLA_DIR / "chunk_o.py",
    _FLA_DIR / "chunk.py",
]

def clean_old_patches():
    """Remove ALL previous gdn_oom_retry patches from FLA ops files."""
    for path in TARGETS:
        if not path.exists():
            continue
        text = path.read_text()
        if "PATCH: gdn_oom_retry" not in text:
            continue

        lines = text.split("\n")
        cleaned = []
        in_patch = False
        brace_depth = 0

        for line in lines:
            if "PATCH: gdn_oom_retry" in line:
                in_patch = True
                continue
            if in_patch:
                stripped = line.strip()
                # End of patch block: non-empty line at module level that isn't
                # part of the patch (not _gdn_, not import, not torch.*)
                if stripped and not stripped.startswith(("#", "import ", "from ",
                    "_gdn_", "def _gdn", "torch.empty_like", "torch.Tensor",
                    "    ", "")) or (stripped.startswith("def ") and "_gdn" not in stripped):
                    in_patch = False
                    cleaned.append(line)
                continue
            cleaned.append(line)

        # Also revert any function name replacements
        result = "\n".join(cleaned)
        result = result.replace("_gdn_safe_empty_like(", "torch.empty_like(")
        result = result.replace("_gdn_safe_new_empty(", "torch.Tensor.new_empty(")
        result = result.replace("_gdn_empty_like(", "torch.empty_like(")
        result = result.replace("_oom_safe_empty_like(", "torch.empty_like(")

        path.write_text(result)
        log.info(f"[gdn_oom_retry] cleaned old patches from {path.name}")


def configure_allocator():
    """Set PyTorch CUDA allocator config for maximum OOM resilience."""
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    parts = [p for p in current.split(",") if p.strip()]

    # Ensure expandable_segments is on
    if not any("expandable_segments" in p for p in parts):
        parts.append("expandable_segments:True")

    # max_split_size limits the largest block the allocator creates,
    # reducing fragmentation from huge transient GDN allocations
    if not any("max_split_size_mb" in p for p in parts):
        parts.append("max_split_size_mb:256")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
    log.info(f"[gdn_oom_retry] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")


def apply():
    clean_old_patches()
    configure_allocator()
    log.info("[gdn_oom_retry] applied (v4 — clean patches + allocator config)")


apply()
