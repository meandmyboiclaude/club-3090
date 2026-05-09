#!/usr/bin/env python3
"""Skip Qwen3_5 vision tower allocation when running text-only.

Qwen3_5ForConditionalGeneration and Qwen3_5MoeForConditionalGeneration
instantiate Qwen3_VisionTransformer (~0.86 GiB) even when
--language-model-only is set.  This patch replaces the ViT constructor
with PPMissingLayer() when language_model_only=True, reclaiming VRAM
for KV cache.
"""
import sys
from pathlib import Path

TARGET_PATHS = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py",
    "/home/user/engines/vllm/vllm/model_executor/models/qwen3_5.py",
]

OLD = '''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )'''

NEW = '''        if multimodal_config.language_model_only:
            # NOEDEL-OVERNIGHT: skip vision tower — saves ~0.86 GiB VRAM
            self.visual = PPMissingLayer()
        else:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )'''

MARKER = "# NOEDEL-OVERNIGHT: skip vision tower"


def patch(path):
    p = Path(path)
    if not p.exists():
        return f"missing: {path}"
    text = p.read_text()
    if MARKER in text:
        return f"already-applied: {path}"
    if OLD not in text:
        return f"anchor-not-found: {path}"
    # No count= limit: anchor is identical in both Qwen3_5ForConditionalGeneration
    # and Qwen3_5MoeForConditionalGeneration — one replace() patches both sites.
    p.write_text(text.replace(OLD, NEW))
    return f"patched: {path}"


found = 0
for path in TARGET_PATHS:
    result = patch(path)
    print(f"[vision_tower_skip] {result}", flush=True)
    if result.startswith(("patched", "already-applied")):
        found += 1

if found == 0:
    print("[vision_tower_skip] WARNING: no targets patched", file=sys.stderr)
    sys.exit(0)  # non-fatal
