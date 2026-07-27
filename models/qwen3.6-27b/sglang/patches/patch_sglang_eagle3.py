#!/usr/bin/env python3
"""patch_sglang_eagle3.py -- enable EAGLE-3 speculative decoding for the
Qwen3.6-class target (Qwen3_5ForConditionalGeneration) in SGLang.

Just run it -- it locates your installed SGLang and patches it in place:

    python3 patch_sglang_eagle3.py

(or pass an explicit path to .../sglang/srt/models/qwen3_5.py)

Problem
-------
SGLang serves Qwen3.6-27B via `Qwen3_5ForConditionalGeneration`. Its
`set_eagle3_layers_to_capture()` is inherited from
`Qwen3VLForConditionalGeneration`, which configures aux-hidden capture by
assigning `self.model.layers_to_capture` -- a plain list read only by the base
Qwen3-VL text model (`Qwen3LLMModel.forward`). But the Qwen3.5/3.6 decoder
(`Qwen3_5ForCausalLM`) captures differently: per-layer `_is_layer_to_capture`
attributes, set by `set_dflash_layers_to_capture()`. The inherited method never
sets those attrs, so EAGLE-3 capture silently no-ops; the decoder returns a bare
tensor and the wrapper forward crashes unpacking `(hidden_states, aux)`. DFlash
works only because the decoder ships `set_dflash_layers_to_capture()`; the
EAGLE-3 equivalent was never added. (Silent mechanism mismatch, not an
AttributeError.)

Fix
---
Add two methods to `sglang/srt/models/qwen3_5.py` (no other file is touched):
  1. `Qwen3_5ForCausalLM.set_eagle3_layers_to_capture` -- decoder level; marks
     `_is_layer_to_capture` (mechanically identical to `set_dflash_layers_to_capture`).
  2. `Qwen3_5ForConditionalGeneration.set_eagle3_layers_to_capture` -- overrides
     the inherited Qwen3-VL version to route through (1).

Idempotent (keys on the `[EAGLE3-PATCH]` marker), writes a `.eagle3-bak`
backup, and AST-validates the result. Verified against SGLang 0.5.12 (current
PyPI release) and the `main` branch. Exits non-zero on any anchor mismatch
(SGLang version drift) or post-patch parse failure.
"""
import ast
import importlib.util
import os
import shutil
import sys

MARKER = "[EAGLE3-PATCH]"

# --- Patch 1: dense decoder (Qwen3_5ForCausalLM) -------------------------
DECODER_ANCHOR = (
    "    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):\n"
    "        self.layers_to_capture = layers_to_capture\n"
    "        for layer_id in self.layers_to_capture:\n"
    '            setattr(self.layers[layer_id], "_is_layer_to_capture", True)\n'
)
DECODER_NEW = DECODER_ANCHOR + (
    "\n"
    "    def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):\n"
    f"        # {MARKER} EAGLE-3 aux-hidden capture for the dense Qwen3.5/3.6\n"
    "        # decoder. Same mechanism as DFlash: mark the decoder layers whose\n"
    "        # residual-stream input is captured and fed to the EAGLE-3 drafter.\n"
    "        self.layers_to_capture = layers_to_capture\n"
    "        for layer_id in self.layers_to_capture:\n"
    '            setattr(self.layers[layer_id], "_is_layer_to_capture", True)\n'
)

# --- Patch 2: multimodal wrapper (Qwen3_5ForConditionalGeneration) -------
WRAPPER_ANCHOR = (
    "class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):"
)
WRAPPER_NEW = (
    "    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):\n"
    f"        # {MARKER} Route EAGLE-3 aux-hidden capture through the dense\n"
    "        # Qwen3_5ForCausalLM decoder's per-layer mechanism. The inherited\n"
    "        # Qwen3VLForConditionalGeneration version sets model.layers_to_capture\n"
    "        # as a plain list, which only the base Qwen3LLMModel.forward reads;\n"
    "        # the Qwen3.5/3.6 decoder captures via per-layer _is_layer_to_capture\n"
    "        # attrs instead. layer_ids are HF-style 'after layer k'; the +1\n"
    "        # converts to SGLang's 'capture before layer k+1' convention.\n"
    "        self.capture_aux_hidden_states = True\n"
    "        if layer_ids is None:\n"
    '            text_cfg = getattr(self.config, "text_config", self.config)\n'
    "            num_layers = text_cfg.num_hidden_layers\n"
    "            offset_ids = [2, num_layers // 2, num_layers - 3]\n"
    "        else:\n"
    "            offset_ids = [val + 1 for val in layer_ids]\n"
    "        self.model.set_eagle3_layers_to_capture(offset_ids)\n"
    "\n"
    "\n"
    + WRAPPER_ANCHOR
)


def locate_qwen3_5():
    """Find qwen3_5.py inside the installed SGLang package."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not getattr(spec, "submodule_search_locations", None):
        sys.exit("ERROR: the 'sglang' package was not found. Activate the env "
                 "where SGLang is installed, or pass the qwen3_5.py path "
                 "explicitly.")
    base = list(spec.submodule_search_locations)[0]
    path = os.path.join(base, "srt", "models", "qwen3_5.py")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found -- SGLang >= 0.5.10 is required "
                 "(Qwen3.6 support).")
    return path


def main():
    if len(sys.argv) == 1:
        path = locate_qwen3_5()
        print(f"located SGLang model file: {path}")
    elif len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        sys.exit("usage: patch_sglang_eagle3.py [path-to qwen3_5.py]")

    with open(path) as f:
        src = f.read()

    if MARKER in src:
        print(f"{path}: already patched ({MARKER} present) -- no change.")
        return

    n_dec = src.count(DECODER_ANCHOR)
    if n_dec != 1:
        sys.exit(f"FAIL patch 1: decoder anchor matched {n_dec}x (expected 1).\n"
                 "  This SGLang's qwen3_5.py differs from the tested versions "
                 "(0.5.12, main). Please report your SGLang version as an issue.")
    n_wrap = src.count(WRAPPER_ANCHOR)
    if n_wrap != 1:
        sys.exit(f"FAIL patch 2: wrapper anchor matched {n_wrap}x (expected 1).\n"
                 "  This SGLang's qwen3_5.py differs from the tested versions.")

    patched = src.replace(DECODER_ANCHOR, DECODER_NEW, 1)
    patched = patched.replace(WRAPPER_ANCHOR, WRAPPER_NEW, 1)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        sys.exit(f"FAIL: patched file does not parse: {e}")
    n_new = patched.count("def set_eagle3_layers_to_capture")
    if n_new != 2:
        sys.exit(f"FAIL: expected 2 set_eagle3_layers_to_capture defs, found {n_new}")

    bak = path + ".eagle3-bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"backup written: {bak}")
    with open(path, "w") as f:
        f.write(patched)

    print(f"{path}: patched OK")
    print("  + Qwen3_5ForCausalLM.set_eagle3_layers_to_capture            (dense decoder)")
    print("  + Qwen3_5ForConditionalGeneration.set_eagle3_layers_to_capture (mm wrapper)")
    print("SGLang can now serve Qwen3.6-class targets with --speculative-algorithm EAGLE3.")


if __name__ == "__main__":
    main()
