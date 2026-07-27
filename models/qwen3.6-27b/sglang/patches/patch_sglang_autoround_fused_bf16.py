#!/usr/bin/env python3
"""Patch SGLang v0.5.12 AutoRound mixed-precision fused layers.

SGLang passes each model's ``packed_modules_mapping`` into quantization configs,
but ``AutoRoundConfig.from_config`` drops that field. For Qwen3.6 AutoRound this
means split checkpoint tensors such as ``in_proj_b`` / ``in_proj_a`` are not
recognized as the fused SGLang module ``in_proj_ba``. The checkpoint marks both
split shards as FP/BF16 via ``extra_config``, but SGLang quantizes the fused
module anyway and later crashes in GPTQ-Marlin repack because ``size_n=96`` is
not divisible by Marlin's 64-wide tile.

This startup patch keeps the packed-module mapping and lets AutoRound evaluate
the split shard configs before deciding whether a fused layer is quantized.
"""
import ast
import importlib.util
import os
import shutil
import sys

MARKER = "[AUTOROUND-FUSED-BF16-PATCH]"

INIT_SIG_ANCHOR = (
    '        backend: str = "auto",\n'
    "    ) -> None:\n"
    "        super().__init__()\n"
)
INIT_SIG_NEW = (
    '        backend: str = "auto",\n'
    "        packed_modules_mapping: Optional[dict[str, list[str]]] = None,\n"
    "    ) -> None:\n"
    "        super().__init__()\n"
)

ATTR_ANCHOR = (
    "        self.data_type = data_type\n"
    "        self.backend = backend\n"
    "        self.pack_factor = Fraction(32, weight_bits)\n"
)
ATTR_NEW = (
    "        self.data_type = data_type\n"
    "        self.backend = backend\n"
    f"        self.packed_modules_mapping = packed_modules_mapping or {{}}  # {MARKER}\n"
    "        self.pack_factor = Fraction(32, weight_bits)\n"
)

FROM_CONFIG_ANCHOR = (
    "            backend=cls.get_from_keys_or(\n"
    '                config, ["backend", "vllm_backend", "sglang_backend"], "auto"\n'
    "            ),\n"
    "        )\n"
)
FROM_CONFIG_NEW = (
    "            backend=cls.get_from_keys_or(\n"
    '                config, ["backend", "vllm_backend", "sglang_backend"], "auto"\n'
    "            ),\n"
    "            packed_modules_mapping=cls.get_from_keys_or(\n"
    '                config, ["packed_modules_mapping"], {}\n'
    "            ),\n"
    "        )\n"
)

GET_CONFIG_ANCHOR = (
    "        def get_config(name: str, quantized: bool = True):\n"
    "            if not self.extra_config:\n"
)
GET_CONFIG_NEW = (
    "        def name_aliases(name: str):\n"
    f"            # {MARKER} Qwen3.6 configs use HF names while SGLang params use\n"
    "            # model.layers.* after stripping model.language_model.* during load.\n"
    "            aliases = [name]\n"
    '            if name.startswith("model."):\n'
    '                aliases.append("model.language_model." + name[len("model."):])\n'
    '            if name.startswith("model.language_model."):\n'
    '                aliases.append("model." + name[len("model.language_model."):])\n'
    "            return aliases\n"
    "\n"
    "        def get_config(name: str, quantized: bool = True):\n"
    "            if not self.extra_config:\n"
)

EXACT_ANCHOR = (
    "            # Exact match first\n"
    "            if name in self.extra_config:\n"
    "                cfg = self.extra_config[name]\n"
    "                return (\n"
    "                    cfg.get(\"bits\", self.weight_bits if quantized else 16),\n"
    "                    cfg.get(\"group_size\", self.group_size if quantized else -1),\n"
    "                    cfg.get(\"sym\", self.sym if quantized else True),\n"
    "                )\n"
)
EXACT_NEW = (
    "            # Exact match first, including HF/SGLang prefix aliases.\n"
    "            for candidate in name_aliases(name):\n"
    "                if candidate in self.extra_config:\n"
    "                    cfg = self.extra_config[candidate]\n"
    "                    return (\n"
    "                        cfg.get(\"bits\", self.weight_bits if quantized else 16),\n"
    "                        cfg.get(\"group_size\", self.group_size if quantized else -1),\n"
    "                        cfg.get(\"sym\", self.sym if quantized else True),\n"
    "                    )\n"
)

REGEX_ANCHOR = "                    if re.fullmatch(pattern, name):\n"
REGEX_NEW = (
    "                    if any(\n"
    "                        re.fullmatch(pattern, candidate)\n"
    "                        for candidate in name_aliases(name)\n"
    "                    ):\n"
)

BLOCK_ANCHOR = (
    "            quantized = any(\n"
    "                layer_name.startswith(name) for name in self.block_name_to_quantize\n"
    "            )\n"
)
BLOCK_NEW = (
    "            quantized = any(\n"
    "                candidate.startswith(name)\n"
    "                for candidate in name_aliases(layer_name)\n"
    "                for name in self.block_name_to_quantize\n"
    "            )\n"
)


def locate_auto_round():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not getattr(spec, "submodule_search_locations", None):
        sys.exit("ERROR: the 'sglang' package was not found.")
    base = list(spec.submodule_search_locations)[0]
    path = os.path.join(base, "srt", "layers", "quantization", "auto_round.py")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found -- expected SGLang v0.5.12 layout.")
    return path


def replace_once(src, old, new, label):
    count = src.count(old)
    if count != 1:
        sys.exit(f"FAIL {label}: anchor matched {count}x (expected 1).")
    return src.replace(old, new, 1)


def main():
    if len(sys.argv) == 1:
        path = locate_auto_round()
        print(f"located SGLang AutoRound file: {path}")
    elif len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        sys.exit("usage: patch_sglang_autoround_fused_bf16.py [path-to auto_round.py]")

    with open(path) as f:
        src = f.read()

    if MARKER in src:
        print(f"{path}: already patched ({MARKER} present) -- no change.")
        return

    patched = replace_once(src, INIT_SIG_ANCHOR, INIT_SIG_NEW, "init signature")
    patched = replace_once(patched, ATTR_ANCHOR, ATTR_NEW, "init attrs")
    patched = replace_once(patched, FROM_CONFIG_ANCHOR, FROM_CONFIG_NEW, "from_config")
    patched = replace_once(patched, GET_CONFIG_ANCHOR, GET_CONFIG_NEW, "name aliases")
    patched = replace_once(patched, EXACT_ANCHOR, EXACT_NEW, "extra_config exact")
    patched = replace_once(patched, REGEX_ANCHOR, REGEX_NEW, "extra_config regex")
    patched = replace_once(patched, BLOCK_ANCHOR, BLOCK_NEW, "block_name_to_quantize")

    try:
        ast.parse(patched)
    except SyntaxError as e:
        sys.exit(f"FAIL: patched file does not parse: {e}")

    bak = path + ".autoround-fused-bf16-bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"backup written: {bak}")
    with open(path, "w") as f:
        f.write(patched)

    print(f"{path}: patched OK")
    print("  + AutoRoundConfig keeps packed_modules_mapping")
    print("  + fused layers consult split-shard extra_config before quantizing")
    print("  + model.language_model.* and model.* names are treated as aliases")


if __name__ == "__main__":
    main()
