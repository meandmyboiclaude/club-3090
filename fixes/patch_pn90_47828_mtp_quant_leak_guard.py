#!/usr/bin/env python3
"""PN90 — vllm#47828 (Qwen3.5 MTP × GPTQ/AWQ quant-config leak) — DOCUMENTED
CONDITIONAL NO-OP for our dense pin.

Upstream PR #47828 fixes Qwen3.5-MoE MTP under GPTQ/AWQ: the main model's
quant_config leaks into the bf16 MTP decoder layers because FusedMoE.__init__
resolves quant_config via get_current_vllm_config() (the global config)
instead of the explicitly passed one. The fix wraps the MTP layer
construction in set_current_vllm_config(<copy with quant_config=None>).

WHY THIS IS A NO-OP ON OUR PIN (nightly-9e57de71, dev1060), verified in-image
2026-07-13:

1. Our model (XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-MTP-BF16) is DENSE:
   hf_text_config.model_type == "qwen3_5_text", so Qwen3_5DecoderLayer builds
   Qwen3NextMLP — never Qwen3NextSparseMoeBlock/FusedMoE. The leak site
   (get_current_vllm_config() inside FusedMoE.__init__) is unreachable.
2. The dense path passes quant_config EXPLICITLY with a full prefix
   (mtp.layers.N.self_attn / mtp.layers.N.mlp), and the checkpoint's GPTQ
   quantization_config.dynamic contains '-:.*mtp.*' — every MTP submodule
   resolves to UnquantizedLinearMethod. MTP n=3 is live and working on :8020.
3. Our quant name ("gptq"/"gptq_marlin") IS in the PR's guard list, so
   applying the guard would merely reproduce the same unquantized outcome the
   dynamic override already produces (redundant, not wrong).

BELT-AND-BRACES SELF-ACTIVATION: on every boot this patcher re-checks the
dense-safety invariants in qwen3_5.py (Qwen3_5DecoderLayer passing
quant_config= explicitly to both the attention and the dense MLP). If a
future image rebuild breaks those invariants (i.e. the dense path starts
resolving quant through the global config, so the prefix-based dynamic skip
can no longer protect the MTP layers), the full #47828 guard IS applied to
qwen3_5_mtp.py. Anchor miss in that activation path is FAIL-LOUD.

Retire when upstream contains the #47828 guard (detected via
set_current_vllm_config / _needs_noquant in Qwen3_5MultiTokenPredictor) —
this patcher then self-retires.
"""
import pathlib
import sys

LOG = "[pn90-mtp-quant-leak-guard]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET_MTP = VLLM / "model_executor/models/qwen3_5_mtp.py"
TARGET_DENSE = VLLM / "model_executor/models/qwen3_5.py"
MARKER = "# PN90:"

# --- dense-safety invariants (inside class Qwen3_5DecoderLayer) -------------
# As long as BOTH constructs receive quant_config explicitly with a
# {prefix}-derived name, the checkpoint's GPTQ dynamic override
# ('-:.*mtp.*') un-quantizes every MTP submodule and the leak cannot bite.
INV_QUANT_FROM_CFG = "        quant_config = vllm_config.quant_config\n"
INV_ATTN = (
    "            self.self_attn = Qwen3NextAttention(\n"
    "                config,\n"
    "                model_config=model_config,\n"
    "                cache_config=cache_config,\n"
    "                quant_config=quant_config,\n"
    '                prefix=f"{prefix}.self_attn",\n'
)
INV_MLP = (
    "            self.mlp = Qwen3NextMLP(\n"
    "                hidden_size=config.hidden_size,\n"
    "                intermediate_size=config.intermediate_size,\n"
    "                hidden_act=config.hidden_act,\n"
    "                quant_config=quant_config,\n"
    '                prefix=f"{prefix}.mlp",\n'
)

# --- activation patch (only used when invariants are broken) ----------------
LAYERS_OLD = (
    "        self.layers = torch.nn.ModuleList(\n"
    "            Qwen3_5DecoderLayer(\n"
    "                vllm_config,\n"
    '                layer_type="full_attention",\n'
    '                prefix=f"{prefix}.layers.{idx}",\n'
    "            )\n"
    "            for idx in range(self.num_mtp_layers)\n"
    "        )\n"
)
LAYERS_NEW = (
    "        # PN90: vllm#47828 backport — MTP layers use unquantized bf16\n"
    "        # weights even when the main model is GPTQ/AWQ-quantized.\n"
    "        # Override the global vllm_config so constructors that consult\n"
    "        # get_current_vllm_config() (e.g. FusedMoE.__init__) see\n"
    "        # quant_config=None instead of the GPTQ config.\n"
    "        # dataclasses.replace() does NOT work because\n"
    "        # VllmConfig.__post_init__ re-resolves quant_config from\n"
    "        # model_config; set it AFTER construction via object.__setattr__.\n"
    "        _needs_noquant = (\n"
    "            quant_config is not None\n"
    '            and hasattr(quant_config, "get_name")\n'
    "            and quant_config.get_name()\n"
    "            in (\n"
    '                "gptq",\n'
    '                "auto_gptq",\n'
    '                "gptq_marlin",\n'
    '                "awq",\n'
    '                "awq_marlin",\n'
    "            )\n"
    "        )\n"
    "\n"
    "        if _needs_noquant:\n"
    "            import copy\n"
    "\n"
    "            from vllm.config import set_current_vllm_config\n"
    "\n"
    "            mtp_vllm_config = copy.copy(vllm_config)\n"
    '            object.__setattr__(mtp_vllm_config, "quant_config", None)\n'
    "            logger.info(\n"
    '                "MTP: bypassing %s quantization for MTP layers (bf16 weights)",\n'
    "                quant_config.get_name(),\n"
    "            )\n"
    "            with set_current_vllm_config(mtp_vllm_config):\n"
    "                self.layers = torch.nn.ModuleList(\n"
    "                    Qwen3_5DecoderLayer(\n"
    "                        mtp_vllm_config,\n"
    '                        layer_type="full_attention",\n'
    '                        prefix=f"{prefix}.layers.{idx}",\n'
    "                    )\n"
    "                    for idx in range(self.num_mtp_layers)\n"
    "                )\n"
    "        else:\n"
    "            self.layers = torch.nn.ModuleList(\n"
    "                Qwen3_5DecoderLayer(\n"
    "                    vllm_config,\n"
    '                    layer_type="full_attention",\n'
    '                    prefix=f"{prefix}.layers.{idx}",\n'
    "                )\n"
    "                for idx in range(self.num_mtp_layers)\n"
    "            )\n"
)


def decoder_layer_slice(text: str) -> str:
    """Body of class Qwen3_5DecoderLayer up to the next top-level def/class."""
    key = "class Qwen3_5DecoderLayer"
    if key not in text:
        return ""
    body = text.split(key, 1)[1]
    for stop in ("\nclass ", "\n@support_torch_compile", "\ndef "):
        idx = body.find(stop)
        if idx != -1:
            body = body[:idx]
    return body


def main() -> int:
    for path in (TARGET_MTP, TARGET_DENSE):
        if not path.exists():
            print(f"{LOG} FATAL: {path} not present", file=sys.stderr)
            return 1

    mtp_text = TARGET_MTP.read_text()
    if MARKER in mtp_text:
        print(f"{LOG} already applied (idempotent)")
        return 0

    # Self-retire: upstream #47828 guard landed.
    if "set_current_vllm_config" in mtp_text or "_needs_noquant" in mtp_text:
        print(
            f"{LOG} upstream drift: #47828 quant-bypass already present in "
            f"qwen3_5_mtp.py — self-retire (no-op)"
        )
        return 0

    dense_text = TARGET_DENSE.read_text()
    layer_body = decoder_layer_slice(dense_text)
    invariants_ok = (
        layer_body != ""
        and INV_QUANT_FROM_CFG in layer_body
        and INV_ATTN in layer_body
        and INV_MLP in layer_body
    )

    if invariants_ok:
        # Documented no-op: dense path passes quant_config explicitly with
        # mtp-prefixed names; checkpoint dynamic '-:.*mtp.*' un-quantizes all
        # MTP submodules. The #47828 leak (FusedMoE via
        # get_current_vllm_config) is unreachable on a dense model.
        print(
            f"{LOG} verified NO-OP: dense-safety invariants intact in "
            f"Qwen3_5DecoderLayer (explicit quant_config= on self_attn + "
            f"Qwen3NextMLP); #47828 leak unreachable on dense "
            f"qwen3_5_text — guard not applied"
        )
        return 0

    # Invariants broken -> the prefix-based dynamic skip may no longer
    # protect the MTP layers. Self-activate: apply the full #47828 guard.
    print(
        f"{LOG} dense-safety invariants BROKEN in qwen3_5.py — "
        f"self-activating the #47828 quant-bypass guard",
        file=sys.stderr,
    )
    if LAYERS_OLD not in mtp_text:
        print(
            f"{LOG} FATAL: anchor-not-found (MTP layer construction) while "
            f"self-activating — upstream refactor; re-derive #47828 backport "
            f"before boot (MTP layers may silently load GPTQ-quantized)",
            file=sys.stderr,
        )
        return 1
    if mtp_text.count(LAYERS_OLD) != 1:
        print(
            f"{LOG} FATAL: ambiguous anchor (MTP layer construction)",
            file=sys.stderr,
        )
        return 1
    TARGET_MTP.write_text(mtp_text.replace(LAYERS_OLD, LAYERS_NEW, 1))
    print(
        f"{LOG} applied: #47828 quant-bypass guard around MTP layer "
        f"construction (self-activated, invariants were broken)"
    )
    return 0


sys.exit(main())
