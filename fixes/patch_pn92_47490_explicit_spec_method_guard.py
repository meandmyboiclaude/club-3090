#!/usr/bin/env python3
"""PN92 — never silently override an explicitly configured speculative method.

Backport of vllm#47490 (GH issue #47486) to nightly-9e57de71 (dev1060),
minus the PR's test files. Guards our --speculative-config
'{"method":"mtp",...}' from silent replacement by checkpoint auto-detection.

Upstream bugs fixed:
1. A draft-model path containing "eagle-"/"eagle3"/"dflash"/"dspark" silently
   rerouted an explicit method onto the eagle-family path.
2. An explicit method with a structurally different drafter checkpoint
   (e.g. medusa vs mtp) was silently converted instead of raising.
3. The terminal error blamed the (supported) method instead of the
   unrecognized checkpoint.
4. EAGLEConfig raised a raw AttributeError when the draft config had no
   vocab_size (transformers_utils/configs/eagle.py).

Mechanics (mirrors the PR): record method_was_explicit before any
normalization in SpeculativeConfig.__post_init__; replace the in-line
auto-detect chain with _resolve_draft_method(), which auto-detects only for
a defaulted method and VALIDATES an explicit one (raising an actionable
ValueError on mismatch); medusa's model_type injection is narrowed to
legacy checkpoints that declare no model_type (_raw_config_has_model_type).

Safety for our boot (traced in-image 2026-07-13): with method='mtp' and
model=None, self.model becomes the target repo; compose_draft_hf_overrides
-> hf_config_override rewrites qwen3_5 -> qwen3_5_mtp, so
_detect_draft_method() returns "mtp" (qwen3_5_mtp in MTPModelTypes) ==
explicit method -> passes through, and the num_speculative_tokens>1 warning
is preserved verbatim.

Anchors verified byte-exact in-image 2026-07-13 (after genesis: P70/P75
touch the ngram-validation region of speculative.py, no overlap with these
anchors). Retire when upstream contains _resolve_draft_method — this
patcher then self-retires.
"""
import pathlib
import sys

LOG = "[pn92-explicit-spec-method-guard]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET_SPEC = VLLM / "config/speculative.py"
TARGET_EAGLE = VLLM / "transformers_utils/configs/eagle.py"
MARKER = "# PN92:"

# --- sub-patch A: record explicitness before normalization -------------------
A_OLD = (
    "        # infer method from user args\n"
    "        # Check if the model field contains a custom module path (e.g., 'pkg.Mod')\n"
)
A_NEW = (
    "        # infer method from user args\n"
    "        # PN92: vllm#47490 — record whether the user explicitly configured\n"
    "        # a method, before any normalization below. Auto-detection from the\n"
    "        # draft checkpoint may only fill in a missing method, never\n"
    "        # override an explicit one.\n"
    "        method_was_explicit = self.method is not None\n"
    "\n"
    "        # Check if the model field contains a custom module path (e.g., 'pkg.Mod')\n"
)

# --- sub-patch B: narrow medusa model_type injection to legacy checkpoints ---
B_OLD = (
    "                draft_hf_overrides: HfOverrides\n"
    '                if self.method == "medusa":\n'
    '                    draft_hf_overrides = {"model_type": "medusa"}\n'
)
B_NEW = (
    "                # PN92: vllm#47490 — only legacy checkpoints (no declared\n"
    "                # model_type) get the injection, so method validation can\n"
    "                # see what the checkpoint really is.\n"
    "                is_legacy_medusa = (\n"
    '                    self.method == "medusa" and not self._raw_config_has_model_type()\n'
    "                )\n"
    "                draft_hf_overrides: HfOverrides = SpeculativeConfig.hf_config_override\n"
    "                if is_legacy_medusa:\n"
    '                    draft_hf_overrides = {"model_type": "medusa"}\n'
)

# --- sub-patch C: vocab-size alignment follows the same narrowing ------------
C_OLD = (
    "                # omit vocab_size in config.json, so MedusaConfig falls back to\n"
    "                # its default (32001). Align with the target model's vocab size\n"
    "                # to avoid shape mismatches when loading LM-head weights.\n"
    '                if self.method == "medusa":\n'
)
C_NEW = (
    "                # omit vocab_size in config.json, so MedusaConfig falls back to\n"
    "                # its default (32001). Align with the target model's vocab size\n"
    "                # to avoid shape mismatches when loading LM-head weights.\n"
    "                if is_legacy_medusa:  # PN92: vllm#47490\n"
)

# --- sub-patch D: replace the in-line auto-detect chain -----------------------
D_OLD = (
    "                # Automatically detect the method\n"
    '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):\n'
    "                    pass\n"
    "                # examples:\n"
    "                # yuhuili/EAGLE-LLaMA3-Instruct-8B\n"
    "                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B\n"
    "                # AngelSlim/Qwen3-8B_eagle3\n"
    "                # deepseek-ai/dspark_qwen3_8b_block7\n"
    '                elif "eagle-" in self.draft_model_config.model.lower():\n'
    '                    self.method = "eagle"\n'
    '                elif "eagle3" in self.draft_model_config.model.lower():\n'
    '                    self.method = "eagle3"\n'
    '                elif "dflash" in self.draft_model_config.model.lower():\n'
    '                    self.method = "dflash"\n'
    "                elif (\n"
    '                    "dspark" in self.draft_model_config.model.lower()\n'
    '                    or "Qwen3DSparkModel" in self.draft_model_config.architectures\n'
    "                ):\n"
    '                    self.method = "dspark"\n'
    '                elif self.draft_model_config.hf_config.model_type == "medusa":\n'
    '                    self.method = "medusa"\n'
    '                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":\n'
    '                    self.method = "mlp_speculator"\n'
    "                elif self.draft_model_config.hf_config.model_type in get_args(\n"
    "                    MTPModelTypes\n"
    "                ):\n"
    '                    self.method = "mtp"\n'
    "                    if (\n"
    "                        self.num_speculative_tokens > 1\n"
    "                        and self.draft_model_config.hf_config.model_type\n"
    '                        != "step3p5_mtp"\n'
    "                    ):\n"
    "                        logger.warning(\n"
    '                            "Enabling num_speculative_tokens > 1 will run "\n'
    '                            "multiple times of forward on same MTP layer"\n'
    '                            ",which may result in lower acceptance rate"\n'
    "                        )\n"
    '                elif self.method == "draft_model":\n'
    "                    pass\n"
    "                else:\n"
    "                    raise NotImplementedError(\n"
    "                        f\"Unsupported speculative method: '{self.method}'\"\n"
    "                    )\n"
)
D_NEW = (
    "                # PN92: vllm#47490 — resolve the speculative method: honor\n"
    "                # an explicitly configured method (validating it against\n"
    "                # the draft checkpoint), otherwise auto-detect it.\n"
    "                self._resolve_draft_method(method_was_explicit)\n"
)

# --- sub-patch E: insert the resolution helpers ------------------------------
E_OLD = (
    "            return AttentionBackendEnum[value.upper()]\n"
    "        return value\n"
    "\n"
    '    @model_validator(mode="after")\n'
    "    def _verify_args(self) -> Self:\n"
)
E_NEW = (
    "            return AttentionBackendEnum[value.upper()]\n"
    "        return value\n"
    "\n"
    "    # PN92: vllm#47490 backport — helpers below.\n"
    "    def _raw_config_has_model_type(self) -> bool:\n"
    '        """Whether the draft checkpoint\'s raw config.json declares a\n'
    "        ``model_type``.\n"
    "\n"
    "        Old-format Medusa checkpoints (e.g. FasterDecoding/medusa-*) omit it,\n"
    "        and only those need the medusa ``model_type`` injection; checkpoints\n"
    "        that declare their own type must keep it so that method validation\n"
    "        can see what the checkpoint really is.\n"
    '        """\n'
    "        from transformers import PretrainedConfig\n"
    "\n"
    "        try:\n"
    "            config_dict, _ = PretrainedConfig.get_config_dict(\n"
    "                self.model,\n"
    "                revision=self.revision,\n"
    "                trust_remote_code=self.target_model_config.trust_remote_code,\n"
    "            )\n"
    "        except Exception:\n"
    "            return False\n"
    '        return "model_type" in config_dict\n'
    "\n"
    "    def _detect_draft_method(self) -> SpeculativeMethod | None:\n"
    '        """Best-effort detection of the speculative method implied by the\n'
    "        draft checkpoint, in the legacy detection order (name hints first).\n"
    '        """\n'
    "        # NOTE: this mirrors the legacy detection order exactly, with the\n"
    "        # name hints first: DeepSeek EAGLE heads are structurally MTP-typed\n"
    '        # (e.g. model_type="deepseek_mtp") and rely on the name hint to route\n'
    "        # to the eagle path.\n"
    "        model_name = self.draft_model_config.model.lower()\n"
    '        if "eagle-" in model_name:\n'
    '            return "eagle"\n'
    '        if "eagle3" in model_name:\n'
    '            return "eagle3"\n'
    '        if "dflash" in model_name:\n'
    '            return "dflash"\n'
    '        if "dspark" in model_name or "Qwen3DSparkModel" in (\n'
    "            self.draft_model_config.architectures or []\n"
    "        ):\n"
    '            return "dspark"\n'
    "        hf_config = self.draft_model_config.hf_config\n"
    '        model_type = getattr(hf_config, "model_type", None)\n'
    '        if model_type == "medusa":\n'
    '            return "medusa"\n'
    '        if model_type == "mlp_speculator":\n'
    '            return "mlp_speculator"\n'
    "        if model_type in get_args(MTPModelTypes):\n"
    '            return "mtp"\n'
    "        return None\n"
    "\n"
    "    def _method_mismatch_msg(self, detected: str | None) -> str:\n"
    "        hf_config = self.draft_model_config.hf_config\n"
    "        if detected is not None:\n"
    '            looks = f"looks like a {detected!r} checkpoint"\n'
    '            hint = f" Use method={detected!r} if that is what you intended."\n'
    "        else:\n"
    "            looks = (\n"
    '                "is not recognized as any drafter type (a plain causal LM "\n'
    "                \"can be used with method='draft_model')\"\n"
    "            )\n"
    '            hint = ""\n'
    "        return (\n"
    '            f"speculative_config requested method={self.method!r}, but the "\n'
    '            f"draft model {self.draft_model_config.model!r} {looks} "\n'
    "            f\"(model_type={getattr(hf_config, 'model_type', None)!r}, \"\n"
    "            f\"architectures={getattr(hf_config, 'architectures', None)}). \"\n"
    '            f"Pass a matching method or a matching draft checkpoint.{hint}"\n'
    "        )\n"
    "\n"
    "    def _resolve_draft_method(self, method_was_explicit: bool) -> None:\n"
    '        """Set or validate ``self.method`` against the draft checkpoint.\n'
    "\n"
    "        Auto-detection only applies when the user did not explicitly\n"
    "        configure a method; an explicitly configured method is validated\n"
    "        against the checkpoint and never silently overridden.\n"
    '        """\n'
    "        detected = self._detect_draft_method()\n"
    "\n"
    "        if not method_was_explicit:\n"
    '            # self.method defaulted to "draft_model"; adopt the detection.\n'
    "            if detected is not None:\n"
    "                self.method = detected\n"
    '        elif self.method in ("eagle", "eagle3", "dflash", "dspark"):\n'
    "            # Eagle-family checkpoints often cannot be detected structurally,\n"
    "            # and some legitimately carry an MTP model_type (e.g. DeepSeek\n"
    "            # EAGLE heads), so trust the explicit method; EAGLEConfig rejects\n"
    "            # incompatible configs with an actionable error. Only a\n"
    "            # structural medusa/mlp_speculator checkpoint can never work.\n"
    '            if detected in ("medusa", "mlp_speculator"):\n'
    "                raise ValueError(self._method_mismatch_msg(detected))\n"
    '        elif self.method == "draft_model":\n'
    "            # draft_model accepts any standalone causal LM, but a drafter\n"
    "            # head (medusa/mlp_speculator/mtp) cannot run standalone.\n"
    '            if detected in ("medusa", "mlp_speculator", "mtp"):\n'
    "                raise ValueError(self._method_mismatch_msg(detected))\n"
    "        elif self.method != detected:\n"
    "            # medusa / mlp_speculator / mtp were requested explicitly.\n"
    "            raise ValueError(self._method_mismatch_msg(detected))\n"
    "\n"
    "        if (\n"
    '            self.method == "mtp"\n'
    "            and self.num_speculative_tokens > 1\n"
    '            and self.draft_model_config.hf_config.model_type != "step3p5_mtp"\n'
    "        ):\n"
    "            logger.warning(\n"
    '                "Enabling num_speculative_tokens > 1 will run "\n'
    '                "multiple times of forward on same MTP layer"\n'
    '                ",which may result in lower acceptance rate"\n'
    "            )\n"
    "\n"
    '    @model_validator(mode="after")\n'
    "    def _verify_args(self) -> Self:\n"
)

SPEC_SUBS = (
    ("explicitness-record", A_OLD, A_NEW),
    ("legacy-medusa-narrowing", B_OLD, B_NEW),
    ("medusa-vocab-align", C_OLD, C_NEW),
    ("detect-chain-replacement", D_OLD, D_NEW),
    ("resolution-helpers", E_OLD, E_NEW),
)

# --- eagle.py: actionable vocab_size error (PR bug 4) -------------------------
EAGLE_OLD = (
    "        if self.model is None:\n"
    "            self.truncated_vocab_size = None\n"
    "        else:\n"
    "            self.truncated_vocab_size = (\n"
)
EAGLE_NEW = (
    "        if self.model is None:\n"
    "            self.truncated_vocab_size = None\n"
    "        else:\n"
    "            # PN92: vllm#47490 — actionable error instead of a raw\n"
    "            # AttributeError when the draft config has no vocab_size.\n"
    '            if truncated_vocab_size is None and not hasattr(self.model, "vocab_size"):\n'
    "                raise ValueError(\n"
    '                    "The draft model config "\n'
    '                    f"({type(self.model).__name__}) does not define "\n'
    "                    \"'vocab_size', so it does not look like an EAGLE draft \"\n"
    '                    "checkpoint. If the draft model is a different drafter "\n'
    "                    \"type (e.g. an MTP head), set the matching 'method' in \"\n"
    '                    "speculative_config."\n'
    "                )\n"
    "            self.truncated_vocab_size = (\n"
)


def apply_subs(path, subs, log_name):
    text = path.read_text()
    for name, old, new in subs:
        if old not in text:
            print(
                f"{LOG} FATAL: anchor-not-found ({log_name}:{name}) — upstream "
                f"refactor; re-derive before boot (explicit method:'mtp' loses "
                f"its silent-override guard)",
                file=sys.stderr,
            )
            return None
        if text.count(old) != 1:
            print(
                f"{LOG} FATAL: ambiguous anchor ({log_name}:{name}, "
                f"{text.count(old)} matches)",
                file=sys.stderr,
            )
            return None
        text = text.replace(old, new, 1)
    return text


def main() -> int:
    for path in (TARGET_SPEC, TARGET_EAGLE):
        if not path.exists():
            print(f"{LOG} FATAL: {path} not present", file=sys.stderr)
            return 1

    spec_text = TARGET_SPEC.read_text()
    if MARKER in spec_text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    if "_resolve_draft_method" in spec_text:
        print(
            f"{LOG} upstream drift: _resolve_draft_method already present — "
            f"self-retire (no-op)"
        )
        return 0

    new_spec = apply_subs(TARGET_SPEC, SPEC_SUBS, "speculative.py")
    if new_spec is None:
        return 1

    eagle_text = TARGET_EAGLE.read_text()
    if MARKER in eagle_text or "does not look like an EAGLE draft" in eagle_text:
        new_eagle = None  # already patched / upstream landed
    else:
        if EAGLE_OLD not in eagle_text:
            print(
                f"{LOG} FATAL: anchor-not-found (eagle.py:vocab-size-guard) — "
                f"upstream refactor; re-derive before boot",
                file=sys.stderr,
            )
            return 1
        if eagle_text.count(EAGLE_OLD) != 1:
            print(f"{LOG} FATAL: ambiguous anchor (eagle.py)", file=sys.stderr)
            return 1
        new_eagle = eagle_text.replace(EAGLE_OLD, EAGLE_NEW, 1)

    # Both files validated -> write together (no partial application).
    TARGET_SPEC.write_text(new_spec)
    if new_eagle is not None:
        TARGET_EAGLE.write_text(new_eagle)
    print(
        f"{LOG} applied: explicit speculative method can no longer be "
        f"silently overridden (vllm#47490 backport; 5 sub-patches in "
        f"speculative.py + eagle.py vocab_size guard)"
    )
    return 0


sys.exit(main())
