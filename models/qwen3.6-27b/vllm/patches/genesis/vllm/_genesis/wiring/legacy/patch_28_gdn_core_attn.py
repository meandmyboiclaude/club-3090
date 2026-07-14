# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 28 — GDN core_attn_out prealloc (CRIT-HW-1 correct form).

Architecture
------------
Per master-plan CRIT-HW-1 ("P28 MUST pre-allocate at `__init__`, NEVER
lazy in forward"), this module does TWO things:

  1. **Class-method monkey-patch on `GatedDeltaNet.__init__`**. After the
     original __init__ runs, we call `gdn_core_attn_manager.attach_buffer`
     which allocates `self._genesis_gdn_core_attn_buf` (tensor OR None).
     This runs EAGER, once per module, outside any torch.compile trace —
     so device probes, env reads, dict lookups, logging are all safe.

  2. **Text-patch on `forward_cuda`**. The original `torch.zeros(...)`
     line is replaced with a pure-tensor conditional slice:

         core_attn_out = (
             self._genesis_gdn_core_attn_buf[:num_tokens].zero_()
             if self._genesis_gdn_core_attn_buf is not None
             else torch.zeros(
                 (num_tokens, self.num_v_heads // self.tp_size,
                  self.head_v_dim),
                 dtype=hidden_states.dtype, device=hidden_states.device,
             )
         )

     Both branches are pure tensor ops. The `is not None` guard resolves
     at trace time against a constant module attribute — `torch.dynamo`
     compiles only the selected branch and everything stays in-graph.

Platform compatibility
----------------------
  - NVIDIA CUDA SM ≥ 8.0 with the attribute set → pre-allocated slice.
  - All others (attribute is None) → fall-through `torch.zeros`
    identical to upstream behavior.

Upstream drift detection
------------------------
If `_genesis_gdn_core_attn_buf` already appears in the file OR upstream
lands its own buffer-pool fix, we skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p28_gdn_core_attn")

GENESIS_P28_MARKER = "Genesis P28 GDN core_attn_out prealloc v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "_genesis_gdn_core_attn_buf",
    "gdn_core_attn_out_buffer",
    "gdn_core_attn_prealloc",
]


# Anchor: disambiguates from forward_xpu's identical line via the
# preceding "see discussions in https://github.com/vllm-project/vllm/pull/28182"
# comment (unique to forward_cuda).
_OLD_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        core_attn_out = torch.zeros(\n"
    "            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )"
)

_NEW_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        # [Genesis P28] Pre-allocated buffer attached by attach_buffer()\n"
    "        # at module __init__ (see vllm._genesis.kernels.gdn_core_attn_manager).\n"
    "        # Both branches are pure tensor ops — fully torch.dynamo-safe.\n"
    "        core_attn_out = (\n"
    "            self._genesis_gdn_core_attn_buf[:num_tokens].zero_()\n"
    "            if getattr(self, '_genesis_gdn_core_attn_buf', None) is not None\n"
    "            else torch.zeros(\n"
    "                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                dtype=hidden_states.dtype,\n"
    "                device=hidden_states.device,\n"
    "            )\n"
    "        )"
)


# ── Sub-patch 2 (2026-07-14, BUG-072 root fix): IN-SOURCE attach ──────
# The runtime __init__ wrap (_wrap_gdn_init) is a process-local
# monkeypatch: it lives only in the ENTRYPOINT python process, while
# EngineCore spawns fresh and re-imports vllm from the patched FILES —
# so engine-side instances never got _genesis_gdn_core_attn_buf, the
# torch.zeros else-branch ran INSIDE the compiled piecewise region, and
# inductor's static planning lifted it into 17 persistent pow2-padded
# buffers (17 × [8192, 24, 256] fp16 = 1.63 GiB — the BUG-072 residual
# resident list; pow2(4128)=8192). Appending the attach IN-SOURCE at the
# end of __init__ makes it run in every process; the manager registry is
# shape-keyed, so all 17 GDN layers SHARE one right-sized buffer
# (~48 MB at the 4160 budget) and the compiled region allocates nothing.
# Bonus: __init__ runs inside the vLLM config context (the line above
# the anchor uses get_current_vllm_config), so the P73 resolver sees the
# real max_num_batched_tokens. Anchor byte-verified count==1 on the
# installed dev1060 file (2026-07-14).

_INIT_ATTACH_OLD = (
    "        compilation_config = get_current_vllm_config().compilation_config\n"
    "        if prefix in compilation_config.static_forward_context:\n"
    "            raise ValueError(f\"Duplicate layer name: {prefix}\")\n"
    "        compilation_config.static_forward_context[prefix] = self\n"
)

_INIT_ATTACH_NEW = (
    "        compilation_config = get_current_vllm_config().compilation_config\n"
    "        if prefix in compilation_config.static_forward_context:\n"
    "            raise ValueError(f\"Duplicate layer name: {prefix}\")\n"
    "        compilation_config.static_forward_context[prefix] = self\n"
    "        # [Genesis P28] in-source attach — must run in EVERY process\n"
    "        # (EngineCore spawns fresh; a runtime __init__ wrap does not\n"
    "        # survive). Shape-keyed registry => one shared buffer across\n"
    "        # all GDN layers; the forward's prealloc branch then engages\n"
    "        # and the compiled region stops lifting torch.zeros into\n"
    "        # persistent pow2-padded statics (BUG-072).\n"
    "        try:\n"
    "            from vllm._genesis.kernels.gdn_core_attn_manager import (\n"
    "                attach_buffer as _genesis_p28_attach,\n"
    "            )\n"
    "            _genesis_p28_attach(self)\n"
    "        except Exception:\n"
    "            pass  # degraded: forward falls back to eager torch.zeros\n"
)


# ── Sub-patch 3 (2026-07-14, BUG-072 kill-shot): POST-LOAD re-attach ──
# Sub-patch 2 moved the attach in-source into __init__, but __init__
# runs BEFORE weights land on CUDA: _guess_module_device() returns None,
# attach_buffer() bails silently, the buffer stays None on every GDN
# layer, and forward's torch.zeros fallback runs INSIDE the compiled
# piecewise region — inductor lifts it into 17 persistent pow2-padded
# statics (17 × [8192, 24, 256] fp16 = 1.63 GiB; pow2(4128)=8192).
# dev799's measured 0.09 GiB cudagraph pool ≈ exactly ONE shared buffer,
# i.e. the attach engaged there; the dev1060 qwen3_next refactor shifted
# __init__/device-move ordering and the probe now always misses.
# Fix: re-attach right after load_model()'s "Model loading took" log —
# weights are on-device, before any compile/capture — so the shape-keyed
# registry engages: ONE right-sized shared buffer (~51 MB @ budget 4160)
# across all GDN layers, and the compiled region allocates nothing.

GENESIS_P28_RUNNER_MARKER = "Genesis P28 post-load re-attach v7.1"

_RUNNER_ATTACH_OLD = (
    "        logger.info_once(\n"
    "            \"Model loading took %s GiB memory and %.6f seconds\",\n"
    "            format_gib(self.model_memory_usage),\n"
    "            time_after_load - time_before_load,\n"
    "        )\n"
)

_RUNNER_ATTACH_NEW = (
    "        logger.info_once(\n"
    "            \"Model loading took %s GiB memory and %.6f seconds\",\n"
    "            format_gib(self.model_memory_usage),\n"
    "            time_after_load - time_before_load,\n"
    "        )\n"
    "        # [Genesis P28 sub-patch 3, BUG-072] post-load re-attach: the\n"
    "        # __init__-time attach runs before weights land on CUDA (device\n"
    "        # probe -> None), so re-call attach_buffer here where the model\n"
    "        # is fully on-device. Shape-keyed registry => one shared buffer\n"
    "        # across all GDN layers; the compiled forward then uses the\n"
    "        # prealloc branch instead of lifting torch.zeros into per-layer\n"
    "        # pow2-padded statics.\n"
    "        try:\n"
    "            from vllm._genesis.kernels.gdn_core_attn_manager import (\n"
    "                attach_buffer as _genesis_p28_attach,\n"
    "            )\n"
    "            _p28_names = (\n"
    "                \"QwenGatedDeltaNetAttention\",\n"
    "                \"GatedDeltaNetAttention\",\n"
    "                \"GatedDeltaNet\",\n"
    "            )\n"
    "            _p28_live = 0\n"
    "            _p28_roots = [self.model]\n"
    "            _p28_drafter = getattr(\n"
    "                getattr(self, \"drafter\", None), \"model\", None)\n"
    "            if _p28_drafter is not None:\n"
    "                _p28_roots.append(_p28_drafter)\n"
    "            for _p28_root in _p28_roots:\n"
    "                for _p28_mod in _p28_root.modules():\n"
    "                    if type(_p28_mod).__name__ in _p28_names:\n"
    "                        _genesis_p28_attach(_p28_mod)\n"
    "                        if getattr(_p28_mod,\n"
    "                                   \"_genesis_gdn_core_attn_buf\",\n"
    "                                   None) is not None:\n"
    "                            _p28_live += 1\n"
    "            logger.info(\n"
    "                \"[Genesis P28] post-load re-attach: buffer live on \"\n"
    "                \"%d GDN module(s)\", _p28_live)\n"
    "        except Exception as _p28_e:\n"
    "            logger.warning(\n"
    "                \"[Genesis P28] post-load re-attach failed: %s\", _p28_e)\n"
)


def _make_runner_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P28 post-load re-attach",
        target_file=target,
        marker=GENESIS_P28_RUNNER_MARKER,
        sub_patches=[
            TextPatch(
                name="p28_postload_reattach",
                anchor=_RUNNER_ATTACH_OLD,
                replacement=_RUNNER_ATTACH_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["_genesis_p28_attach"],
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P28 GDN core_attn_out prealloc",
        target_file=target,
        marker=GENESIS_P28_MARKER,
        sub_patches=[
            TextPatch(
                name="p28_core_attn_out_alloc",
                anchor=_OLD_ALLOC,
                replacement=_NEW_ALLOC,
                required=True,
            ),
            TextPatch(
                name="p28_init_attach_in_source",
                anchor=_INIT_ATTACH_OLD,
                replacement=_INIT_ATTACH_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


# ─── Runtime init wrap ─────────────────────────────────────────────────────
# Wraps `GatedDeltaNet.__init__` so every new instance gets its buffer
# attached after the original init completes. Idempotent.

_INIT_WRAPPED_ATTR = "_genesis_p28_init_wrapped"


# Candidate class names across vLLM versions. Older baselines named the
# class `GatedDeltaNet`; post-2026-04 renamed to `GatedDeltaNetAttention`
# (to reflect the PluggableLayer / MambaBase mixin). We try both and use
# whichever imports cleanly.
# 1033ffac (dev491): concrete class is QwenGatedDeltaNetAttention(GatedDeltaNetAttention)
# in mamba/gdn/qwen_gdn_linear_attn.py — wrap the concrete subclass first.
_CANDIDATE_CLASS_NAMES = (
    "QwenGatedDeltaNetAttention",
    "GatedDeltaNetAttention",
    "GatedDeltaNet",
)


def _resolve_gdn_class():
    """Import the GDN class, trying known names. Returns class or None."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
        )
    except Exception as e:
        log.info("[Genesis P28] gdn_linear_attn module not importable: %s", e)
        return None
    for name in _CANDIDATE_CLASS_NAMES:
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    log.info(
        "[Genesis P28] none of %s found in gdn_linear_attn "
        "(upstream may have renamed the class — update _CANDIDATE_CLASS_NAMES)",
        list(_CANDIDATE_CLASS_NAMES),
    )
    return None


def _wrap_gdn_init() -> bool:
    """Monkey-patch the GDN class's `__init__`. Return True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False

    if getattr(cls.__init__, _INIT_WRAPPED_ATTR, False):
        return True  # already wrapped (idempotent)

    orig_init = cls.__init__

    from vllm._genesis.kernels.gdn_core_attn_manager import attach_buffer

    def _genesis_wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            attach_buffer(self)
        except Exception as e:
            log.warning(
                "[Genesis P28] attach_buffer in __init__ failed: %s "
                "(module will fall back to eager alloc on first forward)",
                e,
            )
            if not hasattr(self, "_genesis_gdn_core_attn_buf"):
                self._genesis_gdn_core_attn_buf = None

    setattr(_genesis_wrapped_init, _INIT_WRAPPED_ATTR, True)
    setattr(_genesis_wrapped_init, "_genesis_p28_original_init", orig_init)
    cls.__init__ = _genesis_wrapped_init
    log.info(
        "[Genesis P28] wrapped %s.__init__ to attach "
        "_genesis_gdn_core_attn_buf on each instance",
        cls.__name__,
    )
    return True


def is_applied() -> bool:
    """Verify init wrap is live (used by verify_live_rebinds)."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    return getattr(cls.__init__, _INIT_WRAPPED_ATTR, False)


def revert() -> bool:
    """Restore original __init__. Returns True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    cur = cls.__init__
    if not getattr(cur, _INIT_WRAPPED_ATTR, False):
        return False
    orig = getattr(cur, "_genesis_p28_original_init", None)
    if orig is None:
        return False
    cls.__init__ = orig
    return True


def apply() -> tuple[str, str]:
    """Apply P28 wiring: warm-up caches + text-patch forward + wrap __init__.

    Never raises.
    """
    # Step 0: warm up the module-level caches (should_apply, env budget)
    # so traced forward paths never have to do device probes or env reads.
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import warm_up
        warm_up()
    except Exception as e:
        log.info("[Genesis P28] warm_up failed (non-fatal): %s", e)

    # P53 (v7.9): Hybrid-active dispatch gate. GDN attention only exists
    # on hybrid models (Qwen3-Next, Mamba2 variants). On pure-attention
    # models the text-patch anchor won't even match, but skipping early
    # keeps dispatch logs clean.
    try:
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip("P28 GDN core-attn forward rewire", "pure-attention model (no GDN)")
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P28] model_detect probe failed (proceeding): %s", e)

    # Step 1: text-patch forward_cuda
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    # APPLIED or IDEMPOTENT — proceed to init wrap.

    # Step 2: wrap __init__ so new GDN instances get the buffer attached.
    init_ok = _wrap_gdn_init()

    # Step 3 [2026-07-14 BUG-072]: post-load re-attach in gpu_model_runner —
    # the ONLY attach that runs with weights on-device (see sub-patch 3
    # comment). Non-fatal: without it P28 degrades to the eager fallback.
    runner_note = ""
    try:
        runner_patcher = _make_runner_patcher()
        if runner_patcher is None:
            runner_note = "; runner re-attach skipped (file not found)"
            log.warning("[Genesis P28] gpu_model_runner.py not found — "
                        "post-load re-attach NOT installed")
        else:
            r_result, r_failure = runner_patcher.apply()
            if r_result == TextPatchResult.FAILED:
                runner_note = "; runner re-attach FAILED"
                log.warning("[Genesis P28] post-load re-attach text-patch "
                            "failed: %s",
                            r_failure.reason if r_failure else "unknown")
            elif r_result == TextPatchResult.SKIPPED:
                runner_note = "; runner re-attach skipped"
                log.info("[Genesis P28] post-load re-attach skipped: %s",
                         r_failure.reason if r_failure else "unknown")
            else:
                runner_note = "; post-load re-attach installed"
    except Exception as e:
        runner_note = "; runner re-attach errored"
        log.warning("[Genesis P28] post-load re-attach wiring errored: %s", e)

    if result == TextPatchResult.APPLIED:
        reason = "forward_cuda patched + __init__ wrapped" if init_ok \
            else "forward_cuda patched, __init__ wrap skipped"
    else:
        reason = "already applied (idempotent)" if init_ok \
            else "idempotent; init wrap skipped"
    return "applied", reason + runner_note
