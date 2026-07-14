# SPDX-License-Identifier: Apache-2.0
"""PN77 — FP8 lm_head EmbeddingMethod subclass (Phase E.5 architectural redesign).

Replaces the broken Phase E.2-3 design (load_weights post-hook + raw
`nn.Parameter(...)` swap that orphans `weight_loader` callback) with the
canonical vllm extension point:

  1. Duck-typed protocol matching `UnquantizedEmbeddingMethod` interface →
     `Genesis_FP8_LMHead_EmbeddingMethod` (no `class X(UnquantizedEmbeddingMethod):`
     literal inheritance — see line 187+ explanation. Audit A-13 honesty fix
     2026-05-06: docstring corrected from "Subclass" since current vllm pin
     has zero `isinstance(quant_method, UnquantizedEmbeddingMethod)` checks
     in the active code path; protocol matching is sufficient and avoids
     parent's `__init__` semantics on running PROD.)
  2. Override `process_weights_after_loading(layer)` — vllm calls this hook
     AFTER all weight loading, after `tie_weights`, with `device_loading_context`
     already active. Use `replace_parameter()` (vllm's canonical primitive) to
     preserve `weight_loader` attribute through Parameter swap.
  3. Override `apply(layer, x, bias)` — hardware-tier dispatch:
       - Ampere (sm86):  weight-only FP8 via `apply_fp8_marlin_linear` (Marlin)
       - Ada/Hopper/Blackwell (sm89+): native FP8 GEMM via `torch._scaled_mm`
       - Fallback: cast-back to bf16, original GEMM (covers CPU/ROCm/old GPUs)

WHY THIS ARCHITECTURE
======================

Boot failure of Phase E.2-3 (env=1):
   `lm_head.weight = nn.Parameter(weight_fp8)` orphans the `weight_loader`
   callback that `set_weight_attrs(weight, {"weight_loader": ...})` registered
   in `create_weights`. On any post-hook re-touch of that Parameter (e.g.
   the SECOND iteration of `lm_head.weight` shard load through multimodal
   wrapper recursion `Qwen3_5MoeForConditionalGeneration → Qwen3_5ForCausalLMBase`),
   `default_weight_loader` is used instead → no TP-sharding → assertion fail
   `(248320, 5120) → (124160, 5120)`.

`vllm.model_executor.utils.replace_parameter` solves this by COPYING the
old Parameter's `weight_loader` onto the new one. Reference impl:
`Fp8LinearMethod.process_weights_after_loading` (`fp8.py:530`).

HARDWARE TIER DISPATCH
======================

Decision is made ONCE at `process_weights_after_loading` time, cached on
the layer as `_genesis_pn77_path = "marlin" | "scaled_mm" | "cast_back"`.
Per-forward dispatch is a single attribute read.

  Tier A (Ampere sm86 — A5000/3090): `apply_fp8_marlin_linear` requires a
    one-time `prepare_fp8_layer_for_marlin` repack at hook time. After
    repack, `layer.weight` is int32-packed (NOT FP8 dtype). Per-forward
    Marlin kernel dequant'ts back inside the GEMM.

  Tier B (Ada/Hopper sm89+): keep raw FP8 e4m3fn weight + PER-CHANNEL scale.
    Per-forward `torch._scaled_mm` with rowwise scales (scale_a=(M,1),
    scale_b=(1,N)) → bf16/fp16 output; falls back to unit-scale GEMM →
    fp32 → explicit per-column scale multiply on builds without rowwise
    kernels. Native FP8 GEMM, ~1.3-2× FLOPs over BF16.

    ORDERING-FIX 2026-07-14: this tier previously collapsed the per-channel
    scale to per-tensor via `weight_scale.amax()`. Since compress() divides
    each vocab ROW by its own scale, dequantizing with the max scale
    multiplied every logit by `smax/s_i` — measured p50 5.5×, max 12.3× on
    Qwopus3.6-27B — a per-row distortion that reorders near-tie logits.
    Under an xgrammar bitmask the model cannot take recovery tokens, so a
    spuriously-boosted digit repeats until max_tokens (boot-14 canary:
    11/12 grammar rounds degenerate with PN77=1, 12/12 clean with PN77=0).
    Offline A/B on the real lm_head (248320×5120): grammar-masked argmax
    agreement vs BF16 ref 0.55 → 0.99; kendall-tau@64 0.30 → 0.93.

  Fallback: cast weight FP8→BF16 per call, original GEMM. ~3 ms per call on
    248K vocab × 5120 hidden. Acceptable for sampling step (one matmul/token).

DRIFT-RESISTANCE
=================

When upstream lands PR #41000 (config-driven `lm_head_quantized: true` in
`Fp8Config`), the wiring text-patch detects `lm_head_quantized` marker in
`fp8.py` source and self-retires. Genesis takes a back seat to upstream.

Author: Sandermage (Sander) Barzov Aleksandr — Ukraine, Odessa.
References:
  - vllm PR #35696 (lucaspirola, OPEN) — naive load_weights hook (mirrored in
    Phase E.2 design, broken due to weight_loader orphan)
  - vllm PR #41000 (webcodes-cz, OPEN) — config-driven Fp8Config dispatch
    (the architecture upstream is converging on)
  - `Fp8LinearMethod.process_weights_after_loading` (fp8.py:530) — REFERENCE
    implementation of the `replace_parameter` + Marlin tier dispatch pattern
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.kernels.lm_head_fp8_method")

# Marker attribute: set on layer once compression is done.
PN77_APPLIED_MARKER = "_already_called_process_weights_after_loading"
PN77_PATH_ATTR = "_genesis_pn77_path"  # "marlin" | "scaled_mm" | "cast_back"

ENV_FLAG = "GENESIS_ENABLE_PN77_FP8_LM_HEAD"

# ─── Ordering-fidelity guard (2026-07-14) ─────────────────────────────
# Per-channel e4m3 quantization keeps SCALE right, but rows with strong
# WITHIN-row outliers still starve the small elements: the row scale is
# set by the outlier, the rest of the row lands in e4m3's denormal range
# (or flushes to zero outright) and the row DIRECTION — what logit
# ordering depends on — is destroyed. Two-component guard on a row
# sample at swap time; refuse the swap (clean skip, BF16 stays) when
# either trips:
#
#   1. roundtrip relative L2 error p99 > bound — catches uniformly noisy
#      rows. NOTE this metric alone is blind to outlier-starved rows: the
#      outlier dominates both the norm and the error, capping rel-L2 near
#      e4m3's own ~3% while the rest of the row is zeroed (verified
#      empirically on a synthetic outlier head). Hence component 2.
#   2. zero-flush fraction p99 > bound — fraction of nonzero elements per
#      row whose quantized value flushes to exactly 0. Directly detects
#      exponent/mantissa starvation regardless of norm domination.
#
# Healthy heads sit far below both — Qwopus3.6-27B measures rel-L2
# p50=0.026 / p99=0.027 / max=0.028 and ~0 flush (bf16 rows don't span
# the ~6 orders of intra-row magnitude needed to flush e4m3 denormals).
PN77_RELERR_BOUND_ENV = "GENESIS_PN77_MAX_ROW_RELERR"
PN77_RELERR_BOUND_DEFAULT = 0.08
PN77_FLUSH_BOUND = 0.5     # p99 of per-row zero-flush fraction
PN77_RELERR_SAMPLE_ROWS = 4096

# Data-ptrs of non-LMHead vocab-embedding weights seen by the PWAL walker.
# `ParallelLMHead.tie_weights` shares the SAME Parameter object with
# embed_tokens (`layer.weight = embed_tokens.weight`), so a data_ptr match
# identifies a tied head. named_modules() walks in registration (pre-)order
# and embed_tokens registers before lm_head on every supported model, so
# the registry is populated before the head is visited. Best-effort: if a
# future model registers the head first, the guard misses and the swap
# proceeds — that case is only a VRAM no-win (embed keeps its own BF16
# Parameter; nothing is corrupted).
_SEEN_EMBEDDING_WEIGHT_PTRS: set[int] = set()


def _is_enabled() -> bool:
    """Read env once; opt-in default OFF."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _detect_hardware_tier() -> str:
    """Return tier identifier: 'marlin' | 'scaled_mm' | 'cast_back'.

    Decision matrix:
      sm86 (Ampere consumer/A5000) → marlin (weight-only FP8)
      sm80 (Ampere datacenter A100) → marlin (same — sm80+ supported)
      sm89/90 (Ada/Hopper) → scaled_mm (native FP8 GEMM via torch._scaled_mm)
      sm100+ (Blackwell) → scaled_mm (native; FP4-accumulator may come later)
      else → cast_back fallback
    """
    if not torch.cuda.is_available():
        return "cast_back"
    cap = torch.cuda.get_device_capability()
    if cap is None:
        return "cast_back"
    major, minor = cap[0], cap[1]
    sm = major * 10 + minor
    if sm >= 89:
        return "scaled_mm"  # Ada/Hopper/Blackwell — native FP8
    if sm >= 80:
        return "marlin"  # Ampere — weight-only FP8 via Marlin
    return "cast_back"


def _is_lm_head(layer) -> bool:
    """Detect ParallelLMHead vs (regular) VocabParallelEmbedding.

    ParallelLMHead inherits VocabParallelEmbedding; the ONLY runtime
    difference is the .apply() call site (LogitsProcessor) vs .embedding().
    Class-name match is the cheapest reliable signal.
    """
    cls_name = type(layer).__name__
    return cls_name == "ParallelLMHead" or cls_name.endswith("LMHead")


def maybe_swap_pn77_quant_method(layer, current_method):
    """Hook invoked from text-patched `process_weights_after_loading` walker.

    Swaps `layer.quant_method` to `Genesis_FP8_LMHead_EmbeddingMethod` if:
      - env GENESIS_ENABLE_PN77_FP8_LM_HEAD=1
      - `layer` is a ParallelLMHead (not regular embed_tokens)
      - hardware supports a useful FP8 path (not 'cast_back' fallback only)
      - current method is `UnquantizedEmbeddingMethod` (don't override
        if a real quant config already chose another method)

    Returns the method that should actually be used (swapped or original).
    NEVER raises — fallback to original on any failure.
    """
    try:
        if not _is_enabled():
            return current_method
        if not _is_lm_head(layer):
            # Record vocab-embedding weight ptrs for the tied-head guard
            # below. Cheap: one set-insert per embedding module per load.
            if type(layer).__name__ == "VocabParallelEmbedding":
                w = getattr(layer, "weight", None)
                if w is not None:
                    _SEEN_EMBEDDING_WEIGHT_PTRS.add(w.data_ptr())
            return current_method
        # Only swap pristine UnquantizedEmbeddingMethod (don't override real quant)
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )
        if not isinstance(current_method, UnquantizedEmbeddingMethod):
            return current_method
        # Idempotency
        if isinstance(current_method, Genesis_FP8_LMHead_EmbeddingMethod):
            return current_method
        # Skip if weight already FP8 (native checkpoint)
        weight = getattr(layer, "weight", None)
        if weight is not None and weight.dtype == torch.float8_e4m3fn:
            return current_method
        # Tied-embeddings guard (2026-07-14): tie_weights shares the SAME
        # Parameter object between lm_head and embed_tokens. Compressing a
        # tied head rebinds lm_head.weight to a new FP8 Parameter while
        # embed_tokens keeps the BF16 one — zero VRAM win, plus the model
        # now decodes through a quantized head it never needed. Skip loudly.
        if weight is not None and weight.data_ptr() in _SEEN_EMBEDDING_WEIGHT_PTRS:
            log.warning(
                "[PN77] SKIP: lm_head weight is TIED to embed_tokens "
                "(shared Parameter, data_ptr match) — FP8 swap would save "
                "no VRAM and only add quantization error. Keeping BF16. "
                "(tie_word_embeddings checkpoints are out of PN77 scope.)"
            )
            return current_method
        # Skip on cast_back-only hardware (no real win beyond fallback path)
        # Actually — cast-back still SAVES VRAM, so include it. Just slower per call.
        new_method = Genesis_FP8_LMHead_EmbeddingMethod()
        layer.quant_method = new_method
        log.info(
            "[PN77] swapped lm_head.quant_method UnquantizedEmbeddingMethod → "
            "Genesis_FP8_LMHead_EmbeddingMethod (tier=%s, weight=%s)",
            _detect_hardware_tier(),
            tuple(weight.shape) if weight is not None else "?",
        )
        return new_method
    except Exception as e:
        log.warning(
            "[PN77] swap helper failed (%s) — keeping original method",
            type(e).__name__,
        )
        return current_method


class Genesis_FP8_LMHead_EmbeddingMethod:
    """FP8 lm_head method — drop-in replacement for `UnquantizedEmbeddingMethod`.

    Pattern mirrors `Fp8LinearMethod.process_weights_after_loading` from
    `vllm/model_executor/layers/quantization/fp8.py:530`. The important
    invariant: use `replace_parameter()` (vllm's canonical helper) which
    PRESERVES the `weight_loader` attribute on the new Parameter, so
    subsequent re-loads (multimodal wrapper recursion, MTP head sync, etc.)
    continue to TP-shard correctly.

    INHERITANCE NOTE: we do NOT inherit from `UnquantizedEmbeddingMethod`
    directly to avoid CPU-path coupling in `process_weights_after_loading`.
    Instead we duplicate the small `create_weights` and `embedding` methods
    (these are the surface vllm reads). The quant_method protocol is duck-typed.
    """

    # ─── Init / shape setup ────────────────────────────────────────────

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list,
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Identical to UnquantizedEmbeddingMethod.create_weights — load BF16
        weights first; we only convert to FP8 in process_weights_after_loading."""
        from vllm.model_executor.utils import set_weight_attrs

        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # ─── Post-load mutation: BF16 → FP8 conversion ────────────────────

    def process_weights_after_loading(self, layer) -> None:
        """vllm canonical hook. Fires AFTER all weight load + tie_weights done.

        Uses `replace_parameter()` to swap weight Parameter while preserving
        the `weight_loader` callback (proven pattern from Fp8LinearMethod).

        Sets `output_size_per_partition`/`input_size_per_partition`/`orig_dtype`
        attrs that `prepare_fp8_layer_for_marlin` requires (these come from
        ColumnParallelLinear naturally but ParallelLMHead uses different names
        like `num_embeddings_per_partition` / `embedding_dim`).
        """
        if getattr(layer, PN77_APPLIED_MARKER, False):
            return  # Idempotent

        try:
            from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import compress
            from vllm.model_executor.utils import replace_parameter
        except Exception as e:
            log.warning(
                "[PN77] import failed (%s) — keeping BF16 lm_head", type(e).__name__,
            )
            return

        weight = layer.weight
        if weight.dtype == torch.float8_e4m3fn:
            log.info("[PN77] lm_head already FP8 — skipping compression")
            setattr(layer, PN77_APPLIED_MARKER, True)
            return

        # Save ORIGINAL dtype before compress — Marlin's prep needs `layer.orig_dtype`
        # to cast scales back. ParallelLMHead doesn't have it natively (set by us).
        orig_dtype = weight.dtype

        # Compress: BF16/FP16 → FP8 e4m3fn + per-channel scale
        try:
            weight_fp8, scale = compress(weight.data)
        except Exception as e:
            log.warning(
                "[PN77] compress() failed (%s) — keeping BF16 lm_head",
                type(e).__name__,
            )
            return

        # Ordering-fidelity guard (2026-07-14): refuse the swap when the
        # per-channel scheme itself cannot hold logit ordering (within-row
        # outliers → mantissa starvation). Loud, precise, clean skip — the
        # silent-regression class is exactly what we are eliminating.
        try:
            relerr_bound = float(
                os.environ.get(PN77_RELERR_BOUND_ENV, "") or PN77_RELERR_BOUND_DEFAULT
            )
        except ValueError:
            relerr_bound = PN77_RELERR_BOUND_DEFAULT
        try:
            n_rows = weight.shape[0]
            idx = torch.randint(
                0, n_rows,
                (min(PN77_RELERR_SAMPLE_ROWS, n_rows),),
                device=weight.device,
            )
            w_ref = weight.data[idx].float()
            w_deq = weight_fp8[idx].float() * scale[idx].unsqueeze(1)
            rel = (w_deq - w_ref).norm(dim=1) / w_ref.norm(dim=1).clamp(min=1e-20)
            rel_p99 = torch.quantile(rel, 0.99).item()
            rel_max = rel.max().item()
            nonzero = w_ref.abs() > 0
            flushed = (w_deq == 0) & nonzero
            flush_frac = flushed.sum(dim=1).float() / nonzero.sum(dim=1).clamp(min=1)
            flush_p99 = torch.quantile(flush_frac, 0.99).item()
            flush_max = flush_frac.max().item()
            del w_ref, w_deq, rel, idx, nonzero, flushed, flush_frac
        except Exception as e:
            log.warning(
                "[PN77] fidelity guard errored (%s: %s) — refusing swap, "
                "keeping BF16 lm_head (fail-closed)",
                type(e).__name__, str(e)[:120],
            )
            del weight_fp8, scale
            return
        if rel_p99 > relerr_bound or flush_p99 > PN77_FLUSH_BOUND:
            log.warning(
                "[PN77] SKIP: FP8 lm_head swap REFUSED by ordering-fidelity "
                "guard — per-row roundtrip rel-L2 p99=%.4f (max=%.4f, bound "
                "%.4f via %s) / zero-flush fraction p99=%.4f (max=%.4f, "
                "bound %.2f). This checkpoint's head has rows the per-channel "
                "e4m3 scheme cannot represent without reordering near-tie "
                "logits (degenerate loops under grammar masks). Keeping BF16 "
                "lm_head; no VRAM saving on this model.",
                rel_p99, rel_max, relerr_bound, PN77_RELERR_BOUND_ENV,
                flush_p99, flush_max, PN77_FLUSH_BOUND,
            )
            del weight_fp8, scale
            return
        log.info(
            "[PN77] fidelity guard PASS: per-row roundtrip rel-L2 p99=%.4f "
            "max=%.4f (bound %.4f); zero-flush p99=%.4f (bound %.2f)",
            rel_p99, rel_max, relerr_bound, flush_p99, PN77_FLUSH_BOUND,
        )

        # Tier dispatch — decide path ONCE, cache on layer
        tier = _detect_hardware_tier()
        setattr(layer, PN77_PATH_ATTR, tier)

        # Replace weight Parameter — preserves weight_loader via replace_parameter
        try:
            replace_parameter(layer, "weight", weight_fp8)
            # Also register scale as Parameter (for symmetric reload behavior).
            # Use replace_parameter even though `weight_scale` doesn't pre-exist —
            # it handles the missing-old-param case (just creates new).
            scale_param = torch.nn.Parameter(scale, requires_grad=False)
            layer.register_parameter("weight_scale", scale_param)
        except Exception as e:
            log.error(
                "[PN77] Parameter replacement failed (%s) — model state may be "
                "inconsistent; recommend container restart with env=0",
                type(e).__name__,
            )
            return

        # Set Marlin-required attrs (ParallelLMHead lacks ColumnParallelLinear's
        # naming convention). These attrs are what prepare_fp8_layer_for_marlin
        # reads:
        #   - output_size_per_partition: vocab_per_rank (N dim of GEMM)
        #   - input_size_per_partition: hidden_size (K dim)
        #   - orig_dtype: BF16/FP16 dtype to cast scales back into
        # ParallelLMHead.weight has shape (n, k) = (vocab_per_rank, hidden) —
        # this is `size_k_first=False` layout for Marlin prep.
        if not hasattr(layer, "output_size_per_partition"):
            layer.output_size_per_partition = getattr(
                layer, "num_embeddings_per_partition", weight_fp8.shape[0]
            )
        if not hasattr(layer, "input_size_per_partition"):
            layer.input_size_per_partition = getattr(
                layer, "embedding_dim", weight_fp8.shape[1]
            )
        if not hasattr(layer, "orig_dtype"):
            layer.orig_dtype = orig_dtype

        # Tier-specific post-processing
        if tier == "marlin":
            try:
                self._prepare_marlin(layer)
            except Exception as e:
                # Capture FULL exception detail for diagnosis (was just type name)
                import traceback
                log.warning(
                    "[PN77] Marlin prepare failed (%s: %s) — falling back to "
                    "cast_back tier. Full trace:\n%s",
                    type(e).__name__, str(e)[:200],
                    "".join(traceback.format_exception(type(e), e, e.__traceback__))[:1500],
                )
                setattr(layer, PN77_PATH_ATTR, "cast_back")

        setattr(layer, PN77_APPLIED_MARKER, True)
        log.info(
            "[PN77] lm_head compressed BF16→FP8: shape=%s, tier=%s, "
            "saved ~%.0f MiB/rank",
            tuple(weight_fp8.shape),
            getattr(layer, PN77_PATH_ATTR, "cast_back"),
            weight_fp8.numel() / (1024 * 1024),  # FP8=1byte
        )

        # VRAM-aware cleanup (2026-05-07): on 27B (large vocab×hidden) the
        # compress + Marlin repack flow creates large intermediate tensors
        # (BF16 → FP8 → packed int32 → marlin format) that the PyTorch caching
        # allocator may keep in non-split blocks until next empty_cache. vllm's
        # own empty_cache happens later in capture_model, but in our 27B+TQ
        # k8v4 measurement that wasn't enough — VRAM stayed +2 GB above
        # baseline. Explicit cleanup HERE forces the freed BF16/FP32/FP8/int32
        # intermediates back to OS BEFORE the rest of vllm load proceeds.
        # Cost: ~10-100 ms one-shot (per-layer hook call). Win: real VRAM save.
        del weight_fp8, scale, weight
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # cleanup is best-effort, never fail the patch

    def _prepare_marlin(self, layer):
        """One-time Marlin repack for Ampere weight-only FP8 path.

        Use `size_k_first=False` because ParallelLMHead.weight has the natural
        nn.Linear layout `(out, in) = (n, k) = (vocab_per_rank, hidden)`,
        opposite to FP8LinearMethod's intermediate which gets transposed first.
        """
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)

    # ─── Forward dispatch ──────────────────────────────────────────────

    def apply(self, layer, x, bias=None):
        """Forward pass — dispatch by tier flag set in process_weights_after_loading."""
        # If process_weights_after_loading hasn't run (e.g. env disabled mid-flight),
        # treat as plain UnquantizedEmbeddingMethod.
        if not getattr(layer, PN77_APPLIED_MARKER, False):
            return self._unquant_apply(layer, x, bias)

        tier = getattr(layer, PN77_PATH_ATTR, "cast_back")
        if tier == "marlin":
            return self._apply_marlin(layer, x, bias)
        if tier == "scaled_mm":
            return self._apply_scaled_mm(layer, x, bias)
        return self._apply_cast_back(layer, x, bias)

    # ─── Tier-specific apply implementations ──────────────────────────

    def _unquant_apply(self, layer, x, bias):
        """Bypass — used when marker not set (env disabled or pre-PN77 state)."""
        import vllm.envs as envs
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
        from vllm.platforms import current_platform

        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            from vllm.model_executor.layers.batch_invariant import (
                linear_batch_invariant,
            )
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def _apply_marlin(self, layer, x, bias):
        """Ampere weight-only FP8 via Marlin.

        Use persistent `output_size_per_partition`/`input_size_per_partition`
        attrs we set in `process_weights_after_loading` — `layer.weight.shape`
        is no longer valid after Marlin repacked it to int32-packed format.
        """
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    def _apply_scaled_mm(self, layer, x, bias):
        """Ada/Hopper/Blackwell native FP8 GEMM — PER-CHANNEL weight dequant.

        [Genesis PN77 ORDERING-FIX 2026-07-14] The previous implementation
        collapsed the per-channel `weight_scale` (one scale per vocab row,
        produced by compress()) into a single per-tensor scale via
        `weight_scale.amax()`. That is not "lossy but acceptable": since each
        row was DIVIDED by its own scale at quantize time, dequantizing all
        rows with max(s) multiplies logit_i by `max(s)/s_i` — a per-row
        factor measured at p50 5.5× / max 12.3× on Qwopus3.6-27B's lm_head.
        Absolute-magnitude-favoured tokens flip near-ties; under a grammar
        bitmask the flip is unrecoverable and decoding degenerates into
        digit loops (boot-14 canary 11/12 failures). Fix: keep the scales
        per-channel end-to-end.

        Primary path: rowwise `torch._scaled_mm` (scale_a shape (M,1),
        scale_b shape (1,N), both fp32 contiguous) — verified supported on
        sm89 / torch 2.11 (rel-err vs BF16 ref ≈ FP8 noise floor).
        Fallback (older torch / no rowwise kernel): unit-scale GEMM with
        fp32 output, then an explicit per-column multiply by
        `x_scale * weight_scale` — mathematically identical, one extra
        (M, N) fp32 elementwise pass which is negligible next to the GEMM.
        The sub-path is probed ONCE and cached on the layer.

        Bias is applied AFTER dequantization in both sub-paths (adding it
        inside the GEMM epilogue would place it before the per-channel
        multiply in the fallback). lm_head bias is None in practice.
        """
        # Cast x to FP8 with its own per-tensor scale (1-D, never 0-D — see
        # the vllm#44912 scale-rank note: Inductor rejects 0-D scales).
        x_amax = x.abs().amax().clamp(min=1e-12)
        x_scale = (x_amax / 448.0).to(torch.float32).reshape(1)
        x_fp8 = (x / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # Per-channel scales as a (1, N) fp32 row vector. weight_scale is a
        # (N,) fp32 Parameter registered by process_weights_after_loading;
        # .view(1, -1) on a contiguous 1-D tensor is zero-copy + contiguous.
        w_scale_row = layer.weight_scale.to(torch.float32).view(1, -1)

        sub_path = getattr(layer, "_genesis_pn77_scaled_mm_sub_path", None)
        if sub_path != "unit_scale":
            try:
                out = torch._scaled_mm(
                    x_fp8,
                    layer.weight.t(),
                    scale_a=x_scale.expand(x_fp8.shape[0], 1).contiguous(),
                    scale_b=w_scale_row.contiguous(),
                    out_dtype=x.dtype,
                )
                if sub_path is None:
                    layer._genesis_pn77_scaled_mm_sub_path = "rowwise"
                    log.info("[PN77] scaled_mm tier: rowwise per-channel path active")
                if bias is not None:
                    out = out + bias
                return out
            except Exception as e:
                if sub_path == "rowwise":
                    raise  # probed-good path failing later is a real error
                layer._genesis_pn77_scaled_mm_sub_path = "unit_scale"
                log.warning(
                    "[PN77] rowwise _scaled_mm unavailable (%s: %s) — using "
                    "unit-scale GEMM + explicit per-channel multiply",
                    type(e).__name__, str(e)[:120],
                )

        # Fallback: unit-scale FP8 GEMM → fp32 raw products, then exact
        # per-channel dequant: logits = raw * x_scale * weight_scale[col].
        unit = getattr(layer, "_genesis_pn77_unit_scale", None)
        if unit is None or unit.device != x.device:
            unit = torch.ones(1, dtype=torch.float32, device=x.device)
            layer._genesis_pn77_unit_scale = unit
        raw = torch._scaled_mm(
            x_fp8,
            layer.weight.t(),
            scale_a=unit,
            scale_b=unit,
            out_dtype=torch.float32,
        )
        out = raw.mul_(w_scale_row).mul_(x_scale).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def _apply_cast_back(self, layer, x, bias):
        """Fallback — decompress to x.dtype, normal GEMM."""
        from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import decompress
        weight = decompress(layer.weight, layer.weight_scale, output_dtype=x.dtype)
        import vllm.envs as envs
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
        from vllm.platforms import current_platform

        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            from vllm.model_executor.layers.batch_invariant import (
                linear_batch_invariant,
            )
            return linear_batch_invariant(x, weight, bias)
        # Build temporary "layer" facade with decompressed weight for dispatch
        # (dispatch_unquantized_gemm may read other layer attrs).
        return dispatch_unquantized_gemm()(layer, x, weight, bias)

    # ─── Embedding path (unchanged from Unquant) ──────────────────────

    def embedding(self, layer, input_):
        """Embedding lookup — for embed_tokens path only.

        ParallelLMHead generally doesn't go through this (embed_tokens does),
        but we provide it for protocol completeness. Decompress if FP8.
        """
        import torch.nn.functional as F
        if layer.weight.dtype == torch.float8_e4m3fn:
            from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import decompress
            weight = decompress(
                layer.weight, layer.weight_scale, output_dtype=torch.bfloat16,
            )
            return F.embedding(input_, weight)
        return F.embedding(input_, layer.weight)
