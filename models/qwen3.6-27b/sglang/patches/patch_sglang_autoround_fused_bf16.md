# patch_sglang_autoround_fused_bf16.py

Local vendor patch for `lmsysorg/sglang:v0.5.12`.

## Problem

`Lorbus/Qwen3.6-27B-int4-AutoRound` keeps the Qwen3-Next DeltaNet
`linear_attn.in_proj_b` and `linear_attn.in_proj_a` tensors in FP/BF16 via
AutoRound `extra_config` entries:

```text
model.language_model.layers.N.linear_attn.in_proj_b -> bits: 16
model.language_model.layers.N.linear_attn.in_proj_a -> bits: 16
```

SGLang represents those two checkpoint tensors as one fused module,
`linear_attn.in_proj_ba`. In v0.5.12, `AutoRoundConfig.from_config()` drops the
model's `packed_modules_mapping`, so `get_layer_config()` does not know that
`in_proj_ba` should consult the split `in_proj_b` / `in_proj_a` configs.

Result: SGLang incorrectly builds `in_proj_ba` as GPTQ-Marlin INT4. The layer has
`size_n = 2 * linear_num_value_heads = 96`, which is not divisible by Marlin's
`tile_n_size = 64`, so post-load repacking crashes:

```text
gptq_marlin_repack.cuh:309: size_n = 96 is not divisible by tile_n_size = 64
```

The repeated `Parameter model.layers.*.linear_attn.in_proj_ba.weight not found in
params_dict` warnings are the same bug showing up earlier: the checkpoint has
BF16 split weights to load, but the model was built with quantized `qweight`
parameters instead of a fused BF16 `weight` parameter.

## Fix

The startup patch edits `sglang/srt/layers/quantization/auto_round.py` in place:

- preserves `packed_modules_mapping` passed by SGLang's model loader;
- lets fused AutoRound modules consult split-shard `extra_config` entries;
- treats `model.language_model.*` and SGLang's internal `model.*` names as
  aliases.

With the patch, `in_proj_ba` sees both split shards as `bits: 16`, receives
`UnquantizedLinearMethod`, and loads the BF16 fused weight through the existing
Qwen3.5 packed weight loader. No kernel rebuild and no weight rewrite are
required.

## Scope

This patch targets the AutoRound GPTQ path. The alternate AWQ checkpoint uses
SGLang's `compressed-tensors` loader, which already preserves
`packed_modules_mapping` and calls the shared fused-layer ignore helper.

## Validation

Static validation (no GPU needed):

```bash
docker run --rm \
  -v <repo>/models/qwen3.6-27b/sglang/patches/patch_sglang_autoround_fused_bf16.py:/patch.py:ro \
  --entrypoint /usr/bin/python3 \
  lmsysorg/sglang:v0.5.12 \
  /patch.py
```

Expected output includes:

```text
AutoRoundConfig keeps packed_modules_mapping
fused layers consult split-shard extra_config before quantizing
```

## Live validation result (2026-05-20)

The Marlin failure class is **fixed** by this patch. Confirmed across multiple boots on a single RTX 3090 (Ampere sm_86, driver 595.71.05, CUDA 13.2):

- No `Parameter ... linear_attn.in_proj_ba.weight not found in params_dict` warnings.
- No `gptq_marlin_repack.cuh:309: size_n = 96 is not divisible by tile_n_size = 64` assertion.
- Target loads cleanly: `target loaded: Qwen3_5ForConditionalGeneration, quant=auto-round, bits=4`.
- Mamba cache + KV cache allocate successfully.

## Operational caveats (must-know for anyone running this)

These are independent of the Marlin fix — they apply regardless of this patch but are needed for the full EAGLE-3 stack to actually serve:

1. **BF16 EAGLE-3 drafter must opt out of target's INT4 quantization.** Without this flag, the drafter (a small BF16 model) silently inherits `--quantization auto-round` and fails to load:

   ```yaml
   --speculative-draft-model-quantization unquant
   ```

2. **Single-3090 EAGLE-3 is genuinely VRAM-tight on 24 GB.** Target (~18 GB after quant) + BF16 drafter (~4 GB) + Mamba state + KV cache leaves ~0–2 GB headroom for cuda-graph capture. Workable knob ranges from 2026-05-20 exploration on a 24 GB single card:

   ```yaml
   --kv-cache-dtype fp8_e5m2          # ~50% KV savings vs default BF16
   --max-running-requests 1           # smallest valid graph-capture surface
   --max-mamba-cache-size 1
   --max-total-tokens 8192
   --disable-radix-cache
   --mamba-scheduler-strategy no_buffer
   --disable-cuda-graph
   environment:
     PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
   ```

3. **`--cpu-offload-gb 2` on single 3090 is currently incompatible with Qwen3-Next.** SGLang's OffloaderV1 hits `ValueError: functional_call got multiple values for keys ['linear_attn.attn.dt_bias', 'linear_attn.dt_bias'], which are tied. Consider using tie_weights=False` on first forward. Avoid CPU offload until SGLang's offloader handles tied DeltaNet params.

4. **Dual GPU is the cleaner full-stack path.** Splitting the target across 2× 3090 (TP=2) eliminates the need for CPU offload entirely; drafter resides on rank 0 only, KV splits across cards. See `models/qwen3.6-27b/sglang/compose/dual/autoround-int4/eagle3-experimental.yml`.

## Implementation notes

The fix is Python-level (no kernel rebuild) and works as a runtime sidecar applied at container startup. Pinned to `lmsysorg/sglang:v0.5.12` per `AGENTS.md` engine-image-pinning policy. The patch is idempotent: re-running it on an already-patched container is a no-op.
