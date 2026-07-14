#!/usr/bin/env python3
"""TurboQuant 3-bit KV store/load bit-parity harness (tq-lane).

Validates TurboQuant quantized-KV numerics of the pinned vLLM image
(nightly-9e57de7197f234f9d9187715d96e07e007048c0f, dev1060) against an
independent pure-torch reference implementation of the TQ packing spec.

Upstream ships NO python/torch reference path for TQ (checked: the tq module
is config.py + centroids.py + the two Triton kernel files). The reference
quantize/dequantize below is therefore implemented from the packing spec in
  vllm/v1/attention/ops/triton_turboquant_store.py   (_tq_fused_store_mse)
  vllm/v1/attention/ops/triton_turboquant_decode.py  (_tq_full_dequant_kv)
and is written op-for-op so that the packed BYTES match the kernel exactly
(same fp32 arithmetic, same truncation semantics, same bit layout).

Shapes are hardcoded from the served model's config.json
(XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-MTP-BF16, text_config):
    num_attention_heads = 24
    num_key_value_heads = 4        (GQA group 6)
    head_dim            = 256
    layer_types: GDN hybrid — full_attention every 4th layer (16 of 64).
    Only those full-attention layers have a KV cache; the GDN/linear layers
    carry conv/recurrent state instead. The harness therefore models a
    single full-attention layer, which is exactly what the TQ backend sees.

Presets covered: turboquant_3bit_nc (live on :8020) and turboquant_k3v4_nc.
FP8-key presets are skipped (not served; different storage path).

Checks (each prints PASS/FAIL + max abs/rel error):
  1. store-bytes parity      — Triton store bytes == reference quantizer bytes
  2. roundtrip error bounds  — store -> Triton dequant vs original K/V
                               (values: exact per-element bound scale/2;
                                keys: 3-bit Lloyd-Max distortion envelope)
  3. dequant parity          — Triton _tq_full_dequant_kv vs reference dequant
                               on the same cache bytes (fp16-rounding exact)
  4. determinism             — two independent store+dequant+decode runs are
                               bitwise identical
  5. continuation (q<=128)   — TQ decode-kernel continuation path vs reference
  6. continuation (q>128)    — _continuation_prefill (dequant+flash) path vs
                               reference; plus the vllm#43357 workspace-lock
                               repro (informational: reports FIXED when the
                               PN95 pool is active, REPRODUCED on stock)
  7. mixed batch (PN86/#46461) — one long first-chunk prefill owning both
                               batch maxima + one continuation request;
                               EXPECTED TO FAIL on the stock image (fast path
                               drops the continuation's cached prefix) and
                               PASS once PN86 is applied.

Run with pytest or directly (python3 tq_parity_harness.py). Requires GPU +
the pinned image's vllm tree. `--selfcheck` runs a CPU-only reference
round-trip sanity pass (no GPU, no Triton).

VRAM budget: shapes kept small (<= 2K tokens, 64 blocks); peak allocated is
printed at the end and stays well under 1 GB. Uses its own CUDA context —
safe to run alongside the live server as long as the card has ~1 GB free.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import torch

# ----------------------------------------------------------------------------
# Served-model geometry (hardcoded from config.json — see module docstring)
# ----------------------------------------------------------------------------
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5
BLOCK_SIZE = 64  # in backend get_supported_kernel_block_sizes()
NUM_BLOCKS = 72  # 72*64 = 4608 slots — covers every check below

PRESETS = ("turboquant_3bit_nc", "turboquant_k3v4_nc")

SEED = 20260713

_RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str) -> None:
    _RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _err_stats(a: torch.Tensor, b: torch.Tensor) -> str:
    a32, b32 = a.float(), b.float()
    abs_err = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-6)
    return (
        f"max_abs={abs_err.max().item():.3e} "
        f"max_rel={(abs_err / denom).max().item():.3e}"
    )


# ----------------------------------------------------------------------------
# Reference implementation of the TQ packing spec (pure torch)
# ----------------------------------------------------------------------------
def _fp16_to_bytes(t: torch.Tensor) -> torch.Tensor:
    """fp16 tensor (any shape) -> uint8 little-endian byte pairs, last dim *2."""
    return t.half().contiguous().view(torch.uint8)


def _bytes_to_fp16(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Two uint8 tensors (LE lo/hi) -> fp16 values (as fp32)."""
    raw = lo.to(torch.int32) | (hi.to(torch.int32) << 8)
    return ((raw + 32768) % 65536 - 32768).to(torch.int16).view(torch.float16).float()


def _pack_3bit(idx: torch.Tensor) -> torch.Tensor:
    """(N, D) int32 in [0,7] -> (N, D//8*3) uint8; element j of each group of 8
    occupies bits [3j, 3j+3) of a 24-bit LE word (matches _tq_fused_store_mse)."""
    N, D = idx.shape
    grp = idx.reshape(N, D // 8, 8)
    shifts = torch.arange(8, device=idx.device, dtype=torch.int32) * 3
    packed24 = (grp << shifts).sum(dim=2)  # (N, D//8) int32
    b0 = (packed24 & 0xFF).to(torch.uint8)
    b1 = ((packed24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed24 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=2).reshape(N, -1)


def _unpack_3bit(data: torch.Tensor, D: int) -> torch.Tensor:
    """(N, bytes) uint8 -> (N, D) int32 (bit layout as _tq_full_dequant_kv:
    element d reads the LE 16-bit window at byte d*3//8, shift d*3%8)."""
    N = data.shape[0]
    d_offs = torch.arange(D, device=data.device, dtype=torch.long)
    bit_off = d_offs * 3
    byte_idx = bit_off // 8
    shift = (bit_off % 8).to(torch.int32)
    lo = data[:, byte_idx].to(torch.int32)
    # kernel loads byte_idx+1 unconditionally; last element's +1 stays inside
    # the slot (norm/scale bytes follow) — for the ref we pad one zero byte,
    # which only matters for bits above the 3-bit mask anyway.
    padded = torch.cat([data, torch.zeros(N, 1, dtype=data.dtype, device=data.device)], 1)
    hi = padded[:, byte_idx + 1].to(torch.int32)
    return ((lo | (hi << 8)) >> shift) & 0x7


def _pack_4bit(idx: torch.Tensor) -> torch.Tensor:
    """(N, D) int32 in [0,15] -> (N, D//2) uint8, low nibble = even element."""
    pairs = idx.reshape(idx.shape[0], -1, 2)
    return ((pairs[:, :, 0] & 0xF) | ((pairs[:, :, 1] & 0xF) << 4)).to(torch.uint8)


def _unpack_4bit(data: torch.Tensor, D: int) -> torch.Tensor:
    d_offs = torch.arange(D, device=data.device, dtype=torch.long)
    byte_idx = d_offs // 2
    shift = ((d_offs % 2) * 4).to(torch.int32)
    return (data[:, byte_idx].to(torch.int32) >> shift) & 0xF


class RefTQ:
    """Reference TQ quantizer/dequantizer for the MSE-key presets."""

    def __init__(self, preset: str, device: torch.device):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )

        self.cfg = TurboQuantConfig.from_cache_dtype(preset, HEAD_DIM)
        assert not self.cfg.key_fp8, "harness covers MSE-key presets only"
        self.device = device
        self.D = HEAD_DIM
        self.mse_bits = self.cfg.key_mse_bits
        self.vqb = self.cfg.effective_value_quant_bits
        self.mse_bytes = math.ceil(self.D * self.mse_bits / 8)
        self.val_bytes = math.ceil(self.D * self.vqb / 8)
        self.kps = self.cfg.key_packed_size  # mse_bytes + 2 (norm fp16)
        self.slot = self.cfg.slot_size
        self.slot_aligned = self.cfg.slot_size_aligned
        self.vmax_level = (1 << self.vqb) - 1

        self.centroids = get_centroids(self.D, self.cfg.centroid_bits).to(
            device=device, dtype=torch.float32
        )
        c_sorted, _ = self.centroids.sort()
        self.midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        # Hadamard rotation — same construction as the backend (Sylvester).
        H = torch.tensor([[1.0]])
        while H.shape[0] < self.D:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        self.Pi = (H / math.sqrt(self.D)).to(device)  # symmetric: Pi == PiT
        self.PiT = self.Pi

    # -- quantize (mirrors triton_turboquant_store MSE path, op-for-op) -----
    def quantize_slot_bytes(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """(N, Hk, D) fp16 K/V -> (N*Hk, slot_size) uint8 packed slots."""
        N, Hk, D = key.shape
        NH = N * Hk
        # keys: exact same op sequence as the launcher (fp32, same order)
        k_flat = key.float().reshape(NH, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = x_hat @ self.PiT
        idx = torch.bucketize(y, self.midpoints, right=True).to(torch.int32)
        idx = torch.minimum(idx, torch.tensor(2**self.mse_bits - 1, device=y.device))
        if self.mse_bits == 3:
            key_data = _pack_3bit(idx)
        else:
            key_data = _pack_4bit(idx)
        norm_bytes = _fp16_to_bytes(norms.squeeze(1)).reshape(NH, 2)

        # values: fp32 min/max, scale, round-half-up via +0.5 truncation
        v_flat = value.float().reshape(NH, D)
        vmin = v_flat.min(dim=1, keepdim=True).values
        vmax = v_flat.max(dim=1, keepdim=True).values
        levels = float(self.vmax_level)
        scale = (vmax - vmin) / levels
        scale = torch.where(scale > 1e-8, scale, torch.full_like(scale, 1e-8))
        q = ((v_flat - vmin) / scale + 0.5).to(torch.int32).clamp(0, self.vmax_level)
        if self.vqb == 3:
            val_data = _pack_3bit(q)
        else:
            val_data = _pack_4bit(q)
        sc_bytes = _fp16_to_bytes(scale.squeeze(1)).reshape(NH, 2)
        zr_bytes = _fp16_to_bytes(vmin.squeeze(1)).reshape(NH, 2)

        return torch.cat([key_data, norm_bytes, val_data, sc_bytes, zr_bytes], dim=1)

    # -- dequantize (mirrors _tq_full_dequant_kv, op-for-op) ----------------
    def dequant_slot_bytes(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(M, >=slot_size) uint8 -> (K_rotated (M, D) fp32, V (M, D) fp32).

        K is returned in ROTATED space (as the kernel emits it); callers apply
        `k @ Pi` to return to model space, exactly like _continuation_prefill.
        """
        M = slots.shape[0]
        key_data = slots[:, : self.mse_bytes]
        if self.mse_bits == 3:
            idx = _unpack_3bit(key_data, self.D)
        else:
            idx = _unpack_4bit(key_data, self.D)
        c = self.centroids[idx.long()]
        if self.cfg.norm_correction:
            inv = 1.0 / torch.sqrt((c * c).sum(dim=1, keepdim=True) + 1e-16)
            c = c * inv
        vec_norm = _bytes_to_fp16(slots[:, self.mse_bytes], slots[:, self.mse_bytes + 1])
        k_rot = vec_norm.unsqueeze(1) * c

        val_data = slots[:, self.kps : self.kps + self.val_bytes]
        if self.vqb == 3:
            vq = _unpack_3bit(val_data, self.D)
        else:
            vq = _unpack_4bit(val_data, self.D)
        sc_base = self.kps + self.val_bytes
        v_scale = _bytes_to_fp16(slots[:, sc_base], slots[:, sc_base + 1])
        v_zero = _bytes_to_fp16(slots[:, sc_base + 2], slots[:, sc_base + 3])
        v = vq.float() * v_scale.unsqueeze(1) + v_zero.unsqueeze(1)
        return k_rot, v


# ----------------------------------------------------------------------------
# GPU-side helpers (vllm kernels)
# ----------------------------------------------------------------------------
class KernelEnv:
    def __init__(self, preset: str):
        import vllm.v1.attention.backends.turboquant_attn as tqmod
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
            TurboQuantAttentionImpl,
            TurboQuantMetadata,
        )

        self.tqmod = tqmod
        self.TurboQuantMetadata = TurboQuantMetadata
        self.device = torch.device("cuda:0")
        self.preset = preset
        self.ref = RefTQ(preset, self.device)

        # Stub the engine config lookups the impl (and PN79, if applied) make.
        # max_cudagraph_capture_size=0 forces the workspace fallback path in
        # PN79-patched images, so the workspace checks below stay meaningful.
        stub = SimpleNamespace(
            attention_config=SimpleNamespace(tq_max_kv_splits_for_cuda_graph=16),
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=0),
        )
        tqmod.get_current_vllm_config = lambda: stub

        self.impl = TurboQuantAttentionImpl(
            num_heads=NUM_Q_HEADS,
            head_size=HEAD_DIM,
            scale=SCALE,
            num_kv_heads=NUM_KV_HEADS,
            kv_cache_dtype=preset,
        )
        self.layer = torch.nn.Module()
        cache_shape = TurboQuantAttentionBackend.get_kv_cache_shape(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, preset
        )
        # native layout (num_blocks, num_kv_heads, block_size, slot_aligned)
        self.cache_shape = cache_shape

    def new_cache(self) -> torch.Tensor:
        return torch.zeros(self.cache_shape, dtype=torch.uint8, device=self.device)

    def store(self, cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              slot_mapping: torch.Tensor) -> None:
        """Mirror do_kv_cache_update: native cache in, transposed for the kernel."""
        self.impl.do_kv_cache_update(
            self.layer, key.reshape(key.shape[0], -1), value.reshape(value.shape[0], -1),
            cache, slot_mapping,
        )

    def read_slots(self, cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Gather packed slot bytes (all heads) -> (num_slots, Hk, slot_aligned)."""
        blk = torch.div(slots, BLOCK_SIZE, rounding_mode="floor").long()
        off = (slots % BLOCK_SIZE).long()
        return cache[blk, :, off, :]  # (n, Hk, slot_aligned)

    def triton_dequant(self, cache: torch.Tensor, block_table: torch.Tensor,
                       length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run _tq_full_dequant_kv over positions [0, length) of one request."""
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _tq_full_dequant_kv,
            _use_fp8_e4b15,
        )
        import triton as _tr

        kvc = cache.transpose(1, 2)  # (nb, bs, Hk, slot) — kernel layout
        ref = self.ref
        alloc = math.ceil(length / BLOCK_SIZE) * BLOCK_SIZE
        k_out = torch.empty(1, NUM_KV_HEADS, alloc, HEAD_DIM,
                            dtype=torch.float16, device=self.device)
        v_out = torch.empty_like(k_out)
        grid = (alloc, NUM_KV_HEADS)
        _tq_full_dequant_kv[grid](
            kvc, block_table, ref.centroids, k_out, v_out,
            k_out.stride(0), k_out.stride(1), k_out.stride(2),
            v_out.stride(0), v_out.stride(1), v_out.stride(2),
            kvc.stride(0), kvc.stride(1), kvc.stride(2),
            block_table.stride(0),
            HEAD_DIM=HEAD_DIM, BLOCK_SIZE=BLOCK_SIZE, NUM_KV_HEADS=NUM_KV_HEADS,
            MSE_BYTES=ref.mse_bytes, KPS=ref.kps, VQB=ref.vqb,
            VAL_DATA_BYTES=ref.val_bytes, MSE_BITS=ref.mse_bits, KEY_FP8=0,
            BLOCK_D=_tr.next_power_of_2(HEAD_DIM),
            NORM_CORRECTION=1 if ref.cfg.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(0), num_warps=4,
        )
        return k_out[0, :, :length], v_out[0, :, :length]  # (Hk, len, D)


def _ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   cached_len: int) -> torch.Tensor:
    """fp32 GQA attention: q (Lq, Hq, D) over k/v (Lk, Hk, D); query i sees
    positions [0, cached_len + i]."""
    Lq, Hq, D = q.shape
    Lk, Hk, _ = k.shape
    g = Hq // Hk
    q32 = q.float().permute(1, 0, 2)                       # (Hq, Lq, D)
    k32 = k.float().permute(1, 0, 2).repeat_interleave(g, 0)
    v32 = v.float().permute(1, 0, 2).repeat_interleave(g, 0)
    scores = torch.einsum("hqd,hkd->hqk", q32, k32) * SCALE
    q_pos = torch.arange(Lq, device=q.device).unsqueeze(1) + cached_len
    k_pos = torch.arange(Lk, device=q.device).unsqueeze(0)
    scores = scores.masked_fill((k_pos > q_pos).unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,hkd->qhd", scores.softmax(dim=-1), v32)


def _meta(env: KernelEnv, seq_lens: list[int], q_lens: list[int],
          slot_mapping: torch.Tensor, block_table: torch.Tensor):
    dev = env.device
    cum = [0]
    for ql in q_lens:
        cum.append(cum[-1] + ql)
    qsl = torch.tensor(cum, dtype=torch.int32, device=dev)
    return env.TurboQuantMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        slot_mapping=slot_mapping,
        block_table=block_table,
        query_start_loc=qsl,
        num_actual_tokens=int(sum(q_lens)),
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        is_prefill=max(q_lens) > 1,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
    )


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def _rand_kv(n: int, device, seed_off: int = 0):
    g = torch.Generator(device="cpu").manual_seed(SEED + seed_off)
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, generator=g).half().to(device)
    v = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, generator=g).half().to(device)
    return k, v


def check_store_bytes(env: KernelEnv) -> None:
    n = 512
    k, v = _rand_kv(n, env.device)
    cache = env.new_cache()
    slots = torch.arange(n, dtype=torch.int32, device=env.device)
    env.store(cache, k, v, slots)
    got = env.read_slots(cache, slots)[:, :, : env.ref.slot]  # meaningful bytes only
    want = env.ref.quantize_slot_bytes(k, v).reshape(n, NUM_KV_HEADS, env.ref.slot)
    mism = (got != want).sum().item()
    _record(
        f"{env.preset}: store-bytes parity (kernel vs reference quantizer)",
        mism == 0,
        f"{mism}/{want.numel()} bytes differ",
    )


def check_roundtrip(env: KernelEnv) -> None:
    n = 512
    k, v = _rand_kv(n, env.device, 1)
    cache = env.new_cache()
    slots = torch.arange(n, dtype=torch.int32, device=env.device)
    env.store(cache, k, v, slots)
    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=env.device).unsqueeze(0)
    k_deq, v_deq = env.triton_dequant(cache, bt, n)  # (Hk, n, D), K rotated
    k_deq = (k_deq.reshape(-1, HEAD_DIM) @ env.ref.Pi.half()).reshape(
        NUM_KV_HEADS, n, HEAD_DIM
    ).transpose(0, 1)  # back to (n, Hk, D) model space
    v_deq = v_deq.transpose(0, 1)

    # values: exact per-element bound |v_hat - v| <= scale/2 (+ fp16 slack)
    v32 = v.float()
    vmin = v32.amin(dim=2, keepdim=True)
    scale = (v32.amax(dim=2, keepdim=True) - vmin) / env.ref.vmax_level
    # fp16 slack: scale/zero stored as fp16 (rel 2^-11) + final fp16 result cast
    bound = scale / 2 + scale * 2e-3 + 8e-3
    verr = (v_deq.float() - v32).abs()
    v_ok = bool((verr <= bound).all())
    _record(
        f"{env.preset}: roundtrip VALUE bound (|err| <= scale/2)",
        v_ok,
        f"{_err_stats(v_deq, v)} worst_margin={(verr - bound).max().item():.3e}",
    )

    # keys: Lloyd-Max distortion envelope (3-bit: rel L2 ~0.19 expected)
    k32 = k.float().reshape(-1, HEAD_DIM)
    kd32 = k_deq.float().reshape(-1, HEAD_DIM)
    rel = (kd32 - k32).norm(dim=1) / k32.norm(dim=1).clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(kd32, k32, dim=1)
    k_ok = bool((rel.max() < 0.35) and (cos.min() > 0.90))
    _record(
        f"{env.preset}: roundtrip KEY envelope (3-bit Lloyd-Max)",
        k_ok,
        f"rel_l2 max={rel.max().item():.4f} mean={rel.mean().item():.4f} "
        f"cos min={cos.min().item():.4f} | {_err_stats(k_deq, k)}",
    )


def check_dequant_parity(env: KernelEnv) -> None:
    n = 512
    k, v = _rand_kv(n, env.device, 2)
    cache = env.new_cache()
    slots = torch.arange(n, dtype=torch.int32, device=env.device)
    env.store(cache, k, v, slots)
    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=env.device).unsqueeze(0)
    k_tr, v_tr = env.triton_dequant(cache, bt, n)  # (Hk, n, D) fp16, K rotated

    slot_bytes = env.read_slots(cache, slots).reshape(n * NUM_KV_HEADS, -1)
    k_ref, v_ref = env.ref.dequant_slot_bytes(slot_bytes)  # (n*Hk, D) fp32
    k_ref = k_ref.reshape(n, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1).half()
    v_ref = v_ref.reshape(n, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1).half()

    k_ok = torch.allclose(k_tr.float(), k_ref.float(), atol=2e-3, rtol=1e-2)
    v_ok = torch.allclose(v_tr.float(), v_ref.float(), atol=2e-3, rtol=1e-2)
    _record(
        f"{env.preset}: dequant parity K (triton vs reference, same bytes)",
        k_ok, _err_stats(k_tr, k_ref),
    )
    _record(
        f"{env.preset}: dequant parity V (triton vs reference, same bytes)",
        v_ok, _err_stats(v_tr, v_ref),
    )


def check_determinism(env: KernelEnv) -> None:
    n = 384
    k, v = _rand_kv(n, env.device, 3)
    slots = torch.arange(n, dtype=torch.int32, device=env.device)
    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=env.device).unsqueeze(0)
    outs, deqs, caches = [], [], []
    for _ in range(2):
        cache = env.new_cache()
        env.store(cache, k, v, slots)
        caches.append(cache.clone())
        kd, vd = env.triton_dequant(cache, bt, n)
        deqs.append((kd.clone(), vd.clone()))
        q = torch.randn(4, NUM_Q_HEADS, HEAD_DIM,
                        generator=torch.Generator(device="cpu").manual_seed(SEED + 9),
                        ).half().to(env.device)
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        env.impl._ensure_on_device(env.layer, env.device)
        out = triton_turboquant_decode_attention(
            query=q, kv_cache=cache.transpose(1, 2),
            block_table=bt.expand(4, -1),
            seq_lens=torch.full((4,), n, dtype=torch.int32, device=env.device),
            Pi=env.layer._tq_Pi, centroids=env.layer._tq_centroids, scale=SCALE,
            mse_bits=env.ref.mse_bits, key_packed_size=env.ref.kps,
            value_quant_bits=env.ref.vqb, key_fp8=False,
            norm_correction=env.ref.cfg.norm_correction, PiT=env.layer._tq_PiT,
        )
        outs.append(out.clone())
    ok = (
        torch.equal(caches[0], caches[1])
        and torch.equal(deqs[0][0], deqs[1][0])
        and torch.equal(deqs[0][1], deqs[1][1])
        and torch.equal(outs[0], outs[1])
    )
    _record(
        f"{env.preset}: determinism (store bytes / dequant / decode, 2 runs)",
        ok,
        "bitwise identical" if ok else "MISMATCH between runs",
    )


def _run_continuation(env: KernelEnv, cached_len: int, q_len: int):
    """Store prefix, then run _prefill_attention for the continuation chunk.
    Returns (impl_out, ref_out) both (q_len, Hq, D)."""
    seq_len = cached_len + q_len
    k_all, v_all = _rand_kv(seq_len, env.device, 4 + q_len)
    g = torch.Generator(device="cpu").manual_seed(SEED + 5 + q_len)
    q = torch.randn(q_len, NUM_Q_HEADS, HEAD_DIM, generator=g).half().to(env.device)

    cache = env.new_cache()
    slots_all = torch.arange(seq_len, dtype=torch.int32, device=env.device)
    env.store(cache, k_all[:cached_len], v_all[:cached_len], slots_all[:cached_len])
    env.store(cache, k_all[cached_len:], v_all[cached_len:], slots_all[cached_len:])

    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=env.device).unsqueeze(0)
    meta = _meta(env, [seq_len], [q_len], slots_all[cached_len:], bt)
    env.impl._ensure_on_device(env.layer, env.device)
    out = env.impl._prefill_attention(
        q, k_all[cached_len:], v_all[cached_len:], cache.transpose(1, 2), meta,
        env.layer._tq_Pi, env.layer._tq_centroids, env.layer._tq_PiT,
        layer=env.layer,
    )

    # reference: dequant what the path actually consumes
    slot_bytes = env.read_slots(cache, slots_all).reshape(seq_len * NUM_KV_HEADS, -1)
    k_rot, v_deq = env.ref.dequant_slot_bytes(slot_bytes)
    k_deq = (k_rot.half() @ env.ref.Pi.half()).float()  # fp16 rotate-back like impl
    k_deq = k_deq.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM)
    v_deq = v_deq.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM)
    if q_len <= 128:
        # decode-kernel path reads current chunk from the cache too
        k_ref, v_ref = k_deq, v_deq
    else:
        # _continuation_prefill: dequant prefix + RAW current chunk
        k_ref = torch.cat([k_deq[:cached_len], k_all[cached_len:].float()])
        v_ref = torch.cat([v_deq[:cached_len], v_all[cached_len:].float()])
    ref = _ref_attention(q, k_ref.half(), v_ref.half(), cached_len)
    return out, ref


def check_continuation_decode_path(env: KernelEnv) -> None:
    out, ref = _run_continuation(env, cached_len=700, q_len=96)  # <=128 threshold
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1, HEAD_DIM), ref.reshape(-1, HEAD_DIM), dim=1
    )
    ok = bool(
        torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2) and cos.min() > 0.995
    )
    _record(
        f"{env.preset}: continuation q<=128 (TQ decode-kernel path) vs reference",
        ok, f"{_err_stats(out, ref)} cos_min={cos.min().item():.5f}",
    )


def check_continuation_prefill_path(env: KernelEnv) -> None:
    from vllm.v1.worker import workspace as ws

    ws.reset_workspace_manager()
    ws.init_workspace_manager(env.device)
    out, ref = _run_continuation(env, cached_len=900, q_len=300)  # >128 threshold
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1, HEAD_DIM), ref.reshape(-1, HEAD_DIM), dim=1
    )
    ok = bool(
        torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2) and cos.min() > 0.995
    )
    _record(
        f"{env.preset}: continuation q>128 (_continuation_prefill path) vs reference",
        ok, f"{_err_stats(out, ref)} cos_min={cos.min().item():.5f}",
    )

    # --- vllm#43357 repro (informational, always PASS-recorded with status) ---
    # Lock the workspace at its current (small) size, then push a continuation
    # whose dequant buffers exceed it. Stock image: AssertionError ("Workspace
    # is locked but allocation ... requires ... Workspace growth is not
    # allowed"). PN95-patched image: no workspace involvement -> no error.
    status = "no-crash (dequant buffers not on locked workspace — PN95 active?)"
    try:
        ws.lock_workspace()
        _run_continuation(env, cached_len=2200, q_len=300)
        crashed = False
    except AssertionError as e:
        crashed = "Workspace is locked" in str(e)
        status = f"REPRODUCED #43357: {str(e)[:120]}..."
    finally:
        ws.unlock_workspace()
    del crashed
    _record(f"{env.preset}: vllm#43357 lock repro (informational)", True, status)
    ws.reset_workspace_manager()
    ws.init_workspace_manager(env.device)


def check_mixed_batch(env: KernelEnv) -> None:
    """PN86/#46461: continuation req + first-chunk req that owns both maxima."""
    cached_a, q_a = 500, 300   # request A: continuation
    q_b = 1200                 # request B: first chunk, seq_len == q_len == max
    dev = env.device

    # PN86-patched images route request A through _continuation_prefill,
    # which requires an (unlocked) workspace manager.
    from vllm.v1.worker import workspace as ws
    if not ws.is_workspace_manager_initialized():
        ws.init_workspace_manager(dev)

    ka, va = _rand_kv(cached_a + q_a, dev, 20)
    kb, vb = _rand_kv(q_b, dev, 21)
    g = torch.Generator(device="cpu").manual_seed(SEED + 22)
    qa = torch.randn(q_a, NUM_Q_HEADS, HEAD_DIM, generator=g).half().to(dev)
    qb = torch.randn(q_b, NUM_Q_HEADS, HEAD_DIM, generator=g).half().to(dev)

    cache = env.new_cache()
    # request A pages: blocks 0..12 ; request B pages: blocks 32..50
    slots_a = torch.arange(cached_a + q_a, dtype=torch.int32, device=dev)
    slots_b = 32 * BLOCK_SIZE + torch.arange(q_b, dtype=torch.int32, device=dev)
    env.store(cache, ka[:cached_a], va[:cached_a], slots_a[:cached_a])
    # store both requests' current chunks (as do_kv_cache_update would)
    env.store(cache, torch.cat([ka[cached_a:], kb]), torch.cat([va[cached_a:], vb]),
              torch.cat([slots_a[cached_a:], slots_b]))

    max_blocks = NUM_BLOCKS
    bt = torch.zeros(2, max_blocks, dtype=torch.int32, device=dev)
    bt[0] = torch.arange(max_blocks)
    bt[1] = 32 + torch.arange(max_blocks)
    meta = _meta(env, [cached_a + q_a, q_b], [q_a, q_b],
                 torch.cat([slots_a[cached_a:], slots_b]), bt)
    assert meta.max_query_len == meta.max_seq_len == q_b  # the #46461 trap

    env.impl._ensure_on_device(env.layer, env.device)
    out = env.impl._prefill_attention(
        torch.cat([qa, qb]), torch.cat([ka[cached_a:], kb]),
        torch.cat([va[cached_a:], vb]), cache.transpose(1, 2), meta,
        env.layer._tq_Pi, env.layer._tq_centroids, env.layer._tq_PiT,
        layer=env.layer,
    )
    out_a, out_b = out[:q_a], out[q_a:]

    # references (per request, consuming what the correct path consumes)
    seq_a = cached_a + q_a
    slot_bytes = env.read_slots(cache, slots_a).reshape(seq_a * NUM_KV_HEADS, -1)
    k_rot, v_deq = env.ref.dequant_slot_bytes(slot_bytes)
    k_deq = (k_rot.half() @ env.ref.Pi.half()).reshape(seq_a, NUM_KV_HEADS, HEAD_DIM)
    v_deq = v_deq.reshape(seq_a, NUM_KV_HEADS, HEAD_DIM)
    k_ref_a = torch.cat([k_deq[:cached_a].float(), ka[cached_a:].float()]).half()
    v_ref_a = torch.cat([v_deq[:cached_a].float(), va[cached_a:].float()]).half()
    ref_a = _ref_attention(qa, k_ref_a, v_ref_a, cached_a)
    ref_b = _ref_attention(qb, kb, vb, 0)

    ok_b = torch.allclose(out_b.float(), ref_b, atol=5e-2, rtol=5e-2)
    cos_a = torch.nn.functional.cosine_similarity(
        out_a.float().reshape(-1, HEAD_DIM), ref_a.reshape(-1, HEAD_DIM), dim=1
    ).min()
    ok_a = bool(
        torch.allclose(out_a.float(), ref_a, atol=5e-2, rtol=5e-2) and cos_a > 0.995
    )
    note = "" if ok_a else " <- EXPECTED on stock image (#46461); PN86 fixes this"
    _record(
        f"{env.preset}: mixed batch — first-chunk request", ok_b, _err_stats(out_b, ref_b),
    )
    _record(
        f"{env.preset}: mixed batch — continuation request (PN86/#46461 case)",
        ok_a, f"{_err_stats(out_a, ref_a)} cos_min={cos_a.item():.5f}{note}",
    )


# ----------------------------------------------------------------------------
# CPU-only self-check of the reference implementation (no GPU / no Triton)
# ----------------------------------------------------------------------------
def selfcheck_cpu() -> int:
    dev = torch.device("cpu")
    rc = 0
    for preset in PRESETS:
        ref = RefTQ(preset, dev)
        g = torch.Generator().manual_seed(SEED)
        k = torch.randn(64, NUM_KV_HEADS, HEAD_DIM, generator=g).half()
        v = torch.randn(64, NUM_KV_HEADS, HEAD_DIM, generator=g).half()
        slots = ref.quantize_slot_bytes(k, v)
        assert slots.shape[1] == ref.slot, (slots.shape, ref.slot)
        k_rot, v_deq = ref.dequant_slot_bytes(slots)
        k_deq = (k_rot @ ref.Pi).reshape(64, NUM_KV_HEADS, HEAD_DIM)
        v_deq = v_deq.reshape(64, NUM_KV_HEADS, HEAD_DIM)
        v32 = v.float()
        vmin = v32.amin(dim=2, keepdim=True)
        scale = (v32.amax(dim=2, keepdim=True) - vmin) / ref.vmax_level
        v_ok = bool(((v_deq - v32).abs() <= scale / 2 + scale * 2e-3 + 2e-3).all())
        rel = ((k_deq - k.float()).reshape(-1, HEAD_DIM).norm(dim=1)
               / k.float().reshape(-1, HEAD_DIM).norm(dim=1)).max()
        k_ok = rel < 0.35
        _record(f"{preset}: CPU selfcheck (reference-only roundtrip)",
                v_ok and bool(k_ok), f"key rel_l2 max={rel.item():.4f} value bound "
                f"{'held' if v_ok else 'VIOLATED'}")
        rc |= 0 if (v_ok and k_ok) else 1
    return rc


# ----------------------------------------------------------------------------
# pytest entry points
# ----------------------------------------------------------------------------
_ENVS: dict[str, KernelEnv] = {}


def _env(preset: str) -> KernelEnv:
    if preset not in _ENVS:
        _ENVS[preset] = KernelEnv(preset)
    return _ENVS[preset]


def _gpu_required():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA required")


def test_store_bytes():
    _gpu_required()
    for p in PRESETS:
        check_store_bytes(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-2:])


def test_roundtrip():
    _gpu_required()
    for p in PRESETS:
        check_roundtrip(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-4:])


def test_dequant_parity():
    _gpu_required()
    for p in PRESETS:
        check_dequant_parity(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-4:])


def test_determinism():
    _gpu_required()
    for p in PRESETS:
        check_determinism(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-2:])


def test_continuation_paths():
    _gpu_required()
    for p in PRESETS:
        check_continuation_decode_path(_env(p))
        check_continuation_prefill_path(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-6:])


def test_mixed_batch():
    _gpu_required()
    for p in PRESETS:
        check_mixed_batch(_env(p))
    assert all(ok for _, ok, _ in _RESULTS[-4:])


def main() -> int:
    if "--selfcheck" in sys.argv:
        return selfcheck_cpu()
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available (use --selfcheck for CPU-only ref pass)")
        return 2
    torch.cuda.reset_peak_memory_stats()
    for preset in PRESETS:
        env = _env(preset)
        check_store_bytes(env)
        check_roundtrip(env)
        check_dequant_parity(env)
        check_determinism(env)
        check_continuation_decode_path(env)
        check_continuation_prefill_path(env)
        check_mixed_batch(env)
    peak = torch.cuda.max_memory_allocated() / 2**20
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed | "
          f"peak VRAM (torch-allocated): {peak:.0f} MiB")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
