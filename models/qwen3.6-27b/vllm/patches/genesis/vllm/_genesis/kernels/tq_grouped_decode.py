# SPDX-License-Identifier: Apache-2.0
"""P40 — GQA-grouped TurboQuant decode stage1 kernel (k8v4 + MSE-key TQ3/TQ4).

Ports the `_tq_grouped_decode_stage1` Triton kernel from upstream PR
vllm-project/vllm#40792 (OPEN as of 2026-04-24) so Genesis can harvest
the +10-27% decode throughput on Qwen3-32B-class GQA configurations
without waiting for merge.

Motivation (upstream PR body)
-----------------------------
The stock `_tq_decode_stage1` kernel (in
`vllm/v1/attention/ops/triton_turboquant_decode.py`) launches ONE CTA
per `(batch, head, kv_split)` — `Hq = 64` on Qwen3.6-35B-A3B means 64
CTAs per batch per split, with every CTA redundantly loading the same
KV tile because `Hq = KV_GROUP_SIZE × Hk`. The grouped kernel batches
up to `BLOCK_H = 16` Q heads that share one KV head, loading K/V
ONCE and computing QK/PV via `tl.dot` on float16 — roughly 4× fewer
KV loads, 2× arithmetic intensity on tensor cores.

PR author measured:
  - Qwen3-32B @ A100 PCIe: +27% decode tok/s (k8v4)
  - Qwen3-32B @ H100:      +16% decode tok/s (k8v4)

v7.73 MSE-key generalization (2026-07-13)
-----------------------------------------
The upstream port covered ONLY the `turboquant_k8v4` preset: FP8 keys
(`key_fp8=True`) and 4-bit uniform values (`tl.static_assert(VQB == 4)`).
The MSE-quantized-key presets (`turboquant_3bit_nc`, `turboquant_4bit_nc`,
`turboquant_k3v4_nc`) silently fell back to the scalar kernel — on a
TQ3 deployment (Qwen3.6-27B, 24 q / 4 kv heads × head_dim 256, GQA
group 6) that scalar kernel was measured at ~13% of achievable
bandwidth and 60% of the decode step at 32K ctx (torch-profiler
2026-07-13, see club-3090 diagnostics/tq-lane/PROFILE-ANALYSIS.md).

This revision generalizes the grouped kernel with the two missing
dequant paths, both ported op-for-op from the scalar
`_tq_decode_stage1`:

  * MSE keys (`KEY_FP8=0`): 3/4-bit index unpack -> centroid gather ->
    optional norm-correction -> per-token fp16 norm scale. Scores are
    `(q_rot . c) * vec_norm * ATTN_SCALE`, computed via `tl.dot` on
    fp16 inputs with fp32 accumulation (the scalar kernel does the dot
    product in fp32; the fp16-input dot introduces <=2^-11 relative
    input rounding — validated against the scalar kernel and the
    tq-lane parity harness reference).
  * 3-bit values (`VQB == 3`): 8 values per 3 bytes, little-endian
    16-bit window unpack — identical bit layout to the scalar kernel
    and `_tq_full_dequant_kv`.

The compile-time guard is now `tl.static_assert(VQB == 3 or VQB == 4)`;
the dispatcher (`should_use_grouped_kernel`) routes:

  key_fp8=True  -> VQB == 4 only            (original upstream scope)
  key_fp8=False -> VQB in {3, 4} and MSE_BITS in {3, 4}

Anything else falls back to the upstream scalar kernel unchanged.

Launch tuning (v7.73)
---------------------
Upstream hard-coded BLOCK_H=16 / BLOCK_KV=16 / num_warps=4 /
num_stages=2 (A100/H100-tuned, k8v4). Those remain the k8v4 defaults.
The MSE path resolves its launch config through
`resolve_launch_config()`: a small table keyed on
(head_dim, kv_group_size) with env overrides

    GENESIS_P40_BLOCK_H     (16/32 — tl.dot needs >=16)
    GENESIS_P40_BLOCK_KV    (16/32/64)
    GENESIS_P40_NUM_WARPS   (1/2/4/8)
    GENESIS_P40_NUM_STAGES  (1..4)

Invalid values fall back to the table default (NEVER raise — Genesis
guards). The (256, 6) entry — Qwen3.6-27B TQ3, q_len 1 (draft) and 4
(MTP K+1 verify) — was swept on RTX 4090 SM 8.9 (2026-07-13): see the
table inline. NUM_KV_SPLITS is NOT tunable here: it is part of the
launch grid and must stay constant for cudagraph capture.

Opt-in gate
-----------
Enabled via `GENESIS_ENABLE_P40=1`. OFF by default so the first
production deployment must explicitly benchmark correctness and
throughput before flipping on. Once we have GPU bench data confirming
+10% or better on our setup, we flip the default to on.

Correctness guardrails
----------------------
- `tl.static_assert(VQB == 3 or VQB == 4)` fires at compile-time if
  misused; MSE_BITS similarly guarded.
- Dispatcher enters the grouped path only for `kv_group_size > 1` and
  a supported (key format, VQB) combination; everything else retains
  the scalar kernel.
- `torch.empty`-allocated scratch (mid_o, output, lse) matches stage2
  output layout byte-for-byte — no change to stage2 kernel or return
  path required.
- Parity gate: club-3090 `diagnostics/tq-lane/tq_parity_harness.py`
  must pass with the grouped path active (store-bytes, roundtrip
  bounds, dequant parity, determinism, continuation q<=128 / q>128,
  mixed batch) before any deployment flips the flag.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.4 implementation (opt-in); v7.73 MSE-key/TQ3 generalization
"""
from __future__ import annotations

import logging
import os


log = logging.getLogger("genesis.tq_grouped_decode")

_ENV_ENABLE = "GENESIS_ENABLE_P40"


def _read_env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


_ENABLED_AT_IMPORT: bool = _read_env_enabled()


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA + SM ≥ 8.0 (Ampere+) + opt-in env."""
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    if not _ENABLED_AT_IMPORT:
        return False
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def _build_grouped_kernel():
    """Define the Triton kernel lazily so import on CPU-only hosts works.

    Triton import is guarded — on hosts without CUDA the import fails;
    we catch and return None so the wiring layer can fall through to
    the upstream scalar kernel.
    """
    try:
        from vllm.triton_utils import tl, triton
    except Exception:
        try:
            import triton
            import triton.language as tl
        except Exception:
            return None

    @triton.jit
    def _tq_grouped_decode_stage1(
        Q_rot_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        Centroids_ptr,
        Mid_o_ptr,
        stride_qb,
        stride_qh,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_bt_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        Q_HEAD_NUM: tl.constexpr,
        MSE_BITS: tl.constexpr,
        MSE_BYTES: tl.constexpr,
        KPS: tl.constexpr,
        VQB: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        ATTN_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_H: tl.constexpr,
        KEY_FP8: tl.constexpr,
        NORM_CORRECTION: tl.constexpr = 0,
        FP8_E4B15: tl.constexpr = 0,
    ):
        """GQA-grouped TQ decode stage1 (FP8 keys OR MSE keys; VQB 3/4).

        Each CTA processes up to BLOCK_H Q heads that share one KV head,
        loading K/V once and computing scores via `tl.dot`.

        Dequant math is ported op-for-op from the scalar
        `_tq_decode_stage1` (see module docstring); the only intentional
        numeric difference is fp16 tl.dot inputs (fp32 accumulate).
        """
        tl.static_assert(
            (VQB == 3) or (VQB == 4),
            "grouped kernel supports 3- and 4-bit uniform values only",
        )
        if not KEY_FP8:
            tl.static_assert(
                (MSE_BITS == 3) or (MSE_BITS == 4),
                "grouped kernel supports 3-/4-bit MSE keys only",
            )

        bid = tl.program_id(0)
        head_group_id = tl.program_id(1)
        sid = tl.program_id(2)

        heads_per_kv_head: tl.constexpr = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
        kv_head = head_group_id // heads_per_kv_head
        group_idx = head_group_id % heads_per_kv_head
        cur_head = (
            kv_head * KV_GROUP_SIZE
            + group_idx * BLOCK_H
            + tl.arange(0, BLOCK_H)
        )
        mask_h = (cur_head < (kv_head + 1) * KV_GROUP_SIZE) & (
            cur_head < Q_HEAD_NUM
        )

        seq_len = tl.load(Seq_lens_ptr + bid)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_start >= split_end:
            out_base = (
                bid * stride_mid_b
                + cur_head * stride_mid_h
                + sid * stride_mid_s
            )
            tl.store(
                Mid_o_ptr + out_base + HEAD_DIM,
                float("-inf"),
                mask=mask_h,
            )
            return

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)

        q_base = (
            bid * stride_qb + cur_head[:, None] * stride_qh + d_offs[None, :]
        )
        q_rot = tl.load(
            Q_rot_ptr + q_base,
            mask=mask_h[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_f16 = q_rot.to(tl.float16)

        # Loop-invariant unpack index vectors (compile-time specialized).
        if not KEY_FP8:
            mse_bit_off = d_offs * MSE_BITS
            mse_byte_idx = mse_bit_off // 8
            mse_bit_shift = mse_bit_off % 8
            mse_mask = (1 << MSE_BITS) - 1
        if VQB == 3:
            val_bit_off = d_offs * 3
            val_byte_idx = val_bit_off // 8
            val_bit_shift = val_bit_off % 8

        m_prev = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        for start_n in range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end

            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=kv_mask, other=0,
            ).to(tl.int64)

            slot_bases = (
                block_nums * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_head, tl.int64) * stride_cache_head
            )

            # ========================================================
            # KEY DEQUANT + SCORES: [BLOCK_H, BLOCK_KV]
            # ========================================================
            if KEY_FP8:
                k_addrs = slot_bases[:, None] + d_offs[None, :]
                k_raw = tl.load(
                    KV_cache_ptr + k_addrs,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                )
                if FP8_E4B15:
                    k_float = k_raw.to(
                        tl.float8e4b15, bitcast=True).to(tl.float32)
                else:
                    k_float = k_raw.to(
                        tl.float8e4nv, bitcast=True).to(tl.float32)

                scores = tl.dot(q_f16, tl.trans(k_float.to(tl.float16)))
                scores = (scores * ATTN_SCALE).to(tl.float32)
            else:
                # MSE keys: unpack indices, gather centroids, norm-scale.
                mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
                mse_raw0 = tl.load(
                    KV_cache_ptr + mse_addrs0,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_raw1 = tl.load(
                    KV_cache_ptr + mse_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16_key = mse_raw0 | (mse_raw1 << 8)
                mse_idx = (raw16_key >> mse_bit_shift[None, :]) & mse_mask

                c_vals = tl.load(
                    Centroids_ptr + mse_idx,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )

                if NORM_CORRECTION:
                    c_norm_sq = tl.sum(
                        tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                        axis=1,
                    )
                    c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                    c_vals = c_vals * c_inv_norm[:, None]

                # fp16 vector norms stored at MSE_BYTES offset.
                norm_bases = slot_bases + MSE_BYTES
                n_lo = tl.load(
                    KV_cache_ptr + norm_bases, mask=kv_mask, other=0,
                ).to(tl.uint16)
                n_hi = tl.load(
                    KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0,
                ).to(tl.uint16)
                vec_norms = (
                    (n_lo | (n_hi << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )

                scores = tl.dot(q_f16, tl.trans(c_vals.to(tl.float16)))
                scores = scores.to(tl.float32) * (
                    vec_norms * ATTN_SCALE)[None, :]

            scores = tl.where(
                mask_h[:, None] & kv_mask[None, :], scores, -float("inf"),
            )

            n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max[:, None])

            # ========================================================
            # VALUE DEQUANT: [BLOCK_KV, BLOCK_D]
            # ========================================================
            val_bases = slot_bases + KPS

            if VQB == 3:
                val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
                val_raw0 = tl.load(
                    KV_cache_ptr + val_addrs0,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                val_raw1 = tl.load(
                    KV_cache_ptr + val_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16_val = val_raw0 | (val_raw1 << 8)
                v_idx = (
                    (raw16_val >> val_bit_shift[None, :]) & 0x7
                ).to(tl.float32)
            else:  # VQB == 4
                vb_idx = d_offs // 2
                vb_shift = (d_offs % 2) * 4
                val_addrs = val_bases[:, None] + vb_idx[None, :]
                val_raw = tl.load(
                    KV_cache_ptr + val_addrs,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(
                KV_cache_ptr + sc_bases, mask=kv_mask, other=0,
            ).to(tl.uint16)
            sc_hi = tl.load(
                KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0,
            ).to(tl.uint16)
            v_scales = (
                (sc_lo | (sc_hi << 8))
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            zr_lo = tl.load(
                KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0,
            ).to(tl.uint16)
            zr_hi = tl.load(
                KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0,
            ).to(tl.uint16)
            v_zeros = (
                (zr_lo | (zr_hi << 8))
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

            acc = acc * re_scale[:, None] + tl.dot(
                p.to(tl.float16), values.to(tl.float16),
            ).to(tl.float32)
            l_prev = l_prev * re_scale + tl.sum(p, 1)
            m_prev = n_e_max

        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        out_base = (
            bid * stride_mid_b
            + cur_head[:, None] * stride_mid_h
            + sid * stride_mid_s
        )
        tl.store(
            Mid_o_ptr + out_base + d_offs[None, :],
            acc / safe_l[:, None],
            mask=mask_h[:, None] & d_mask[None, :],
        )
        lse = m_prev + tl.log(safe_l)
        tl.store(
            Mid_o_ptr
            + bid * stride_mid_b
            + cur_head * stride_mid_h
            + sid * stride_mid_s
            + HEAD_DIM,
            lse,
            mask=mask_h,
        )

    return _tq_grouped_decode_stage1


# Lazy accessor — built once per process on first use, cached.
_CACHED_KERNEL = None


def get_grouped_kernel():
    """Return the compiled grouped-decode Triton kernel (None on non-CUDA).

    Build is deferred to first call so CPU-only test environments don't
    fail at import time (Triton + CUDA only available on GPU hosts).
    """
    global _CACHED_KERNEL
    if _CACHED_KERNEL is None:
        _CACHED_KERNEL = _build_grouped_kernel()
    return _CACHED_KERNEL


def should_use_grouped_kernel(
    kv_group_size: int,
    key_fp8: bool,
    value_quant_bits: int,
    mse_bits: int | None = None,
) -> bool:
    """Dispatcher decision: route to grouped kernel iff supported.

    Routing (v7.73):
      key_fp8=True  -> VQB == 4 only (upstream #40792 scope, unchanged)
      key_fp8=False -> VQB in {3, 4} AND mse_bits in {3, 4}
                       (MSE-key generalization; mse_bits=None is treated
                        as unknown and rejected — caller must pass it)
    Returns False on any unsupported config so the caller falls back to
    the original scalar kernel (correctness preserved).
    """
    if not should_apply():
        return False
    if kv_group_size <= 1:
        return False
    if key_fp8:
        # FP8-key path: 4-bit values only (original upstream port).
        return value_quant_bits == 4
    # MSE-key path (v7.73): 3-/4-bit keys, 3-/4-bit values.
    if value_quant_bits not in (3, 4):
        return False
    if mse_bits not in (3, 4):
        return False
    return True


# Launch-parameter constants — match upstream PR #40792 hard-coded values.
# These remain the k8v4 (FP8-key) defaults; the MSE path resolves through
# resolve_launch_config() below.
BLOCK_H = 16
BLOCK_KV = 16
NUM_WARPS = 4
NUM_STAGES = 2

# MSE-path launch table, keyed on (head_dim, kv_group_size).
# (256, 6) = Qwen3.6-27B TQ3 (24 q / 4 kv heads, head_dim 256), swept on
# RTX 4090 (SM 8.9) 2026-07-13 with the tq-lane microbench at 350-token
# and 32K contexts, B=1 (q_len 1 draft) and B=8 (2 seqs × q_len 4 MTP
# K+1 verify bucket). Sweep findings (54 configs): num_stages=1 is
# decisive at long ctx (S>=2 costs 2-3× — multi-stage smem buffering of
# the byte-gather tiles backfires); BLOCK_KV=16 beats 32/64; num_warps=2
# wins the B=8 bucket (902 us vs scalar 2595 us @32K = 2.9×), W=4/8
# within ~20% at B=1. See club-3090
# diagnostics/tq-lane/PROFILE-ANALYSIS.md Target 1 for the source data.
_MSE_LAUNCH_TABLE: dict[tuple[int, int], tuple[int, int, int, int]] = {
    # (head_dim, group): (BLOCK_H, BLOCK_KV, num_warps, num_stages)
    (256, 6): (16, 16, 2, 1),
}
_MSE_LAUNCH_DEFAULT: tuple[int, int, int, int] = (
    BLOCK_H, BLOCK_KV, NUM_WARPS, NUM_STAGES,
)

_VALID_BLOCK_H = {16, 32}       # tl.dot needs M >= 16
_VALID_BLOCK_KV = {16, 32, 64}  # tl.dot needs K >= 16
_VALID_NUM_WARPS = {1, 2, 4, 8}


def _env_int(name: str, valid: set[int] | None = None,
             lo: int | None = None, hi: int | None = None) -> int | None:
    """Parse an int env override; None if unset/invalid (never raise)."""
    env = os.environ.get(name, "").strip()
    if not env or not env.isdigit():
        return None
    v = int(env)
    if valid is not None and v not in valid:
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def resolve_launch_config(
    head_dim: int,
    kv_group_size: int,
    key_fp8: bool,
) -> tuple[int, int, int, int]:
    """Resolve (BLOCK_H, BLOCK_KV, num_warps, num_stages) for a launch.

    k8v4 keeps the upstream constants; MSE presets consult the tuned
    table, then env overrides (validated; invalid values are ignored).
    """
    if key_fp8:
        base = _MSE_LAUNCH_DEFAULT
    else:
        base = _MSE_LAUNCH_TABLE.get(
            (head_dim, kv_group_size), _MSE_LAUNCH_DEFAULT)
    block_h = _env_int("GENESIS_P40_BLOCK_H", _VALID_BLOCK_H) or base[0]
    block_kv = _env_int("GENESIS_P40_BLOCK_KV", _VALID_BLOCK_KV) or base[1]
    num_warps = _env_int("GENESIS_P40_NUM_WARPS", _VALID_NUM_WARPS) or base[2]
    num_stages = _env_int(
        "GENESIS_P40_NUM_STAGES", lo=1, hi=4) or base[3]
    return block_h, block_kv, num_warps, num_stages
