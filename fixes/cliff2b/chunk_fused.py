# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Fused GDN chunk kernel — eliminates h and v_new materialization.
# Fixes Cliff 2b OOM (github.com/vllm-project/vllm Issue #TBD).

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp
from .utils import FLA_CHUNK_SIZE, use_cuda_graph


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_fused_kernel(
    q,
    k,
    v,
    w,
    g,
    gk,
    o,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    stride_qk = Hg * K
    stride_v = H * V
    stride_w = H * K
    stride_o = H * V

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    w += (bos * H + i_h) * K
    o += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h

    # Initialize recurrent state b_h in registers [BV, 64] x ceil(K/64)
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K
        p_h0_1 = tl.make_block_ptr(p_h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(p_h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(p_h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(p_h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):

        # STAGE 1: v_corr = v_raw - w @ h  (UN-GATED)
        p_w1 = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_wh = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))

        if K > 64:
            p_w2 = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_wh += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))

        if K > 128:
            p_w3 = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w3 = tl.load(p_w3, boundary_check=(0, 1))
            b_wh += tl.dot(b_w3, tl.trans(b_h3).to(b_w3.dtype))

        if K > 192:
            p_w4 = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w4 = tl.load(p_w4, boundary_check=(0, 1))
            b_wh += tl.dot(b_w4, tl.trans(b_h4).to(b_w4.dtype))

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v_intra = tl.load(p_v, boundary_check=(0, 1)) - b_wh

        # STAGE 2: Cross-chunk output: o_cross = q @ h^T (PRE-update h)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)

        p_q1 = tl.make_block_ptr(
            q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_o += tl.dot(b_q1, tl.trans(b_h1).to(b_q1.dtype))

        if K > 64:
            p_q2 = tl.make_block_ptr(
                q, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_q2 = tl.load(p_q2, boundary_check=(0, 1))
            b_o += tl.dot(b_q2, tl.trans(b_h2).to(b_q2.dtype))

        if K > 128:
            p_q3 = tl.make_block_ptr(
                q, (T, K), (stride_qk, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_q3 = tl.load(p_q3, boundary_check=(0, 1))
            b_o += tl.dot(b_q3, tl.trans(b_h3).to(b_q3.dtype))

        if K > 192:
            p_q4 = tl.make_block_ptr(
                q, (T, K), (stride_qk, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_q4 = tl.load(p_q4, boundary_check=(0, 1))
            b_o += tl.dot(b_q4, tl.trans(b_h4).to(b_q4.dtype))

        # STAGE 3: Intra-chunk output using UN-GATED b_v_intra
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        p_k1 = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_A += tl.dot(b_q1, b_k1)

        if K > 64:
            p_k2 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_A += tl.dot(b_q2, b_k2)

        if K > 128:
            p_k3 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k3 = tl.load(p_k3, boundary_check=(0, 1))
            b_A += tl.dot(b_q3, b_k3)

        if K > 192:
            p_k4 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k4 = tl.load(p_k4, boundary_check=(0, 1))
            b_A += tl.dot(b_q4, b_k4)

        if USE_G:
            p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * exp(b_g[:, None] - b_g[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        # CRITICAL: b_v_intra is UN-GATED here
        b_o = b_o * scale + tl.dot(b_A.to(b_v_intra.dtype), b_v_intra) * scale

        # STAGE 4: Store output
        p_o = tl.make_block_ptr(
            o, (T, V), (stride_o, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # STAGE 5: Gate v_corr for recurrence (deferred — un-gated value dead
        # after STAGE 3, safe to mutate in-place)
        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1

        if USE_G:
            b_g_last_raw = tl.load(g + last_idx * H)
            b_v_intra = b_v_intra * tl.where(m_t, exp(b_g_last_raw - b_g), 0)[:, None]

        # STAGE 6: Decay recurrent state and update (b_v_intra now GATED)
        if USE_G:
            b_g_last_exp = exp(b_g_last_raw)
            b_h1 *= b_g_last_exp
            if K > 64:
                b_h2 *= b_g_last_exp
            if K > 128:
                b_h3 *= b_g_last_exp
            if K > 192:
                b_h4 *= b_g_last_exp

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[None, :]

        b_v_gated_k = b_v_intra.to(k.dtype.element_ty)

        p_k_rec1 = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k_rec1 = tl.load(p_k_rec1, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k_rec1, b_v_gated_k))

        if K > 64:
            p_k_rec2 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k_rec2 = tl.load(p_k_rec2, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k_rec2, b_v_gated_k))

        if K > 128:
            p_k_rec3 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k_rec3 = tl.load(p_k_rec3, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k_rec3, b_v_gated_k))

        if K > 192:
            p_k_rec4 = tl.make_block_ptr(
                k, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k_rec4 = tl.load(p_k_rec4, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k_rec4, b_v_gated_k))

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * V * K
        p_ht1 = tl.make_block_ptr(p_ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(p_ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht3 = tl.make_block_ptr(p_ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht4 = tl.make_block_ptr(p_ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    o_buf: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused forward pass: eliminates h and v_new materialization."""
    B, T, Hg, K = q.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is not None:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
    else:
        N = B

    if o_buf is not None:
        o = o_buf
    else:
        o = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)

    final_state = (
        torch.empty(N, H, V, K, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_fused_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        g=g,
        gk=gk,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state
