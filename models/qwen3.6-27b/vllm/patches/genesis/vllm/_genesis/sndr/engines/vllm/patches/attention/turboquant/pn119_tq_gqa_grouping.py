# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN119 — TurboQuant k8v4 GQA head grouping kernel.

Backport of [vllm#40792](https://github.com/vllm-project/vllm/pull/40792)
by `hoseung2` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

Adds the GQA-grouped variant of TurboQuant decode stage-1 kernel
``_tq_grouped_decode_stage1`` (~255 lines of new Triton code) and
updates the dispatch in ``triton_turboquant_decode_attention`` to
select the grouped kernel when GQA is active. The grouped kernel
handles BOTH FP8 keys (k8v4) and MSE-quantized keys (FIX 2 — the
Gemma ``turboquant_4bit_nc`` preset, ``key_fp8=False``); both route
through ``tl.dot`` tensor cores. The gate is
``kv_group_size > 1 and value_quant_bits == 4`` (3-bit-value presets
still fall back to the scalar kernel).

The upstream PR measured **+16.5% – 27.2% TPS** on A100 / H100 with
GQA-ratio ∈ {4, 8, 24}. Our 27B and 35B both run **GQA-ratio 8**
(num_q_heads=32, num_kv_heads=4) so the win should be near the high
end on Ampere SM 8.6 hardware.

The grouped kernel:
  - Loads K once per ``BLOCK_H`` query-head tile and shares it across
    that whole tile of q-heads (the q-heads of one GQA group all read
    the same K vectors).
  - Uses ``tl.dot`` instead of element-wise products → routes through
    tensor cores instead of CUDA cores → 4-8× FLOPS density.
  - For MSE-quantized keys (``key_fp8=False``), reconstructs the same
    ``k_float = vec_norms * centroids`` tile the scalar kernel implies
    (FIX 2), so the dot product is numerically equivalent to the
    scalar ``scores = vec_norms * sum(q_rot * c) * scale``.
  - Falls back to the legacy ``_tq_decode_stage1`` kernel for every
    preset the grouped V-path cannot handle (3-bit values, and — on
    KVQ-carrying pins — non-uniform value codebooks / value outliers).

================================================================
2026-07-25 — WHEEL-v2 RE-ANCHOR (raw diff → sniffed TextPatcher)
================================================================

PN119 used to ship its two hunks as a bundled unified diff
(``pn119_kernel.diff``) applied through ``patch(1)`` behind a whole-file
md5 guard. On the wheel-v2 image (``dev1474cherrymax-1757-20260725``)
that stopped working:

    md5 v1/attention/ops/triton_turboquant_decode.py
      dev1060cherry-20260713      e93d6f9eb591e0b68a50b0fc2eb689c3
      dev1474cherry-1711   (v1)   e93d6f9eb591e0b68a50b0fc2eb689c3
      dev1474cherrymax-1757 (v2)  916313835a97c192208c443d55f9baaa   <- KVQ squash

    patch -p1 --dry-run < pn119_kernel.diff
      wheel v1 : diff APPLIES
      wheel v2 : Hunk #1 succeeded at 354 (offset 41); Hunk #2 FAILED at 804

Root cause is the SAME KVQ/nuqv squash that broke P101: ``cherry-max``
threaded three new arguments through the decode entry point and into the
stage-1 kernel launch —

    val_cent                (positional, after ``centroids``)
    VALUE_NUQ=value_nuq_flag
    N_VAL_CENTROIDS=n_val_centroids
    N_OUTLIERS=n_outliers

— and PN119's hunk #2 REWRITES exactly that launch (it moves the scalar
call into an ``else:`` branch and re-indents it). A raw diff cannot see
the new arguments, so it rejects; and a naive re-anchor that only fixed
the ANCHOR would have re-emitted the launch WITHOUT those four
arguments, silently dropping the nuqv / outlier presets on the scalar
path. That is the P101 lesson: when a patch rewrites a call, the
REPLACEMENT has to carry the new arguments too.

Fix shape (the P101 92647851/748d2a0b + P89 6edf8386 precedent): PN119 is
now a content-sniffed :class:`TextPatcher` with two sub-patches, and the
tree stays DUAL-PIN — ``dev1060cherry-20260713`` and wheel v1 must keep
booting because prod rollback depends on them, so nothing is re-anchored
in place:

  * ``pn119_grouped_kernel_insert`` — inserts the grouped Triton kernel
    between the end of ``_tq_decode_stage1`` and the pre-dequant kernel
    comment banner. That region is byte-identical on all three pins, so
    it needs no sniff.
  * ``pn119_dispatch_grouped`` — rewrites the stage-1 launch. Anchor AND
    replacement are assembled from the same building blocks by
    :func:`_dispatch_sub_patch`, and the KVQ block is spliced into BOTH
    or NEITHER. The scalar ``else:`` branch is produced by re-indenting
    the *anchor's own* launch text, so it is structurally impossible for
    the rewrite to drop an argument the target had.

The sniff itself is wrapped in ``except Exception``: an unreadable target
degrades to the plain (non-KVQ) anchors, i.e. an ordinary anchor miss and
a graceful skip — never a boot-killing exception. ``resolve_vllm_file()``
returns a **str**, so every read goes through ``Path(...)``.

Grouped-path gate on KVQ pins. The vendored grouped kernel implements the
4-bit UNIFORM value path only: it has no Lloyd-Max value codebook and no
KVQ-3 outlier scatter-gather. Upstream's own preset table
(``model_executor/layers/quantization/turboquant/config.py``) gives every
``value_nuq`` / ``value_outlier_pct`` preset ``value_quant_bits: 3``, so
``value_quant_bits == 4`` already excludes them — but on KVQ pins the
emitted gate ALSO tests ``not value_nuq_flag and n_outliers == 0`` so a
future 4-bit KVQ preset can never be silently routed through a kernel
that would ignore its codebook and its outliers. Belt and braces; the
cost is one constant-folded boolean per decode call.

================================================================
SAFETY MODEL
================================================================

- **Byte-exact anchors, no md5 gate.** The old whole-file md5 guard was
  already known-bad (a sibling TQ patch — PN14, which lands first — edits
  an unrelated region and shifts the whole-file md5 every boot). Anchors
  are the guard now: a drifted region simply does not match and PN119
  skips.
- **Idempotency**: ``TextPatcher`` prepends its marker line and returns
  IDEMPOTENT on the next boot.
- **Upstream absorption probe**: if ``_tq_grouped_decode_stage1`` is in
  the target but our vendor tag is not, upstream landed #40792 and PN119
  self-retires loudly. Verified NOT the case today — a package-wide grep
  of the wheel-v2 image finds zero hits for that symbol. The probe is
  deliberately implemented in :func:`apply` rather than as an
  ``upstream_drift_markers`` entry, because we emit that symbol
  ourselves and a bare marker would be a self-collision.
- **No fallback path needed**: PN119 is additive on top of an unchanged
  dispatch entry point. If it does not apply, vLLM keeps using the
  original scalar kernel.
- ``pn119_kernel.diff`` is RETAINED beside this module as the upstream
  provenance artifact (it is the literal source of the kernel text
  below). It is no longer applied by anything.

================================================================
HW GATE
================================================================

This patch is *active* on Ampere (SM 8.x) and Hopper (SM 9.x). It is
*not* expected to crash on newer HW, but the win was only measured
on A100/H100. Operators may disable with ``GENESIS_DISABLE_PN119=1``.

NOTE: P18B_TEXT anchors on THIS patch's output (requires_patches chain;
PN119 boot-dispatches BEFORE P18B_TEXT). When the emitted launch text
changes, re-check ``p18b_kernel_literals_textpatch.py``.

================================================================

Author: Genesis backport, original by hoseung2.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn119_tq_gqa_grouping")

GENESIS_PN119_MARKER = (
    "Genesis PN119 TurboQuant k8v4 GQA head grouping (backport: vllm#40792)"
)

# Emitted inside the inserted kernel's banner. Lets the upstream-absorption
# probe distinguish OUR ``_tq_grouped_decode_stage1`` from a future native one
# without making the probe collide with our own output.
GENESIS_PN119_VENDOR_TAG = "[Genesis PN119 vendor of vllm#40792]"

_UPSTREAM_GROUPED_SYMBOL = "_tq_grouped_decode_stage1"

_TARGET_REL = "v1/attention/ops/triton_turboquant_decode.py"


def _target_path() -> Path | None:
    """``resolve_vllm_file`` returns a **str** (or None) — wrap in Path."""
    p = resolve_vllm_file(_TARGET_REL)
    if p is None:
        return None
    return Path(p)


def _read_target() -> str | None:
    """Best-effort read of the target. Never raises."""
    try:
        target = _target_path()
        if target is None:
            return None
        return target.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — a sniff must never kill a boot
        log.warning(
            "[PN119] target sniff failed (%s: %s) — falling back to the "
            "plain (non-KVQ) anchors",
            type(e).__name__, e,
        )
        return None


# ─────────────────────────────────────────────────────────────────────
# Sub-patch 1 — insert the grouped kernel.
#
# The insertion point is the blank gap between the last statement of
# ``_tq_decode_stage1`` and the pre-dequant kernel banner. Byte-identical
# on dev1060cherry-20260713, wheel v1 and wheel v2 (the KVQ squash edits
# the value-dequant block INSIDE the loop, not the tail), so this
# sub-patch needs no content sniff. count==1 verified on all three pins.
# ─────────────────────────────────────────────────────────────────────

_KERNEL_ANCHOR = (
    "    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)\n"
    "\n"
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16\n"
)

# Verbatim from ``pn119_kernel.diff`` hunk #1 (upstream vllm#40792), plus the
# vendor-tag banner line.
_GROUPED_KERNEL_SRC = '''\
# ---------------------------------------------------------------------------
# Stage 1 (grouped): GQA head grouping + tl.dot tensor-core scoring
# [Genesis PN119 vendor of vllm#40792] — vendored, not upstream-native.
# ---------------------------------------------------------------------------


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
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KEY_FP8: tl.constexpr = 1,
    MSE_BITS: tl.constexpr = 0,
    MSE_BYTES: tl.constexpr = 0,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
):
    """GQA-grouped TQ decode stage1 for FP8 keys AND MSE-quantized keys.

    Each CTA processes up to BLOCK_H Q heads that share one KV head,
    loading K/V once and computing scores via `tl.dot`.

    Key dequant is selected by KEY_FP8:
      * KEY_FP8=1 — FP8 (E4M3) keys: bitcast raw bytes to fp8 -> fp32.
      * KEY_FP8=0 — MSE keys (`turboquant_4bit_nc` etc.): unpack the
        per-(token, dim) centroid indices, gather centroids, apply the
        optional unit-norm correction, then scale each token's vector
        by its stored key-norm (`vec_norms`). The resulting k_float tile
        feeds the SAME `tl.dot` path. This is algebraically identical to
        the scalar kernel's `scores = vec_norms * sum(q_rot * c) * scale`
        because `vec_norms * sum(q*c) == sum(q * (vec_norms*c))`.
    Values are 4-bit uniform (VQB==4) in both cases.
    """
    bid = tl.program_id(0)
    head_group_id = tl.program_id(1)
    sid = tl.program_id(2)

    # Map head_group_id → KV head + Q head range.
    # CTAs are partitioned per KV head: each KV head owns
    # `heads_per_kv_head = ceil(KV_GROUP_SIZE / BLOCK_H)` consecutive CTAs.
    # This keeps every CTA confined to a single KV head even when
    # KV_GROUP_SIZE > BLOCK_H and not a multiple of it.
    heads_per_kv_head: tl.constexpr = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    kv_head = head_group_id // heads_per_kv_head
    group_idx = head_group_id % heads_per_kv_head
    cur_head = kv_head * KV_GROUP_SIZE + group_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (kv_head + 1) * KV_GROUP_SIZE) & (cur_head < Q_HEAD_NUM)

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        # Still must write valid -inf LSE for masked heads
        out_base = bid * stride_mid_b + cur_head * stride_mid_h + sid * stride_mid_s
        tl.store(Mid_o_ptr + out_base + HEAD_DIM, float("-inf"), mask=mask_h)
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load Q: [BLOCK_H, BLOCK_D]
    q_base = bid * stride_qb + cur_head[:, None] * stride_qh + d_offs[None, :]
    q_rot = tl.load(
        Q_rot_ptr + q_base,
        mask=mask_h[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators: [BLOCK_H]
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
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # K DEQUANT → k_float [BLOCK_KV, BLOCK_D] (FP8 or MSE keys).
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            # MSE-quantized keys: tiled version of the scalar per-token
            # unpack. Build the SAME k_float = vec_norms * centroids tile
            # the scalar kernel implies, so the tl.dot below reproduces
            # `vec_norms * sum(q_rot * c_vals) * ATTN_SCALE` exactly.
            mse_bit_off = d_offs * MSE_BITS
            mse_byte_idx = mse_bit_off // 8
            mse_bit_shift = mse_bit_off % 8
            mse_umask = (1 << MSE_BITS) - 1
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
            mse_raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (mse_raw16 >> mse_bit_shift[None, :]) & mse_umask

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

            # Per-token key norms at MSE_BYTES offset (fp16 -> fp32).
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(
                KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0
            ).to(tl.uint16)
            vec_norms = (
                (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            k_float = vec_norms[:, None] * c_vals

        # scores = q_rot @ k_float^T : [BLOCK_H, BLOCK_KV]
        scores = tl.dot(q_rot.to(tl.float16), tl.trans(k_float.to(tl.float16)))
        scores = (scores * ATTN_SCALE).to(tl.float32)
        scores = tl.where(mask_h[:, None] & kv_mask[None, :], scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX: [BLOCK_H]
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])

        # ============================================================
        # V DEQUANT → values [BLOCK_KV, BLOCK_D] (4-bit uniform; VQB==3
        # is an MSE-only path handled by the original scalar kernel).
        # ============================================================
        tl.static_assert(VQB == 4, "grouped kernel only supports 4-bit values")
        val_bases = slot_bases + KPS

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
        sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
            tl.uint16
        )
        zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # ACCUMULATE: acc += p @ values via tl.dot
        # ============================================================
        acc = acc * re_scale[:, None] + tl.dot(
            p.to(tl.float16), values.to(tl.float16)
        ).to(tl.float32)
        l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    # Store partial results per Q head
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out_base = (
        bid * stride_mid_b + cur_head[:, None] * stride_mid_h + sid * stride_mid_s
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


'''

_KERNEL_REPLACEMENT = (
    "    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)\n"
    "\n"
    "\n"
    + _GROUPED_KERNEL_SRC
    + "# ---------------------------------------------------------------------------\n"
    "# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16\n"
)


# ─────────────────────────────────────────────────────────────────────
# Sub-patch 2 — dispatch rewrite (CONTENT-SNIFFED, dual-pin).
#
# Building blocks. The KVQ pieces are present on wheel v2
# (dev1474cherrymax) and absent on dev1060cherry-20260713 / wheel v1.
# Byte-verified: the plain shape is count==1 on the old pins and count==0
# on v2; the KVQ shape is the exact mirror.
# ─────────────────────────────────────────────────────────────────────

_DISPATCH_PREAMBLE = (
    "    # Stage 1: split-KV tiled attention scoring + value accumulation\n"
    "    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)\n"
)

_SCALAR_LAUNCH_HEAD = (
    "    BLOCK_KV = 4\n"
    "    grid = (B, Hq, NUM_KV_SPLITS)\n"
    "    _tq_decode_stage1[grid](\n"
    "        q_rot,\n"
    "        kv_cache,\n"
    "        block_table,\n"
    "        seq_lens,\n"
    "        centroids,\n"
)

# KVQ-1: the value codebook pointer, positional, right after ``centroids``.
_SCALAR_LAUNCH_VAL_CENT = "        val_cent,\n"

_SCALAR_LAUNCH_MID = (
    "        mid_o,\n"
    "        q_rot.stride(0),\n"
    "        q_rot.stride(1),\n"
    "        kv_cache.stride(0),\n"
    "        kv_cache.stride(1),\n"
    "        kv_cache.stride(2),\n"
    "        block_table.stride(0),\n"
    "        mid_o.stride(0),\n"
    "        mid_o.stride(1),\n"
    "        mid_o.stride(2),\n"
    "        NUM_KV_HEADS=Hk,\n"
    "        HEAD_DIM=D,\n"
    "        BLOCK_SIZE=block_size,\n"
    "        NUM_KV_SPLITS=NUM_KV_SPLITS,\n"
    "        KV_GROUP_SIZE=kv_group_size,\n"
    "        MSE_BITS=mse_bits,\n"
    '        MSE_BYTES=cfg["mse_bytes"],\n'
    "        KPS=key_packed_size,\n"
    "        VQB=value_quant_bits,\n"
    '        VAL_DATA_BYTES=cfg["val_data_bytes"],\n'
    "        ATTN_SCALE=scale,\n"
    '        BLOCK_D=cfg["BLOCK_D"],\n'
    "        BLOCK_KV=BLOCK_KV,\n"
    "        KEY_FP8=1 if key_fp8 else 0,\n"
    "        NORM_CORRECTION=1 if norm_correction else 0,\n"
    "        FP8_E4B15=fp8_e4b15,\n"
)

# KVQ-1 / KVQ-3 constexprs, between FP8_E4B15 and the launch tune.
_SCALAR_LAUNCH_KVQ = (
    "        VALUE_NUQ=value_nuq_flag,\n"
    "        N_VAL_CENTROIDS=n_val_centroids,\n"
    "        N_OUTLIERS=n_outliers,\n"
)

_SCALAR_LAUNCH_TAIL = (
    "        num_warps=1,\n"
    "        num_stages=1,\n"
    "    )\n"
)

# The grouped launch. Emitted at ``if``-body depth (8-space call, 12-space
# kwargs), which is the shape P18B_TEXT's GQA anchor keys on.
_GROUPED_SETUP = (
    "    BLOCK_H = 16\n"
    "    BLOCK_KV_GROUPED = 16\n"
    "    heads_per_kv_head = triton.cdiv(kv_group_size, BLOCK_H)\n"
    "    head_groups = Hk * heads_per_kv_head\n"
    "\n"
    "    # Grouped tl.dot path: GQA (kv_group_size > 1) with 4-bit values.\n"
    "    # Admits BOTH FP8 keys (k8v4) and MSE-quantized keys (e.g. the\n"
    "    # Gemma `turboquant_4bit_nc` preset, key_fp8=False). The kernel's\n"
    "    # K-dequant branches on KEY_FP8; the V-path requires VQB==4, so we\n"
    "    # exclude 3-bit-value presets here.\n"
)

_GROUPED_LAUNCH = (
    "        grid = (B, head_groups, NUM_KV_SPLITS)\n"
    "        _tq_grouped_decode_stage1[grid](\n"
    "            q_rot,\n"
    "            kv_cache,\n"
    "            block_table,\n"
    "            seq_lens,\n"
    "            centroids,\n"
    "            mid_o,\n"
    "            q_rot.stride(0),\n"
    "            q_rot.stride(1),\n"
    "            kv_cache.stride(0),\n"
    "            kv_cache.stride(1),\n"
    "            kv_cache.stride(2),\n"
    "            block_table.stride(0),\n"
    "            mid_o.stride(0),\n"
    "            mid_o.stride(1),\n"
    "            mid_o.stride(2),\n"
    "            HEAD_DIM=D,\n"
    "            BLOCK_SIZE=block_size,\n"
    "            NUM_KV_SPLITS=NUM_KV_SPLITS,\n"
    "            KV_GROUP_SIZE=kv_group_size,\n"
    "            Q_HEAD_NUM=Hq,\n"
    "            KPS=key_packed_size,\n"
    "            VQB=value_quant_bits,\n"
    '            VAL_DATA_BYTES=cfg["val_data_bytes"],\n'
    "            ATTN_SCALE=scale,\n"
    '            BLOCK_D=cfg["BLOCK_D"],\n'
    "            BLOCK_KV=BLOCK_KV_GROUPED,\n"
    "            BLOCK_H=BLOCK_H,\n"
    "            KEY_FP8=1 if key_fp8 else 0,\n"
    "            MSE_BITS=mse_bits,\n"
    '            MSE_BYTES=cfg["mse_bytes"],\n'
    "            NORM_CORRECTION=1 if norm_correction else 0,\n"
    "            FP8_E4B15=fp8_e4b15,\n"
    "            num_warps=4,\n"
    "            num_stages=2,\n"
    "        )\n"
)

# Gate, plain pins: only GQA + 4-bit uniform values reach the grouped kernel.
_GATE_PLAIN = "    if kv_group_size > 1 and value_quant_bits == 4:\n"

# Gate, KVQ pins: additionally refuse any non-uniform / outlier preset. The
# vendored grouped kernel has neither a Lloyd-Max value codebook nor the
# KVQ-3 outlier scatter-gather, so routing such a preset through it would
# silently drop the codebook and the exact outliers. Redundant today
# (every value_nuq / value_outlier preset is 3-bit) — deliberately kept as
# a structural guard against a future 4-bit KVQ preset.
_GATE_KVQ = (
    "    if (\n"
    "        kv_group_size > 1\n"
    "        and value_quant_bits == 4\n"
    "        # [Genesis PN119] the grouped kernel implements the 4-bit\n"
    "        # UNIFORM value path only — no Lloyd-Max codebook, no KVQ-3\n"
    "        # outlier scatter-gather. Never route those presets here.\n"
    "        and not value_nuq_flag\n"
    "        and n_outliers == 0\n"
    "    ):\n"
)

_ELSE_HEAD = (
    "    else:\n"
    "        # MHA (kv_group_size==1) and every preset the grouped kernel\n"
    "        # cannot serve: use the original scalar kernel, unchanged\n"
    "        # except for indentation.\n"
)


def _indent4(block: str) -> str:
    """Indent every non-blank line of ``block`` by four spaces."""
    return "".join(
        ("    " + line) if line.strip() else line
        for line in block.splitlines(keepends=True)
    )


def _dispatch_sub_patch(kvq: bool) -> TextPatch:
    """Build the dispatch sub-patch for a KVQ / non-KVQ target.

    The scalar launch text is assembled ONCE and used twice: verbatim as
    the tail of the anchor, and re-indented as the body of the ``else:``
    branch in the replacement. That makes it structurally impossible for
    the rewrite to drop an argument the target carried (the P101 lesson —
    fixing only the anchor silently strips the KVQ kwargs from the
    rewritten call and the nuqv presets stop taking effect).
    """
    scalar_launch = (
        _SCALAR_LAUNCH_HEAD
        + (_SCALAR_LAUNCH_VAL_CENT if kvq else "")
        + _SCALAR_LAUNCH_MID
        + (_SCALAR_LAUNCH_KVQ if kvq else "")
        + _SCALAR_LAUNCH_TAIL
    )
    anchor = _DISPATCH_PREAMBLE + scalar_launch
    replacement = (
        _DISPATCH_PREAMBLE
        + _GROUPED_SETUP
        + (_GATE_KVQ if kvq else _GATE_PLAIN)
        + _GROUPED_LAUNCH
        + _ELSE_HEAD
        + _indent4(scalar_launch)
    )
    return TextPatch(
        name="pn119_dispatch_grouped" + ("_kvq" if kvq else ""),
        anchor=anchor,
        replacement=replacement,
        required=False,
    )


def _make_patcher(target: Path, content: str | None) -> TextPatcher:
    """Assemble the patcher, sniffing ``content`` for the KVQ launch shape.

    Both dispatch variants are ``required=False`` with required-at-least-one
    semantics (the kernel's ``no_applicable_sub_patches`` SKIP fires when
    every sub-patch misses) — the P85 / P18B convention. Exactly one of them
    can match any given target: the plain and KVQ launch shapes are mutually
    exclusive by construction.
    """
    kvq_first = bool(content is not None and _SCALAR_LAUNCH_KVQ in content)
    variants = [True, False] if kvq_first else [False, True]
    return TextPatcher(
        patch_name=(
            "PN119 v1/attention/ops/triton_turboquant_decode.py — "
            "TurboQuant k8v4 GQA head grouping kernel (vllm#40792)"
        ),
        target_file=str(target),
        marker=GENESIS_PN119_MARKER,
        sub_patches=[
            TextPatch(
                name="pn119_grouped_kernel_insert",
                anchor=_KERNEL_ANCHOR,
                replacement=_KERNEL_REPLACEMENT,
                required=True,
            ),
            *[_dispatch_sub_patch(kvq) for kvq in variants],
        ],
        # Deliberately EMPTY: the one string worth watching for
        # (``_tq_grouped_decode_stage1``) is emitted by this very patch, so
        # listing it here would be a self-collision. The absorption probe
        # lives in apply() and tests for the symbol WITHOUT our vendor tag.
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    """Apply PN119 — TurboQuant k8v4 GQA head grouping kernel."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN119")
    log_decision("PN119", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = _target_path()
    if target is None or not target.is_file():
        return "skipped", "triton_turboquant_decode.py not found"

    content = _read_target()

    # Upstream-absorption probe. Our own inserted kernel carries the vendor
    # tag, so this only fires for a genuinely native implementation.
    if (
        content is not None
        and _UPSTREAM_GROUPED_SYMBOL in content
        and GENESIS_PN119_VENDOR_TAG not in content
        and GENESIS_PN119_MARKER not in content
    ):
        return "skipped", (
            f"upstream_merged — {_UPSTREAM_GROUPED_SYMBOL!r} present in the "
            "target without a Genesis vendor tag: vllm#40792 landed natively, "
            "PN119 self-retires"
        )

    # Idempotency FIRST — a warm restart must report IDEMPOTENT, not trip the
    # pre-gate below (our own applied output no longer contains the pristine
    # dispatch anchor, which would read as a scary "drifted" skip).
    if content is not None and GENESIS_PN119_MARKER in content:
        return "applied", "idempotent (marker present)"

    # Pre-gate (the P85 convention): sub-patch 1 is required=True, so a
    # dispatch-only drift would otherwise write the kernel, set the marker
    # and report APPLIED while the grouped kernel sat unreachable — a silent
    # capability loss behind a green log line. Refuse BEFORE any write.
    if content is not None and not any(
        _dispatch_sub_patch(kvq).anchor in content for kvq in (False, True)
    ):
        return "skipped", (
            "PN119: dispatch anchor absent in both the plain and the KVQ "
            "shape — the stage-1 launch in triton_turboquant_decode_attention "
            "drifted. Refusing to insert an unreachable kernel."
        )

    patcher = _make_patcher(target, content)

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001 — never raise out of an apply hook
        return "failed", f"PN119 apply exception: {type(e).__name__}: {e}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "idempotent (marker present)"

    if result == TextPatchResult.SKIPPED:
        why = failure.reason if failure else "anchor drift"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN119: {why}{detail}"

    if result == TextPatchResult.FAILED:
        why = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN119: {why}{detail}"

    applied = ", ".join(patcher.applied_sub_patches) or "(unknown)"
    return "applied", (
        "PN119 applied: TurboQuant k8v4 GQA-grouped decode stage1 kernel "
        "inserted; dispatch in triton_turboquant_decode_attention routed to "
        f"the grouped variant for GQA-ratio > 1 [{applied}]. Upstream "
        "measured +16-27% TPS on A100/H100 GQA-{4,8,24}."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    content = _read_target()
    if content is None:
        return False
    return GENESIS_PN119_MARKER in content
