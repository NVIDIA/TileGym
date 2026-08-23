# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Qwen2VL Multimodal Rotary Position Embedding (M-RoPE) kernel (CuTile backend).

Half-split layout: left half of head_dim = real part, right half = imaginary part.
Three RoPE sections: temporal [0, t_end), height [t_end, h_end), width [h_end, hd//2).
cos/sin shape: (3, bsz, seq_len, head_dim).

Two kernel variants selected at runtime via the ALIGNED flag (mirrors rope.py):

  ALIGNED case (power-of-2 head_dim AND all head counts match tile sizes AND
  contiguous Q/K):
    _qwen2vl_mrope_4d_ct -- operates on Q/K in original
    (bsz, n_heads, seq_len, head_dim) layout using ct.load block loads (block index
    0/1 in the last dim selects the real/imag half). No host-side transpose or
    contiguous copy is needed, and Q/K traffic is a coalesced TMA load instead of a
    per-element gather. Grid: (bsz * seq_len,).

  Non-ALIGNED case (e.g. non-power-of-2 head_dim):
    _qwen2vl_mrope_kernel -- gather/scatter kernel operating on a host-side
    transposed+contiguous (bsz, seq_len, n_heads, head_dim) copy with column masking
    to head_dim//2 so non-power-of-2 head_dim stays out-of-bounds safe.
    Grid: (bsz, seq_len).
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

ConstInt = ct.Constant[int]


@ct.kernel
def _qwen2vl_mrope_4d_ct(
    Q,  # (bsz, n_q_heads, seq_len, head_dim) -- original layout, head_dim = 2*TILE_HD
    K,  # (bsz, n_k_heads, seq_len, head_dim)
    COS,  # (3, bsz, seq_len, head_dim) -- first TILE_HD elements per section are the cos values
    SIN,  # (3, bsz, seq_len, head_dim)
    seq_len: ConstInt,
    MROPE_SECTION_T: ConstInt,
    MROPE_SECTION_H: ConstInt,
    sin_sign: ct.Constant[float],
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """ALIGNED fast path: no host-side transpose/contiguous, coalesced TMA loads."""
    pid = ct.bid(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    t_end = MROPE_SECTION_T
    h_end = t_end + MROPE_SECTION_H

    # Select the M-RoPE section per column along head_dim//2: temporal [0, t_end),
    # height [t_end, h_end), width [h_end, hd//2). COS/SIN are (3, bsz, seq, hd);
    # block index (section, batch, seq, 0) grabs [0, TILE_HD) = the rotation values.
    col = ct.arange(TILE_HD, dtype=ct.int32)[None, :]  # (1, TILE_HD)
    in_t = col < t_end
    in_h = col < h_end
    t_cos = ct.load(COS, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    h_cos = ct.load(COS, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    w_cos = ct.load(COS, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    t_sin = ct.load(SIN, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    h_sin = ct.load(SIN, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    w_sin = ct.load(SIN, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    cos_row = ct.where(in_t, t_cos, ct.where(in_h, h_cos, w_cos))
    sin_row = ct.where(in_t, t_sin, ct.where(in_h, h_sin, w_sin)) * sin_sign

    # Q in (bsz, n_q_heads, seq_len, head_dim): index (b, 0, s, 0) = real half,
    # index (b, 0, s, 1) = imag half (block 1 starts at element TILE_HD = head_dim//2).
    q_r = ct.load(Q, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    q_i = ct.load(Q, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    # Rotate in fp32 (cos_row/sin_row are fp32), round only the final result.
    q_r_f = q_r.astype(ct.float32)
    q_i_f = q_i.astype(ct.float32)
    new_q_r = (q_r_f * cos_row - q_i_f * sin_row).astype(Q.dtype)
    new_q_i = (q_i_f * cos_row + q_r_f * sin_row).astype(Q.dtype)
    ct.store(Q, index=(batch_idx, 0, seq_idx, 0), tile=new_q_r.reshape((1, TILE_QH, 1, TILE_HD)))
    ct.store(Q, index=(batch_idx, 0, seq_idx, 1), tile=new_q_i.reshape((1, TILE_QH, 1, TILE_HD)))

    # K in (bsz, n_k_heads, seq_len, head_dim)
    k_r = ct.load(K, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    k_i = ct.load(K, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    k_r_f = k_r.astype(ct.float32)
    k_i_f = k_i.astype(ct.float32)
    new_k_r = (k_r_f * cos_row - k_i_f * sin_row).astype(K.dtype)
    new_k_i = (k_i_f * cos_row + k_r_f * sin_row).astype(K.dtype)
    ct.store(K, index=(batch_idx, 0, seq_idx, 0), tile=new_k_r.reshape((1, TILE_KH, 1, TILE_HD)))
    ct.store(K, index=(batch_idx, 0, seq_idx, 1), tile=new_k_i.reshape((1, TILE_KH, 1, TILE_HD)))


@ct.kernel
def _qwen2vl_mrope_kernel(
    query,  # 1D flat, len = bsz*sl*n_qh*hd
    key,  # 1D flat, len = bsz*sl*n_kh*hd
    cos,  # 1D flat, len = 3*bsz*sl*hd
    sin,  # 1D flat
    sl,
    BS_SL,  # bsz * sl  (slab stride = BS_SL * HEAD_DIM, computed in-kernel)
    N_QH: ConstInt,
    N_KH: ConstInt,
    MROPE_SECTION_T: ConstInt,
    MROPE_SECTION_H: ConstInt,
    BACKWARD: ct.Constant[bool],
    HEAD_DIM: ConstInt,
    HEAD_DIM_HALF: ConstInt,
    TILE_HD: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    HD_POW2: ct.Constant[bool],
):
    batch_idx = ct.bid(0)
    seq_idx = ct.bid(1)

    t_end = MROPE_SECTION_T
    h_end = t_end + MROPE_SECTION_H

    FLAT = TILE_QH * TILE_HD
    row_1d = ct.arange(TILE_QH, dtype=ct.int32)
    col_1d = ct.arange(TILE_HD, dtype=ct.int32)
    flat_row = ct.broadcast_to(row_1d[:, None], (TILE_QH, TILE_HD)).reshape((FLAT,))
    flat_col = ct.broadcast_to(col_1d[None, :], (TILE_QH, TILE_HD)).reshape((FLAT,))

    # Issue Q+K gathers first to start DRAM fetch, then cos/sin gathers can overlap.
    q_token_off = (batch_idx * sl + seq_idx) * (N_QH * HEAD_DIM)
    q_r_idx = q_token_off + flat_row * HEAD_DIM + flat_col
    q_i_idx = q_r_idx + HEAD_DIM_HALF
    if HD_POW2:
        q_mask = flat_row < N_QH
    else:
        q_mask = (flat_row < N_QH) & (flat_col < HEAD_DIM_HALF)
    q_r = ct.gather(query, q_r_idx, mask=q_mask, check_bounds=False, latency=2)
    q_i = ct.gather(query, q_i_idx, mask=q_mask, check_bounds=False, latency=2)

    # K indices (computed early so K gathers can fire right after Q's)
    FLAT_K = TILE_KH * TILE_HD
    krow_1d = ct.arange(TILE_KH, dtype=ct.int32)
    k_flat_row = ct.broadcast_to(krow_1d[:, None], (TILE_KH, TILE_HD)).reshape((FLAT_K,))
    k_flat_col = ct.broadcast_to(col_1d[None, :], (TILE_KH, TILE_HD)).reshape((FLAT_K,))
    k_token_off = (batch_idx * sl + seq_idx) * (N_KH * HEAD_DIM)
    k_r_idx = k_token_off + k_flat_row * HEAD_DIM + k_flat_col
    k_i_idx = k_r_idx + HEAD_DIM_HALF
    if HD_POW2:
        k_mask = k_flat_row < N_KH
    else:
        k_mask = (k_flat_row < N_KH) & (k_flat_col < HEAD_DIM_HALF)
    k_r = ct.gather(key, k_r_idx, mask=k_mask, check_bounds=False, latency=2)
    k_i = ct.gather(key, k_i_idx, mask=k_mask, check_bounds=False, latency=2)

    token_cs_off = (batch_idx * sl + seq_idx) * HEAD_DIM
    slab_stride = BS_SL * HEAD_DIM
    t_idx = flat_col + token_cs_off
    h_idx = t_idx + slab_stride
    w_idx = h_idx + slab_stride
    t_cos = ct.gather(cos, t_idx, check_bounds=False, latency=2)
    t_sin = ct.gather(sin, t_idx, check_bounds=False, latency=2)
    h_cos = ct.gather(cos, h_idx, check_bounds=False, latency=2)
    h_sin = ct.gather(sin, h_idx, check_bounds=False, latency=2)
    w_cos = ct.gather(cos, w_idx, check_bounds=False, latency=2)
    w_sin = ct.gather(sin, w_idx, check_bounds=False, latency=2)

    in_t = flat_col < t_end
    in_h = flat_col < h_end
    cos_row = ct.where(in_t, t_cos, ct.where(in_h, h_cos, w_cos))
    sin_row = ct.where(in_t, t_sin, ct.where(in_h, h_sin, w_sin))
    if BACKWARD:
        sin_row = -sin_row

    # Rotate in fp32 (cos_row/sin_row are fp32) and round only the final result,
    # Casting cos/sin down to bf16 first would
    # round the intermediate product/subtraction and fail bf16 tolerance.
    q_r_f = q_r.astype(ct.float32)
    q_i_f = q_i.astype(ct.float32)
    new_q_r = (q_r_f * cos_row - q_i_f * sin_row).astype(query.dtype)
    new_q_i = (q_i_f * cos_row + q_r_f * sin_row).astype(query.dtype)

    # Reuse Q's cos_row when FLAT_K <= FLAT (common case: TILE_KH <= TILE_QH).
    if TILE_KH <= TILE_QH:
        cos_k = ct.extract(cos_row, (0,), shape=(FLAT_K,))
        sin_k = ct.extract(sin_row, (0,), shape=(FLAT_K,))
    else:
        t_idx_k = k_flat_col + token_cs_off
        h_idx_k = t_idx_k + slab_stride
        w_idx_k = h_idx_k + slab_stride
        t_cos_k = ct.gather(cos, t_idx_k, check_bounds=False, latency=2)
        t_sin_k = ct.gather(sin, t_idx_k, check_bounds=False, latency=2)
        h_cos_k = ct.gather(cos, h_idx_k, check_bounds=False, latency=2)
        h_sin_k = ct.gather(sin, h_idx_k, check_bounds=False, latency=2)
        w_cos_k = ct.gather(cos, w_idx_k, check_bounds=False, latency=2)
        w_sin_k = ct.gather(sin, w_idx_k, check_bounds=False, latency=2)
        in_t_k = k_flat_col < t_end
        in_h_k = k_flat_col < h_end
        cos_k_raw = ct.where(in_t_k, t_cos_k, ct.where(in_h_k, h_cos_k, w_cos_k))
        sin_k_raw = ct.where(in_t_k, t_sin_k, ct.where(in_h_k, h_sin_k, w_sin_k))
        if BACKWARD:
            sin_k_raw = -sin_k_raw
        cos_k = cos_k_raw
        sin_k = sin_k_raw
    # Rotate in fp32 (cos_k/sin_k are fp32), round only the final result.
    k_r_f = k_r.astype(ct.float32)
    k_i_f = k_i.astype(ct.float32)
    new_k_r = (k_r_f * cos_k - k_i_f * sin_k).astype(key.dtype)
    new_k_i = (k_i_f * cos_k + k_r_f * sin_k).astype(key.dtype)
    ct.scatter(query, q_r_idx, new_q_r, mask=q_mask, check_bounds=False, latency=1)
    ct.scatter(query, q_i_idx, new_q_i, mask=q_mask, check_bounds=False, latency=1)
    ct.scatter(key, k_r_idx, new_k_r, mask=k_mask, check_bounds=False, latency=1)
    ct.scatter(key, k_i_idx, new_k_i, mask=k_mask, check_bounds=False, latency=1)


def _is_aligned(q, k, n_q_head, n_kv_head, head_dim_half, TILE_HD, TILE_QH, TILE_KH):
    """ALIGNED fast path: power-of-2 head_dim / head counts and contiguous Q/K, so the
    4D block-load kernel can run in-place on the original (bsz, n_heads, seq, hd) layout
    without any host transpose+contiguous copy."""
    return (
        (TILE_HD == head_dim_half)
        and (TILE_QH == n_q_head)
        and (TILE_KH == n_kv_head)
        and q.is_contiguous()
        and k.is_contiguous()
    )


def _qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):
    # q/k in: (bsz, n_heads, seq_len, head_dim)
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    if _is_aligned(q, k, n_q_head, n_kv_head, head_dim_half, TILE_HD, TILE_QH, TILE_KH):
        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        ct.launch(
            torch.cuda.current_stream(),
            (batch_size * seq_len,),
            _qwen2vl_mrope_4d_ct,
            (
                q,
                k,
                cos_c,
                sin_c,
                int(seq_len),
                int(mrope_section[0]),
                int(mrope_section[1]),
                1.0,
                int(TILE_QH),
                int(TILE_KH),
                int(TILE_HD),
            ),
        )
        return q, k, cos, sin

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)
    bs_sl = batch_size * seq_len

    cos = cos.contiguous()
    sin = sin.contiguous()

    grid = (batch_size, seq_len)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel,
        (
            q.view(-1),
            k.view(-1),
            cos.view(-1),
            sin.view(-1),
            int(seq_len),
            int(bs_sl),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            False,
            int(head_dim),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(TILE_HD == head_dim_half),
        ),
    )

    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def _qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section):
    # dq/dk in: (bsz, n_heads, seq_len, head_dim)
    batch_size, n_q_head, seq_len, head_dim = dq.shape
    n_kv_head = dk.shape[1]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    if _is_aligned(dq, dk, n_q_head, n_kv_head, head_dim_half, TILE_HD, TILE_QH, TILE_KH):
        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        ct.launch(
            torch.cuda.current_stream(),
            (batch_size * seq_len,),
            _qwen2vl_mrope_4d_ct,
            (
                dq,
                dk,
                cos_c,
                sin_c,
                int(seq_len),
                int(mrope_section[0]),
                int(mrope_section[1]),
                -1.0,  # backward negates sin
                int(TILE_QH),
                int(TILE_KH),
                int(TILE_HD),
            ),
        )
        return dq, dk

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)
    bs_sl = batch_size * seq_len

    grid = (batch_size, seq_len)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel,
        (
            dq.view(-1),
            dk.view(-1),
            cos.view(-1),
            sin.view(-1),
            int(seq_len),
            int(bs_sl),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            True,
            int(head_dim),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(TILE_HD == head_dim_half),
        ),
    )

    return dq.transpose(1, 2), dk.transpose(1, 2)


class Qwen2VLMRopeCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        q, k, cos, sin = _qwen2vl_mrope_forward(q, k, cos, sin, mrope_section)
        ctx.save_for_backward(cos, sin)
        ctx.mrope_section = mrope_section
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        mrope_section = ctx.mrope_section
        dq, dk = _qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section)
        return dq, dk, None, None, None, None


@register_impl("liger.qwen2vl_mrope", backend="cutile")
def qwen2vl_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    unsqueeze_dim: int = 1,
    **kwargs,
) -> tuple:
    return Qwen2VLMRopeCuTileFunction.apply(q, k, cos, sin, mrope_section, unsqueeze_dim)
