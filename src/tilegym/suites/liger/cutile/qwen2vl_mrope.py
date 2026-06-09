# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py

"""
Qwen2VL Multimodal Rotary Position Embedding (M-RoPE) kernel (CuTile backend).

Half-split layout: left half of head_dim = real part, right half = imaginary part.
Three RoPE sections: temporal [0, t_end), height [t_end, h_end), width [h_end, hd//2).
cos/sin shape: (3, bsz, seq_len, head_dim).
Grid: (bsz * seq_len,) — one program per token.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _qwen2vl_mrope_kernel(
    query,  # (bsz, seq_len, n_qh, 2, head_dim_half)
    key,  # (bsz, seq_len, n_kh, 2, head_dim_half)
    cos,  # (3, bsz, seq_len, hd)
    sin,  # (3, bsz, seq_len, hd)
    sl,
    N_QH: ConstInt,
    N_KH: ConstInt,
    MROPE_SECTION_T: ConstInt,
    MROPE_SECTION_H: ConstInt,
    sin_sign,
    HEAD_DIM_HALF: ConstInt,
    TILE_HD: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    ALIGNED: ct.Constant[bool],
):
    pid = ct.bid(0)
    batch_idx = pid // sl
    seq_idx = pid % sl

    t_end = MROPE_SECTION_T
    h_end = t_end + MROPE_SECTION_H

    # Load cos/sin for 3 sections: temporal, height, width.
    # When ALIGNED (TILE_HD == head_dim_half, i.e. power-of-2), skip zero-padding
    # for the hardware TMA fast path. Otherwise use PAD_ZERO for safety.
    if ALIGNED:
        t_cos = ct.load(cos, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
        t_sin = ct.load(sin, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
        h_cos = ct.load(cos, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
        h_sin = ct.load(sin, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
        w_cos = ct.load(cos, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
        w_sin = ct.load(sin, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD)).reshape((1, TILE_HD))
    else:
        t_cos = ct.load(cos, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )
        t_sin = ct.load(sin, index=(0, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )
        h_cos = ct.load(cos, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )
        h_sin = ct.load(sin, index=(1, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )
        w_cos = ct.load(cos, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )
        w_sin = ct.load(sin, index=(2, batch_idx, seq_idx, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        )

    # Section masks
    d_idx = ct.arange(TILE_HD, dtype=ct.int32)
    t_mask = d_idx < t_end
    h_mask = ct.bitwise_and((d_idx >= t_end), (d_idx < h_end))
    w_mask = ct.bitwise_and((d_idx >= h_end), (d_idx < HEAD_DIM_HALF))

    t_f = ct.astype(t_mask, ct.float32)
    h_f = ct.astype(h_mask, ct.float32)
    w_f = ct.astype(w_mask, ct.float32)

    cos_row = t_cos * t_f + h_cos * h_f + w_cos * w_f
    sin_row = (t_sin * t_f + h_sin * h_f + w_sin * w_f) * sin_sign

    # Process Q: load all heads at once
    if ALIGNED:
        q_r = ct.load(query, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_QH, 1, TILE_HD)).reshape(
            (TILE_QH, TILE_HD)
        )
        q_i = ct.load(query, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_QH, 1, TILE_HD)).reshape(
            (TILE_QH, TILE_HD)
        )
    else:
        q_r = ct.load(
            query, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_QH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_QH, TILE_HD))
        q_i = ct.load(
            query, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_QH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_QH, TILE_HD))
    new_q_r = q_r * cos_row - q_i * sin_row
    new_q_i = q_i * cos_row + q_r * sin_row
    ct.store(
        query,
        index=(batch_idx, seq_idx, 0, 0, 0),
        tile=new_q_r.reshape((1, 1, TILE_QH, 1, TILE_HD)).astype(query.dtype),
    )
    ct.store(
        query,
        index=(batch_idx, seq_idx, 0, 1, 0),
        tile=new_q_i.reshape((1, 1, TILE_QH, 1, TILE_HD)).astype(query.dtype),
    )

    # Process K: load all heads at once
    if ALIGNED:
        k_r = ct.load(key, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_KH, 1, TILE_HD)).reshape(
            (TILE_KH, TILE_HD)
        )
        k_i = ct.load(key, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_KH, 1, TILE_HD)).reshape(
            (TILE_KH, TILE_HD)
        )
    else:
        k_r = ct.load(
            key, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_KH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_KH, TILE_HD))
        k_i = ct.load(
            key, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_KH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_KH, TILE_HD))
    new_k_r = k_r * cos_row - k_i * sin_row
    new_k_i = k_i * cos_row + k_r * sin_row
    ct.store(
        key,
        index=(batch_idx, seq_idx, 0, 0, 0),
        tile=new_k_r.reshape((1, 1, TILE_KH, 1, TILE_HD)).astype(key.dtype),
    )
    ct.store(
        key,
        index=(batch_idx, seq_idx, 0, 1, 0),
        tile=new_k_i.reshape((1, 1, TILE_KH, 1, TILE_HD)).astype(key.dtype),
    )


def _qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)
    # ALIGNED: both TILE_HD == head_dim_half (power-of-2) AND
    # TILE_QH == n_q_head AND TILE_KH == n_kv_head — no padding needed anywhere
    ALIGNED = (TILE_HD == head_dim_half) and (TILE_QH == n_q_head) and (TILE_KH == n_kv_head)

    n_row = batch_size * seq_len

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    q_5d = q.reshape(batch_size, seq_len, n_q_head, 2, head_dim_half)
    k_5d = k.reshape(batch_size, seq_len, n_kv_head, 2, head_dim_half)

    grid = (n_row,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel,
        (
            q_5d,
            k_5d,
            cos,
            sin,
            int(seq_len),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            float(1.0),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(ALIGNED),
        ),
    )

    q_out = q_5d.reshape(batch_size, seq_len, n_q_head, head_dim)
    k_out = k_5d.reshape(batch_size, seq_len, n_kv_head, head_dim)
    return q_out.transpose(1, 2), k_out.transpose(1, 2), cos, sin


def _qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = next_power_of_2(head_dim_half)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)
    ALIGNED = (TILE_HD == head_dim_half) and (TILE_QH == n_q_head) and (TILE_KH == n_kv_head)

    n_row = batch_size * seq_len

    dq = dq.contiguous()
    dk = dk.contiguous()

    dq_5d = dq.reshape(batch_size, seq_len, n_q_head, 2, head_dim_half)
    dk_5d = dk.reshape(batch_size, seq_len, n_kv_head, 2, head_dim_half)

    grid = (n_row,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel,
        (
            dq_5d,
            dk_5d,
            cos,
            sin,
            int(seq_len),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            float(-1.0),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(ALIGNED),
        ),
    )

    dq_out = dq_5d.reshape(batch_size, seq_len, n_q_head, head_dim)
    dk_out = dk_5d.reshape(batch_size, seq_len, n_kv_head, head_dim)
    return dq_out.transpose(1, 2), dk_out.transpose(1, 2)


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
