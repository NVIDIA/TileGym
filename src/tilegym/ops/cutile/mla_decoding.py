# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import numpy as np
import math

import torch
import cuda.tile as ct

from tilegym.backend import register_impl
from cuda.tile._numeric_semantics import RoundingMode as RMd

# Type aliases for constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel
def naive_absorb_mla_transpose(
    Q,
    QPE,
    K,
    V,
    KPE,
    Out,
    L,
    sm_scale: float,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    TILE_H: ConstInt,
    TILE_N: ConstInt,
    TILE_KPE: ConstInt,  # TILE_KPE = position embedding size
    S_kv: ConstInt,
    EVEN_N: ConstBool,
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y
    qk_scale = sm_scale * INV_LOG_2

    # Initialize accumulation variables
    m_prev = ct.full((TILE_H,), -math.inf, dtype=ct.float32)
    l_prev = ct.full((TILE_N, TILE_H), 1.0, dtype=ct.float32)
    acc = ct.zeros((TILE_D, TILE_H), dtype=ct.float32)

    # Load query and query position encoding
    q = ct.load(
        Q,
        index=(batch_idx, 0, bid_x),
        shape=(1, TILE_D, TILE_H),
        order=(0, 2, 1),
        allow_tma=True,
    )
    q = ct.reshape(q, (TILE_D, TILE_H))
    qpe = ct.load(
        QPE,
        index=(batch_idx, 0, bid_x),
        shape=(1, TILE_KPE, TILE_H),
        order=(0, 2, 1),
        allow_tma=True,
    )
    qpe = ct.reshape(qpe, (TILE_KPE, TILE_H))

    # Loop over key-value pairs and update accumulator
    end_n = S_kv
    cnt = 0
    mask_start = S_kv // TILE_N * TILE_N
    offs_n = ct.arange(TILE_N, dtype=ct.int32)
    for curr_n in range(0, end_n, TILE_N):
        # Load key and compute Q@K^T
        k = ct.load(
            K,
            index=(batch_idx, cnt, 0),
            shape=(1, TILE_N, TILE_D),
            allow_tma=True,
        )
        k = ct.reshape(k, (TILE_N, TILE_D))
        qk = ct.full((TILE_N, TILE_H), 0.0, dtype=ct.float32)
        qk = ct.mma(k, q, qk)

        # Load key position encoding and compute QPE@KPE^T
        kpe = ct.load(
            KPE,
            index=(batch_idx, cnt, 0),
            shape=(1, TILE_N, TILE_KPE),
            allow_tma=True,
        )
        kpe = ct.reshape(kpe, (TILE_N, TILE_KPE))
        qk = ct.mma(kpe, qpe, qk)

        if not EVEN_N and curr_n >= mask_start:
            mask = (curr_n + offs_n[:, None]) < S_kv
            qk = ct.where(mask, qk, -1.0e6)

        # Apply scaling and compute attention scores
        # qk = ct.astype(qk, ct.float32)
        m_ij = ct.maximum(m_prev, ct.max(qk, 0) * qk_scale)
        qk = qk * qk_scale - m_ij[None, :]

        # Compute attention weights and update running statistics
        p = ct.exp2(qk)
        alpha = ct.exp2(m_prev - m_ij)
        l_prev = l_prev * alpha[None, :] + p
        acc = acc * alpha[None, :]

        # Load value and compute attention @ value
        v = ct.load(
            V,
            index=(cnt, batch_idx, 0),
            shape=(TILE_N, 1, TILE_D),
            order=(1, 0, 2),
            allow_tma=True,
        )
        v = ct.reshape(v, (TILE_N, TILE_D))
        v = ct.transpose(v, axis0=1, axis1=0)
        p = ct.astype(p, Q.dtype)
        acc = ct.mma(v, p, acc)
        m_prev = m_ij
        cnt += 1

    # Finalize attention computation
    l_prev = ct.sum(l_prev, 0)
    acc = ct.truediv(acc, (l_prev[None, :]), flush_to_zero=True, rounding_mode=RMd.APPROX)
    l_prev = m_prev + ct.log2(l_prev)

    # Store results
    l_prev = ct.reshape(l_prev, (1, TILE_H))
    ct.store(L, index=(batch_idx, bid_x), tile=l_prev, allow_tma=True)

    acc = ct.astype(acc, Out.dtype)
    acc = ct.transpose(acc, axis0=1, axis1=0)
    acc = ct.reshape(acc, (1, TILE_H, TILE_D))
    ct.store(Out, index=(batch_idx, bid_x, 0), tile=acc, allow_tma=True)


class _mla_decoding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        qpe,
        kv,
        kpe,
        sm_scale,
        TILE_H,
        TILE_N,
        num_ctas,
    ):
        # Setup stride and shape
        B, num_head, TILE_D = q.shape
        TILE_KPE = kpe.shape[2]
        S_kv = kv.shape[1]
        o = torch.empty_like(q)
        l = torch.empty((B, num_head), device=q.device, dtype=torch.float32)

        # Launch fmha fwd kernel
        grid = (math.ceil(num_head / TILE_H), B, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            naive_absorb_mla_transpose,
            (
                q,
                qpe,
                kv,
                kv,
                kpe,
                o,
                l,
                sm_scale,
                TILE_D,
                TILE_H,
                TILE_N,
                TILE_KPE,
                S_kv,
                (S_kv % TILE_N) == 0,
            ),
        )
        return o, l

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError()


@register_impl("mla_decoding", backend="cutile")
def mla_decoding(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: float,
    transpose: bool = True,
    **kwargs,
) -> torch.Tensor:
    TILE_H = 16
    TILE_N = 128
    num_ctas = None  # Let compiler auto-pick
    assert transpose == True, "CuTile MLA Decoding only supports transpose=True"
    if sm_scale is None:
        sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))
    return _mla_decoding.apply(q, qpe, kv, kpe, sm_scale, TILE_H, TILE_N, num_ctas)
