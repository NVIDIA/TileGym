# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import numpy as np
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl

# Type aliases for constants
ConstInt = ct.Constant[int]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel
def prefill_mla(
    Q,
    QPE,
    K,
    KPE,
    V,
    Out,
    qk_scale: float,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    TILE_KPE: ConstInt,  # TILE_KPE = position embedding size
    H: int,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    if QUERY_GROUP_SIZE > 0:
        off_kv_h = head_idx // QUERY_GROUP_SIZE
    else:
        off_kv_h = head_idx
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = ct.expand_dims(offs_m, 1)
    offs_n = ct.arange(TILE_N, dtype=ct.int32)
    offs_n = ct.expand_dims(offs_n, 0)

    # Initialize m, l, acc
    m_i = ct.full((TILE_M,), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M,), 1.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load q
    q = ct.load(
        Q,
        index=(batch_idx, head_idx, bid_x, 0),
        shape=(1, 1, TILE_M, TILE_D),
    )
    q = ct.reshape(q, (TILE_M, TILE_D))

    # Load qpe
    qpe = ct.load(
        QPE,
        index=(batch_idx, head_idx, bid_x, 0),
        shape=(1, 1, TILE_M, TILE_KPE),
    )
    qpe = ct.reshape(qpe, (TILE_M, TILE_KPE))

    # Stage 1 inline:
    start_m = bid_x
    lo, mask_start, hi = 0, start_m * TILE_M, (start_m + 1) * TILE_M
    mask_start = mask_start // TILE_N
    hi = ct.cdiv(hi, TILE_N)
    for j in range(0, hi):
        curr_n = j * TILE_N
        # Compute qk
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
        )
        k = ct.reshape(k, (TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Add position embedding contribution
        kpe = ct.load(
            KPE,
            index=(batch_idx, 0, 0, j),
            shape=(1, 1, TILE_KPE, TILE_N),
            order=(0, 1, 3, 2),
        )
        kpe = ct.reshape(kpe, (TILE_KPE, TILE_N))
        qk = ct.mma(qpe, kpe, qk)

        # Apply mask
        if j >= mask_start:
            mask = offs_m >= (curr_n + offs_n)
            qk = ct.where(mask, qk, -1.0e6)

        # Stage 1 special handling
        m_ij = ct.maximum(m_i, ct.max(qk, axis=-1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        # Scale acc
        acc = acc * alpha[:, None]

        # Compute pv
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
        )
        v = ct.reshape(v, (TILE_N, TILE_D))
        p = ct.astype(p, Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i[:, None], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.reshape(acc, (1, 1, TILE_M, TILE_D))
    acc = ct.astype(acc, Q.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, qpe, k, kpe, v, sm_scale, IS_CAUSAL, kernel_configs):
        assert IS_CAUSAL, "CuTile MLA only supports IS_CAUSAL=True"
        # Setup stride and shape
        B, H, S_qo, TILE_D = q.shape
        TILE_KPE = qpe.shape[3]
        assert k.shape == v.shape
        num_head_kv = k.shape[1]
        S_kv = k.shape[2]
        o = torch.empty_like(q)

        if H == num_head_kv:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)
        # Launch fmha fwd kernel
        grid = (math.ceil(S_qo / kernel_configs.get("TILE_M", 256)), B * H, 1)
        TILE_M = kernel_configs.get("TILE_M", 256)
        TILE_N = kernel_configs.get("TILE_N", 128)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            prefill_mla,
            (
                q,
                qpe,
                k,
                kpe,
                v,
                o,
                sm_scale,
                TILE_D,
                TILE_KPE,
                H,
                TILE_M,
                TILE_N,
                query_group_size,
            ),
        )
        ctx.save_for_backward(q, k, v, o)
        ctx.sm_scale = sm_scale
        ctx.shapes = (B, H, S_qo, S_kv)
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass is not implemented for CuTile MLA")


class Attention:
    def __init__(self, IS_CAUSAL, kernel_configs):
        self.IS_CAUSAL = IS_CAUSAL
        self.kernel_configs = kernel_configs

    def __call__(self, q, k, v, sm_scale, qpe=None, kpe=None):
        c = _attention.apply(
            q,
            qpe,
            k,
            kpe,
            v,
            sm_scale,
            self.IS_CAUSAL,
            self.kernel_configs,
        )
        return c


def tile_mla(q, k, v, qpe, kpe, is_causal, scaling, **kwargs):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

    defaults = {"TILE_M": 256, "TILE_N": 128}
    user_cfg = kwargs.get("kernel_configs")
    if user_cfg is None:
        kernel_configs = defaults
    else:
        kernel_configs = {**defaults, **user_cfg}
    attention = Attention(is_causal, kernel_configs)
    o = attention(q, k, v, scaling, qpe, kpe)
    return o


register_impl("mla", "cutile")(tile_mla)
