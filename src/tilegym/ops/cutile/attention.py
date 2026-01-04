# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)

# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# --- FMHA Kernel Implementation ---
@ct.kernel(occupancy=2)
def fmha_kernel(
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    input_pos: int,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool,
):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape(
        (TILE_M, TILE_D)
    )  # [TILE_M, TILE_D]

    # Loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # When kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # When kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            # Out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # Causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -math.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # Attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # Scale acc
        acc = acc * alpha  # [TILE_M, TILE_N]

        # --- Compute PV product ---
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def _fmha_autotune_configs():
    """
    Iterator of autotune configurations for FMHA kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
    else:
        # sm100 (Blackwell)
        yield SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)


def cutile_autotune_fmha(
    stream,
    q,
    k,
    v,
    o,
    sm_scale,
    input_pos,
    hidden_size,
    num_heads,
    query_group_size,
    is_causal,
    EVEN_K,
):
    batch_size, _, q_len, _ = q.shape
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(q_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_kernel,
        args_fn=lambda cfg: (
            q,
            k,
            v,
            o,
            sm_scale,
            input_pos,
            hidden_size,
            num_heads,
            cfg.TILE_M,
            cfg.TILE_N,
            query_group_size,
            is_causal,
            EVEN_K,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_fmha_autotune_configs,
    )
    return o


def tile_prefill_fmha(q, k, v, sm_scale, is_causal=True, kernel_configs=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv

    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q)

    input_pos = 0  # prefill, causal

    max_tile_n = max(cfg.TILE_N for cfg in _fmha_autotune_configs())
    EVEN_K = (k_len % max_tile_n) == 0
    return cutile_autotune_fmha(
        torch.cuda.current_stream(),
        q,
        k,
        v,
        o,
        sm_scale,
        input_pos,
        hidden_size,
        num_heads,
        query_group_size,
        is_causal,
        EVEN_K,
    )


def tile_fmha(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    kernel_configs = kwargs.get("kernel_configs", None)
    o = tile_prefill_fmha(q, k, v, scaling, is_causal, kernel_configs)
    return o


# Register cutile implementation for fmha
register_impl("fmha", "cutile")(tile_fmha)
