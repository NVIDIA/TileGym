# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import numpy as np
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl

INV_LOG_2 = 1.0 / math.log(2)

# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel(occupancy=2)
def attention_sink_kernel(
    Q,
    K,
    V,
    Sinks,
    Out,
    Start_q,
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    N_KV_CTX: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    BANDWIDTH: ConstInt,
):
    """
    CuTile kernel for Fused Multi-Head Attention with Sink Tokens.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Load start_q from tensor
    start_q_tile = ct.load(Start_q, index=(0,), shape=(1,))
    start_q = start_q_tile.reshape(()).astype(np.int32)

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Load attention sink value for this head
    sink = ct.load(Sinks, index=(head_idx,), shape=(1,))
    sink = sink.astype(np.float32)
    sink = sink.reshape(())
    sink_scaled = sink * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m = offs_m[:, None]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Initialize online softmax accumulators
    # m_i starts at sink_scaled so sink contributes to max
    m_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32) + sink_scaled
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    # Compute loop bounds using loaded start_q
    if BANDWIDTH > 0:
        lo = ct.maximum(0, start_q + bid_x * TILE_M - BANDWIDTH)
        hi = start_q + (bid_x + 1) * TILE_M
    else:
        lo = 0
        hi = start_q + (bid_x + 1) * TILE_M

    hi = ct.minimum(hi, N_KV_CTX)

    Tc = ct.cdiv(hi, TILE_N)
    start_block = lo // TILE_N

    # Loop over K, V blocks
    for j in range(start_block, Tc):
        start_n = j * TILE_N
        offs_n = start_n + offs_n_tile

        # Load K transposed - increased latency hint from 2 to 6 based on IR analysis
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=6,
        )
        k = k.reshape((TILE_D, TILE_N))

        # Compute QK
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply causal masking
        query_pos = start_q + offs_m
        causal_mask = offs_n > query_pos

        oob_mask = offs_n >= N_KV_CTX
        mask = causal_mask | oob_mask

        if BANDWIDTH > 0:
            too_old = offs_n < (query_pos - BANDWIDTH + 1)
            mask = mask | too_old

        qk = qk + ct.where(mask, -1.0e6, 0.0)

        # Online Softmax Update
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)

        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V - increased latency hint from 4 to 6 based on IR analysis
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=6,
        ).reshape((TILE_N, TILE_D))

        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Add sink contribution to softmax denominator
    sink_exp = ct.exp2(sink_scaled - m_i, flush_to_zero=True)
    z = l_i + sink_exp

    # Final Normalization and Store
    acc = ct.truediv(acc, z, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def _attention_sink_autotune_configs():
    """
    Iterator of autotune configurations for attention_sink kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    for TILE_M in [256, 128, 64]:
        for TILE_N in [128, 64]:
            for occupancy in [1, 2, 4]:
                yield SimpleNamespace(TILE_M=TILE_M, TILE_N=TILE_N, num_ctas=1, occupancy=occupancy)


def _cutile_autotune_attention_sink(
    stream,
    q,
    k,
    v,
    sinks,
    o,
    start_q,
    sm_scale,
    head_dim,
    n_heads,
    n_kv_ctx,
    repeat_kv,
    bandwidth,
):
    """Autotuned kernel launch."""
    batch_size, _, n_ctx, _ = q.shape

    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(n_ctx / cfg.TILE_M),
            batch_size * n_heads,
            1,
        ),
        kernel=attention_sink_kernel,
        args_fn=lambda cfg: (
            q,
            k,
            v,
            sinks,
            o,
            start_q,
            sm_scale,
            head_dim,
            n_heads,
            n_kv_ctx,
            cfg.TILE_M,
            cfg.TILE_N,
            repeat_kv,
            bandwidth,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_attention_sink_autotune_configs,
    )


def attention_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
    **kwargs,
):
    """
    Attention with sink tokens using CuTile.
    """
    # Extract dimensions
    bs, n_ctx, n_kv_heads, repeat_kv, head_dim = query.shape
    _, n_kv_ctx, _, _ = key.shape
    n_heads = n_kv_heads * repeat_kv

    # Reshape query to merge kv_heads and repeat_kv
    q = query.view(bs, n_ctx, n_heads, head_dim)
    k = key.view(bs, n_kv_ctx, n_kv_heads, head_dim)
    v = value.view(bs, n_kv_ctx, n_kv_heads, head_dim)

    # Transpose to [bs, heads, seq_len, head_dim]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # Allocate output
    o = torch.empty_like(q)

    # Ensure start_q is a tensor on GPU (CRITICAL: avoid .item() which causes sync)
    if isinstance(start_q, torch.Tensor):
        start_q_tensor = start_q.to(torch.int32).contiguous()
        if start_q_tensor.device.type != "cuda":
            start_q_tensor = start_q_tensor.cuda()
    else:
        start_q_tensor = torch.tensor([int(start_q)], dtype=torch.int32, device=query.device)

    # Bandwidth (0 means no sliding window)
    bandwidth = sliding_window if sliding_window is not None else 0

    # Use autotune
    enable_autotune = os.environ.get("DISABLE_AUTOTUNE", "0") != "1"

    if enable_autotune:
        _cutile_autotune_attention_sink(
            torch.cuda.current_stream(),
            q,
            k,
            v,
            sinks,
            o,
            start_q_tensor,
            sm_scale,
            head_dim,
            n_heads,
            n_kv_ctx,
            repeat_kv,
            bandwidth,
        )
    else:
        # Default configuration
        TILE_M = 128
        TILE_N = 128
        grid = (math.ceil(n_ctx / TILE_M), bs * n_heads, 1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_sink_kernel,
            (
                q,
                k,
                v,
                sinks,
                o,
                start_q_tensor,
                sm_scale,
                head_dim,
                n_heads,
                n_kv_ctx,
                TILE_M,
                TILE_N,
                repeat_kv,
                bandwidth,
            ),
        )

    # Transpose back and reshape output
    o = o.transpose(1, 2).contiguous()
    o = o.view(bs, n_ctx, n_heads * head_dim)

    return o


# Register cutile implementation for attention_sink
register_impl("attention_sink", "cutile")(attention_sink)
