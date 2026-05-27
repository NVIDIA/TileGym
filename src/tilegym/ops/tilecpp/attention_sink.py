# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Attention Sink implementation.
Implements attention with attention sinks for streaming/infinite context.
Supports sliding window attention with configurable bandwidth.
"""

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl

from .utils._cuda_utils import TileCppKernel
from .utils._dump_types import dump_kernel_types


def _cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# Define kernel
_attn_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention_sink.cuh",
    kernel_name="attention_sink_fwd_kernel",
)


def _launch_attn_fwd_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    m: torch.Tensor,
    out: torch.Tensor,
    start_q: int,
    Z: int,
    H: int,
    N_Q_CTX: int,
    N_KV_CTX: int,
    bandwidth: int,
    HEAD_DIM: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """Launch the attention sink forward kernel."""
    dump_kernel_types("attention_sink_fwd_kernel", q, k, v)
    dtype = q.dtype

    has_bandwidth = bandwidth is not None and bandwidth > 0

    kernel, _, _ = _attn_fwd_kernel.get_kernel(
        dtype=dtype,
        template_params=[HEAD_DIM, BLOCK_M, BLOCK_N, has_bandwidth],
        signature="{T}*, {T}*, {T}*, {T}*, float, float*, {T}*, int, int, int, int, int, int",
    )

    grid = (_cdiv(N_Q_CTX, BLOCK_M), Z * H, 1)

    sinks_ptr = np.uint64(sinks.data_ptr()) if sinks is not None else np.uint64(0)
    bandwidth_val = bandwidth if bandwidth is not None else 0

    _attn_fwd_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(q.data_ptr()),
            np.uint64(k.data_ptr()),
            np.uint64(v.data_ptr()),
            sinks_ptr,
            np.float32(sm_scale),
            np.uint64(m.data_ptr()),
            np.uint64(out.data_ptr()),
            np.int32(start_q),
            np.int32(Z),
            np.int32(H),
            np.int32(N_Q_CTX),
            np.int32(N_KV_CTX),
            np.int32(bandwidth_val),
        ],
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape
        n_heads = n_kv_heads * repeat_kv
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        # The kernel uses BLOCK_N = 64 with static_assert(BLOCK_N <= HEAD_DIM),
        # so head_dim < 64 will fail to compile via nvcc.
        assert HEAD_DIM_K in {64, 128, 256}, (
            f"tilecpp attention_sink supports head_dim in {{64, 128, 256}}, got {HEAD_DIM_K}"
        )

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        BLOCK_M = 64
        BLOCK_N = 64
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        # pad q to multiple of its block size in the n_ctx dimension (-2)
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        # pad k and v to multiple of their block size in the n_kv_ctx dimension
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)

        start_q_val = start_q.item() if isinstance(start_q, torch.Tensor) else int(start_q)

        _launch_attn_fwd_kernel(
            q,
            k,
            v,
            sinks,
            sm_scale,
            M,
            o,
            start_q_val,
            bs,
            n_heads,
            n_ctx + m_pad_size,
            n_kv_ctx + n_pad_size,
            bandwidth,
            HEAD_DIM_K,
            BLOCK_M,
            BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass for tilecpp attention_sink is not implemented.")


attention = _Attention.apply


@register_impl("attention_sink", backend="tilecpp")
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
    CUDA Tile C++ Attention Sink.

    Implements attention with attention sinks for streaming/infinite context.
    Supports sliding window attention with configurable bandwidth.

    Args:
        query: Query tensor [bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM]
        key: Key tensor [bs, n_kv_ctx, n_kv_heads, HEAD_DIM]
        value: Value tensor [bs, n_kv_ctx, n_kv_heads, HEAD_DIM]
        sinks: Attention sinks per head [H]
        sm_scale: Softmax scale
        sliding_window: Sliding window bandwidth (None for full attention)
        start_q: Starting position for queries

    Returns:
        Output tensor [bs, n_ctx, n_heads * HEAD_DIM]
    """
    # Coerce a Python int default into a 1-element int32 tensor on the input device.
    if isinstance(start_q, torch.Tensor):
        start_q_tensor = start_q.to(torch.int32).contiguous()
        if start_q_tensor.device.type != "cuda":
            start_q_tensor = start_q_tensor.cuda()
    else:
        start_q_tensor = torch.tensor([int(start_q)], dtype=torch.int32, device=query.device)

    return attention(query, key, value, sinks, sm_scale, sliding_window, start_q_tensor)
