# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Flash Decode Operator
Implements attention decode for inference with grouped query attention
using CUDA C++ tile kernels.

Optimized to use proper tensor layouts for efficient tensor_span + partition_view access.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

# =============================================================================
# Helper Functions
# =============================================================================


def next_power_of_2(n: int) -> int:
    """Return the next power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =============================================================================
# Kernel Definition
# =============================================================================


_attention_decode_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "flash_decode.cuh",
    kernel_name="attention_decode_kernel_optimized",
)


# =============================================================================
# Kernel Launcher
# =============================================================================


def _launch_attention_decode_kernel_optimized(
    Q: torch.Tensor,  # [B, H_kv, Q_per_KV, D]
    K: torch.Tensor,  # [B, H_kv, S_kv, D]
    V: torch.Tensor,  # [B, H_kv, S_kv, D]
    Att_Out: torch.Tensor,  # [B, H_kv, Q_per_KV, splits, D]
    LSE_Out: torch.Tensor,  # [B, H_kv, Q_per_KV, splits]
    softmax_scale: float,
    B: int,
    H_kv: int,
    S_kv: int,
    num_q_head_per_kv: int,
    query_group_block_size: int,
    kv_len_per_split: int,
    head_dim: int,
    block_n: int,
    num_kv_splits: int,
    grid: tuple,
):
    """Launch the optimized attention_decode kernel."""
    dump_kernel_types("attention_decode_kernel_optimized", Q, K, V, Att_Out, LSE_Out)

    # Template params: T, B, H_KV, S_KV, Q_PER_KV, QUERY_GROUP_BLOCK_SIZE, KV_LEN_PER_SPLIT, HEAD_DIM, BLOCK_N, NUM_KV_SPLITS
    kernel, _, _ = _attention_decode_kernel.get_kernel(
        dtype=Q.dtype,
        template_params=[
            B,
            H_kv,
            S_kv,
            num_q_head_per_kv,
            query_group_block_size,
            kv_len_per_split,
            head_dim,
            block_n,
            num_kv_splits,
        ],
        signature="const {T}*, const {T}*, const {T}*, {T}*, float*, float",
    )

    _attention_decode_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(Att_Out.data_ptr()),
            np.uint64(LSE_Out.data_ptr()),
            np.float32(softmax_scale),
        ],
    )


# =============================================================================
# SplitK Reduce - use the CUDA Tile C++ kernel implementation
# =============================================================================


from tilegym.ops.tilecpp.splitk_reduce import splitk_reduce as _tilecpp_splitk_reduce


def splitk_reduce(Att_Mid_Out: torch.Tensor, LSE_Out: torch.Tensor, O: torch.Tensor, seq_len: int):
    """
    Reduce split-K attention results using the CUDA Tile C++ kernel.

    Args:
        Att_Mid_Out: Intermediate attention output [B, H, NUM_SPLITS, D]
        LSE_Out: Log-sum-exp values [B, H, NUM_SPLITS] (in log2 scale)
        O: Output tensor [B, H, D]
        seq_len: Sequence length (unused, kept for compatibility)
    """
    _tilecpp_splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)


# =============================================================================
# Main Function
# =============================================================================


class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, kv_len_per_split=None):
        """
        Grouped Query Attention implementation using optimized kernel with
        tensor_span + partition_view for efficient memory access.

        Args:
            Q: Query tensor of shape [batch_size, num_q_heads, 1, head_dim]
            K: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            V: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            softmax_scale: Scale factor for attention computation
            kv_len_per_split: Optional KV length per split for parallelization

        Returns:
            O: Output tensor of shape [batch_size, num_q_heads, 1, head_dim]
        """
        # Get dimensions
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        # Calculate grouped attention parameters
        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_block_size = max(8, next_power_of_2(num_q_head_per_kv))

        # Reshape Q for grouped query attention: [B, H_kv, Q_per_KV, D]
        # This makes memory layout contiguous for partition_view access
        Q_grouped = Q.view(batch_size, num_kv_heads, num_q_head_per_kv, head_dim).contiguous()
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim).contiguous()
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim).contiguous()

        # Calculate number of blocks
        BLOCK_N = 256  # Optimal block size for larger sequences

        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            NUM_KV_SPLITS = max(1, NUM_SMS // (batch_size * num_kv_heads))
            BLOCK_SIZE = max(BLOCK_N, next_power_of_2(cdiv(seq_len, NUM_KV_SPLITS)))
            NUM_KV_SPLITS = cdiv(seq_len, BLOCK_SIZE)
        else:
            NUM_KV_SPLITS = cdiv(seq_len, kv_len_per_split)
            BLOCK_SIZE = kv_len_per_split

        assert BLOCK_SIZE == next_power_of_2(BLOCK_SIZE)

        # Round up head_dim to next power of 2
        HEAD_DIM = next_power_of_2(head_dim)

        # Allocate intermediate results with 5D layout for contiguous access
        device = Q.device
        Att_Mid_Out_5D = torch.empty(
            (batch_size, num_kv_heads, num_q_head_per_kv, NUM_KV_SPLITS, head_dim),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out_4D = torch.empty(
            (batch_size, num_kv_heads, num_q_head_per_kv, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )

        # Prepare output
        O = torch.empty(batch_size, num_q_heads, head_dim, device=device, dtype=Q.dtype)

        # Launch kernel
        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)

        _launch_attention_decode_kernel_optimized(
            Q=Q_grouped,
            K=K,
            V=V,
            Att_Out=Att_Mid_Out_5D,
            LSE_Out=LSE_Out_4D,
            softmax_scale=softmax_scale,
            B=batch_size,
            H_kv=num_kv_heads,
            S_kv=seq_len,
            num_q_head_per_kv=num_q_head_per_kv,
            query_group_block_size=query_group_block_size,
            kv_len_per_split=BLOCK_SIZE,
            head_dim=HEAD_DIM,
            block_n=BLOCK_N,
            num_kv_splits=NUM_KV_SPLITS,
            grid=grid,
        )

        # Reshape outputs for splitk_reduce: [B, H_qo, splits, D]
        Att_Mid_Out = Att_Mid_Out_5D.view(batch_size, num_q_heads, NUM_KV_SPLITS, head_dim)
        LSE_Out = LSE_Out_4D.view(batch_size, num_q_heads, NUM_KV_SPLITS)

        # Reduce kernel splitk results
        splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)
        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Attention backward is not implemented yet")


attention_decode = _attention_decode.apply


@register_impl("fmha_decode", backend="tilecpp")
def fmha_decode(q, k, v, sm_scale, kv_len_per_split=None, **kwargs):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    o = attention_decode(q, k, v, sm_scale, kv_len_per_split)
    return o
