# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ MLA Decoding with Split-KV Implementation

Implements Multi-head Latent Attention with split-KV for efficient parallel processing.
Optimized to use tensor_span + partition_view with contiguous tensor layouts.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.splitk_reduce import splitk_reduce as _tilecpp_splitk_reduce
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_dtype_info
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
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
# Kernel Definitions
# =============================================================================

_mla_decode_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "mla_decoding_split_kv.cuh",
    kernel_name="naive_absorb_mla_transpose",
)


def _get_cpp_type(dtype: torch.dtype) -> str:
    """Get C++ type string for a torch dtype."""
    type_map = {
        torch.float32: "float",
        torch.float16: "__half",
        torch.bfloat16: "__nv_bfloat16",
    }
    return type_map.get(dtype, "float")


def _launch_mla_decode_kernel(
    Q: torch.Tensor,
    QPE: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    KPE: torch.Tensor,
    Att_Out: torch.Tensor,
    LSE_Out: torch.Tensor,
    sm_scale: float,
    B: int,
    NUM_HEADS: int,
    S_kv: int,
    kv_len_per_split: int,
    TILE_D: int,
    TILE_H: int,
    TILE_N: int,
    TILE_KPE: int,
    NUM_KV_SPLITS: int,
):
    """Launch the MLA decode split-KV kernel with all dimensions as template params."""
    dtype = Q.dtype
    cpp_type = _get_cpp_type(dtype)

    dump_kernel_types("naive_absorb_mla_transpose", Q, QPE, K, V, KPE)

    # All dimensions as template parameters for optimal code generation
    # Template: T, B, NUM_HEADS, S_KV, TILE_D, TILE_H, TILE_N, TILE_KPE, NUM_KV_SPLITS, KV_LEN_PER_SPLIT, EVEN_N
    EVEN_N = S_kv % TILE_N == 0
    template_params = [B, NUM_HEADS, S_kv, TILE_D, TILE_H, TILE_N, TILE_KPE, NUM_KV_SPLITS, kv_len_per_split, EVEN_N]

    kernel, _, _ = _mla_decode_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature=f"const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, {cpp_type}*, float*, float",
    )

    num_head_groups = cdiv(NUM_HEADS, TILE_H)
    grid = (num_head_groups, B, NUM_KV_SPLITS)

    _mla_decode_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(QPE.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(KPE.data_ptr()),
            np.uint64(Att_Out.data_ptr()),
            np.uint64(LSE_Out.data_ptr()),
            np.float32(sm_scale),
        ],
    )


# =============================================================================
# Autograd Function
# =============================================================================


class _mla_decoding_split_kv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, QPE, KV, KPE, sm_scale, kv_len_per_split=None):
        """
        MLA Decoding with Split-KV Forward.

        Args:
            Q: Query tensor [B, NUM_HEADS, HEAD_DIM]
            QPE: Query positional embedding [B, NUM_HEADS, KPE_DIM]
            KV: Key-Value tensor [B, S_kv, HEAD_DIM]
            KPE: Key positional embedding [B, S_kv, KPE_DIM]
            sm_scale: Softmax scale
            kv_len_per_split: KV length per split (optional)

        Returns:
            O: Output tensor [B, NUM_HEADS, HEAD_DIM]
        """
        B, NUM_HEADS, HEAD_DIM = Q.shape
        TILE_KPE = QPE.shape[-1]
        S_kv = KV.shape[1]
        TILE_D = HEAD_DIM

        TILE_H = 16  # Heads per tile
        TILE_N = 128  # KV sequence per tile

        # per-split KV span down/up through next_power_of_2(S_kv // estimate).
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_split_kv_estimated = max(1, NUM_SMS // B)
            kv_len_per_split = next_power_of_2(S_kv // num_split_kv_estimated)
            kv_len_per_split = max(kv_len_per_split, TILE_N)

        assert kv_len_per_split == next_power_of_2(kv_len_per_split)
        assert kv_len_per_split >= TILE_N
        NUM_KV_SPLITS = cdiv(S_kv, kv_len_per_split)

        # Allocate intermediate tensors with contiguous layout
        Att_Out = torch.empty(
            (B, NUM_HEADS, NUM_KV_SPLITS, TILE_D),
            device=Q.device,
            dtype=Q.dtype,
        )
        LSE_Out = torch.empty(
            (B, NUM_HEADS, NUM_KV_SPLITS),
            device=Q.device,
            dtype=torch.float32,
        )
        O = torch.empty_like(Q)

        # Launch decode kernel
        _launch_mla_decode_kernel(
            Q,
            QPE,
            KV,
            KV,
            KPE,  # K=V=KV
            Att_Out,
            LSE_Out,
            sm_scale,
            B,
            NUM_HEADS,
            S_kv,
            kv_len_per_split,
            TILE_D,
            TILE_H,
            TILE_N,
            TILE_KPE,
            NUM_KV_SPLITS,
        )

        _tilecpp_splitk_reduce(Att_Out, LSE_Out, O, S_kv)

        return O.reshape((B, NUM_HEADS, TILE_D))

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("MLA Decoding Split-KV backward is not implemented yet")


@register_impl("mla_decoding_split_kv", backend="tilecpp")
def mla_decoding_split_kv(q, qpe, kv, kpe, sm_scale=None, kv_len_per_split=None, **kwargs):
    """
    MLA Decoding with Split-KV interface

    Args:
        q: Query tensor [batch_size, seq_len, head_dim]
        qpe: Query positional embedding [batch_size, seq_len, kpe_dim]
        kv: Key-Value tensor [batch_size, kv_seq_len, head_dim]
        kpe: Key positional embedding [batch_size, kv_seq_len, kpe_dim]
        sm_scale: Softmax scale (defaults to 1/sqrt(head_dim + kpe_dim))
        kv_len_per_split: kv_len_per_split
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        o: Output tensor [batch_size, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    o = _mla_decoding_split_kv.apply(q, qpe, kv, kpe, sm_scale, kv_len_per_split)
    return o
