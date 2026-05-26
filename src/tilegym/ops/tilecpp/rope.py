# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ RoPE (Rotary Position Embedding) Operation
Implements RoPE forward and backward passes using CUDA C++ tile kernels.

"""

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types


def _next_power_of_2(n):
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


# Define kernels
_rope_forward_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "rope.cuh",
    kernel_name="rope_kernel",
)

_rope_backward_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "rope.cuh",
    kernel_name="rope_backward_kernel",
)


def rope_forward(q, k, cos, sin, rope_dim=None):
    """
    Apply rotary position encoding in forward pass using CUDA kernel.

    Supports both full RoPE (rope_dim is None, defaults to head_dim) and
    partial RoPE (rope_dim < head_dim). The kernel uses tile-space indexing
    so only the first ``rope_dim`` elements of head_dim are touched; the tail
    host-side slice or concat needed).

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor (modified in-place)
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor (modified in-place)
        cos: [*, seq_len, rope_dim] or [*, seq_len, rope_dim/2] - Cosine values
        sin: [*, seq_len, rope_dim] or [*, seq_len, rope_dim/2] - Sine values
        rope_dim: Number of head dims to rotate (None = full head_dim)

    Returns:
        (q, k, cos, sin) with q/k rotated in-place.
    """
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    if rope_dim is None:
        rope_dim = head_dim
    half_rope_dim = rope_dim // 2

    # Ensure cos/sin are 3-D: (cos_bs, seq_len, rope_dim).
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # cos/sin must arrive as [*, seq_len, rope_dim] (duplicate-frequency layout).
    # The half-frequency layout cos.shape[-1] == half_rope_dim is not supported.
    if cos.shape[-1] != rope_dim:
        raise NotImplementedError(
            f"tilecpp RoPE expects cos/sin last dim to equal rope_dim ({rope_dim}); "
            f"got {cos.shape[-1]}. The half-frequency layout is not supported."
        )

    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    cos_bs = cos.shape[0]

    cos = cos.reshape(cos_bs, seq_len, 2, half_rope_dim)
    sin = sin.reshape(cos_bs, seq_len, 2, half_rope_dim)

    BLOCK_HD = _next_power_of_2(half_rope_dim)
    BLOCK_QH = _next_power_of_2(n_q_head)
    BLOCK_KH = _next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    dtype = q.dtype
    dump_kernel_types("rope_kernel", q, k, cos, sin)

    # Template params: T, BATCH, Q_HEADS, K_HEADS, BLOCK_QH, BLOCK_KH, BLOCK_HD,
    #                  HALF_ROPE_DIM, HEAD_DIM, COS_BS, SEQ_LEN
    template_params = [
        batch_size,
        n_q_head,
        n_kv_head,
        BLOCK_QH,
        BLOCK_KH,
        BLOCK_HD,
        half_rope_dim,
        head_dim,
        cos_bs,
        seq_len,
    ]

    kernel, _, _ = _rope_forward_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature="{T}*, {T}*, {T}*, {T}*",
    )

    _rope_forward_kernel.launch(
        grid=n_row,
        kernel=kernel,
        args=[
            np.uint64(q.data_ptr()),
            np.uint64(k.data_ptr()),
            np.uint64(cos.data_ptr()),
            np.uint64(sin.data_ptr()),
        ],
    )

    return (
        q,  # already [B, n_q_head, seq_len, head_dim]
        k,
        cos.reshape(original_cos_shape),
        sin.reshape(original_sin_shape),
    )


def rope_backward(dq, dk, cos, sin, rope_dim=None):
    """
    Apply rotary position encoding backward pass using CUDA kernel.

    Supports full (rope_dim is None) and partial (rope_dim < head_dim) rope.
    Gradients for the passthrough tail are left unchanged (in-place).
    """
    batch_size, n_q_head, seq_len, head_dim = dq.shape
    n_kv_head = dk.shape[1]
    if rope_dim is None:
        rope_dim = head_dim
    half_rope_dim = rope_dim // 2

    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    cos_bs = cos.shape[0]

    if cos.shape[-1] != rope_dim:
        raise NotImplementedError(
            f"tilecpp RoPE expects cos/sin last dim to equal rope_dim ({rope_dim}); "
            f"got {cos.shape[-1]}. The half-frequency layout is not supported."
        )

    cos = cos.reshape(cos_bs, seq_len, 2, half_rope_dim)
    sin = sin.reshape(cos_bs, seq_len, 2, half_rope_dim)

    BLOCK_HD = _next_power_of_2(half_rope_dim)
    BLOCK_QH = _next_power_of_2(n_q_head)
    BLOCK_KH = _next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    dq = dq.contiguous()
    dk = dk.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    dtype = dq.dtype
    dump_kernel_types("rope_backward_kernel", dq, dk, cos, sin)

    template_params = [
        batch_size,
        n_q_head,
        n_kv_head,
        BLOCK_QH,
        BLOCK_KH,
        BLOCK_HD,
        half_rope_dim,
        head_dim,
        cos_bs,
        seq_len,
    ]

    kernel, _, _ = _rope_backward_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature="{T}*, {T}*, {T}*, {T}*",
    )

    _rope_backward_kernel.launch(
        grid=n_row,
        kernel=kernel,
        args=[
            np.uint64(dq.data_ptr()),
            np.uint64(dk.data_ptr()),
            np.uint64(cos.data_ptr()),
            np.uint64(sin.data_ptr()),
        ],
    )

    return dq, dk


class TileCppRopeFunction(torch.autograd.Function):
    """
    CUDA C++ Tile implementation of the Rotary Positional Embedding (RoPE) operation.

    This implements the HuggingFace Llama & Mistral version, whose rotation matrix
    is slightly different than the original RoPE paper.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rope_dim=None):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (*, seq_len, rope_dim)  - rope_dim==head_dim for full RoPE
        sin size: (*, seq_len, rope_dim)
        rope_dim: number of head_dim elements to rotate; None == head_dim
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin, rope_dim=rope_dim)
        ctx.save_for_backward(cos, sin)
        ctx.rope_dim = rope_dim
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin, rope_dim=ctx.rope_dim)
        return dq, dk, None, None, None, None, None


@register_impl("apply_rope_base", backend="tilecpp")
def apply_rope_base(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, partial_rotary_factor=1.0):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Cosine tensor
        sin: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Sine tensor
        position_ids: Optional - Position IDs tensor, default None
        unsqueeze_dim: Optional - Dimension to unsqueeze, default 1
        partial_rotary_factor: Fraction of head dims to rotate (default 1.0 = full RoPE)

    Returns:
        Query and key tensor pair with RoPE applied
    """
    rope_dim = None
    if partial_rotary_factor < 1.0:
        head_dim = q.shape[-1]
        rope_dim = int(head_dim * partial_rotary_factor)
        assert cos.shape[-1] == rope_dim, (
            f"cos last dim ({cos.shape[-1]}) must equal int(head_dim * partial_rotary_factor) "
            f"= int({head_dim} * {partial_rotary_factor}) = {rope_dim}"
        )
    return TileCppRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, rope_dim)


@register_impl("get_apply_rope_func", backend="tilecpp")
def get_apply_rope_func(model="llama"):
    if model in ("llama", "qwen2", "gemma3", "gpt-oss"):
        return apply_rope_base
    elif model == "qwen3_5":

        def wrapper(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rope_base(q, k, cos, sin, partial_rotary_factor=0.25)

        return wrapper
    elif model == "deepseek":

        def wrapper(q, k, freqs_cis):
            cos, sin = freqs_cis.real, freqs_cis.imag

            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            return apply_rope_base(q, k, cos, sin)

        return wrapper

    else:
        raise NotImplementedError(f"tilecpp RoPE does not support model={model!r}.")
