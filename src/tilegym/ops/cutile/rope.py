# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def rope_kernel(
    q,
    k,
    cos,
    sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    # q size: (bsz, seq_len, num_q_heads, 2, head_dim)
    # k size: (bsz, seq_len, num_kv_heads, 2, head_dim)
    # cos size: (1, seq_len, *, head_dim) or (bsz, seq_len, , head_dim)
    cos_bs = cos.shape[0]

    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # ####################################################################
    # Load cos and sin values
    # ####################################################################
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))

    # ####################################################################
    # Process Q tensor
    # ####################################################################
    q_tile_1 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype),
    )
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype),
    )

    # ####################################################################
    # Process K tensor
    # ####################################################################
    k_tile_1 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype),
    )
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype),
    )


def rope_forward(q, k, cos, sin):
    """
    Apply rotary position encoding in forward pass

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Cosine values
        sin: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Sine values

    Returns:
        Query and key tensors with RoPE applied
    """
    # Calculate padded dimensions
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    q = q.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    assert cos.shape[-1] == head_dim // 2 or cos.shape[-1] == head_dim, (
        f"cos.shape[-1]: {cos.shape[-1]}, head_dim: {head_dim}"
    )
    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2)

    half_head_dim = q.shape[-1]
    TILE_HD = next_power_of_2(half_head_dim)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len
    grid = (n_row, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        rope_kernel,
        (
            q,
            k,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            TILE_QH,
            TILE_KH,
            TILE_HD,
        ),
    )

    return (
        q.reshape(batch_size, n_q_head, seq_len, head_dim),
        k.reshape(batch_size, n_kv_head, seq_len, head_dim),
        cos.reshape(original_cos_shape),
        sin.reshape(original_sin_shape),
    )


class TileRopeFunction(torch.autograd.Function):
    """
    CUDA Tile implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        Backward pass not yet implemented
        """
        raise NotImplementedError("Backward pass is not implemented for TileRopeFunction")


@register_impl("apply_rope_base", backend="cutile")
def apply_rope_base(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Cosine tensor
        sin: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Sine tensor
        position_ids: Optional - Position IDs tensor, default None
        unsqueeze_dim: Optional - Dimension to unsqueeze, default 1

    Returns:
        Query and key tensor pair with RoPE applied
    """
    return TileRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


@register_impl("get_apply_rope_func", backend="cutile")
def get_apply_rope_func(model="llama"):
    if model == "llama" or model == "qwen2":
        return apply_rope_base
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
        raise ValueError(f"Unsupported model: {model}")
