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
    TILE_RD: ConstInt,
):
    """
    Unified RoPE kernel operating in-place on 4-D Q/K tensors.

    Works for both full RoPE (TILE_RD = head_dim // 2) and partial RoPE
    (TILE_RD = rope_dim // 2 < head_dim // 2).  Tile-space index
    ``dim_tile=0`` selects ``[0 : TILE_RD]`` and ``dim_tile=1`` selects
    ``[TILE_RD : 2*TILE_RD]``, so only the first ``2*TILE_RD`` elements
    of head_dim are rotated; the rest pass through unchanged.

    q shape: (bsz, num_q_heads, seq_len, head_dim)  — in-place
    k shape: (bsz, num_kv_heads, seq_len, head_dim)  — in-place
    cos shape: (cos_bs, seq_len, rope_dim)            — 3-D
    sin shape: (cos_bs, seq_len, rope_dim)            — 3-D
    """
    cos_bs = cos.shape[0]

    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # ####################################################################
    # Load cos and sin values — first half of rope_dim
    # ####################################################################
    cos_row = ct.load(cos, index=(cos_batch_idx, row_idx, 0), shape=(1, 1, TILE_RD), padding_mode=PAD_ZERO).reshape(
        (1, TILE_RD)
    )
    sin_row = ct.load(sin, index=(cos_batch_idx, row_idx, 0), shape=(1, 1, TILE_RD), padding_mode=PAD_ZERO).reshape(
        (1, TILE_RD)
    )

    # ####################################################################
    # Process Q tensor — first half [0:rope_dim//2] and second half
    # [rope_dim//2:rope_dim] via tile-space dim_tile indexing
    # ####################################################################
    q_tile_1 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 0),
        shape=(1, TILE_QH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_RD))
    q_tile_2 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 1),
        shape=(1, TILE_QH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_RD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 0),
        tile=new_q_tile_1.reshape((1, TILE_QH, 1, TILE_RD)).astype(q.dtype),
    )
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 1),
        tile=new_q_tile_2.reshape((1, TILE_QH, 1, TILE_RD)).astype(q.dtype),
    )

    # ####################################################################
    # Process K tensor — same pattern
    # ####################################################################
    k_tile_1 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 0),
        shape=(1, TILE_KH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_RD))
    k_tile_2 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 1),
        shape=(1, TILE_KH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_RD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 0),
        tile=new_k_tile_1.reshape((1, TILE_KH, 1, TILE_RD)).astype(k.dtype),
    )
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 1),
        tile=new_k_tile_2.reshape((1, TILE_KH, 1, TILE_RD)).astype(k.dtype),
    )


def rope_forward(q, k, cos, sin, rope_dim=None):
    """
    Apply rotary position encoding **in-place** on 4-D Q/K tensors.

    Supports both full RoPE (rope_dim is None) and partial RoPE
    (rope_dim < head_dim).  The unified kernel uses tile-space indexing so
    only the first ``rope_dim`` elements are touched; no host-side reshape,
    slice, or concatenation is needed.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor (modified in-place)
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor (modified in-place)
        cos: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Cosine values
        sin: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Sine values
        rope_dim: Number of head dimensions to rotate (None = full head_dim)

    Returns:
        (q, k, cos, sin) — q and k rotated in-place.
    """
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]

    if rope_dim is None:
        rope_dim = head_dim
    half_rope_dim = rope_dim // 2

    # Ensure cos/sin are 3-D: (cos_bs, seq_len, rope_dim)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    TILE_RD = next_power_of_2(half_rope_dim)
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
            TILE_RD,
        ),
    )

    return q, k, cos, sin


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
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rope_dim=None):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, rope_dim) or (bsz, seq_len, rope_dim)  — rope_dim == head_dim for full RoPE
        sin size: same as cos
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin, rope_dim=rope_dim)
        ctx.save_for_backward(cos, sin)
        ctx.rope_dim = rope_dim
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        Backward pass via inverse rotation.
        """
        cos, sin = ctx.saved_tensors
        dq = dq.contiguous()
        dk = dk.contiguous()
        dq_out, dk_out, _, _ = rope_forward(dq, dk, cos, -sin, rope_dim=ctx.rope_dim)
        return dq_out, dk_out, None, None, None, None, None


@register_impl("apply_rope_base", backend="cutile")
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
    return TileRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, rope_dim)


@register_impl("get_apply_rope_func", backend="cutile")
def get_apply_rope_func(model="llama"):
    if model == "llama" or model == "qwen2" or model == "gemma3" or model == "gpt-oss":
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
        raise ValueError(f"Unsupported model: {model}")
