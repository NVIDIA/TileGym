# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import register_impl


@triton.jit
def _triton_rope_kernel_tma(
    Q_desc,
    K_desc,
    cos_desc,
    sin_desc,
    # Q tensor parameters
    cos_bs: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    dtype = Q_desc.type.block_type.element_ty
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    row_idx = pid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    cos_row = cos_desc.load([cos_batch_idx, row_idx, 0, 0]).reshape((1, BLOCK_HD))
    sin_row = sin_desc.load([cos_batch_idx, row_idx, 0, 0]).reshape((1, BLOCK_HD))
    q_tile_1 = Q_desc.load([batch_idx, 0, row_idx, 0, 0]).reshape((BLOCK_QH, BLOCK_HD))
    q_tile_2 = Q_desc.load([batch_idx, 0, row_idx, 1, 0]).reshape((BLOCK_QH, BLOCK_HD))
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    Q_desc.store(
        [batch_idx, 0, row_idx, 0, 0],
        new_q_tile_1.reshape((1, BLOCK_QH, 1, 1, BLOCK_HD)).to(dtype),
    )
    Q_desc.store(
        [batch_idx, 0, row_idx, 1, 0],
        new_q_tile_2.reshape((1, BLOCK_QH, 1, 1, BLOCK_HD)).to(dtype),
    )
    k_tile_1 = K_desc.load([batch_idx, 0, row_idx, 0, 0]).reshape((BLOCK_KH, BLOCK_HD))
    k_tile_2 = K_desc.load([batch_idx, 0, row_idx, 1, 0]).reshape((BLOCK_KH, BLOCK_HD))
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    K_desc.store(
        [batch_idx, 0, row_idx, 0, 0],
        new_k_tile_1.reshape((1, BLOCK_KH, 1, 1, BLOCK_HD)).to(dtype),
    )
    K_desc.store(
        [batch_idx, 0, row_idx, 1, 0],
        new_k_tile_2.reshape((1, BLOCK_KH, 1, 1, BLOCK_HD)).to(dtype),
    )


# Adapted from Liger-Kernel: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py


@triton.jit
def _rope_kernel(
    q,
    q_row_stride,
    k,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
    rope_hd: tl.constexpr = 0,
    pad_rope_hd: tl.constexpr = 0,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)
    # cos size: (1, seq_len, rope_hd) or (bsz, seq_len, rope_hd)  — rope_hd == hd for full RoPE
    # stride: (seq_len * rope_hd, rope_hd, 1)

    # When rope_hd == 0, fall back to full-dim RoPE (rope_hd == hd)
    effective_rope_hd: tl.constexpr = hd if rope_hd == 0 else rope_hd
    effective_pad_rope_hd: tl.constexpr = pad_hd if pad_rope_hd == 0 else pad_rope_hd

    # Cast to int64 to prevent int32 overflow in pointer offset arithmetic.
    # For large configs (e.g. bsz=8, seq_len=65536, n_qh=32, hd=256),
    # pid * q_row_stride exceeds INT32_MAX and wraps to a negative offset.
    pid = tl.program_id(0).to(tl.int64)

    # Locate start address
    q = q + pid * q_row_stride
    k = k + pid * k_row_stride

    # ####################################################################
    # Get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # Program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    batch_idx = pid // sl
    cos_row_idx = pid % sl
    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    cos_offsets = tl.arange(0, effective_pad_rope_hd // 2)
    cos_mask = cos_offsets < effective_rope_hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

    # ####################################################################
    # Process Q and K tensors: load the left and right half of the rotary
    # portion for the current token.  When rope_hd < hd (partial RoPE),
    # only the first rope_hd dims are touched; the rest pass through.
    # ####################################################################
    # Left half of the rotary portion  [0 : rope_hd//2]
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, effective_pad_rope_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, effective_pad_rope_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, effective_pad_rope_hd // 2)[None, :] < effective_rope_hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, effective_pad_rope_hd // 2)[None, :] < effective_rope_hd // 2
    )
    q_tile_1 = tl.load(q + first_half_q_offsets, mask=first_q_mask, other=0).to(tl.float32)
    k_tile_1 = tl.load(k + first_half_k_offsets, mask=first_k_mask, other=0).to(tl.float32)

    # Right half of the rotary portion  [rope_hd//2 : rope_hd]
    second_half_q_offsets = first_half_q_offsets + (effective_rope_hd // 2)
    second_half_k_offsets = first_half_k_offsets + (effective_rope_hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q + second_half_q_offsets, mask=second_q_mask, other=0).to(tl.float32)
    k_tile_2 = tl.load(k + second_half_k_offsets, mask=second_k_mask, other=0).to(tl.float32)

    if not BACKWARD_PASS:
        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # Process Q tensor
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(
            q + first_half_q_offsets,
            new_q_tile_1.to(sin_row.dtype),
            mask=first_q_mask,
        )
        tl.store(
            q + second_half_q_offsets,
            new_q_tile_2.to(sin_row.dtype),
            mask=second_q_mask,
        )

        # Process K tensor
        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(
            k + first_half_k_offsets,
            new_k_tile_1.to(sin_row.dtype),
            mask=first_k_mask,
        )
        tl.store(
            k + second_half_k_offsets,
            new_k_tile_2.to(sin_row.dtype),
            mask=second_k_mask,
        )
    else:
        # with some math, we can get:
        # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        # Process Q tensor
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(
            q + first_half_q_offsets,
            new_q_tile_1.to(sin_row.dtype),
            mask=first_q_mask,
        )
        tl.store(
            q + second_half_q_offsets,
            new_q_tile_2.to(sin_row.dtype),
            mask=second_q_mask,
        )

        # Process K tensor
        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(
            k + first_half_k_offsets,
            new_k_tile_1.to(sin_row.dtype),
            mask=first_k_mask,
        )
        tl.store(
            k + second_half_k_offsets,
            new_k_tile_2.to(sin_row.dtype),
            mask=second_k_mask,
        )


def _rope_forward_tma(q, k, cos, sin):
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    q = q.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    assert cos.shape[-1] == head_dim // 2 or cos.shape[-1] == head_dim, (
        f"cos.shape[-1]: {cos.shape[-1]}, head_dim: {head_dim}"
    )
    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    # Here use contiguous to avoid the TensorDescriptor error: "strides must be 16-byte aligned"
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2).contiguous()
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2).contiguous()
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2).contiguous()
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2).contiguous()

    half_head_dim = q.shape[-1]
    BLOCK_HD = triton.next_power_of_2(half_head_dim)
    BLOCK_QH = triton.next_power_of_2(n_q_head)
    BLOCK_KH = triton.next_power_of_2(n_kv_head)
    # Calculate grid size
    n_row = batch_size * seq_len

    # Launch kernel

    # Create tensor descriptors
    desc_q = TensorDescriptor(
        q,
        shape=q.shape,
        strides=q.stride(),
        block_shape=[1, BLOCK_QH, 1, 1, BLOCK_HD],
    )
    desc_k = TensorDescriptor(
        k,
        shape=k.shape,
        strides=k.stride(),
        block_shape=[1, BLOCK_KH, 1, 1, BLOCK_HD],
    )
    desc_cos = TensorDescriptor(
        cos,
        shape=cos.shape,
        strides=cos.stride(),
        block_shape=[1, 1, 1, BLOCK_HD],
    )
    desc_sin = TensorDescriptor(
        sin,
        shape=sin.shape,
        strides=sin.stride(),
        block_shape=[1, 1, 1, BLOCK_HD],
    )

    _triton_rope_kernel_tma[(n_row,)](
        desc_q,
        desc_k,
        desc_cos,
        desc_sin,
        # Q tensor parameters
        cos.shape[0],
        seq_len,
        BLOCK_QH,
        BLOCK_KH,
        BLOCK_HD,
    )
    return (
        q.reshape(batch_size, n_q_head, seq_len, head_dim),
        k.reshape(batch_size, n_kv_head, seq_len, head_dim),
        cos.reshape(original_cos_shape),
        sin.reshape(original_sin_shape),
    )


def _rope_forward(q, k, cos, sin, rope_dim=None):
    """
    Apply rotary position encoding in forward pass

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Cosine values
             (rope_dim == head_dim for full RoPE)
        sin: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Sine values
        rope_dim: Number of head dimensions to rotate (0 = full head_dim)

    Returns:
        Query and key tensors with RoPE applied
    """
    # Transpose it back to the physical shape because Triton looks at the physical storage
    # Note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # Partial RoPE support
    if rope_dim is None:
        rope_dim = head_dim
    pad_rope_hd = triton.next_power_of_2(rope_dim)

    n_row = batch_size * seq_len

    # Ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]

    _rope_kernel[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BACKWARD_PASS=False,
        rope_hd=rope_dim,
        pad_rope_hd=pad_rope_hd,
    )
    return (
        q.transpose(1, 2),
        k.transpose(1, 2),
        cos,
        sin,
    )


def _rope_backward(dq, dk, cos, sin, rope_dim=None):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    cos_batch_size = cos.shape[0]
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # Partial RoPE support
    if rope_dim is None:
        rope_dim = head_dim
    pad_rope_hd = triton.next_power_of_2(rope_dim)

    n_row = batch_size * seq_len

    # Ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    # Backward is similar to forward except swapping few ops
    _rope_kernel[(n_row,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BACKWARD_PASS=True,
        rope_hd=rope_dim,
        pad_rope_hd=pad_rope_hd,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class _TritonRopeFunction(torch.autograd.Function):
    """
    Triton implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, use_tma=True, rope_dim=None):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, rope_dim) or (bsz, seq_len, rope_dim)  — rope_dim == head_dim for full RoPE
        sin size: same as cos
        """
        if use_tma and rope_dim is None:
            # TMA path only supports full RoPE; fall back to standard kernel for partial
            q, k, cos, sin = _rope_forward_tma(q, k, cos, sin)
        else:
            q, k, cos, sin = _rope_forward(q, k, cos, sin, rope_dim=rope_dim)
        ctx.save_for_backward(cos, sin)
        ctx.rope_dim = rope_dim
        return q, k

    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = _rope_backward(dq, dk, cos, sin, rope_dim=ctx.rope_dim)
        return dq, dk, None, None, None, None, None, None


@register_impl("apply_rope_base", backend="triton")
def apply_rope_base(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, use_tma=True, partial_rotary_factor=1.0):
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
    return _TritonRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, use_tma, rope_dim)


@register_impl("get_apply_rope_func", backend="triton")
def get_apply_rope_func(model="llama"):
    def is_use_tma(s):
        return s > 128

    if model == "llama" or model == "qwen2" or model == "gemma3" or model == "gpt-oss":

        def wrapper(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, use_tma=True):
            return apply_rope_base(q, k, cos, sin, use_tma=is_use_tma(q.shape[2]))

        return wrapper
    elif model == "qwen3_5":

        def wrapper(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rope_base(
                q,
                k,
                cos,
                sin,
                use_tma=False,
                partial_rotary_factor=0.25,
            )

        return wrapper
    elif model == "deepseek":

        def wrapper(q, k, freqs_cis):
            cos, sin = freqs_cis.real, freqs_cis.imag

            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            return apply_rope_base(q, k, cos, sin, use_tma=is_use_tma(s))

        return wrapper

    else:
        raise ValueError(f"Unsupported model: {model}")
