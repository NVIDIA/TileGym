# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Optional
from typing import Tuple

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

ConstInt = ct.Constant[int]


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@ct.kernel
def _per_token_group_quant_8bit_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride: ConstInt,
    N: ConstInt,
    eps: ConstInt,
    bit8_min: ConstInt,
    bit8_max: ConstInt,
    BLOCK: ConstInt,
    y_array_size: ConstInt,
    y_q_array_size: ConstInt,
    y_s_array_size: ConstInt,
):
    """Per-token-group quantization kernel (row-major scales)."""
    g_id = ct.bid(0)

    # Compute base offsets for this group
    y_base = g_id * y_stride
    y_q_base = g_id * y_stride

    # Create column offsets and mask
    cols = ct.arange(BLOCK, dtype=ct.int32)
    mask = cols < N

    # Load input values with gather (element-level access with mask)
    y_indices = y_base + cols
    y = ct.gather(y_ptr, (y_indices,), check_bounds=True, padding_value=0.0)
    y = ct.astype(y, ct.float32)

    # Compute absmax
    abs_y = ct.abs(y)
    _absmax = ct.max(abs_y, axis=0)
    _absmax = ct.maximum(_absmax, eps)

    # Compute scale and inverse scale
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s

    # Quantize: clamp(y * y_s_inv, bit8_min, bit8_max)
    y_q = ct.minimum(ct.maximum(y * y_s_inv, bit8_min), bit8_max)
    y_q = ct.astype(y_q, y_q_ptr.dtype)

    # Store quantized values with mask (use OOB offsets for invalid positions)
    oob_offset = ct.full((BLOCK,), y_q_array_size, dtype=ct.int32)
    y_q_indices = y_q_base + cols
    y_q_indices_masked = ct.where(mask, y_q_indices, oob_offset)
    ct.scatter(y_q_ptr, (y_q_indices_masked,), y_q, check_bounds=True)

    # Store scale (single scalar per group)
    y_s_idx = g_id
    oob_scalar = ct.full((), y_s_array_size, dtype=ct.int32)
    s_idx_masked = ct.where(y_s_idx < y_s_array_size, y_s_idx, oob_scalar)
    ct.scatter(y_s_ptr, (s_idx_masked,), y_s)


@ct.kernel
def _per_token_group_quant_8bit_colmajor_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size: ConstInt,
    y_num_columns: ConstInt,
    y_row_stride: ConstInt,
    y_s_col_stride: ConstInt,
    eps: ConstInt,
    bit8_min: ConstInt,
    bit8_max: ConstInt,
    scale_ue8m0: ConstInt,
    BLOCK: ConstInt,
    y_array_size: ConstInt,
    y_q_array_size: ConstInt,
    y_s_array_size: ConstInt,
):
    """Per-token-group quantization kernel (column-major scales)."""
    groups_per_row = y_num_columns // group_size

    g_id = ct.bid(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    # Compute base offsets
    y_base = row * y_row_stride + group_id * group_size
    y_q_base = g_id * group_size
    y_s_offset = group_id * y_s_col_stride + row

    # Create column offsets and mask
    cols = ct.arange(BLOCK, dtype=ct.int32)
    mask = cols < group_size

    # Load input values
    y_indices = y_base + cols
    y = ct.gather(y_ptr, (y_indices,), check_bounds=True, padding_value=0.0)
    y = ct.astype(y, ct.float32)

    # Compute absmax
    abs_y = ct.abs(y)
    _absmax = ct.max(abs_y, axis=0)
    _absmax = ct.maximum(_absmax, eps)

    # Compute scale
    y_s = _absmax / bit8_max

    # Optional: round scale to power of 2 (UE8M0)
    if scale_ue8m0:
        abs_y_s = ct.abs(y_s)
        safe_y_s = ct.maximum(abs_y_s, 1e-10)
        y_s = ct.exp2(ct.ceil(ct.log2(safe_y_s)))

    # Quantize: clamp(y / y_s, bit8_min, bit8_max)
    y_q = ct.minimum(ct.maximum(y / y_s, bit8_min), bit8_max)
    y_q = ct.astype(y_q, y_q_ptr.dtype)

    # Store quantized values with mask
    oob_offset = ct.full((BLOCK,), y_q_array_size, dtype=ct.int32)
    y_q_indices = y_q_base + cols
    y_q_indices_masked = ct.where(mask, y_q_indices, oob_offset)
    ct.scatter(y_q_ptr, (y_q_indices_masked,), y_q, check_bounds=True)

    # Store scale (single scalar)
    oob_scalar = ct.full((), y_s_array_size, dtype=ct.int32)
    s_idx_masked = ct.where(y_s_offset < y_s_array_size, y_s_offset, oob_scalar)
    ct.scatter(y_s_ptr, (s_idx_masked,), y_s)


def _ceil_align(x: int, align: int) -> int:
    return (x + align - 1) // align * align


@register_impl("flashinfer.quant.per_token_group_quant_8bit", backend="cutile")
def per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token group 8-bit quantization (FP8 or INT8) - CuTile implementation."""
    if dst_dtype is None:
        dst_dtype = torch.float8_e4m3fn
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible by `group_size` {group_size}"
    )
    assert x.is_contiguous(), "`x` must be contiguous"
    if scale_tma_aligned or scale_ue8m0:
        assert column_major_scales, "scale_tma_aligned or scale_ue8m0 requires column_major_scales=True"

    if dst_dtype == torch.int8:
        info = torch.iinfo(dst_dtype)
        bit8_min = float(info.min)
        bit8_max = float(info.max)
    else:
        info = torch.finfo(dst_dtype)
        bit8_min = info.min
        bit8_max = info.max

    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        num_groups = x.shape[-1] // group_size
        num_tokens = x.shape[-2] if x.dim() >= 2 else x.shape[0]
        if scale_tma_aligned:
            # TMA-friendly layout: (num_groups, aligned_num_tokens), align to 4 floats (16B)
            aligned_size = _ceil_align(num_tokens, 4)
            x_s_raw = torch.empty(
                (num_groups, aligned_size),
                device=x.device,
                dtype=torch.float32,
            )
            x_s_col_stride = aligned_size
        else:
            shape = (num_groups,) + x.shape[:-1]
            x_s_raw = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
            x_s_col_stride = x_s_raw.stride(1)
        x_s = x_s_raw
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = next_power_of_2(N)

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    # Flatten tensors for gather/scatter access
    x_flat = x.contiguous().view(-1)
    x_q_flat = x_q.view(-1)
    x_s_flat = x_s.contiguous().view(-1) if not column_major_scales else x_s

    if column_major_scales:
        # Use a contiguous flat view of x_s for scatter
        # x_s may be non-contiguous (permuted), so we use as_strided to get the raw storage
        x_s_for_kernel = (
            x_s_raw.view(-1)
            if x_s_raw.is_contiguous()
            else torch.as_strided(x_s_raw, (x_s_raw.numel(),), (1,), storage_offset=x_s_raw.storage_offset())
        )

        ct.launch(
            stream,
            grid,
            _per_token_group_quant_8bit_colmajor_kernel,
            (
                x_flat,
                x_q_flat,
                x_s_for_kernel,
                group_size,
                x.shape[-1],
                x.stride(-2) if x.dim() >= 2 else x.shape[-1],
                x_s_col_stride,
                eps,
                bit8_min,
                bit8_max,
                1 if scale_ue8m0 else 0,
                BLOCK,
                x_flat.numel(),
                x_q_flat.numel(),
                x_s_for_kernel.numel(),
            ),
        )
        if scale_tma_aligned:
            x_s = x_s_raw[:, :num_tokens].t().contiguous()
    else:
        assert not scale_ue8m0

        ct.launch(
            stream,
            grid,
            _per_token_group_quant_8bit_kernel,
            (
                x_flat,
                x_q_flat,
                x_s_flat,
                group_size,
                N,
                eps,
                bit8_min,
                bit8_max,
                BLOCK,
                x_flat.numel(),
                x_q_flat.numel(),
                x_s_flat.numel(),
            ),
        )

    return x_q, x_s
