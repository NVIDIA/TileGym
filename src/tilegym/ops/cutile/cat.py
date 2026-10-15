# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import logging
from typing import List
from typing import Tuple
from typing import Union

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

logger = logging.getLogger(__name__)

# Number of tensor slots in _cat_copy_multi_tma_kernel_4.
# Changing this requires rewriting the kernel signature and dispatch branches.
_CAT_KERNEL_SLOTS = 4


@ct.kernel
def _cat_copy_multi_tma_kernel_4(
    out_2d,
    in_2d_a,
    in_2d_b,
    in_2d_c,
    in_2d_d,
    dim_size_in_a,
    dim_size_in_b,
    dim_size_in_c,
    dim_size_in_d,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    dim_prod_post,
    BLOCK_INNER: ct.Constant[int],
    SLICE_DIVISOR: ct.Constant[int],
):
    """
    Four-slot multi-tensor copy kernel using 2D view load/store for input and output.
    Supports arbitrary tensor rank via the 2D projection (dim_prod_pre, dim_prod_post).

    Each kernel launch handles up to four input tensors. The branch selects the
    input view and cat-dimension metadata, while the load/store tile shape stays
    fixed across all branches.

    Memory layout:
    - in_2d_*: [dim_prod_pre, dim_size_in_* * dim_prod_post]
    - out_2d:  [dim_prod_pre, dim_size_out * dim_prod_post]

    Grid: (dim_prod_pre, ceil(max(dim_size_in_* * dim_prod_post) / BLOCK_INNER), num_tensors)
    - bid(0): pre_idx   -- outer dimension block index
    - bid(1): inner block index within the selected input tensor
    - bid(2): tensor slot within this group of up to four inputs
    """
    bid_pre = ct.bid(0)
    bid_inner = ct.bid(1)
    tensor_idx = ct.bid(2)

    if tensor_idx == 0:
        in_2d = in_2d_a
        dim_size_in = dim_size_in_a
        dim_offset = dim_offset_a
    elif tensor_idx == 1:
        in_2d = in_2d_b
        dim_size_in = dim_size_in_b
        dim_offset = dim_offset_b
    elif tensor_idx == 2:
        in_2d = in_2d_c
        dim_size_in = dim_size_in_c
        dim_offset = dim_offset_c
    else:
        in_2d = in_2d_d
        dim_size_in = dim_size_in_d
        dim_offset = dim_offset_d

    # 2D TMA load from input: loads a (1, BLOCK_INNER) tile.
    # padding_mode=ZERO handles the last partial block and out-of-bounds pre blocks.
    data = ct.load(
        in_2d,
        index=(bid_pre, bid_inner),
        shape=(1, BLOCK_INNER),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((BLOCK_INNER,))

    # Slice the output 2D tensor along axis=1 to isolate this tensor's region.
    # Array.slice supports runtime start/stop and enables TMA for ct.store.
    slice_start = dim_offset * dim_prod_post
    slice_stop = (dim_offset + dim_size_in) * dim_prod_post
    # SLICE_DIVISOR is 16 only when the host has proved both bounds are
    # divisible by 16; otherwise it is 1. This makes the alignment fact explicit
    # to cuTile without changing runtime scalar specialization.
    slice_start = (slice_start // SLICE_DIVISOR) * SLICE_DIVISOR
    slice_stop = (slice_stop // SLICE_DIVISOR) * SLICE_DIVISOR
    out_slice = out_2d.slice(1, slice_start, slice_stop)

    # 2D TMA store into the sliced output region.
    ct.store(out_slice, index=(bid_pre, bid_inner), tile=data.reshape((1, BLOCK_INNER)))


@register_impl("cat", backend="cutile")
def cat(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0, **kwargs) -> torch.Tensor:
    """
    TileGym CuTile cat operation.

    Concatenates tensors along specified dimension using 2D TMA/view load-store
    operations. Inputs are grouped in batches of up to four tensors per launch.

    Supports arbitrary tensor rank via dim_prod_pre / dim_prod_post projection.

    Args:
        tensors: Tuple or list of tensors to concatenate
        dim: Dimension along which to concatenate (default: 0)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        torch.Tensor: Concatenated result
    """
    logger.debug(f"TileGym CuTile CAT called: {len(tensors)} tensors, dim={dim}")

    if len(tensors) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(tensors) == 1:
        return tensors[0]

    # Handle special case: first tensor is empty and has different ndim
    if len(tensors) == 2:
        first, second = tensors
        if first.numel() == 0 and first.ndim != second.ndim:
            logger.debug("TileGym cat: First tensor is empty with different ndim, returning second tensor")
            return second
        if second.numel() == 0 and first.ndim != second.ndim:
            logger.debug("TileGym cat: Second tensor is empty with different ndim, returning first tensor")
            return first

    # Handle negative dimensions and validate
    tensor_ndim = tensors[0].ndim

    assert dim >= -tensor_ndim and dim < tensor_ndim, f"Invalid dim: {dim}"
    dim %= tensor_ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}")
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    # Type promotion
    device = tensors[0].device
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    tensors = [t.contiguous() if not t.is_contiguous() else t for t in tensors]

    shapes = [t.shape for t in tensors]
    cat_dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(cat_dim_sizes)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    # Compute dim_prod_post: product of dims after cat dim (same for all tensors)
    dim_prod_post = 1
    for d in range(dim + 1, tensors[0].ndim):
        dim_prod_post *= tensors[0].shape[d]

    # Compute dim_prod_pre: product of dims before cat dim (same for all tensors)
    dim_prod_pre = 1
    for d in range(0, dim):
        dim_prod_pre *= tensors[0].shape[d]

    dim_size_out = out_shape[dim]

    # Reshape output as 2D [dim_prod_pre, dim_size_out * dim_prod_post] for TMA writes.
    out_2d = out.reshape(dim_prod_pre, dim_size_out * dim_prod_post)

    # BLOCK_INNER: tile width for the inner dimension.
    # Must be a power of 2 for vectorized view load/store.
    BLOCK_INNER = 1024

    stream = torch.cuda.current_stream()

    unit_inner_entries = []
    nonunit_inner_entries = []
    dim_offset = 0
    for tensor in tensors:
        dim_size_in = tensor.shape[dim]
        inner_size = dim_size_in * dim_prod_post
        entry = (tensor, dim_size_in, dim_offset, inner_size)
        if inner_size == 1:
            unit_inner_entries.append(entry)
        else:
            nonunit_inner_entries.append(entry)
        dim_offset += dim_size_in

    # Keep inputs with unit inner_size in separate launches, avoiding the cuTile
    # TileTypeError that arises when branches conditionally select 2D arrays
    # with incompatible Array types: stride (1, 1) vs (?, 1). Each entry keeps
    # its original output offset, so launch grouping does not affect cat order.
    for bucket in (unit_inner_entries, nonunit_inner_entries):
        i = 0
        while i < len(bucket):
            batch = bucket[i : i + _CAT_KERNEL_SLOTS]
            num_tensors = len(batch)
            in_2ds = []
            sizes = []
            offsets = []
            inners = []

            for tensor, dim_size_in, dim_offset, inner_size in batch:
                in_2ds.append(tensor.reshape(dim_prod_pre, inner_size))
                sizes.append(dim_size_in)
                offsets.append(dim_offset)
                inners.append(inner_size)

            slice_divisor = 16
            for dim_size_in, dim_offset in zip(sizes, offsets):
                slice_start = dim_offset * dim_prod_post
                slice_stop = (dim_offset + dim_size_in) * dim_prod_post
                if slice_start % slice_divisor != 0 or slice_stop % slice_divisor != 0:
                    slice_divisor = 1
                    break

            while len(in_2ds) < _CAT_KERNEL_SLOTS:
                in_2ds.append(in_2ds[0])
                sizes.append(0)
                offsets.append(0)

            grid_inner = (max(inners) + BLOCK_INNER - 1) // BLOCK_INNER
            grid = (dim_prod_pre, grid_inner, num_tensors)

            ct.launch(
                stream,
                grid,
                _cat_copy_multi_tma_kernel_4,
                (
                    out_2d,
                    in_2ds[0],
                    in_2ds[1],
                    in_2ds[2],
                    in_2ds[3],
                    sizes[0],
                    sizes[1],
                    sizes[2],
                    sizes[3],
                    offsets[0],
                    offsets[1],
                    offsets[2],
                    offsets[3],
                    dim_prod_post,
                    BLOCK_INNER,
                    slice_divisor,
                ),
            )

            i += _CAT_KERNEL_SLOTS

    return out
