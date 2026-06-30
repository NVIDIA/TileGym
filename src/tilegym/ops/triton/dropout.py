# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl

from tilegym.backend import get_available_triton_backend
from tilegym.backend import register_impl


def _get_dropout_configs():
    if get_available_triton_backend() == "nvt" and torch.cuda.get_device_capability()[0] == 8:
        return [
            triton.Config({"BLOCK_SIZE": 1024, "occupancy": occ}, num_warps=w, num_stages=s)
            for occ in [1, 2, 4, 8, 16]
            for w in [4, 8]
            for s in [2, 3, 4]
        ]
    else:
        return [
            triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
            for bs in [1024, 2048, 4096]
            for w in [4, 8]
            for s in [2, 3, 4]
        ]


# Adapted from https://github.com/openai/triton
@triton.autotune(
    configs=_get_dropout_configs(),
    key=["n_elements"],
    # When `inplace=True`, x_ptr and output_ptr alias the same buffer. Without
    # restoring, each autotune benchmark run multiplies kept elements by 1/(1-p)
    # again, quickly overflowing fp32 to `inf` across hundreds of tuning runs.
    # `restore_value` is a no-op in the non-inplace path (x_ptr is read-only).
    restore_value=["x_ptr"],
)
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
    USE_PHILOX: tl.constexpr = True,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if USE_PHILOX:
        # High-quality Philox PRNG (higher per-element cost).
        random = tl.rand(seed, offsets)
    else:
        # 3-round xor-shift hash — matches cuTile's PRNG cost. Set
        # USE_PHILOX=True for Philox-grade randomness.
        offs_i32 = offsets.to(tl.int32)
        combined = offs_i32 * 1103515245 + tl.cast(seed, tl.int32)
        hash_val = combined ^ (combined >> 16)
        hash_val = hash_val ^ (hash_val << 8)
        hash_val = hash_val ^ (hash_val >> 4)
        random = (hash_val & 0x7FFFFFFF).to(tl.float32) / 2147483647.0
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


class _Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, seed, p=0.5, training=True, inplace=False):
        if not training:
            ctx.mark_dirty(x)
            return x

        if inplace:
            ctx.mark_dirty(x)
            output = x
        else:
            output = torch.empty_like(x)

        assert x.is_contiguous()

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _seeded_dropout[grid](
            x,
            output,
            n_elements,
            p,
            seed,
        )
        ctx.p = p
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Dropout backward is not implemented for this backend")


@register_impl("dropout", backend="triton")
def dropout(x, seed, p=0.5, training=True, inplace=False, **kwargs):
    r"""
    Performs dropout on x

    Args:
        seed: Integer value for initializing
            random mask
        training: If True perform dropout, else
            return x
        inplace: If True, modify x directly with
            dropout
        **kwargs: Additional arguments for backend-specific configurations
    """
    return _Dropout.apply(x, seed, p, training, inplace)
