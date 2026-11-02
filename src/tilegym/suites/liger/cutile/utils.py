# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch


@ct.function
def _store_loss_normalization(output, count, total_weight, weight_sum):
    denominator = ct.maximum(count, 1.0)
    ct.scatter(output, 0, denominator)
    ct.scatter(output, 1, 1.0 / denominator)
    ct.scatter(output, 2, ct.where(count == 0, 1.0, total_weight))
    ct.scatter(output, 3, weight_sum)


@ct.kernel
def _loss_normalization_partial(
    target,
    weight,
    partial,
    output,
    ignore_index: ct.Constant[int],
    vocab_size: ct.Constant[int],
    HAS_WEIGHT: ct.Constant[bool],
    SINGLE_BLOCK: ct.Constant[bool],
    BLOCK: ct.Constant[int],
):
    block = ct.bid(0)
    indices = block * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    labels = ct.gather(target, indices, padding_value=ignore_index)
    valid = labels != ignore_index
    if vocab_size > 0:
        ct.assert_(~valid | ((labels >= 0) & (labels < vocab_size)), message="Target is out of bounds")
    count = ct.sum(ct.astype(valid, ct.float32), 0)
    if HAS_WEIGHT:
        selected = ct.gather(weight, ct.where(valid, labels, 0), padding_value=0)
        total_weight = ct.sum(ct.where(valid, ct.astype(selected, ct.float32), 0.0), 0)
        weights = ct.gather(weight, indices, padding_value=0)
        weight_sum = ct.sum(ct.astype(weights, ct.float32), 0)
    else:
        total_weight = count
        weight_sum = ct.full((), 0.0, ct.float32)
    if SINGLE_BLOCK:
        _store_loss_normalization(output, count, total_weight, weight_sum)
    else:
        ct.scatter(partial, (0, block), count)
        ct.scatter(partial, (1, block), total_weight)
        ct.scatter(partial, (2, block), weight_sum)


@ct.kernel
def _loss_normalization_reduce(partial, output, BLOCK: ct.Constant[int]):
    indices = ct.arange(BLOCK, dtype=ct.int32)
    count = ct.sum(ct.gather(partial, (0, indices), padding_value=0), 0)
    total_weight = ct.sum(ct.gather(partial, (1, indices), padding_value=0), 0)
    weight_sum = ct.sum(ct.gather(partial, (2, indices), padding_value=0), 0)
    _store_loss_normalization(output, count, total_weight, weight_sum)


def loss_normalization(target, ignore_index, vocab_size=0, weight=None):
    """Normalization terms of a loss over the targets that are not ``ignore_index``.

    Returns float32 device scalars ``(count, 1 / count, total_weight, weight_sum)``.
    ``count`` is the number of non-ignored targets, at least one. ``total_weight`` is
    the summed ``weight`` of those targets (``count`` without weights, one when every
    target is ignored) and ``weight_sum`` the sum of ``weight`` (zero without weights).
    A positive ``vocab_size`` also asserts that non-ignored targets are in range.
    """
    block = 1024
    size = max(target.numel(), weight.numel() if weight is not None else 0)
    blocks = max(1, ct.cdiv(size, block))
    output = torch.empty(4, device=target.device, dtype=torch.float32)
    partial = torch.empty((3, blocks), device=target.device, dtype=torch.float32) if blocks > 1 else 0
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (blocks,),
        _loss_normalization_partial,
        (
            target,
            weight if weight is not None else 0,
            partial,
            output,
            ignore_index,
            vocab_size,
            weight is not None,
            blocks == 1,
            block,
        ),
    )
    if blocks > 1:
        ct.launch(stream, (1,), _loss_normalization_reduce, (partial, output, next_power_of_2(blocks)))
    return output.unbind()


@ct.kernel
def _scale_kernel(x, scale, BLOCK: ct.Constant[int]):
    factor = ct.astype(ct.load(scale, (0,), shape=()), ct.float32)
    tile = ct.load(x, (ct.bid(0),), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(x, (ct.bid(0),), ct.astype(ct.astype(tile, ct.float32) * factor, x.dtype))


def scale_(x, scale):
    """Multiply the dense tensor ``x`` in place by the device scalar ``scale``; returns ``x``."""
    flat = x.permute(sorted(range(x.ndim), key=x.stride, reverse=True)).view(-1)
    block = 4096
    ct.launch(
        torch.cuda.current_stream(), (ct.cdiv(flat.numel(), block),), _scale_kernel, (flat, scale.reshape(1), block)
    )
    return x


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def addmm_f32_(acc, a, b):
    """acc += a @ b on an fp32 accumulator, accumulating inside the GEMM (beta=1)."""
    if a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float16):
        return torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    return acc.addmm_(a.float(), b.float())  # .float() is a no-op for fp32


def mm_f32(a, b):
    """a @ b with an fp32 result; bf16/fp16 operands stay on tensor cores (fp32 accumulate and output)."""
    if a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float16):
        return torch.mm(a, b, out_dtype=torch.float32)
    return a.float() @ b.float()  # .float() is a no-op for fp32
