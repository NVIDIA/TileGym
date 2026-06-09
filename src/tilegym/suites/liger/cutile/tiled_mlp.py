# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tiled MLP (cuTile backend).

Pure Python implementation — no GPU kernel.
Shards input along sequence dimension (dim=-2), applies fn on each shard,
and concatenates. Backward re-computes forward per shard to save memory.

"""

import math
from typing import Callable
from typing import List
from typing import Optional

import torch

from tilegym.backend import register_impl


class _TiledMLPFunctionCT(torch.autograd.Function):
    """Tiled MLP computation (no GPU kernel, memory-efficient via re-computation)."""

    @staticmethod
    def forward(ctx, fn, mlp_module, x, shards, compute_params=None):
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        return torch.cat(output_shards, dim=-2)

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad

        # Chunk along dim=-2 to match forward sharding exactly.
        # Flattening to 2D first and re-chunking would create different
        # row groupings, leading to different GEMM algorithms and relu-mask
        # flips at near-zero activations (up to ~0.09 gradient error).
        x_detached = x.detach()
        x_shards = list(torch.chunk(x_detached, chunks=shards, dim=-2))
        grad_shards = list(torch.chunk(grads[0], chunks=shards, dim=-2))

        # Pre-allocate gradient buffer and chunk it into views aligned with x_shards.
        # This lets a single backward() per shard populate x_shard_leaf.grad AND
        # accumulate weight gradients, halving the number of backward passes vs the
        # previous retain_graph=True + second backward() approach.
        if x_requires_grad:
            x_grad = torch.zeros_like(x_detached)
            x_grad_shards = list(torch.chunk(x_grad, chunks=shards, dim=-2))
        else:
            x_grad = None

        for i, (x_shard, grad_shard) in enumerate(zip(x_shards, grad_shards)):
            x_shard_leaf = x_shard.detach().requires_grad_(x_requires_grad)
            if x_requires_grad:
                # Pre-assign the gradient buffer slice so backward() fills it in-place.
                x_shard_leaf.grad = x_grad_shards[i]
            with torch.enable_grad():
                output = fn(mlp_module, x_shard_leaf)
            # Single backward per shard: accumulates weight gradients AND populates
            # x_shard_leaf.grad (which is a view into x_grad) when x_requires_grad.
            torch.autograd.backward(output, grad_shard)

        return None, None, x_grad, None, None


def _apply_tiled_mlp_ct(fn, mlp_module, x, num_shards=None, compute_params=None):
    if num_shards is None:
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)
    num_shards = max(1, num_shards)
    return _TiledMLPFunctionCT.apply(fn, mlp_module, x, num_shards, compute_params)


@register_impl("liger.tiled_mlp", backend="cutile")
def tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List] = None,
    **kwargs,
) -> torch.Tensor:
    return _apply_tiled_mlp_ct(fn, mlp_module, x, num_shards, compute_params)
