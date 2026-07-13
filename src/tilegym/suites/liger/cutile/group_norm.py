# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Group Normalization kernel (CuTile backend).

Forward:  2D grid (batch_size, num_groups).  Each block computes mean/variance
          over all elements of a group, then normalizes with per-channel W, B.
          mean_stats and rstd_stats are stored for the backward pass.

Backward: 2D grid (batch_size, num_groups).  Each block computes:
          - dw_partial[batch_idx, *channels_in_group] — partial gradient for W
          - db_partial[batch_idx, *channels_in_group] — partial gradient for B
          - DX for its (batch, group) slice
          The host reduces dw = dw_partial.sum(dim=0) and db = db_partial.sum(dim=0).
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE = 65536


@ct.kernel
def _group_norm_fwd_kernel(
    x_input,  # (batch_size * num_channels, hidden_size_per_channel)
    y_output,  # (batch_size * num_channels, hidden_size_per_channel)
    weight,  # (num_channels,)
    bias,  # (num_channels,)
    mean_stats,  # (batch_size * num_groups,) — indexed as [batch*num_groups + group]
    rstd_stats,  # (batch_size * num_groups,)
    NUM_CHANNELS: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
    TOTAL_HIDDEN_SIZE: ct.Constant[int],  # hidden_size_per_channel
    eps,
    BLOCK_SIZE: ct.Constant[int],
):
    """
    Group norm forward.

    Grid: (batch_size, num_groups, 1).
    One block per (batch, group): computes mean/variance over all channels in the
    group, then normalizes with per-channel W and B.
    """
    batch_idx = ct.bid(0)
    group_idx = ct.bid(1)

    group_row = batch_idx * NUM_GROUPS + group_idx  # scalar index for mean_stats/rstd_stats

    # Total elements per group (for normalization denominator)
    N = CHANNELS_PER_GROUP * TOTAL_HIDDEN_SIZE
    inv_N = 1.0 / N  # pre-compute reciprocal to avoid two divisions

    # num_h_chunks is constant for all channels — hoist outside both loops
    num_h_chunks = (TOTAL_HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # OOB positions (col_idx >= hidden_size) get padding_value=0 → contribute 0 (correct)
    sum_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            sum_tile = sum_tile + x_tile
            sum_sq_tile = sum_sq_tile + x_tile * x_tile

    s = ct.sum(sum_tile, 0, keepdims=False)  # scalar
    sq = ct.sum(sum_sq_tile, 0, keepdims=False)  # scalar
    mean = s * inv_N
    variance = sq * inv_N - mean * mean
    rstd = ct.rsqrt(variance + eps)

    # Store mean and rstd
    ct.scatter(mean_stats, group_row, ct.astype(mean, mean_stats.dtype))
    ct.scatter(rstd_stats, group_row, ct.astype(rstd, rstd_stats.dtype))

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)
        b_scalar = ct.astype(ct.load(bias, channel_idx, shape=()), ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            y_tile = (x_tile - mean) * rstd * w_scalar + b_scalar
            ct.scatter(y_output, (row_idx, col_idx), ct.astype(y_tile, y_output.dtype), check_bounds=True)


@ct.kernel
def _group_norm_bwd_kernel(
    x_input,  # (batch_size * num_channels, hidden_size_per_channel)
    upstream,  # (batch_size * num_channels, hidden_size_per_channel) upstream gradient
    weight,  # (num_channels,)
    mean_stats,  # (batch_size * num_groups,)
    rstd_stats,  # (batch_size * num_groups,)
    dx_output,  # (batch_size * num_channels, hidden_size_per_channel) output gradient
    dw_partial,  # (batch_size, num_channels) partial weight gradient — host does .sum(dim=0)
    db_partial,  # (batch_size, num_channels) partial bias gradient — host does .sum(dim=0)
    NUM_CHANNELS: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
    TOTAL_HIDDEN_SIZE: ct.Constant[int],  # hidden_size_per_channel
    BLOCK_SIZE: ct.Constant[int],
):
    """
    Group norm backward.

    Grid: (batch_size, num_groups, 1).
    Each block computes DX for its (batch, group) slice and writes partial
    dw and db for the channels it owns.
    """
    batch_idx = ct.bid(0)
    group_idx = ct.bid(1)

    group_row = batch_idx * NUM_GROUPS + group_idx

    mean = ct.astype(ct.load(mean_stats, group_row, shape=()), ct.float32)
    rstd = ct.astype(ct.load(rstd_stats, group_row, shape=()), ct.float32)

    N = CHANNELS_PER_GROUP * TOTAL_HIDDEN_SIZE
    inv_N = 1.0 / N  # pre-compute reciprocal: one multiply instead of two scalar divisions

    # num_h_chunks is constant for all channels — hoist outside both loops
    num_h_chunks = (TOTAL_HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pass 1: compute c1, c2 and partial dw, db for each channel in the group
    c1_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    c2_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)

        dW_acc_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dB_acc_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            upstream_tile = ct.astype(
                ct.gather(upstream, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            x_hat = (x_tile - mean) * rstd
            wdy = w_scalar * upstream_tile
            c1_tile = c1_tile + x_hat * wdy
            c2_tile = c2_tile + wdy
            dW_acc_tile = dW_acc_tile + upstream_tile * x_hat
            dB_acc_tile = dB_acc_tile + upstream_tile

        # Reduce per-channel partial dW, dB to scalar and write to partial buffer
        dW_val = ct.sum(dW_acc_tile, 0, keepdims=False)
        dB_val = ct.sum(dB_acc_tile, 0, keepdims=False)
        ct.scatter(dw_partial, (batch_idx, channel_idx), ct.astype(dW_val, dw_partial.dtype))
        ct.scatter(db_partial, (batch_idx, channel_idx), ct.astype(dB_val, db_partial.dtype))

    c1 = ct.sum(c1_tile, 0, keepdims=False) * inv_N
    c2 = ct.sum(c2_tile, 0, keepdims=False) * inv_N

    # Pass 2: compute DX = (wdy - (x_hat * c1 + c2)) * rstd
    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            upstream_tile = ct.astype(
                ct.gather(upstream, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            x_hat = (x_tile - mean) * rstd
            wdy = w_scalar * upstream_tile
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            ct.scatter(dx_output, (row_idx, col_idx), ct.astype(dx, dx_output.dtype), check_bounds=True)


class GroupNormCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, num_channels, num_groups, eps):
        if not X.is_contiguous():
            X = X.contiguous()
        if not W.is_contiguous():
            W = W.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()

        shape = X.shape
        batch_size = shape[0]
        channels_per_group = num_channels // num_groups
        hidden_size = X.shape[-1]  # hidden_size_per_channel (spatial dim)

        BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(hidden_size))

        # Reshape to 2D: (batch_size * num_channels, hidden_size)
        X_2d = X.view(batch_size * num_channels, hidden_size).contiguous()
        Y_2d = torch.empty_like(X_2d)
        # Stats kept in fp32 (matches upstream Liger). bf16 stats round-trip through
        # forward -> backward and lose precision in the (x - mean) * rstd step.
        mean_stats = torch.empty(batch_size * num_groups, dtype=torch.float32, device=X.device)
        rstd_stats = torch.empty(batch_size * num_groups, dtype=torch.float32, device=X.device)

        grid = (batch_size, num_groups, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _group_norm_fwd_kernel,
            (
                X_2d,
                Y_2d,
                W,
                B,
                mean_stats,
                rstd_stats,
                int(num_channels),
                int(num_groups),
                int(channels_per_group),
                int(hidden_size),
                float(eps),
                int(BLOCK_SIZE),
            ),
        )

        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.save_for_backward(X_2d, W, B, mean_stats, rstd_stats)
        ctx.shape = shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y_2d.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        X_2d, W, B, mean_stats, rstd_stats = ctx.saved_tensors
        num_channels = ctx.num_channels
        num_groups = ctx.num_groups
        shape = ctx.shape
        BLOCK_SIZE = ctx.BLOCK_SIZE

        batch_size = shape[0]
        hidden_size = shape[-1]
        channels_per_group = num_channels // num_groups

        if not dY.is_contiguous():
            dY = dY.contiguous()
        dY_2d = dY.view(batch_size * num_channels, hidden_size).contiguous()

        dx_2d = torch.empty_like(X_2d)
        dw_partial = torch.zeros(batch_size, num_channels, dtype=W.dtype, device=W.device)
        db_partial = torch.zeros(batch_size, num_channels, dtype=B.dtype, device=B.device)

        grid = (batch_size, num_groups, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _group_norm_bwd_kernel,
            (
                X_2d,
                dY_2d,
                W,
                mean_stats,
                rstd_stats,
                dx_2d,
                dw_partial,
                db_partial,
                int(num_channels),
                int(num_groups),
                int(channels_per_group),
                int(hidden_size),
                int(BLOCK_SIZE),
            ),
        )

        dw = dw_partial.sum(dim=0)
        db = db_partial.sum(dim=0)
        return dx_2d.view(*shape), dw, db, None, None, None


@register_impl("liger.group_norm", backend="cutile")
def group_norm(
    X: torch.Tensor,
    num_channels: int,
    num_groups: int,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-5,
    **kwargs,
) -> torch.Tensor:
    return GroupNormCuTileFunction.apply(X, W, B, num_channels, num_groups, eps)
