# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
Dynamic Tanh (DyT) activation kernel (cuTile backend).

Formula: y = tanh(alpha * x) * gamma + beta

Reference:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/dyt.py

"""

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE = 65536


@ct.kernel
def _dyt_fwd_kernel(
    x_input,  # (M, N) input
    y_output,  # (M, N) output
    alpha_tensor,  # (1,) scalar alpha
    gamma_tensor,  # (N,) per-channel scale
    beta_tensor,  # (N,) per-channel bias (or dummy 1-element tensor when HAVE_BETA=0)
    HAVE_BETA: ct.Constant[int],
    CHECK_BOUNDS: ct.Constant[bool],
    BLOCK_N: ct.Constant[int],
):
    """DyT forward; CHECK_BOUNDS=False requires BLOCK_N == N and a single column block."""
    row_id = ct.bid(1)

    if CHECK_BOUNDS:
        col_indices = ct.arange(BLOCK_N, dtype=ct.int32) + ct.bid(0) * BLOCK_N
        gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)
        x = ct.astype(ct.gather(x_input, (row_id, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
    else:
        col_indices = ct.arange(BLOCK_N, dtype=ct.int32)
        gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=False), ct.float32)
        x = ct.astype(ct.gather(x_input, (row_id, col_indices), check_bounds=False), ct.float32)

    alpha = ct.astype(ct.load(alpha_tensor, 0, shape=()), ct.float32)
    tanh_x = ct.tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        if CHECK_BOUNDS:
            beta = ct.astype(ct.gather(beta_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)
        else:
            beta = ct.astype(ct.gather(beta_tensor, col_indices, check_bounds=False), ct.float32)
        y = y + beta

    ct.scatter(y_output, (row_id, col_indices), ct.astype(y, y_output.dtype), check_bounds=CHECK_BOUNDS)


@ct.kernel
def _dyt_bwd_kernel(
    dy_input,  # (M, N) upstream gradient
    dx_output,  # (M, N) gradient w.r.t. x
    da_partial,  # (NUM_SMS, num_col_blocks) partial d_alpha per block, host reduces
    dg_partial,  # (NUM_SMS, N) partial d_gamma per block row, host reduces
    db_partial,  # (NUM_SMS, N) partial d_beta when HAVE_BETA, host reduces
    x_input,  # (M, N) saved input
    alpha_tensor,  # (1,) scalar alpha
    gamma_tensor,  # (N,) per-channel scale
    HAVE_BETA: ct.Constant[int],
    M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    NUM_SMS: ct.Constant[int],
):
    """DyT backward with per-block gradient partials reduced by the host."""
    col_block = ct.bid(0)
    start_row_id = ct.bid(1)
    col_start = col_block * BLOCK_N
    col_indices = ct.arange(BLOCK_N, dtype=ct.int32) + col_start

    alpha = ct.astype(ct.load(alpha_tensor, 0, shape=()), ct.float32)
    gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)

    da_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
    dg_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
    if HAVE_BETA:
        db_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)

    num_iters = (M + 2 * NUM_SMS - 1) // (2 * NUM_SMS)
    for i in range(num_iters):
        row_a = start_row_id + i * (2 * NUM_SMS)
        row_b = row_a + NUM_SMS
        x_a = ct.astype(ct.gather(x_input, (row_a, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        dy_a = ct.astype(ct.gather(dy_input, (row_a, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        x_b = ct.astype(ct.gather(x_input, (row_b, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        dy_b = ct.astype(ct.gather(dy_input, (row_b, col_indices), check_bounds=True, padding_value=0.0), ct.float32)

        # Padded rows must not accumulate tanh(inf * 0) NaNs.
        if row_a < M:
            tanh_a = ct.tanh(alpha * x_a, rounding_mode=RMd.APPROX)
            tmp_a = (ct.full((BLOCK_N,), 1.0, dtype=ct.float32) - tanh_a * tanh_a) * dy_a * gamma
            if HAVE_BETA:
                db_acc = db_acc + dy_a
            dg_acc = dg_acc + dy_a * tanh_a
            da_acc = da_acc + x_a * tmp_a
            ct.scatter(dx_output, (row_a, col_indices), ct.astype(alpha * tmp_a, dx_output.dtype), check_bounds=True)

        if row_b < M:
            tanh_b = ct.tanh(alpha * x_b, rounding_mode=RMd.APPROX)
            tmp_b = (ct.full((BLOCK_N,), 1.0, dtype=ct.float32) - tanh_b * tanh_b) * dy_b * gamma
            if HAVE_BETA:
                db_acc = db_acc + dy_b
            dg_acc = dg_acc + dy_b * tanh_b
            da_acc = da_acc + x_b * tmp_b
            ct.scatter(dx_output, (row_b, col_indices), ct.astype(alpha * tmp_b, dx_output.dtype), check_bounds=True)

    row_idx_tile = ct.full((BLOCK_N,), start_row_id, dtype=ct.int32)
    ct.scatter(dg_partial, (row_idx_tile, col_indices), dg_acc, check_bounds=True)
    if HAVE_BETA:
        ct.scatter(db_partial, (row_idx_tile, col_indices), db_acc, check_bounds=True)

    da_scalar = ct.full((1,), ct.sum(da_acc, 0, keepdims=False), dtype=ct.float32)
    ct.scatter(
        da_partial,
        (ct.full((1,), start_row_id, dtype=ct.int32), ct.full((1,), col_block, dtype=ct.int32)),
        da_scalar,
    )


_dyt_bwd_kernel_narrow = _dyt_bwd_kernel.replace_hints(num_worker_warps=4)
_dyt_bwd_kernel_wide = _dyt_bwd_kernel.replace_hints(num_worker_warps=8, occupancy=4)


def _dyt_forward_ct(x, alpha, gamma, beta):
    HAVE_BETA = beta is not None
    input_shape = x.shape
    x_2d = x.view(-1, input_shape[-1])
    M, N = x_2d.shape

    BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))
    num_col_blocks = (N + BLOCK_N - 1) // BLOCK_N

    check_bounds = not ((BLOCK_N == N) and (num_col_blocks == 1))

    y = torch.empty_like(x_2d)
    beta_tensor = beta if HAVE_BETA else torch.empty(1, device=x.device, dtype=x.dtype)

    ct.launch(
        torch.cuda.current_stream(),
        (num_col_blocks, M, 1),
        _dyt_fwd_kernel,
        (x_2d, y, alpha, gamma, beta_tensor, int(HAVE_BETA), bool(check_bounds), int(BLOCK_N)),
    )
    return y.view(input_shape)


def _dyt_backward_ct(dy, x, alpha, gamma, beta):
    HAVE_BETA = beta is not None
    input_shape = x.shape
    x_2d = x.view(-1, input_shape[-1])
    dy_2d = dy.view(-1, input_shape[-1])
    M, N = x_2d.shape

    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    narrow = N <= 2048
    BLOCK_N = min(next_power_of_2(N), 256 if narrow else 1024)
    num_col_blocks = (N + BLOCK_N - 1) // BLOCK_N

    dx = torch.empty_like(dy_2d)

    da_partial = torch.zeros(NUM_SMS, num_col_blocks, dtype=torch.float32, device=x.device)
    dg_partial = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)
    db_partial = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device) if HAVE_BETA else None

    db_tensor = db_partial if HAVE_BETA else torch.empty(1, device=x.device, dtype=torch.float32)

    ct.launch(
        torch.cuda.current_stream(),
        (num_col_blocks, NUM_SMS, 1),
        _dyt_bwd_kernel_narrow if narrow else _dyt_bwd_kernel_wide,
        (
            dy_2d,
            dx,
            da_partial,
            dg_partial,
            db_tensor,
            x_2d,
            alpha,
            gamma,
            int(HAVE_BETA),
            int(M),
            int(BLOCK_N),
            int(NUM_SMS),
        ),
    )

    da = da_partial.sum().to(x.dtype).unsqueeze(0)
    dg = dg_partial.sum(0).to(gamma.dtype)
    db = db_partial.sum(0).to(x.dtype) if HAVE_BETA else None
    return dx.view(input_shape), da, dg, db


class _DytFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        x = x.contiguous()
        alpha = alpha.contiguous()
        gamma = gamma.contiguous()
        if beta is not None:
            beta = beta.contiguous()
        y = _dyt_forward_ct(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, alpha, gamma, beta = ctx.saved_tensors
        dy = dy.contiguous()
        dx, dalpha, dgamma, dbeta = _dyt_backward_ct(dy, x, alpha, gamma, beta)
        return dx, dalpha, dgamma, dbeta


@register_impl("liger.dyt", backend="cutile")
def dyt(x, alpha, gamma, beta=None):
    """
    Dynamic Tanh (DyT): y = tanh(alpha * x) * gamma + beta

    Args:
        x: Input tensor of shape (*, N)
        alpha: Scalar learnable parameter, shape (1,)
        gamma: Per-channel scale, shape (N,)
        beta: Optional per-channel bias, shape (N,)

    Returns:
        Output tensor of same shape as x
    """
    return _DytFunction.apply(x, alpha, gamma, beta)
