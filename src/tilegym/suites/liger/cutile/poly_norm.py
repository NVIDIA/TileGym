# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
PolyNorm normalization kernel (CuTile backend).

Formula: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
where norm(u) = u / sqrt(mean(u²) + ε)

Reference:
1. https://github.com/BryceZhuo/PolyCom/
2. https://arxiv.org/pdf/2411.03884

"""

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE = 65536


@ct.kernel(occupancy=4)
def _poly_norm_fwd_kernel(
    x_input,  # (n_rows, n_cols) input
    y_output,  # (n_rows, n_cols) output
    weights,  # (3,) weights [w0, w1, w2]
    bias,  # (1,) scalar bias
    rstd3,  # (n_rows,) cached rstd for x^3 power
    rstd2,  # (n_rows,) cached rstd for x^2 power
    rstd1,  # (n_rows,) cached rstd for x^1 power
    N_COLS: ct.Constant[int],
    eps,
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """Two-pass PolyNorm forward; ALIGNED requires N_COLS divisible by BLOCK_SIZE."""
    row_idx = ct.bid(0)

    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)
    b = ct.astype(ct.load(bias, 0, shape=()), ct.float32)

    sum_sq_3 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_2 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_1 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    num_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False)
        else:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0)
        x_f32 = ct.astype(x_tile, ct.float32)

        x2 = x_f32 * x_f32
        x4 = x2 * x2
        sum_sq_1 = sum_sq_1 + x2
        sum_sq_2 = sum_sq_2 + x4
        sum_sq_3 = sum_sq_3 + x4 * x2

    tot_1, tot_2, tot_3 = ct.reduce(
        (sum_sq_1, sum_sq_2, sum_sq_3),
        0,
        lambda a1, a2, a3, b1, b2, b3: (a1 + b1, a2 + b2, a3 + b3),
        (0.0, 0.0, 0.0),
    )
    rstd_3 = ct.rsqrt(tot_3 * (1.0 / N_COLS) + eps)
    rstd_2 = ct.rsqrt(tot_2 * (1.0 / N_COLS) + eps)
    rstd_1 = ct.rsqrt(tot_1 * (1.0 / N_COLS) + eps)

    ct.scatter(rstd3, row_idx, rstd_3)
    ct.scatter(rstd2, row_idx, rstd_2)
    ct.scatter(rstd1, row_idx, rstd_1)

    w0r3 = w0 * rstd_3
    w1r2 = w1 * rstd_2
    w2r1 = w2 * rstd_1

    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False)
        else:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0)
        x_f32 = ct.astype(x_tile, ct.float32)
        x2 = x_f32 * x_f32

        inner = w0r3 * x_f32 + w1r2
        y_f32 = x2 * inner + w2r1 * x_f32 + b
        if ALIGNED:
            ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile.dtype), check_bounds=False)
        else:
            ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile.dtype), check_bounds=True)


@ct.kernel(occupancy=4)
def _poly_norm_fwd_kernel_sc_large(
    x_input,
    y_output,
    weights,
    bias,
    rstd3,
    rstd2,
    rstd1,
    N_COLS: ct.Constant[int],
    eps,
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """Single-chunk PolyNorm forward with a second input gather for the output pass."""
    row_idx = ct.bid(0)

    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)
    b = ct.astype(ct.load(bias, 0, shape=()), ct.float32)

    col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    if ALIGNED:
        x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False, latency=2)
    else:
        x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0, latency=2)
    x_f32 = ct.astype(x_tile, ct.float32)
    x2 = x_f32 * x_f32
    x4 = x2 * x2

    tot_1, tot_2, tot_3 = ct.reduce(
        (x2, x4, x4 * x2),
        0,
        lambda a1, a2, a3, b1, b2, b3: (a1 + b1, a2 + b2, a3 + b3),
        (0.0, 0.0, 0.0),
    )
    rstd_3 = ct.rsqrt(tot_3 * (1.0 / N_COLS) + eps)
    rstd_2 = ct.rsqrt(tot_2 * (1.0 / N_COLS) + eps)
    rstd_1 = ct.rsqrt(tot_1 * (1.0 / N_COLS) + eps)

    ct.scatter(rstd3, row_idx, rstd_3)
    ct.scatter(rstd2, row_idx, rstd_2)
    ct.scatter(rstd1, row_idx, rstd_1)

    w0r3 = w0 * rstd_3
    w1r2 = w1 * rstd_2
    w2r1 = w2 * rstd_1

    if ALIGNED:
        x_tile2 = ct.gather(x_input, (row_idx, col_indices), check_bounds=False, latency=2)
    else:
        x_tile2 = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0, latency=2)
    x2_f32 = ct.astype(x_tile2, ct.float32)
    x2sq = x2_f32 * x2_f32
    inner = w0r3 * x2_f32 + w1r2
    y_f32 = x2sq * inner + w2r1 * x2_f32 + b
    if ALIGNED:
        ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile2.dtype), check_bounds=False)
    else:
        ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile2.dtype), check_bounds=True)


@ct.kernel(occupancy=2)
def _poly_norm_bwd_kernel(
    dy,  # (n_rows, n_cols) output gradient
    dx,  # (n_rows, n_cols) input gradient (output)
    x_input,  # (n_rows, n_cols) saved input
    weights,  # (3,) weights [w0, w1, w2]
    rstd3,  # (n_rows,) cached rstd for x^3 power
    rstd2,  # (n_rows,) cached rstd for x^2 power
    rstd1,  # (n_rows,) cached rstd for x^1 power
    dwdb_output,  # (4,) global atomic reduction target [dW0, dW1, dW2, dB]
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """Two-pass PolyNorm backward with atomic weight and bias gradient accumulation.

    ALIGNED requires N_COLS divisible by BLOCK_SIZE.
    """
    row_idx = ct.bid(0)

    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)

    rstd_3 = ct.load(rstd3, row_idx, shape=())
    rstd_2 = ct.load(rstd2, row_idx, shape=())
    rstd_1 = ct.load(rstd1, row_idx, shape=())

    S_3_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    S_2_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    S_1_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    dB_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    num_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=False), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=False), ct.float32)
        else:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        S_3_acc = S_3_acc + dy_t * (x * x * x)
        S_2_acc = S_2_acc + dy_t * (x * x)
        S_1_acc = S_1_acc + dy_t * x
        dB_acc = dB_acc + dy_t

    S_3 = ct.sum(S_3_acc, axis=0, keepdims=False)
    S_2 = ct.sum(S_2_acc, axis=0, keepdims=False)
    S_1 = ct.sum(S_1_acc, axis=0, keepdims=False)
    dB_row = ct.sum(dB_acc, axis=0, keepdims=False)

    ct.atomic_add(dwdb_output, 0, rstd_3 * S_3, check_bounds=False)
    ct.atomic_add(dwdb_output, 1, rstd_2 * S_2, check_bounds=False)
    ct.atomic_add(dwdb_output, 2, rstd_1 * S_1, check_bounds=False)
    ct.atomic_add(dwdb_output, 3, dB_row, check_bounds=False)

    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=False), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=False), ct.float32)
        else:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        x2 = x * x
        x3 = x2 * x

        rstd3_cu = rstd_3 * rstd_3 * rstd_3
        grad_3 = w0 * (3.0 * x2 * rstd_3 * dy_t - (3.0 / N_COLS) * x2 * x3 * rstd3_cu * S_3)

        rstd2_cu = rstd_2 * rstd_2 * rstd_2
        grad_2 = w1 * (2.0 * x * rstd_2 * dy_t - (2.0 / N_COLS) * x3 * rstd2_cu * S_2)

        rstd1_cu = rstd_1 * rstd_1 * rstd_1
        grad_1 = w2 * (rstd_1 * dy_t - (1.0 / N_COLS) * x * rstd1_cu * S_1)

        dx_t = grad_3 + grad_2 + grad_1
        if ALIGNED:
            ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_t, dx.dtype), check_bounds=False)
        else:
            ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_t, dx.dtype), check_bounds=True)


_poly_norm_bwd_kernel_large = _poly_norm_bwd_kernel.replace_hints(occupancy=4, num_worker_warps=8)

_fwd_tune_cache: dict = {}

_FWD_BODY = {"mc": _poly_norm_fwd_kernel, "sc": _poly_norm_fwd_kernel_sc_large}


def _fwd_configs(n_cols):
    """Return (body, block size, occupancy, worker warps) candidates.

    The first candidate is the fixed schedule when autotuning is disabled.
    """
    np2 = next_power_of_2(n_cols)
    cfgs = []

    sc_block = min(MAX_FUSED_SIZE, np2)
    if sc_block >= n_cols and sc_block != 4096:
        if sc_block <= 2048:
            sc_occ = (8, 16)
        else:
            sc_occ = (2, 4)
        sc_nww = (4, 8) if sc_block >= 8192 else (4,)
        for occ in sc_occ:
            for nww in sc_nww:
                cfgs.append(("sc", sc_block, occ, nww))

    for blk in (512, 1024, 2048, 4096):
        if blk >= n_cols:
            continue
        for occ in (6, 8) if blk <= 1024 else (4, 8):
            for nww in (4, 8) if blk >= 1024 else (4,):
                cfgs.append(("mc", blk, occ, nww))
    return cfgs


def _tune_forward(stream, grid, launch_args, n_cols):
    """Return the fastest forward schedule across kernel bodies."""
    configs = _fwd_configs(n_cols)
    best = None  # (mean_us, body_key, BLOCK_SIZE, occupancy, num_worker_warps)
    for body_key in dict.fromkeys(k for (k, _, _, _) in configs):
        sub = [
            SimpleNamespace(BLOCK_SIZE=blk, occupancy=occ, num_worker_warps=nww)
            for (k, blk, occ, nww) in configs
            if k == body_key
        ]
        with ct.compiler_timeout(60):
            result = exhaustive_search(
                sub,
                stream,
                lambda cfg: grid,
                _FWD_BODY[body_key],
                lambda cfg: launch_args(cfg.BLOCK_SIZE),
                lambda cfg: {"occupancy": cfg.occupancy, "num_worker_warps": cfg.num_worker_warps},
            )
        m = result.best
        if best is None or m.mean_us < best[0]:
            best = (m.mean_us, body_key, m.config.BLOCK_SIZE, m.config.occupancy, m.config.num_worker_warps)

    _, body_key, blk, occ, nww = best
    return _FWD_BODY[body_key].replace_hints(occupancy=occ, num_worker_warps=nww), blk


class PolyNormCuTileFunction(torch.autograd.Function):
    """
    PolyNorm autograd function with CuTile forward and backward kernels.

    Formula: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
    where norm(u) = u / sqrt(mean(u²) + ε)
    """

    @staticmethod
    def forward(ctx, X, W, B, eps):
        shape = X.shape
        dim = shape[-1]
        X_2d = X.contiguous().view(-1, dim)
        n_rows, n_cols = X_2d.shape

        bias_shape = B.shape
        B = B.reshape(1)

        Y = torch.empty_like(X_2d)
        RSTD3 = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        RSTD2 = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        RSTD1 = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        stream = torch.cuda.current_stream()
        grid = (n_rows, 1, 1)
        Wc = W.contiguous()
        Bc = B.contiguous()

        def launch_args(blk):
            aligned = bool((n_cols % blk) == 0)
            return (X_2d, Y, Wc, Bc, RSTD3, RSTD2, RSTD1, int(n_cols), float(eps), int(blk), aligned)

        if is_autotune_disabled():
            body_key, BLOCK_SIZE, occ, nww = _fwd_configs(n_cols)[0]
            fwd_kernel = _FWD_BODY[body_key].replace_hints(occupancy=occ, num_worker_warps=nww)
        else:
            cache_key = (n_rows, n_cols, X_2d.dtype, str(X_2d.device))
            if cache_key not in _fwd_tune_cache:
                _fwd_tune_cache[cache_key] = _tune_forward(stream, grid, launch_args, n_cols)
            fwd_kernel, BLOCK_SIZE = _fwd_tune_cache[cache_key]

        ct.launch(stream, grid, fwd_kernel, launch_args(BLOCK_SIZE))

        ctx.save_for_backward(X_2d, W, RSTD3, RSTD2, RSTD1)
        ctx.shape = shape
        ctx.bias_shape = bias_shape

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dy):
        X_2d, W, RSTD3, RSTD2, RSTD1 = ctx.saved_tensors
        shape = ctx.shape

        dim = shape[-1]
        dY_2d = dy.contiguous().view(-1, dim)
        n_rows, n_cols = dY_2d.shape

        BWD_MAX_BLOCK = 4096
        BLOCK_SIZE = min(BWD_MAX_BLOCK, next_power_of_2(n_cols))
        aligned = (n_cols % BLOCK_SIZE) == 0

        dx = torch.empty_like(dY_2d)

        dwdb_output = torch.zeros(4, dtype=torch.float32, device=W.device)

        kernel_choice = _poly_norm_bwd_kernel_large if n_cols >= 8192 else _poly_norm_bwd_kernel

        grid = (n_rows, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel_choice,
            (
                dY_2d,
                dx,
                X_2d,
                W,
                RSTD3,
                RSTD2,
                RSTD1,
                dwdb_output,
                int(n_cols),
                int(BLOCK_SIZE),
                bool(aligned),
            ),
        )

        sums = dwdb_output.to(W.dtype)  # (4,) cast to W.dtype
        dW = sums[:3]
        dB = sums[3:4].reshape(ctx.bias_shape)

        return dx.view(*shape), dW, dB, None


@register_impl("liger.poly_norm", backend="cutile")
def poly_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    PolyNorm normalization (CuTile backend).

    Args:
        input: Input tensor of shape (*, H)
        weight: Weight tensor of shape (3,) for [w0, w1, w2]
        bias: Scalar bias tensor of shape (1,)
        eps: Epsilon for numerical stability. Default: 1e-6

    Returns:
        Output tensor of same shape as input
    """
    return PolyNormCuTileFunction.apply(input, weight, bias, eps)
