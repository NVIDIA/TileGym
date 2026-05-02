# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Layer Normalization kernel (CuTile backend).

Forward: row-parallel (one block per row). Single-pass computes sum and sum_sq
  via fold trick → mean and variance. Second pass applies normalization:
  Y = (X - mean) * rstd * W + B. Mean and RSTD cached for backward.

Backward: Split into two kernels for maximum SM parallelism:

  Kernel 1 — DX kernel: grid=(n_rows, 1, 1), one block per row.
    Per row: loads X, DY, W, Mean[row], RSTD[row]; computes x_hat, wdy, c1, c2;
    writes DX[row]. High block count hides memory latency (same as forward).

  Kernel 2 — DW/DB kernel: grid=(sm_count, 1, 1), persistent loop.
    Block b processes rows [b*rpp, (b+1)*rpp); accumulates dW+=dy*x_hat, dB+=dy.
    DW/DB partial buffers: (sm_count, n_cols). Host reduces dim=0.
    Second read of X/DY hits L2 cache (still warm from DX kernel).

  This split avoids the latency-bound regime of the old combined (148-block)
  kernel: the DX kernel launches n_rows blocks (like HF's vectorized kernel),
  giving warp occupancy ~60% and hiding long-scoreboard stalls.
"""

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Cached secondary stream for concurrent DW/DB kernel launch.
# Lazily created per device on first backward call; avoids stream creation overhead.
_dw_stream_cache: dict = {}


def _calculate_settings(n_cols):
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(f"Hidden dimension {n_cols} exceeds maximum supported size of 65536.")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@ct.kernel(occupancy=1)
def _layer_norm_fwd_ct(
    X,  # (n_rows, n_cols) input
    Y,  # (n_rows, n_cols) output
    W,  # (n_cols,) scale
    B,  # (n_cols,) bias
    Mean,  # (n_rows,) cached mean
    RSTD,  # (n_rows,) cached reciprocal std
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """
    Layer norm forward.

    Row-parallel: one block per row. Grid=(n_rows,1,1): every row_idx in [0,n_rows).
    When ALIGNED=True (n_cols is power-of-2), BLOCK_SIZE==n_cols and column accesses
    are exactly in-bounds → check_bounds=False enables hardware TMA path.
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    check_bounds = not ALIGNED

    # ---- Pass 1: compute sum(x) and sum(x^2) via fold trick ----
    # OOB column positions (when ALIGNED=False, last chunk) padded with 0.0 → correct
    sum_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0), ct.float32)
        sum_tile = ct.add(sum_tile, x)
        sum_sq_tile = ct.add(sum_sq_tile, x * x)

    total_sum = ct.sum(sum_tile, 0, keepdims=False)  # scalar
    total_sum_sq = ct.sum(sum_sq_tile, 0, keepdims=False)  # scalar
    mean = total_sum / n_cols
    var = total_sum_sq / n_cols - mean * mean
    rstd = ct.rsqrt(var + eps)  # scalar

    # Cache mean and rstd for backward
    ct.scatter(Mean, row_idx, ct.astype(mean, Mean.dtype))
    ct.scatter(RSTD, row_idx, ct.astype(rstd, RSTD.dtype))

    # ---- Pass 2: Y = (X - mean) * rstd * W + B ----
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0), ct.float32)
        w = ct.astype(ct.gather(W, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
        b = ct.astype(ct.gather(B, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
        y = (x - mean) * rstd * w + b
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y, Y.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _layer_norm_bwd_dx_ct(
    X,  # (n_rows, n_cols) saved input
    DY,  # (n_rows, n_cols) upstream gradient
    W,  # (n_cols,) scale
    Mean,  # (n_rows,) saved mean
    RSTD,  # (n_rows,) saved rstd
    DX,  # (n_rows, n_cols) output gradient w.r.t. input
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """
    Layer norm backward — DX kernel only.

    Grid: (n_rows, 1, 1). One block per row → high concurrency hides latency.
    Each block: load X[row], DY[row], W, Mean[row], RSTD[row]; compute DX[row].
    Does NOT compute DW/DB — handled by separate _layer_norm_bwd_dw_ct kernel.

    When ALIGNED=True: check_bounds=False → hardware TMA path for all accesses.
    """
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    check_bounds = not ALIGNED

    mean = ct.astype(ct.load(Mean, row_idx, shape=(), latency=3), ct.float32)
    rstd = ct.astype(ct.load(RSTD, row_idx, shape=(), latency=3), ct.float32)

    w = ct.astype(ct.gather(W, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
    _lat = 3 if BLOCK_SIZE <= 1024 else 6
    x = ct.astype(
        ct.gather(X, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat), ct.float32
    )
    dy = ct.astype(
        ct.gather(DY, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat), ct.float32
    )

    inv_n_cols = 1.0 / n_cols
    x_hat = (x - mean) * rstd
    wdy = w * dy
    c1 = ct.sum(x_hat * wdy, 0, keepdims=False) * inv_n_cols
    c2 = ct.sum(wdy, 0, keepdims=False) * inv_n_cols
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    ct.scatter(DX, (row_idx, col_idx), ct.astype(dx, DX.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _layer_norm_bwd_dw_ct(
    X,  # (n_rows, n_cols) saved input
    DY,  # (n_rows, n_cols) upstream gradient
    Mean,  # (n_rows,) saved mean
    RSTD,  # (n_rows,) saved rstd
    DWDB_partial,  # (2*num_programs, n_cols) stacked DW [0:num_programs) + DB [num_programs:2*num_programs)
    n_rows: ct.Constant[int],
    n_cols: ct.Constant[int],
    num_programs: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """
    Layer norm backward — DW/DB accumulation kernel.

    Grid: (num_programs, 1, 1). Block b processes rows [b*rpp, min((b+1)*rpp, n_rows)).
    Writes DW partial to row block_id and DB partial to row (num_programs + block_id)
    in the stacked DWDB_partial tensor. Host sums each half in a single call.

    When ALIGNED=True: check_bounds=False → hardware TMA for all accesses.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    check_bounds = not ALIGNED

    dW_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    dB_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri
        if row_idx < n_rows:
            mean = ct.astype(ct.load(Mean, row_idx, shape=(), latency=3), ct.float32)
            rstd = ct.astype(ct.load(RSTD, row_idx, shape=(), latency=3), ct.float32)
            _lat = 3 if BLOCK_SIZE <= 1024 else 6
            x = ct.astype(
                ct.gather(X, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat), ct.float32
            )
            dy = ct.astype(
                ct.gather(DY, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat),
                ct.float32,
            )
            x_hat = (x - mean) * rstd
            dW_acc = ct.add(dW_acc, dy * x_hat)
            dB_acc = ct.add(dB_acc, dy)

    # Write DW to row block_id, DB to row (num_programs + block_id) in stacked buffer.
    # ALIGNED=True: BLOCK_SIZE==n_cols, ct.store is safe (no OOB).
    # ALIGNED=False: BLOCK_SIZE>n_cols, must use scatter with check_bounds=True to
    #   avoid writing OOB padding elements (col_idx already defined above).
    db_row = num_programs + block_id
    if ALIGNED:
        ct.store(DWDB_partial, index=(block_id, 0), tile=dW_acc.reshape((1, BLOCK_SIZE)))
        ct.store(DWDB_partial, index=(db_row, 0), tile=dB_acc.reshape((1, BLOCK_SIZE)))
    else:
        ct.scatter(DWDB_partial, (block_id, col_idx), dW_acc, check_bounds=True)
        ct.scatter(DWDB_partial, (db_row, col_idx), dB_acc, check_bounds=True)


def _layer_norm_forward_ct(X, W, B, eps):
    shape = X.shape
    dim = shape[-1]
    X2d = X.view(-1, dim).contiguous()
    n_rows, n_cols = X2d.shape

    BLOCK_SIZE, _ = _calculate_settings(n_cols)
    aligned = (n_cols & (n_cols - 1)) == 0  # True when n_cols is a power of 2

    Y = torch.empty_like(X2d)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    grid = (n_rows, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _layer_norm_fwd_ct,
        (
            X2d,
            Y,
            W.contiguous(),
            B.contiguous(),
            Mean,
            RSTD,
            int(n_cols),
            float(eps),
            int(BLOCK_SIZE),
            bool(aligned),
        ),
    )

    return Y.view(*shape), X2d, Mean, RSTD, BLOCK_SIZE


def _layer_norm_backward_ct(dY, X, W, B, Mean, RSTD, BLOCK_SIZE, compute_dW=True, compute_dB=True):
    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim).contiguous()
    n_rows, n_cols = dY2d.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    aligned = (n_cols & (n_cols - 1)) == 0  # True when n_cols is a power of 2

    X_contig = X.contiguous()
    W_contig = W.contiguous()

    DX = torch.empty_like(X)

    # Fast path: skip DW/DB entirely if neither W nor B needs gradients.
    # This halves memory traffic and eliminates the secondary kernel + stream sync.
    if not compute_dW and not compute_dB:
        main_stream = torch.cuda.current_stream()
        dx_grid = (n_rows, 1, 1)
        ct.launch(
            main_stream,
            dx_grid,
            _layer_norm_bwd_dx_ct,
            (
                X_contig,
                dY2d,
                W_contig,
                Mean,
                RSTD,
                DX,
                int(n_cols),
                int(BLOCK_SIZE),
                bool(aligned),
            ),
        )
        DX = DX.view(*shape)
        DW = torch.zeros_like(W)
        DB = torch.zeros_like(B)
        return DX, DW, DB

    dw_scale = 1
    num_programs = sm_count * dw_scale
    # Stacked buffer: rows [0:num_programs) = dW partial, rows [num_programs:2*num_programs) = dB partial.
    # One .sum() call reduces both halves in a single CUDA kernel, saving ~5-10us over two calls.
    # Use torch.empty (not zeros): every row is fully written by one block in the DW/DB kernel
    # (grid=(num_programs,), block b writes rows b and num_programs+b), so no initialization needed.
    DWDB_partial = torch.empty(2 * num_programs, n_cols, dtype=torch.float32, device=W.device)

    rows_per_program = math.ceil(n_rows / num_programs)
    dx_grid = (n_rows, 1, 1)
    dw_grid = (num_programs, 1, 1)

    # Launch DX and DW/DB kernels concurrently on two streams.
    # Both kernels only READ X and DY (no write conflict), so parallel execution is safe.
    # DX kernel: n_rows blocks → high concurrency, hides latency.
    # DW/DB kernel: sm_count blocks, persistent loop → shares remaining SM capacity.
    # Two-stream overlap gives 1.2–1.6x speedup over sequential launch.
    main_stream = torch.cuda.current_stream()
    device_idx = X.device.index if X.device.index is not None else 0
    if device_idx not in _dw_stream_cache:
        _dw_stream_cache[device_idx] = torch.cuda.Stream(device=X.device)
    dw_stream = _dw_stream_cache[device_idx]

    ct.launch(
        main_stream,
        dx_grid,
        _layer_norm_bwd_dx_ct,
        (
            X_contig,
            dY2d,
            W_contig,
            Mean,
            RSTD,
            DX,
            int(n_cols),
            int(BLOCK_SIZE),
            bool(aligned),
        ),
    )

    # DW/DB kernel on secondary stream — runs concurrently with DX kernel.
    ct.launch(
        dw_stream,
        dw_grid,
        _layer_norm_bwd_dw_ct,
        (
            X_contig,
            dY2d,
            Mean,
            RSTD,
            DWDB_partial,
            int(n_rows),
            int(n_cols),
            int(num_programs),
            int(rows_per_program),
            int(BLOCK_SIZE),
            bool(aligned),
        ),
    )

    # Run partial sums on dw_stream immediately after DW/DB kernel — overlaps with DX.
    # Single sum over stacked (2*num_programs, n_cols) → (2, n_cols); split into DW/DB.
    with torch.cuda.stream(dw_stream):
        DWDB = DWDB_partial.view(2, num_programs, n_cols).sum(dim=1)

    # Sync dw_stream back to main stream: wait for DW/DB kernel + sums.
    main_stream.wait_stream(dw_stream)

    DX = DX.view(*shape)
    DW = DWDB[0].to(W.dtype)
    DB = DWDB[1].to(B.dtype)
    return DX, DW, DB


class LayerNormFunction(torch.autograd.Function):
    """CuTile autograd wrapper for layer normalization."""

    @staticmethod
    def forward(ctx, X, W, B, eps):
        X = X.contiguous()
        W = W.contiguous()
        B = B.contiguous()
        Y, X_saved, Mean, RSTD, BLOCK_SIZE = _layer_norm_forward_ct(X, W, B, eps)
        ctx.save_for_backward(X_saved, W, B, Mean, RSTD)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        dY = dY.contiguous()
        need_dW = ctx.needs_input_grad[1]
        need_dB = ctx.needs_input_grad[2]
        DX, DW, DB = _layer_norm_backward_ct(
            dY,
            X,
            W,
            B,
            Mean,
            RSTD,
            ctx.BLOCK_SIZE,
            compute_dW=need_dW,
            compute_dB=need_dB,
        )
        return DX, DW if need_dW else None, DB if need_dB else None, None


@register_impl("liger.layer_norm", backend="cutile")
def layer_norm(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-5,
    **kwargs,
) -> torch.Tensor:
    return LayerNormFunction.apply(X, W, B, eps)
