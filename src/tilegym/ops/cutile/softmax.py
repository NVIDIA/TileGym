# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

from .utils import cached_replace_hints
from .utils import next_power_of_2

ConstInt = ct.Constant[int]

_DEFAULT_TMA_OCCUPANCY = 2
_TMA_FWD_OCCUPANCY_CANDIDATES = (2, 4, 6, 8)
# None keeps the compiler default; worker warps above 8 can hang the gather kernels.
_GATHER_WORKER_WARP_CANDIDATES = (None, 4, 8)
_BWD_TMA_HINT_CANDIDATES = (
    SimpleNamespace(occupancy=2, num_worker_warps=None),
    SimpleNamespace(occupancy=4, num_worker_warps=None),
    SimpleNamespace(occupancy=6, num_worker_warps=None),
    SimpleNamespace(occupancy=8, num_worker_warps=None),
    SimpleNamespace(occupancy=1, num_worker_warps=8),
)
_softmax_tune_cache: dict = {}
# Online softmax tile width is fixed so a wide row never becomes one whole-row
# register tile.
_ONLINE_FWD_TILE_SIZE = 16384
_ONLINE_BWD_TILE_SIZE = 16384
# Backward re-reads (y, dy) in its second pass; bounding the rows resident per SM
# keeps that working set inside L2.
_ONLINE_BWD_ROWS_PER_SM = 2
_ONLINE_BWD_HINTS_BY_ARCH = {(10, 7): dict(occupancy=2, num_worker_warps=4)}


@ct.kernel(occupancy=4)
def _softmax_kernel(
    output,
    input,
    N_ROWS: ConstInt,
    TILE_SIZE: ConstInt,
    DIM_COLS: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for row_idx in range(pid, N_ROWS, num_programs):
        # Load the row tile using index-based access
        row = ct.gather(input, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)
        # Convert to float32 for computation
        row = ct.astype(row, ct.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 0, keepdims=True)
        row_minus_max = row - row_max

        # Compute exponential
        numerator = ct.exp(row_minus_max, rounding_mode=RMd.APPROX)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 0, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator, rounding_mode=RMd.APPROX, flush_to_zero=True)
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.scatter(output, (row_idx, offsets), softmax_output, check_bounds=True)


@ct.kernel
def _softmax_kernel_multi_wave_full_row_reg_cached_ldg(
    output,
    input,
    N: ConstInt,
    TILE_SIZE: ConstInt,
):
    """scheduling=multi_wave | coverage=full_row | load=ldg | caching=reg_cached

    Multi-wave softmax: one block per row, no grid-stride loop.
    Uses conditional bounds check — skipped when TILE_SIZE == N (power-of-2 N)."""
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)
    check_bound = TILE_SIZE != N

    row = ct.gather(input, (row_idx, offsets), check_bounds=check_bound, padding_value=-math.inf)
    row = ct.astype(row, ct.float32)

    row_max = ct.max(row, 0, keepdims=True)
    numerator = ct.exp(row - row_max, rounding_mode=RMd.APPROX)
    denominator = ct.sum(numerator, 0, keepdims=True)

    softmax_output = ct.truediv(numerator, denominator, rounding_mode=RMd.APPROX, flush_to_zero=True)
    softmax_output = ct.astype(softmax_output, input.dtype)
    ct.scatter(output, (row_idx, offsets), softmax_output, check_bounds=check_bound)


@ct.kernel
def _softmax_kernel_backward(
    output,
    input,
    do_tensor,
    TILE_SIZE: ConstInt,
    N_COLS: ConstInt,
):
    """One block per row; the whole row is a single tile (TILE_SIZE >= N_COLS)."""
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)
    # Bounds checks compile out when the row exactly fills the tile.
    check_bounds = TILE_SIZE != N_COLS

    # Load the probability and gradient rows; out-of-range columns read as zero
    probs_tile = ct.gather(input, (row_idx, offsets), check_bounds=check_bounds, padding_value=0.0)
    do_tile = ct.gather(do_tensor, (row_idx, offsets), check_bounds=check_bounds, padding_value=0.0)

    # Convert to float32 for computation
    probs_tile = ct.astype(probs_tile, ct.float32)
    do_tile = ct.astype(do_tile, ct.float32)

    # Compute dxhat = probs * do
    dxhat = probs_tile * do_tile

    # Compute the full-row sum of dxhat
    dxhat_sum = ct.sum(dxhat, axis=0, keepdims=True)

    # Compute softmax gradient: dxhat - probs * sum(dxhat)
    probs_times_sum = probs_tile * dxhat_sum
    softmax_grad_output = dxhat - probs_times_sum

    # Convert back to original dtype and store
    softmax_grad_output = ct.astype(softmax_grad_output, input.dtype)
    ct.scatter(output, (row_idx, offsets), softmax_grad_output, check_bounds=check_bounds)


@ct.kernel(occupancy=2, num_worker_warps=8)
def _online_softmax_kernel(
    output,
    input,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt,
    NUM_TILES: ConstInt,
):
    """Online softmax forward: one block per row, walked in NUM_TILES fixed tiles.

    Pass 1 keeps a running max and a running sum rescaled onto it; pass 2 normalizes."""
    row_idx = ct.bid(0)
    # Bounds checks compile out when the row divides evenly into tiles.
    check_bounds = (N_COLS % TILE_SIZE) != 0
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    row_max = ct.full((1,), -math.inf, dtype=ct.float32)
    denominator = ct.full((1,), 0.0, dtype=ct.float32)

    for tile_idx in range(NUM_TILES):
        cols = offsets + tile_idx * TILE_SIZE
        chunk = ct.gather(input, (row_idx, cols), check_bounds=check_bounds, padding_value=-math.inf)
        chunk = ct.astype(chunk, ct.float32)
        new_max = ct.maximum(row_max, ct.max(chunk, 0, keepdims=True))
        denominator = denominator * ct.exp(row_max - new_max, rounding_mode=RMd.APPROX)
        denominator = denominator + ct.sum(ct.exp(chunk - new_max, rounding_mode=RMd.APPROX), 0, keepdims=True)
        row_max = new_max

    inv_denominator = ct.truediv(1.0, denominator, rounding_mode=RMd.APPROX, flush_to_zero=True)

    # Walk pass 2 from the last tile back: the most recently read tiles are the
    # ones most likely still resident in L2.
    for rev_idx in range(NUM_TILES):
        cols = offsets + (NUM_TILES - 1 - rev_idx) * TILE_SIZE
        chunk = ct.gather(input, (row_idx, cols), check_bounds=check_bounds, padding_value=-math.inf)
        chunk = ct.astype(chunk, ct.float32)
        softmax_output = ct.exp(chunk - row_max, rounding_mode=RMd.APPROX) * inv_denominator
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.scatter(output, (row_idx, cols), softmax_output, check_bounds=check_bounds)


@ct.kernel(occupancy=2, num_worker_warps=8)
def _online_softmax_kernel_backward(
    output,
    input,
    do_tensor,
    N_ROWS: ConstInt,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt,
    NUM_TILES: ConstInt,
):
    """Online softmax backward with a persistent grid-stride row loop.

    Pass 1 accumulates the full-row sum(y * dy) over all tiles; pass 2 emits
    y * (dy - sum)."""
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    check_bounds = (N_COLS % TILE_SIZE) != 0
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for row_idx in range(pid, N_ROWS, num_programs):
        dxhat_row_sum = ct.full((1,), 0.0, dtype=ct.float32)

        for tile_idx in range(NUM_TILES):
            cols = offsets + tile_idx * TILE_SIZE
            probs_tile = ct.gather(input, (row_idx, cols), check_bounds=check_bounds, padding_value=0.0)
            do_tile = ct.gather(do_tensor, (row_idx, cols), check_bounds=check_bounds, padding_value=0.0)
            probs_tile = ct.astype(probs_tile, ct.float32)
            do_tile = ct.astype(do_tile, ct.float32)
            dxhat_row_sum = dxhat_row_sum + ct.sum(probs_tile * do_tile, 0, keepdims=True)

        for rev_idx in range(NUM_TILES):
            cols = offsets + (NUM_TILES - 1 - rev_idx) * TILE_SIZE
            probs_tile = ct.gather(input, (row_idx, cols), check_bounds=check_bounds, padding_value=0.0)
            do_tile = ct.gather(do_tensor, (row_idx, cols), check_bounds=check_bounds, padding_value=0.0)
            probs_tile = ct.astype(probs_tile, ct.float32)
            do_tile = ct.astype(do_tile, ct.float32)
            softmax_grad_output = probs_tile * (do_tile - dxhat_row_sum)
            softmax_grad_output = ct.astype(softmax_grad_output, input.dtype)
            ct.scatter(output, (row_idx, cols), softmax_grad_output, check_bounds=check_bounds)


# TMA version with static persistent scheduling
@ct.kernel(occupancy=_DEFAULT_TMA_OCCUPANCY)
def _softmax_kernel_tma(
    output,
    input,
    N_ROWS: ConstInt,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, N_ROWS, num_programs):
        # Load the entire row in one tile (TILE_SIZE >= n_cols by design)
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.NEG_INF)

        # Convert to float32 for computation
        row = ct.astype(row, ct.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = row - row_max

        # Compute exponential
        numerator = ct.exp(row_minus_max, rounding_mode=RMd.APPROX)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 1, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator, rounding_mode=RMd.APPROX, flush_to_zero=True)

        # Convert back to original dtype and store
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.store(output, index=(row_idx, 0), tile=softmax_output)


# Chunked softmax kernel for large tensors (3-pass algorithm)
@experimental_kernel
@ct.kernel(occupancy=4)
def _softmax_kernel_chunked(
    output,
    input,
    N_ROWS: ConstInt,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, N_ROWS, num_programs):
        row_max = ct.full((1,), -math.inf, dtype=ct.float32)
        denominator = ct.full((1,), 0.0, dtype=ct.float32)
        num_chunks = (N_COLS + TILE_SIZE - 1) // TILE_SIZE
        col_offsets_base = ct.arange(TILE_SIZE, dtype=ct.int32)

        # Pass 1: Find maximum
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32) + col_offsets_base
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            chunk_max = ct.max(chunk, 0, keepdims=True)
            row_max = ct.maximum(row_max, chunk_max)

        # Pass 2: First pass to compute denominator (sum of all exp values)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32) + col_offsets_base
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            row_minus_max = chunk - row_max
            numerator = ct.exp(row_minus_max, rounding_mode=RMd.APPROX)
            exponentials_sum = ct.sum(numerator, 0, keepdims=True)
            denominator = denominator + exponentials_sum

        # Reciprocal once per row, multiplied inside the pass-3 chunk loop.
        inv_denominator = ct.truediv(1.0, denominator, rounding_mode=RMd.APPROX, flush_to_zero=True)

        # Pass 3: Compute final softmax
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32) + col_offsets_base
            # Use gather to load chunk (returns 1D tensor like basic kernel)
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            row_minus_max = chunk - row_max
            numerator = ct.exp(row_minus_max, rounding_mode=RMd.APPROX)
            softmax_output = numerator * inv_denominator
            softmax_output = ct.astype(softmax_output, input.dtype)
            # Use scatter with bounds checking to avoid writing padded zeros
            ct.scatter(output, (row_idx, col_indices), softmax_output, check_bounds=True)


@ct.kernel(occupancy=2)
def _softmax_kernel_backward_tma(
    output,
    input,
    do_tensor,
    TILE_SIZE: ConstInt,
):
    """One block per row; the whole row is a single tile (TILE_SIZE >= n_cols).

    Out-of-range columns load as zero and are dropped by the store."""
    row_idx = ct.bid(0)

    # Load the probability and gradient rows
    probs_tile = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    do_tile = ct.load(do_tensor, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)

    # Convert to float32 for computation
    probs_tile = ct.astype(probs_tile, ct.float32)
    do_tile = ct.astype(do_tile, ct.float32)

    # Compute dxhat = probs * do
    dxhat = probs_tile * do_tile

    # Compute the full-row sum of dxhat along axis 1
    dxhat_sum = ct.sum(dxhat, axis=1, keepdims=True)

    # Compute softmax gradient: dxhat - probs * sum(dxhat)
    probs_times_sum = probs_tile * dxhat_sum
    softmax_grad_output = dxhat - probs_times_sum

    # Convert back to original dtype and store
    softmax_grad_output = ct.astype(softmax_grad_output, input.dtype)
    ct.store(output, index=(row_idx, 0), tile=softmax_grad_output)


# Launch patterns for the kernels:
def _cfg_hints(cfg):
    """Map a tuning-config namespace to a cuTile launch-hint dict."""
    hints = {}
    if getattr(cfg, "occupancy", None) is not None:
        hints["occupancy"] = cfg.occupancy
    if getattr(cfg, "num_worker_warps", None) is not None:
        hints["num_worker_warps"] = cfg.num_worker_warps
    return hints


def _launch_softmax_kernel(input, output, TILE_SIZE=1024):
    """
    Launch the basic cuTile softmax kernel with static persistent scheduling

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    num_programs = min(NUM_SM * 4, n_rows)
    grid = (num_programs, 1, 1)
    args = (output, input, n_rows, TILE_SIZE, original_n_cols)

    candidates = [SimpleNamespace(num_worker_warps=w) for w in _GATHER_WORKER_WARP_CANDIDATES]
    if is_autotune_disabled():
        cfg = candidates[0]
    else:
        cache_key = ("gather_fwd", str(input.device), input.dtype, n_rows, original_n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search(
                candidates,
                torch.cuda.current_stream(),
                lambda _: grid,
                _softmax_kernel,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    hints = _cfg_hints(cfg)
    softmax_kernel_forward = cached_replace_hints(_softmax_kernel, **hints) if hints else _softmax_kernel

    ct.launch(torch.cuda.current_stream(), grid, softmax_kernel_forward, args)


def _launch_softmax_kernel_multi_wave_full_row_reg_cached_ldg(input, output, TILE_SIZE):
    n_rows, n_cols = input.shape
    input = input.contiguous()
    output = output.contiguous()

    grid = (n_rows, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_kernel_multi_wave_full_row_reg_cached_ldg,
        (output, input, n_cols, TILE_SIZE),
    )


def _launch_online_softmax_kernel(input, output, TILE_SIZE=_ONLINE_FWD_TILE_SIZE):
    """Launch the online softmax forward kernel: one block per row, bounded tiles."""
    n_rows, n_cols = input.shape
    input = input.contiguous()
    output = output.contiguous()
    num_tiles = (n_cols + TILE_SIZE - 1) // TILE_SIZE
    grid = (n_rows, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _online_softmax_kernel,
        (output, input, n_cols, TILE_SIZE, num_tiles),
    )


def _launch_online_softmax_backward_kernel(input, do, output, TILE_SIZE=_ONLINE_BWD_TILE_SIZE):
    """Launch the online softmax backward kernel: persistent grid, bounded tiles."""
    n_rows, n_cols = input.shape
    input = input.contiguous()
    do = do.contiguous()
    output = output.contiguous()
    num_tiles = (n_cols + TILE_SIZE - 1) // TILE_SIZE
    num_sm = torch.cuda.get_device_properties(input.device).multi_processor_count
    grid = (min(num_sm * _ONLINE_BWD_ROWS_PER_SM, n_rows), 1, 1)
    kernel = _online_softmax_kernel_backward
    arch_hints = _ONLINE_BWD_HINTS_BY_ARCH.get(torch.cuda.get_device_capability(input.device))
    if arch_hints:
        kernel = cached_replace_hints(kernel, **arch_hints)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (output, input, do, n_rows, n_cols, TILE_SIZE, num_tiles),
    )


def _launch_softmax_kernel_tma(
    input,
    output,
    use_online=None,
):
    """
    Launch the TMA cuTile softmax kernel

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        use_online: Whether to use online algorithm
    """
    COL_THRESHOLD = 32768
    # Ensure input is 2D
    if input.dim() == 1:
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    TILE_SIZE = next_power_of_2(n_cols)
    original_n_cols = n_cols

    if use_online is None:
        use_online = n_cols >= COL_THRESHOLD

    if use_online:
        _launch_online_softmax_kernel(input, output)
        return

    # Regular TMA path (single tile per row, persistent scheduling)
    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    args = (output, input, n_rows, original_n_cols, TILE_SIZE)
    candidates = [SimpleNamespace(occupancy=o) for o in _TMA_FWD_OCCUPANCY_CANDIDATES]
    if is_autotune_disabled():
        cfg = candidates[0]
    else:
        cache_key = ("tma_fwd", str(input.device), input.dtype, n_rows, original_n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search(
                candidates,
                torch.cuda.current_stream(),
                lambda candidate: (min(NUM_SM * candidate.occupancy, n_rows), 1, 1),
                _softmax_kernel_tma,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    occupancy = cfg.occupancy
    softmax_kernel_forward = cached_replace_hints(_softmax_kernel_tma, occupancy=occupancy)
    grid = (min(NUM_SM * occupancy, n_rows), 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, softmax_kernel_forward, args)


def _launch_softmax_kernel_chunked(
    input,
    output,
    TILE_SIZE=8192,
):
    """
    Launch the chunked cuTile softmax kernel for large tensors

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing chunks (default 8192)
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    num_programs = min(NUM_SM * 4, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_kernel_chunked,
        (
            output,
            input,
            n_rows,
            original_n_cols,
            TILE_SIZE,
        ),
    )


# Launch functions for backward kernels


def _ensure_cuda_context(device):
    """Make ``device``'s primary CUDA driver context current on the calling thread.

    Backward runs on an autograd worker thread whose driver context stack can be
    empty when no torch kernel has run on it yet, or hold another GPU's context;
    exhaustive_search times configs through the driver API and fails there."""
    try:
        from cuda.bindings import driver as _drv

        err, cur = _drv.cuCtxGetCurrent()
        if err == _drv.CUresult.CUDA_SUCCESS and int(cur) != 0:
            err, cur_device = _drv.cuCtxGetDevice()
            if err == _drv.CUresult.CUDA_SUCCESS and int(cur_device) == device.index:
                return
        err, primary_ctx = _drv.cuDevicePrimaryCtxRetain(device.index)
        if err == _drv.CUresult.CUDA_SUCCESS:
            _drv.cuCtxSetCurrent(primary_ctx)
    except Exception:  # noqa: BLE001
        pass


def _launch_softmax_backward_kernel(input, do, output, TILE_SIZE=None):
    """
    Launch the basic cuTile softmax backward kernel

    Args:
        input: Softmax probabilities tensor of shape (n_rows, n_cols)
        do: Gradient from upstream of shape (n_rows, n_cols)
        output: Contiguous output gradient tensor of shape (n_rows, n_cols)
        TILE_SIZE: Row tile width; must cover the whole row (default: next_power_of_2(n_cols))
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols
    if TILE_SIZE is None:
        TILE_SIZE = next_power_of_2(n_cols)
    assert TILE_SIZE >= n_cols, f"TILE_SIZE ({TILE_SIZE}) must cover n_cols ({n_cols})"
    # The kernel writes into output in place, so it must already be contiguous.
    assert output.is_contiguous(), "softmax backward output must be contiguous"

    # Ensure inputs are contiguous; gather bounds checks handle the tail past n_cols
    input = input.contiguous()
    do = do.contiguous()

    # Calculate grid - single block per row
    grid = (n_rows, 1, 1)
    args = (output, input, do, TILE_SIZE, original_n_cols)

    candidates = [SimpleNamespace(num_worker_warps=w) for w in _GATHER_WORKER_WARP_CANDIDATES]
    if is_autotune_disabled():
        cfg = candidates[0]
    else:
        cache_key = ("gather_bwd", str(input.device), input.dtype, n_rows, original_n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search(
                candidates,
                torch.cuda.current_stream(),
                lambda _: grid,
                _softmax_kernel_backward,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    hints = _cfg_hints(cfg)
    softmax_kernel_backward = (
        cached_replace_hints(_softmax_kernel_backward, **hints) if hints else _softmax_kernel_backward
    )

    ct.launch(torch.cuda.current_stream(), grid, softmax_kernel_backward, args)


def _launch_softmax_backward_kernel_tma(
    input,
    do,
    output,
    use_online=None,
):
    """
    Launch the TMA cuTile softmax backward kernel

    Args:
        input: Softmax probabilities tensor of shape (n_rows, n_cols)
        do: Gradient from upstream of shape (n_rows, n_cols)
        output: Output gradient tensor of shape (n_rows, n_cols)
        use_online: Whether to use online algorithm
    """
    COL_THRESHOLD = 32768

    # Ensure inputs are 2D
    if input.dim() == 1:
        input = input.unsqueeze(0)
        do = do.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        do = do.view(-1, do.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    # Determine block size and kernel type
    if use_online is None:
        use_online = n_cols >= COL_THRESHOLD

    if use_online:
        _launch_online_softmax_backward_kernel(input, do, output)
        return

    TILE_SIZE = next_power_of_2(n_cols)

    # The whole row is one tile; TMA zero-pads the load and clips the store past n_cols.
    # The kernel writes into output in place, so it must already be contiguous.
    assert output.is_contiguous(), "softmax backward output must be contiguous"
    input = input.contiguous()
    do = do.contiguous()
    grid = (n_rows, 1, 1)
    args = (output, input, do, TILE_SIZE)

    if is_autotune_disabled():
        cfg = _BWD_TMA_HINT_CANDIDATES[0]
    else:
        cache_key = ("tma_bwd", str(input.device), input.dtype, n_rows, n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search(
                list(_BWD_TMA_HINT_CANDIDATES),
                torch.cuda.current_stream(),
                lambda _: grid,
                _softmax_kernel_backward_tma,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    softmax_kernel_backward_func = cached_replace_hints(_softmax_kernel_backward_tma, **_cfg_hints(cfg))

    ct.launch(torch.cuda.current_stream(), grid, softmax_kernel_backward_func, args)


# Autograd Function Classes


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        use_tma=False,
        use_online=None,
        use_chunked=False,
        use_multi_wave=False,
    ):
        assert not (use_tma and use_chunked), "Cannot use both TMA and chunked softmax at the same time"
        # TMA may be emulated on this arch; redirect to non-TMA path with a warning.
        if use_tma and torch.cuda.get_device_capability(x.device)[0] < 9:
            import warnings

            warnings.warn(
                "softmax: use_tma=True has no effect on this GPU (TMA is emulated). "
                "Falling back to use_tma=False. Pass use_tma=False to suppress this.",
                UserWarning,
                stacklevel=3,
            )
            use_tma = False
        COL_THRESHOLD = 16384
        n_rows, n_cols = x.shape
        TILE_SIZE = next_power_of_2(n_cols)
        MAX_TILE_SIZE = 8192

        # Automatically choose between regular and online algorithms
        if use_online is None:
            use_online = n_cols >= COL_THRESHOLD

        # Create output tensor; kernels write in place, so it must be contiguous even if x is not
        y = torch.empty_like(x, memory_format=torch.contiguous_format)

        if use_multi_wave:
            _launch_softmax_kernel_multi_wave_full_row_reg_cached_ldg(x, y, TILE_SIZE=TILE_SIZE)
        elif use_chunked:
            # Use chunked kernel (3-pass algorithm for large tensors)
            # Cap TILE_SIZE at 8192 to enable chunking for very large n_cols
            # For smaller n_cols, use next_power_of_2(n_cols) to match data size
            _launch_softmax_kernel_chunked(x, y, TILE_SIZE=min(TILE_SIZE, MAX_TILE_SIZE))
        elif use_tma:
            # Use TMA implementation
            _launch_softmax_kernel_tma(x, y, use_online=use_online)
        elif use_online:
            _launch_online_softmax_kernel(x, y)
        else:
            # Use grid-based implementation
            _launch_softmax_kernel(x, y, TILE_SIZE=TILE_SIZE)
        # Save for backward
        ctx.save_for_backward(y)
        ctx.use_tma = use_tma
        ctx.use_online = use_online
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors

        # Launch on y's device: the stream, SM count and driver context below all
        # follow the current device, which the autograd worker need not have set.
        with torch.cuda.device(y.device):
            _ensure_cuda_context(y.device)

            # Create gradient input tensor; kernels write in place, so it must be contiguous
            grad_input = torch.empty_like(grad_output, memory_format=torch.contiguous_format)

            if ctx.use_tma:
                # Use TMA backward implementation
                _launch_softmax_backward_kernel_tma(
                    y,
                    grad_output,
                    grad_input,
                    ctx.use_online,
                )
            elif ctx.use_online:
                _launch_online_softmax_backward_kernel(y, grad_output, grad_input)
            else:
                # Use grid-based backward implementation
                _launch_softmax_backward_kernel(y, grad_output, grad_input)

        return grad_input, None, None, None, None


@register_impl("softmax", backend="cutile")
def softmax(
    x,
    use_tma=False,
    use_online=None,
    **kwargs,
):
    """
    Performs softmax using cuTile kernels with automatic gradient support

    Args:
        x: Input tensor of shape (M, N)
        use_online: Whether to use online algorithm. If None, auto-choose based on size
        use_tma: Whether to use TMA (Tensor Memory Accelerator) implementation.
                Requires H100+ GPU (compute capability >= 9.0)
        **kwargs: Additional arguments for backend-specific configurations
                  use_chunked: whether to use chunked softmax implementation
                  use_multi_wave: whether to use multi-wave (one block per row)

    Returns:
        Softmax output tensor with gradient support
    """
    use_chunked = kwargs.get("use_chunked", False)
    use_multi_wave = kwargs.get("use_multi_wave", False)
    # Launch on x's device: the launchers use the current device's stream.
    with torch.cuda.device(x.device):
        return _Softmax.apply(
            x,
            use_tma,
            use_online,
            use_chunked,
            use_multi_wave,
        )
