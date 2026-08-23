# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Forward autotune space (occupancy × num_worker_warps): the cumsum scan over the
# whole row is thread-parallel, so large n_cols favour nww=8 (256 threads) while
# small n_cols favour nww=4. exhaustive_search picks the per-shape best on first
# launch and caches it.
_SPARSEMAX_FWD_TUNE_CONFIGS = [SimpleNamespace(occ=o, nww=n) for o in [2, 4, 8] for n in [4, 8]]
_SPARSEMAX_FWD_TUNE_CACHE: dict = {}

# Backward autotune space. Same occ×nww sweep, but occ=1 is EXCLUDED: the bwd kernel
# scatters inside a loop and occupancy=1 there causes intermittent hangs. The best
# config is shape-dependent (e.g. occ4/nww8 wins at n_cols=8192, occ2/nww8 at 32768),
# which is exactly why a per-shape autotune is needed rather than a fixed hint.
_SPARSEMAX_BWD_TUNE_CONFIGS = [SimpleNamespace(occ=o, nww=n) for o in [2, 4, 8] for n in [4, 8]]
_SPARSEMAX_BWD_TUNE_CACHE: dict = {}


# Exact sparsemax threshold (Martins & Astudillo 2016, Alg. 1):
# on the descending-sorted row z, find support size k = max{ j : z_(j) > (cssv_j - 1)/j },
# then tau = (sum_{i<=k} z_(i) - 1)/k. The whole row lives in one BLOCK_SIZE tile so ct.cumsum
# gives the running prefix sum.
@ct.kernel
def _sparsemax_fwd_kernel(
    y_output,
    x_input,
    x_sorted,  # row-wise descending sort of x_input (fp32), produced by torch.sort
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],  # = next_pow2(N_COLS): whole row in one tile for the cumsum
):
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    one_b = ct.full((BLOCK_SIZE,), 1.0, ct.float32)
    zero_b = ct.full((BLOCK_SIZE,), 0.0, ct.float32)
    valid_mask = col_idx < N_COLS
    valid_f = ct.astype(valid_mask, ct.float32)

    z_sorted = ct.astype(
        ct.gather(x_sorted, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
        ct.float32,
    )
    # Masked entries (col >= N_COLS) contribute 0 to the prefix sum / are excluded from support.
    z_valid = z_sorted * valid_f
    cssv = ct.cumsum(z_valid, 0)
    r = ct.astype(col_idx, ct.float32) + one_b
    t_vec = (cssv - one_b) / r
    support = (z_sorted > t_vec) & valid_mask

    # Support size k, clamped to >= 1.
    k_int = ct.maximum(ct.sum(ct.astype(support, ct.int32), 0, keepdims=True), ct.full((1,), 1, ct.int32))
    k = ct.astype(k_int, ct.float32)
    s = ct.sum(ct.where(support, z_sorted, zero_b), 0, keepdims=True)
    tau = (s - ct.full((1,), 1.0, ct.float32)) / k

    x_row = ct.astype(
        ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
        ct.float32,
    )
    y = ct.maximum(x_row - tau, ct.full((BLOCK_SIZE,), 0.0, ct.float32))
    ct.scatter(y_output, (row_idx, col_idx), ct.astype(y, y_output.dtype), check_bounds=True)


@ct.kernel
def _sparsemax_bwd_kernel(
    grad_input,
    output,
    grad_output,
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    row_idx = ct.bid(0)
    n_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE

    go_sum_tile = ct.full((BLOCK_SIZE,), 0.0, ct.float32)
    supp_cnt_tile = ct.full((BLOCK_SIZE,), 0.0, ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        o_tile = ct.astype(ct.gather(output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        go_tile = ct.astype(
            ct.gather(grad_output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )
        supp_f = ct.astype(o_tile > ct.full((BLOCK_SIZE,), 0.0, ct.float32), ct.float32)
        go_sum_tile = go_sum_tile + supp_f * go_tile
        supp_cnt_tile = supp_cnt_tile + supp_f

    go_sum = ct.sum(go_sum_tile, 0, keepdims=False)
    supp_cnt = ct.sum(supp_cnt_tile, 0, keepdims=False)
    mean_go = go_sum / (supp_cnt + 1e-6)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        o_tile = ct.astype(ct.gather(output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        go_tile = ct.astype(
            ct.gather(grad_output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )
        supp_f = ct.astype(o_tile > ct.full((BLOCK_SIZE,), 0.0, ct.float32), ct.float32)
        gi_tile = supp_f * (go_tile - mean_go)
        ct.scatter(grad_input, (row_idx, col_idx), ct.astype(gi_tile, grad_input.dtype), check_bounds=True)


def _get_tuned_fwd_kernel(n_cols, BLOCK_SIZE, n_rows, stream, out_flat, x_flat, x_sorted):
    """Autotune occupancy+nww on first call per shape; return cached kernel afterwards."""
    key = (n_cols, BLOCK_SIZE)
    if key in _SPARSEMAX_FWD_TUNE_CACHE:
        return _SPARSEMAX_FWD_TUNE_CACHE[key]

    result = exhaustive_search(
        _SPARSEMAX_FWD_TUNE_CONFIGS,
        stream,
        lambda cfg: (n_rows, 1, 1),
        _sparsemax_fwd_kernel,
        lambda cfg: (out_flat, x_flat, x_sorted, int(n_cols), int(BLOCK_SIZE)),
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SPARSEMAX_FWD_TUNE_CACHE[key] = _sparsemax_fwd_kernel.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SPARSEMAX_FWD_TUNE_CACHE[key]


def _sparsemax_forward_ct(x: torch.Tensor, dim: int):
    """Exact, sort-based sparsemax forward.

    Sort each row descending (torch.sort), then one kernel computes the exact threshold tau
    via prefix sums and applies max(x - tau, 0). The whole row must fit in one tile
    (BLOCK = next_pow2(n_cols)) so ct.cumsum yields the running prefix sum.
    """
    if dim < 0:
        dim += x.dim()
    x_sw = x.transpose(dim, -1).contiguous()
    n_cols = x_sw.size(-1)
    n_rows = x_sw.numel() // n_cols
    x_flat = x_sw.view(n_rows, n_cols)
    x_sorted = torch.sort(x_flat.float(), dim=-1, descending=True).values

    BLOCK_SIZE = next_power_of_2(n_cols)  # whole row in one tile (required for the cumsum)
    out_flat = torch.empty_like(x_flat)
    stream = torch.cuda.current_stream()
    kernel = _get_tuned_fwd_kernel(n_cols, BLOCK_SIZE, n_rows, stream, out_flat, x_flat, x_sorted)
    ct.launch(
        stream,
        (n_rows, 1, 1),
        kernel,
        (out_flat, x_flat, x_sorted, int(n_cols), int(BLOCK_SIZE)),
    )

    return out_flat.view_as(x_sw).transpose(dim, -1).contiguous(), out_flat


def _get_tuned_bwd_kernel(n_cols, BLOCK_SIZE, n_rows, dtype, device):
    """Autotune occupancy+nww per shape on first call; cache afterwards.

    Uses a fresh stream + dummy tensors — the autograd backward stream is not safe
    for exhaustive_search.
    """
    key = (n_cols, BLOCK_SIZE)
    if key in _SPARSEMAX_BWD_TUNE_CACHE:
        return _SPARSEMAX_BWD_TUNE_CACHE[key]

    tune_stream = torch.cuda.Stream(device=device)
    dx = torch.empty(n_rows, n_cols, dtype=dtype, device=device)
    o = torch.empty(n_rows, n_cols, dtype=dtype, device=device)
    go = torch.empty(n_rows, n_cols, dtype=dtype, device=device)

    result = exhaustive_search(
        _SPARSEMAX_BWD_TUNE_CONFIGS,
        tune_stream,
        lambda cfg: (n_rows, 1, 1),
        _sparsemax_bwd_kernel,
        lambda cfg: (dx, o, go, int(n_cols), int(BLOCK_SIZE)),
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SPARSEMAX_BWD_TUNE_CACHE[key] = _sparsemax_bwd_kernel.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SPARSEMAX_BWD_TUNE_CACHE[key]


def _sparsemax_backward_ct(grad_out: torch.Tensor, out_flat: torch.Tensor, dim: int):
    grad_sw = grad_out.transpose(dim, -1).contiguous()
    n_cols = grad_sw.size(-1)
    n_rows = grad_sw.numel() // n_cols
    go_flat = grad_sw.view(n_rows, n_cols).contiguous()

    BLOCK_SIZE = min(next_power_of_2(n_cols), 4096)
    dx_flat = torch.empty_like(go_flat)
    kernel = _get_tuned_bwd_kernel(n_cols, BLOCK_SIZE, n_rows, go_flat.dtype, go_flat.device)
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        kernel,
        (dx_flat, out_flat, go_flat, int(n_cols), int(BLOCK_SIZE)),
    )

    return dx_flat.view_as(grad_sw).transpose(dim, -1)


class SparsemaxCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        y, out_flat = _sparsemax_forward_ct(x.contiguous(), dim)
        ctx.save_for_backward(out_flat)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (out_flat,) = ctx.saved_tensors
        return _sparsemax_backward_ct(grad_out.contiguous(), out_flat, ctx.dim), None


@register_impl("liger.sparsemax", backend="cutile")
def sparsemax(
    input: torch.Tensor,
    dim: int = -1,
    **kwargs,
) -> torch.Tensor:
    return SparsemaxCuTileFunction.apply(input, dim)
