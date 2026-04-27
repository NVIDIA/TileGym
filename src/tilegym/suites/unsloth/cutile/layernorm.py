# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.
# Also credits: Copyright 2024-present Andrej Karpathy & the llm.c team.

"""
Full LayerNorm (with bias) CuTile kernels.

CuTile kernels:
  - _layernorm_forward_ct_1d: 1D gather/scatter forward with autotune over occupancy.
    Uses cuda.tile_experimental.autotune_launch to search occupancy=[1, 2, 4, 8].
    Low occupancy (occ=1, 102 regs) optimal for sparse grids (small n_rows);
    high occupancy (occ=4, 62 regs) optimal for dense grids (large n_rows).
    Both produce flat load_ptr_tko IR (matching NVT pattern), avoiding tensor_view/
    partition_view abstraction that causes predicate explosion.
  - _layernorm_backward_ct_1d: 1D gather/scatter backward with autotune over occupancy.
    Matches forward pattern: scalar loads for inv_var/mean, 1D reductions.
"""

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import calculate_settings

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]

# Module-level tune caches: (direction, n_rows, n_cols, dtype, TILE_N, device) -> tuned_kernel
_layernorm_fwd_tune_cache: dict = {}
_layernorm_bwd_tune_cache: dict = {}


def _layernorm_forward_ct_1d_body(Y, X, W, b, r, mu, n_cols, eps, TILE_N):
    """
    Shared kernel body for 1D LayerNorm forward.
    Uses ct.gather/ct.scatter producing flat load_ptr_tko IR (matching NVT pattern),
    avoiding tensor_view/partition_view abstraction that causes predicate explosion.

    When TILE_N == n_cols (no padding), check_bounds=False is used on all
    gather/scatter ops since all indices are guaranteed in-bounds (row < n_rows
    by grid launch, offsets < n_cols by construction). This eliminates 44 ISETP
    boundary check instructions (SASS 712→576), ~5% kernel speedup.
    When TILE_N > n_cols (padded), bounds checking is kept and padding_value=0
    ensures correct reductions (padded zeros contribute 0 to sum/sum_sq).

    Modeled after rms_norm_kernel_gather pattern in _cutile_kernels/rms_norm.py.
    """
    row = ct.bid(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    # check_bounds=False when TILE_N == n_cols (compile-time branch, no padding)
    no_padding = TILE_N == n_cols

    # 1D loads via gather — no tensor_view, no partition_view
    x = ct.gather(X, (row, offsets), check_bounds=not no_padding, padding_value=0)
    x = ct.astype(x, ct.float32)

    w = ct.gather(W, offsets, check_bounds=not no_padding, padding_value=0)
    w = ct.astype(w, ct.float32)

    b_val = ct.gather(b, offsets, check_bounds=not no_padding, padding_value=0)
    b_val = ct.astype(b_val, ct.float32)

    sum_x = ct.sum(x, axis=0)  # scalar
    mean_x = sum_x / n_cols

    # Second pass: compute variance from centered values
    xx = x - mean_x
    if TILE_N > n_cols:
        xx_masked = ct.where(ct.arange(TILE_N, dtype=ct.int32) < n_cols, xx, 0.0)
    else:
        xx_masked = xx
    var = ct.sum(xx_masked * xx_masked, axis=0) / n_cols
    inv_var = ct.rsqrt(var + eps)

    ct.scatter(mu, row, mean_x, check_bounds=not no_padding)
    ct.scatter(r, row, inv_var, check_bounds=not no_padding)

    # Normalize, scale, bias — all 1D ops
    output = (xx * inv_var) * w + b_val
    output = ct.astype(output, X.dtype)

    ct.scatter(Y, (row, offsets), output, check_bounds=not no_padding)


@ct.kernel
def _layernorm_forward_ct_1d(
    Y,
    X,
    W,
    b,
    r,
    mu,
    n_cols: ConstInt,
    eps: ConstFloat,
    TILE_N: ConstInt,
):
    """1D LayerNorm forward with autotune over occupancy.

    Bare @ct.kernel — occupancy is injected at runtime by autotune_launch via
    hints_fn. Search space: occupancy=[1, 2, 4, 8].
    Low occupancy (occ=1, 102 regs) optimal for sparse grids (small n_rows);
    high occupancy (occ=4, 62 regs) optimal for dense grids (large n_rows).
    """
    _layernorm_forward_ct_1d_body(Y, X, W, b, r, mu, n_cols, eps, TILE_N)


def _layernorm_backward_ct_1d_body(dX, dY, X, W, r, mu, n_cols, TILE_N):
    """
    1D LayerNorm backward body using ct.gather/ct.scatter.

    Produces flat load_ptr_tko IR (matching NVT pattern), avoiding
    tensor_view/partition_view abstraction that causes predicate explosion.
    Scalar loads for inv_var and mean via ct.gather().item() — no reshapes.

    IMPORTANT: dX and dY must be separate tensors (not aliased) because
    autotune_launch runs the kernel multiple times to benchmark occupancy
    configs. In-place (dX == dY) would corrupt dY on the first trial,
    causing subsequent trials to read garbage and produce explosive outputs.

    Optimized formulation (avoids materializing normed vector):
      xhat = x - mean
      dY_W = dY * W
      dX = inv_var * dY_W - (inv_var / n) * sum(dY_W)
           - (inv_var^3 / n) * sum(dY_W * xhat) * xhat
    This saves TILE_N multiplies vs the original normed = xhat * inv_var.
    """
    row = ct.bid(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    # check_bounds=False when TILE_N == n_cols (compile-time branch, no padding)
    no_padding = TILE_N == n_cols

    # 1D loads via gather — no tensor_view, no partition_view
    # padding_value=0 ensures dY_W=0 at padded positions (no explicit mask needed)
    dy = ct.gather(dY, (row, offsets), check_bounds=not no_padding, padding_value=0)
    dy = ct.astype(dy, ct.float32)

    x = ct.gather(X, (row, offsets), check_bounds=not no_padding, padding_value=0)
    x = ct.astype(x, ct.float32)

    w = ct.gather(W, offsets, check_bounds=not no_padding, padding_value=0)
    w = ct.astype(w, ct.float32)

    # Scalar loads — no reshape needed
    inv_var = ct.gather(r, row, check_bounds=False).item()
    mean = ct.gather(mu, row, check_bounds=False).item()

    xhat = x - mean  # (TILE_N,) — unnormalized centered values
    dY_W = dy * w  # zero at padded positions (dy=0, w=0)

    # 1D reductions — axis=0, no masking needed (dY_W=0 at padded positions)
    sum_dY_W = ct.sum(dY_W, axis=0)
    sum_dY_W_xhat = ct.sum(dY_W * xhat, axis=0)

    # Scalar coefficients (cheap scalar ops, not vector ops)
    c1 = inv_var / n_cols  # inv_var / n
    inv_var_sq = inv_var * inv_var
    c2 = c1 * inv_var_sq * sum_dY_W_xhat  # inv_var^3 / n * sum(dY_W * xhat)

    # Final: dX = inv_var * dY_W - c1 * sum_dY_W - c2 * xhat
    # Equivalent to: inv_var * (dY_W - sum(dY_W)/n - normed * sum(dY_W*normed)/n)
    dX_row = inv_var * dY_W - c1 * sum_dY_W - c2 * xhat
    dX_row = ct.astype(dX_row, dY.dtype)

    ct.scatter(dX, (row, offsets), dX_row, check_bounds=not no_padding)


@ct.kernel
def _layernorm_backward_ct_1d(
    dX,  # output (separate buffer — must not alias dY for autotune safety)
    dY,  # input
    X,
    W,
    r,
    mu,
    n_cols: ConstInt,
    TILE_N: ConstInt,
):
    """1D LayerNorm backward with autotune over occupancy.

    Bare @ct.kernel — occupancy is injected at runtime by autotune_launch via
    hints_fn. Search space: occupancy=[1, 2, 4, 8].
    """
    _layernorm_backward_ct_1d_body(dX, dY, X, W, r, mu, n_cols, TILE_N)


class _Fast_Layernorm_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        TILE_N = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        mu = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        stream = torch.cuda.current_stream()
        fwd_cache_key = (n_rows, n_cols, X.dtype, TILE_N, str(X.device))
        if fwd_cache_key not in _layernorm_fwd_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows,),
                _layernorm_forward_ct_1d,
                lambda cfg: (Y, X, W, b, r, mu, n_cols, eps, TILE_N),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _layernorm_fwd_tune_cache[fwd_cache_key] = ct.kernel(
                _layernorm_forward_ct_1d._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_fwd_kernel = _layernorm_fwd_tune_cache[fwd_cache_key]
        ct.launch(
            stream,
            (n_rows,),
            tuned_fwd_kernel,
            (Y, X, W, b, r, mu, n_cols, eps, TILE_N),
        )

        ctx.eps = eps
        ctx.TILE_N = TILE_N
        ctx.save_for_backward(X, W, r, mu)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r, mu = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        # Separate output buffer — autotune_launch runs the kernel multiple
        # times to benchmark occupancy configs, so in-place (dX == dY) would
        # corrupt dY on the first trial and produce garbage on subsequent ones.
        dX = torch.empty_like(dY)

        stream = torch.cuda.current_stream()
        bwd_cache_key = (n_rows, n_cols, dY.dtype, ctx.TILE_N, str(dY.device))
        if bwd_cache_key not in _layernorm_bwd_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows,),
                _layernorm_backward_ct_1d,
                lambda cfg: (dX, dY, X, W, r, mu, n_cols, ctx.TILE_N),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _layernorm_bwd_tune_cache[bwd_cache_key] = ct.kernel(
                _layernorm_backward_ct_1d._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_bwd_kernel = _layernorm_bwd_tune_cache[bwd_cache_key]
        ct.launch(
            stream,
            (n_rows,),
            tuned_bwd_kernel,
            (dX, dY, X, W, r, mu, n_cols, ctx.TILE_N),
        )

        return dX.view(*shape), None, None, None, None


@register_impl("unsloth.layernorm", backend="cutile")
def layernorm(X, W, b, eps=1e-6):
    return _Fast_Layernorm_CT.apply(X, W, b, eps)
