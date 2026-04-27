# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
RMS LayerNorm CuTile kernels (standard + Gemma variant).

CuTile kernels:
  - _rms_layernorm_forward_ct_1d: 1D gather/scatter forward with autotune over occupancy.
    Uses cuda.tile_experimental.autotune_launch to search occupancy=[1, 2, 4, 8].
    Both produce flat load_ptr_tko IR (matching NVT pattern), avoiding tensor_view/
    partition_view abstraction that causes predicate explosion.
  - _rms_layernorm_backward_ct_1d: 1D gather/scatter backward with autotune over occupancy.
    Matches forward pattern: scalar load for inv_var, 1D reductions.
    Optimized: algebraic refactoring eliminates normed vector (saves TILE_N muls).

Performance notes:
  - Both forward and backward converted from 2D ct.load/ct.store to 1D
    ct.gather/ct.scatter. This produces flat load_ptr_tko IR, avoiding
    tensor_view/partition_view abstraction that causes predicate explosion.
  - Bounds-check elimination: when TILE_N == n_cols, check_bounds=False saves
    ~44 ISETP instructions (~5% speedup).
  - Backward uses scalar inv_var load via ct.gather().item() — no reshapes.
"""

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import calculate_settings

# Module-level tune cache: (direction, n_rows, n_cols, dtype, TILE_N, OFFSET, device) -> tuned_kernel
_rms_layernorm_tune_cache: dict = {}

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


def _rms_layernorm_forward_ct_1d_body(Y, X, W, r, n_cols, eps, OFFSET, TILE_N):
    """
    Shared kernel body for 1D RMS LayerNorm forward.
    Uses ct.gather/ct.scatter producing flat load_ptr_tko IR (matching NVT pattern),
    avoiding tensor_view/partition_view abstraction that causes predicate explosion.

    When TILE_N == n_cols (no padding), check_bounds=False is used on all
    gather/scatter ops since all indices are guaranteed in-bounds (row < n_rows
    by grid launch, offsets < n_cols by construction). This eliminates ~44 ISETP
    boundary check instructions (SASS 712→576), ~5% kernel speedup.
    When TILE_N > n_cols (padded), bounds checking is kept and padding_value=0
    ensures correct reductions (padded zeros contribute 0 to sum_x2).

    Modeled after layernorm.py _layernorm_forward_ct_1d_body pattern.
    """
    row = ct.bid(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    # check_bounds=False when TILE_N == n_cols (compile-time branch, no padding)
    no_padding = TILE_N == n_cols

    # 1D loads via gather — no tensor_view, no partition_view
    x = ct.gather(X, (row, offsets), check_bounds=not no_padding, padding_value=0)
    x = ct.astype(x, ct.float32)

    w = ct.gather(W, offsets, check_bounds=not no_padding, padding_value=0)

    # 1D reduction — axis=0, no reshape needed
    # Padding zeros contribute 0 to sum_x2 — no masking needed.
    x2_sum = ct.sum(x * x, axis=0)  # scalar

    inv_var = ct.rsqrt(x2_sum / n_cols + eps)

    ct.scatter(r, row, inv_var, check_bounds=not no_padding)

    # Normalize and apply weight
    normed = x * inv_var
    if OFFSET != 0.0:
        # Gemma path: compute in f32 (W loaded as f32)
        w_f32 = ct.astype(w, ct.float32)
        output = normed * (OFFSET + w_f32)
        output = ct.astype(output, X.dtype)
    else:
        normed = ct.astype(normed, W.dtype)
        output = normed * w

    ct.scatter(Y, (row, offsets), output, check_bounds=not no_padding)


@ct.kernel
def _rms_layernorm_forward_ct_1d(
    Y,
    X,
    W,
    r,
    n_cols: ConstInt,
    eps: ConstFloat,
    OFFSET: ConstFloat,
    TILE_N: ConstInt,
):
    """1D RMS LayerNorm forward with autotune over occupancy.

    Bare @ct.kernel — occupancy is injected at runtime by autotune_launch via
    hints_fn. Search space: occupancy=[1, 2, 4, 8].
    """
    _rms_layernorm_forward_ct_1d_body(Y, X, W, r, n_cols, eps, OFFSET, TILE_N)


def _rms_layernorm_backward_ct_1d_body(dX, dY, X, W, r, n_cols, OFFSET, TILE_N):
    """
    1D RMS LayerNorm backward body using ct.gather/ct.scatter.

    Produces flat load_ptr_tko IR (matching NVT pattern), avoiding
    tensor_view/partition_view abstraction that causes predicate explosion.
    Scalar load for inv_var via ct.gather().item() — no reshapes.

    Optimized formulation (avoids materializing normed vector):
      dY_W = dY * (OFFSET + W)
      c = (inv_var^3 / n_cols) * sum(dY_W * x)
      dX = inv_var * dY_W - c * x
    This saves TILE_N multiplies vs the original normed = x * inv_var approach.
    Equivalent to: inv_var/n * (n*dY_W - normed * sum(dY_W * normed)).
    """
    row = ct.bid(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    # check_bounds=False when TILE_N == n_cols (compile-time branch, no padding)
    no_padding = TILE_N == n_cols

    # 1D loads via gather — no tensor_view, no partition_view
    dy = ct.gather(dY, (row, offsets), check_bounds=not no_padding, padding_value=0)
    dy = ct.astype(dy, ct.float32)

    x = ct.gather(X, (row, offsets), check_bounds=not no_padding, padding_value=0)
    x = ct.astype(x, ct.float32)

    w = ct.gather(W, offsets, check_bounds=not no_padding, padding_value=0)
    w = ct.astype(w, ct.float32)

    # Scalar load — no reshape needed
    inv_var = ct.gather(r, row, check_bounds=False).item()

    dY_W = dy * (OFFSET + w)  # (TILE_N,)

    # 1D reduction — axis=0, using x directly (not normed)
    sum_dY_W_x = ct.sum(dY_W * x, axis=0)  # scalar

    # Scalar coefficients (cheap scalar ops, not vector ops)
    inv_var_sq = inv_var * inv_var
    c = (inv_var / n_cols) * inv_var_sq * sum_dY_W_x  # inv_var^3 / n * sum(dY_W * x)

    # Final: dX = inv_var * dY_W - c * x
    # Equivalent to: inv_var/n * (n*dY_W - normed * sum(dY_W * normed))
    dX_row = inv_var * dY_W - c * x
    dX_row = ct.astype(dX_row, dX.dtype)

    ct.scatter(dX, (row, offsets), dX_row, check_bounds=not no_padding)


@ct.kernel
def _rms_layernorm_backward_ct_1d(
    dX,
    dY,
    X,
    W,
    r,
    n_cols: ConstInt,
    OFFSET: ConstFloat,
    TILE_N: ConstInt,
):
    """1D RMS LayerNorm backward with autotune over occupancy.

    Bare @ct.kernel — occupancy is injected at runtime by autotune_launch via
    hints_fn. Search space: occupancy=[1, 2, 4, 8].
    """
    _rms_layernorm_backward_ct_1d_body(dX, dY, X, W, r, n_cols, OFFSET, TILE_N)


class _Fast_RMS_Layernorm_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        TILE_N = calculate_settings(n_cols)
        OFFSET = 1.0 if gemma else 0.0

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        stream = torch.cuda.current_stream()
        fwd_cache_key = ("fwd", n_rows, n_cols, X.dtype, TILE_N, OFFSET, str(X.device))
        if fwd_cache_key not in _rms_layernorm_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows,),
                _rms_layernorm_forward_ct_1d,
                lambda cfg: (Y, X, W, r, n_cols, eps, OFFSET, TILE_N),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _rms_layernorm_tune_cache[fwd_cache_key] = ct.kernel(
                _rms_layernorm_forward_ct_1d._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_fwd_kernel = _rms_layernorm_tune_cache[fwd_cache_key]
        ct.launch(
            stream,
            (n_rows,),
            tuned_fwd_kernel,
            (Y, X, W, r, n_cols, eps, OFFSET, TILE_N),
        )

        ctx.eps = eps
        ctx.TILE_N = TILE_N
        ctx.OFFSET = OFFSET
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        # Always allocate separate output buffer — autotune_launch runs the
        # kernel multiple times to benchmark occupancy configs, so in-place
        # (dX == dY) would corrupt dY on the first trial and produce garbage.
        dX = torch.empty_like(dY)

        stream = torch.cuda.current_stream()
        bwd_cache_key = ("bwd", n_rows, n_cols, dY.dtype, ctx.TILE_N, ctx.OFFSET, str(dY.device))
        if bwd_cache_key not in _rms_layernorm_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows,),
                _rms_layernorm_backward_ct_1d,
                lambda cfg: (dX, dY, X, W, r, n_cols, ctx.OFFSET, ctx.TILE_N),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _rms_layernorm_tune_cache[bwd_cache_key] = ct.kernel(
                _rms_layernorm_backward_ct_1d._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_bwd_kernel = _rms_layernorm_tune_cache[bwd_cache_key]
        ct.launch(
            stream,
            (n_rows,),
            tuned_bwd_kernel,
            (dX, dY, X, W, r, n_cols, ctx.OFFSET, ctx.TILE_N),
        )

        return dX.view(*shape), None, None, None


@register_impl("unsloth.rms_layernorm", backend="cutile")
def rms_layernorm(X, W, eps=1e-6, gemma=False):
    return _Fast_RMS_Layernorm_CT.apply(X, W, eps, gemma)
