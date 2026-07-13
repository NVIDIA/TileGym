# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
KL Divergence loss kernel (CuTile backend).

Computes KL(y_true || y_pred) where y_pred is in log-space.

Forward: row-parallel, one block per token row (BT).
  - "none" mode: writes per-element loss to (BT, V) output.
  - Reduce modes (sum/mean/batchmean): accumulates row sum to (BT,) output via fold trick;
    final reduction applied at Python level.

Backward: row-parallel, computes -y_true (or -exp(y_true)) and scatters to gradient tensor.

ALIGNED optimization
====================
When N_FULL_CHUNKS > 0, the first N_FULL_CHUNKS chunks are exactly BLOCK_SIZE elements and
use check_bounds=False (hardware TMA path). Only the final tail chunk (if V % BLOCK_SIZE != 0)
uses check_bounds=True (software bounds checking). This avoids per-element bounds checking
for the majority of chunks and activates the hardware TMA path for maximum bandwidth.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE = 4096  # Use 4096 for better pipelining (31 chunks for V=128256)

_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE,
    "sum": _REDUCTION_MODE_SUM,
    "mean": _REDUCTION_MODE_MEAN,
    "batchmean": _REDUCTION_MODE_BATCHMEAN,
}


@ct.kernel
def _kldiv_fwd_none_ct(
    Y,  # (BT, V) log-probs input
    GT,  # (BT, V) target (probs or log-probs)
    LOSS,  # (BT, V) per-element output
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],  # number of full (non-tail) chunks; use check_bounds=False
):
    """
    Forward kernel for reduction='none'. Grid: (BT, 1, 1).
    Writes per-element KL loss to LOSS[row, col].
    Full chunks use check_bounds=False (hardware TMA); tail chunk uses check_bounds=True.
    """
    row_idx = ct.bid(0)
    # Pre-compute eps_tile once outside the loop (compiler hint: loop-invariant)
    eps_tile = ct.full((BLOCK_SIZE,), eps, dtype=ct.float32)

    # Fast path: full aligned chunks (check_bounds=False -> hardware TMA)
    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        ct.scatter(LOSS, (row_idx, col_idx), ct.astype(loss, LOSS.dtype), check_bounds=False)

    # Slow path: tail chunk only if V is not exactly divisible by BLOCK_SIZE
    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        ct.scatter(LOSS, (row_idx, col_idx), ct.astype(loss, LOSS.dtype), check_bounds=True)


@ct.kernel
def _kldiv_fwd_reduce_ct(
    Y,  # (BT, V) log-probs input
    GT,  # (BT, V) target (probs or log-probs)
    LOSS,  # (BT,) per-row sum output
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],  # number of full (non-tail) chunks; use check_bounds=False
):
    """
    Forward kernel for sum/mean/batchmean reductions. Grid: (BT, 1, 1).
    Computes per-row sum via fold trick and stores to LOSS[row].
    Full chunks use check_bounds=False (hardware TMA); tail chunk uses check_bounds=True.
    """
    row_idx = ct.bid(0)

    loss_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    # Pre-compute eps_tile once outside the loop (compiler hint: loop-invariant)
    eps_tile = ct.full((BLOCK_SIZE,), eps, dtype=ct.float32)

    # Fast path: full aligned chunks (check_bounds=False -> hardware TMA)
    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        loss_acc = ct.add(loss_acc, loss)

    # Slow path: tail chunk only if V is not exactly divisible by BLOCK_SIZE
    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        loss_acc = ct.add(loss_acc, loss)

    row_sum = ct.sum(loss_acc, 0, keepdims=False)
    ct.scatter(LOSS, row_idx, ct.astype(row_sum, LOSS.dtype))


@ct.kernel
def _kldiv_bwd_ct(
    GT,  # (BT, V) target (probs or log-probs)
    GRADS,  # (BT, V) output gradient
    n_cols: ct.Constant[int],
    scale,
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],  # number of full (non-tail) chunks; use check_bounds=False
):
    """
    Backward kernel. Grid: (BT, 1, 1).
    Gradient w.r.t. y_pred: -y_true * scale (or -exp(y_true) * scale for log_target).
    scale fuses grad_output and the reduction normalizer (1/BT or 1/(BT*V)) so that
    both are applied in a single pass, eliminating post-kernel element-wise ops.
    Full chunks use check_bounds=False (hardware TMA); tail chunk uses check_bounds=True.
    """
    row_idx = ct.bid(0)

    # Fast path: full aligned chunks (check_bounds=False -> hardware TMA)
    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            res = -ct.exp(gt) * scale
        else:
            res = -gt * scale

        ct.scatter(GRADS, (row_idx, col_idx), ct.astype(res, GRADS.dtype), check_bounds=False)

    # Slow path: tail chunk only if V is not exactly divisible by BLOCK_SIZE
    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            res = -ct.exp(gt) * scale
        else:
            res = -gt * scale

        ct.scatter(GRADS, (row_idx, col_idx), ct.astype(res, GRADS.dtype), check_bounds=True)


def _kldiv_forward_ct(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(V))
    reduction_int = _str_to_reduction_mode[reduction]
    n_full_chunks = V // BLOCK_SIZE  # full aligned chunks (tail handled separately in kernel)

    grid = (BT, 1, 1)

    if reduction_int == _REDUCTION_MODE_NONE:
        output_tensor = torch.zeros(BT, V, device=y_pred.device, dtype=torch.float32)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _kldiv_fwd_none_ct,
            (
                y_pred,
                y_true,
                output_tensor,
                int(V),
                float(eps),
                int(BLOCK_SIZE),
                int(log_target),
                int(n_full_chunks),
            ),
        )
        return output_tensor
    else:
        row_sums = torch.zeros(BT, device=y_pred.device, dtype=torch.float32)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _kldiv_fwd_reduce_ct,
            (
                y_pred,
                y_true,
                row_sums,
                int(V),
                float(eps),
                int(BLOCK_SIZE),
                int(log_target),
                int(n_full_chunks),
            ),
        )
        if reduction_int == _REDUCTION_MODE_BATCHMEAN:
            return row_sums.sum() / BT
        elif reduction_int == _REDUCTION_MODE_SUM:
            return row_sums.sum(dim=0)
        else:  # mean
            return row_sums.sum() / (BT * V)


def _kldiv_backward_ct(y_true, scale, log_target):
    BT, V = y_true.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(V))
    n_full_chunks = V // BLOCK_SIZE  # full aligned chunks (tail handled separately in kernel)

    new_grads = torch.empty_like(y_true)
    grid = (BT, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _kldiv_bwd_ct,
        (
            y_true,
            new_grads,
            int(V),
            float(scale),
            int(BLOCK_SIZE),
            int(log_target),
            int(n_full_chunks),
        ),
    )

    return new_grads


class KLDivCuTileFunction(torch.autograd.Function):
    """CuTile autograd wrapper for KL divergence loss."""

    @staticmethod
    def forward(ctx, y_pred, y_true, reduction, log_target, eps):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return _kldiv_forward_ct(y_pred, y_true, log_target, reduction, eps)

    @staticmethod
    def backward(ctx, grad_output):
        (y_true,) = ctx.saved_tensors
        BT, V = y_true.shape

        # Compute combined scale: fuse grad_output and reduction normalizer into a
        # single scalar so the kernel can apply both in one pass, eliminating the
        # extra element-wise kernel launches that existed previously.
        if grad_output.numel() == 1:
            scale = grad_output.item()
        else:
            scale = 1.0

        if ctx.reduction == "batchmean":
            scale /= BT
        elif ctx.reduction == "mean":
            scale /= BT * V

        derivative = _kldiv_backward_ct(y_true, scale, ctx.log_target)

        # Non-scalar grad_output (rare: only when reduction="none"): apply separately
        if grad_output.numel() != 1:
            derivative = derivative * grad_output

        return derivative, None, None, None, None


@register_impl("liger.kl_div", backend="cutile")
def kl_div(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    return KLDivCuTileFunction.apply(y_pred, y_true, reduction, log_target, eps)
