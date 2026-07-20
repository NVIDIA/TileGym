# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Total Variation Distance loss kernel (CuTile backend).

TVD(P || Q) = 0.5 * |P - Q|
Gradient w.r.t. P: 0.5 if P > Q, -0.5 if P <= Q

Optional shift_labels for ignore_index masking (zeros out grads for ignored rows).
Reduction modes: none, sum, mean, batchmean.

PERF NOTES
==========
- TMA (ct.load) instead of gather for loads of P and Q.
  TMA is hardware-accelerated and avoids per-element index computation.
- Aligned fast path: when n_cols % BLOCK_SIZE == 0, use TMA store for GRADS/LOSS.
  Unaligned path: ct.scatter with check_bounds=True for GRADS/LOSS writes.
- Stride-misaligned V ((V * itemsize) % 16 != 0): TMA descriptor can't be built and ct.load
  falls back to a serialized per-element path; route to the ct.gather kernel (indexed LDG,
  BLOCK=4096, occupancy=2) instead — reaches ~48% SM throughput and beats the TMA fallback.
- @ct.kernel(occupancy=3): TMA kernels; occupancy=2 for the gather kernel.
- BLOCK_SIZE=8192: fewer loop iterations, TMA handles large tiles efficiently.
  At V=128256: ceil(128256/8192)=16 chunks per row vs 128 at baseline.
- Reduction scale (1/n_non_ignore etc.) is fused into the kernel (grads/loss written
  pre-scaled) — no host-side division pass.
- Grid: (BT, 1, 1) — one block per token row, with persistent scheduling over rows.
"""

from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

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


@ct.kernel(occupancy=3)
def _tv_distance_kernel_ct_aligned(
    P,  # (BT, V) first distribution
    Q,  # (BT, V) second distribution
    LOSS,  # (BT, V) when reduction=none, else (BT,)
    GRADS,  # (BT, V) gradient output
    LABEL,  # (BT,) label tensor, or dummy 1-elem tensor when HAS_LABEL=0
    scale,  # runtime float: fused reduction scale (1/n_non_ignore etc.); 1.0 for sum/none
    n_rows: ct.Constant[int],
    n_cols: ct.Constant[int],
    ignore_index: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    HAS_LABEL: ct.Constant[int],
    reduction: ct.Constant[int],
):
    """
    TVD forward -- aligned fast path (n_cols % BLOCK_SIZE == 0).

    Uses TMA (ct.load/ct.store) for all 2D accesses.
    Persistent scheduling: each block handles multiple rows.
    """
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    n_chunks = n_cols // BLOCK_SIZE

    for row_idx in range(pid, n_rows, num_programs):
        if HAS_LABEL:
            lbl = ct.load(LABEL, row_idx, shape=())
            if lbl == ignore_index:
                for ci in range(n_chunks):
                    ct.store(GRADS, index=(row_idx, ci), tile=ct.full((1, BLOCK_SIZE), 0.0, GRADS.dtype))
                    if reduction == _REDUCTION_MODE_NONE:
                        ct.store(LOSS, index=(row_idx, ci), tile=ct.full((1, BLOCK_SIZE), 0.0, LOSS.dtype))
                if reduction != _REDUCTION_MODE_NONE:
                    ct.scatter(LOSS, row_idx, ct.astype(0.0, LOSS.dtype))
                continue

        loss_acc = ct.full((BLOCK_SIZE,), 0.0, ct.float32)

        for ci in range(n_chunks):
            p = ct.astype(
                ct.load(P, index=(row_idx, ci), shape=(1, BLOCK_SIZE), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BLOCK_SIZE,)
                ),
                ct.float32,
            )
            q = ct.astype(
                ct.load(Q, index=(row_idx, ci), shape=(1, BLOCK_SIZE), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BLOCK_SIZE,)
                ),
                ct.float32,
            )

            diff = p - q
            tv_loss = ct.maximum(diff, 0.0 - diff) * 0.5
            p_gt_q = ct.astype(p > q, ct.float32)
            # Fuse reduction scale into the gradient: grads written pre-scaled,
            # so no host-side grads / n_non_ignore pass is needed. scale == 1.0 for sum/none.
            grad = (2.0 * p_gt_q - 1.0) * 0.5 * scale

            ct.store(GRADS, index=(row_idx, ci), tile=ct.astype(grad, GRADS.dtype).reshape((1, BLOCK_SIZE)))

            if reduction == _REDUCTION_MODE_NONE:
                ct.store(LOSS, index=(row_idx, ci), tile=ct.astype(tv_loss, LOSS.dtype).reshape((1, BLOCK_SIZE)))
            else:
                loss_acc = loss_acc + tv_loss

        if reduction != _REDUCTION_MODE_NONE:
            row_sum = ct.sum(loss_acc, 0, keepdims=False)
            ct.scatter(LOSS, row_idx, ct.astype(row_sum * scale, LOSS.dtype))


@ct.kernel(occupancy=3)
def _tv_distance_kernel_ct(
    P,  # (BT, V) first distribution
    Q,  # (BT, V) second distribution
    LOSS,  # (BT, V) when reduction=none, else (BT,)
    GRADS,  # (BT, V) gradient output
    LABEL,  # (BT,) label tensor, or dummy 1-elem tensor when HAS_LABEL=0
    scale,  # runtime float: fused reduction scale (1/n_non_ignore etc.); 1.0 for sum/none
    n_rows: ct.Constant[int],
    n_cols: ct.Constant[int],
    ignore_index: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    HAS_LABEL: ct.Constant[int],
    reduction: ct.Constant[int],
):
    """
    TVD forward -- general path (arbitrary n_cols).

    Uses TMA ct.load for P and Q (with zero padding for last chunk).
    Uses ct.scatter for GRADS/LOSS writes (bounds-checked for partial last chunk).
    Persistent scheduling: each block handles multiple rows.
    """
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for row_idx in range(pid, n_rows, num_programs):
        if HAS_LABEL:
            lbl = ct.load(LABEL, row_idx, shape=())
            if lbl == ignore_index:
                for ci in range(n_chunks):
                    col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
                    ct.scatter(GRADS, (row_idx, col_idx), ct.full((BLOCK_SIZE,), 0.0, GRADS.dtype), check_bounds=True)
                    if reduction == _REDUCTION_MODE_NONE:
                        ct.scatter(LOSS, (row_idx, col_idx), ct.full((BLOCK_SIZE,), 0.0, LOSS.dtype), check_bounds=True)
                if reduction != _REDUCTION_MODE_NONE:
                    ct.scatter(LOSS, row_idx, ct.astype(0.0, LOSS.dtype))
                continue

        loss_acc = ct.full((BLOCK_SIZE,), 0.0, ct.float32)

        for ci in range(n_chunks):
            col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

            p = ct.astype(
                ct.load(P, index=(row_idx, ci), shape=(1, BLOCK_SIZE), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BLOCK_SIZE,)
                ),
                ct.float32,
            )
            q = ct.astype(
                ct.load(Q, index=(row_idx, ci), shape=(1, BLOCK_SIZE), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BLOCK_SIZE,)
                ),
                ct.float32,
            )

            diff = p - q
            tv_loss = ct.maximum(diff, 0.0 - diff) * 0.5
            p_gt_q = ct.astype(p > q, ct.float32)
            # Fuse reduction scale into the gradient: grads written pre-scaled,
            # so no host-side grads / n_non_ignore pass is needed. scale == 1.0 for sum/none.
            grad = (2.0 * p_gt_q - 1.0) * 0.5 * scale

            ct.scatter(GRADS, (row_idx, col_idx), ct.astype(grad, GRADS.dtype), check_bounds=True)

            if reduction == _REDUCTION_MODE_NONE:
                ct.scatter(LOSS, (row_idx, col_idx), ct.astype(tv_loss, LOSS.dtype), check_bounds=True)
            else:
                loss_acc = loss_acc + tv_loss

        if reduction != _REDUCTION_MODE_NONE:
            row_sum = ct.sum(loss_acc, 0, keepdims=False)
            ct.scatter(LOSS, row_idx, ct.astype(row_sum * scale, LOSS.dtype))


@ct.kernel(occupancy=2)
def _tv_distance_kernel_ct_gather(
    P,  # (BT, V) first distribution
    Q,  # (BT, V) second distribution
    LOSS,  # (BT, V) when reduction=none, else (BT,)
    GRADS,  # (BT, V) gradient output
    LABEL,  # (BT,) label tensor, or dummy 1-elem tensor when HAS_LABEL=0
    scale,  # runtime float: fused reduction scale (1/n_non_ignore etc.); 1.0 for sum/none
    n_rows: ct.Constant[int],
    n_cols: ct.Constant[int],
    ignore_index: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    HAS_LABEL: ct.Constant[int],
    reduction: ct.Constant[int],
):
    """
    TVD forward -- gather path for stride-misaligned V ((V * itemsize) % 16 != 0).

    When rows are not 16-byte aligned, ct.load's TMA descriptor can't be built: it falls
    back to per-element loads but keeps the tile-load structure (2x the loads + many CTA
    BAR.SYNC barriers), which serializes the kernel to ~9% SM throughput. Reading via
    ct.gather (plain indexed LDG, no TMA machinery) with a smaller BLOCK and occupancy=2
    halves the barriers and loads and cuts L2 contention on the uncoalesced access — reaching
    ~48% SM throughput and beating the TMA-fallback on odd V.
    """
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for row_idx in range(pid, n_rows, num_programs):
        if HAS_LABEL:
            lbl = ct.load(LABEL, row_idx, shape=())
            if lbl == ignore_index:
                for ci in range(n_chunks):
                    col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
                    ct.scatter(GRADS, (row_idx, col_idx), ct.full((BLOCK_SIZE,), 0.0, GRADS.dtype), check_bounds=True)
                    if reduction == _REDUCTION_MODE_NONE:
                        ct.scatter(LOSS, (row_idx, col_idx), ct.full((BLOCK_SIZE,), 0.0, LOSS.dtype), check_bounds=True)
                if reduction != _REDUCTION_MODE_NONE:
                    ct.scatter(LOSS, row_idx, ct.astype(0.0, LOSS.dtype))
                continue

        loss_acc = ct.full((BLOCK_SIZE,), 0.0, ct.float32)

        for ci in range(n_chunks):
            col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
            p = ct.astype(ct.gather(P, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
            q = ct.astype(ct.gather(Q, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

            diff = p - q
            tv_loss = ct.maximum(diff, 0.0 - diff) * 0.5
            p_gt_q = ct.astype(p > q, ct.float32)
            grad = (2.0 * p_gt_q - 1.0) * 0.5 * scale

            ct.scatter(GRADS, (row_idx, col_idx), ct.astype(grad, GRADS.dtype), check_bounds=True)

            if reduction == _REDUCTION_MODE_NONE:
                ct.scatter(LOSS, (row_idx, col_idx), ct.astype(tv_loss, LOSS.dtype), check_bounds=True)
            else:
                loss_acc = loss_acc + tv_loss

        if reduction != _REDUCTION_MODE_NONE:
            row_sum = ct.sum(loss_acc, 0, keepdims=False)
            ct.scatter(LOSS, row_idx, ct.astype(row_sum * scale, LOSS.dtype))


def _get_block_size(V: int) -> int:
    # Use 8192 elements per block: fewer loop iterations with TMA.
    # At V=128256: ceil(128256/8192)=16 chunks per row.
    return min(8192, next_power_of_2(V))


def _tvd_forward_ct(p, q, shift_labels, reduction, ignore_index, has_label):
    BT, V = p.shape
    reduction_int = _str_to_reduction_mode[reduction]

    # Row-stride alignment drives the read path. When (V * itemsize) % 16 != 0 (e.g. odd V in
    # bf16), rows aren't 16-byte aligned so ct.load's TMA falls back to a serialized per-element
    # path (2x loads + many CTA barriers, ~9% SM throughput). Route those to the gather kernel
    # (indexed LDG, BLOCK=4096, occ=2), which reaches ~48% SM throughput.
    if (V * p.element_size()) % 16 != 0:
        BLOCK_SIZE = min(4096, next_power_of_2(V))
        kernel = _tv_distance_kernel_ct_gather
    else:
        BLOCK_SIZE = _get_block_size(V)
        kernel = _tv_distance_kernel_ct_aligned if V % BLOCK_SIZE == 0 else _tv_distance_kernel_ct

    label_tensor = shift_labels if has_label else torch.empty(1, device=p.device, dtype=torch.long)

    # Only batchmean/mean need n_non_ignore; skip the .item() sync for sum/none.
    # The scale is fused into the kernel (grads/loss written pre-scaled), so there is no
    # host-side grads / n_non_ignore division pass.
    if reduction_int == _REDUCTION_MODE_BATCHMEAN:
        n_non_ignore = (shift_labels != ignore_index).sum().item() if has_label else BT
        scale = 1.0 / n_non_ignore
    elif reduction_int == _REDUCTION_MODE_MEAN:
        n_non_ignore = (shift_labels != ignore_index).sum().item() if has_label else BT
        scale = 1.0 / (n_non_ignore * V)
    else:  # sum / none
        scale = 1.0

    # empty (not zeros): the kernel writes every position, zeroing ignore rows explicitly.
    grads = torch.empty_like(p)
    out_size = (BT, V) if reduction_int == _REDUCTION_MODE_NONE else (BT,)
    output_tensor = torch.empty(out_size, device=p.device, dtype=torch.float32)

    grid = (BT, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            p,
            q,
            output_tensor,
            grads,
            label_tensor,
            float(scale),
            int(BT),
            int(V),
            int(ignore_index),
            int(BLOCK_SIZE),
            int(has_label),
            int(reduction_int),
        ),
    )

    # Loss and grads are already scaled inside the kernel — no separate division needed.
    if reduction_int in (_REDUCTION_MODE_BATCHMEAN, _REDUCTION_MODE_MEAN):
        return output_tensor.sum(), grads
    elif reduction_int == _REDUCTION_MODE_SUM:
        return output_tensor.sum(dim=0), grads
    else:  # none
        return output_tensor.to(p.dtype), grads


class TVDLossCuTileFunction(torch.autograd.Function):
    """CuTile autograd wrapper for Total Variation Distance loss."""

    @staticmethod
    def forward(ctx, p, q, shift_labels, reduction, ignore_index):
        has_label = shift_labels is not None
        if has_label:
            assert shift_labels.shape == (p.shape[0],), f"shift_labels must have shape (BT,). Got: {shift_labels.shape}"
            shift_labels = shift_labels.contiguous()

        loss, grads = _tvd_forward_ct(p, q, shift_labels, reduction, ignore_index, has_label)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grads,) = ctx.saved_tensors
        if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            return grads, None, None, None, None
        return grads * grad_output, None, None, None, None


@register_impl("liger.tvd", backend="cutile")
def tvd(
    p: torch.Tensor,
    q: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    reduction: str = "batchmean",
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    return TVDLossCuTileFunction.apply(p.contiguous(), q.contiguous(), shift_labels, reduction, ignore_index)
