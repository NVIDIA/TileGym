# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools
import math
from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]
JSD_BLOCK_SIZE = 4096


def _ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _jsd_kernel(
    x,  # (BT, V) log Q (student)
    y,  # (BT, V) log P (teacher)
    loss,  # (BT, V) float32 loss accumulator (pre-initialized to 0)
    dx,  # (BT, V) gradient output (input dtype)
    label,  # (BT,) label tensor, or dummy 1-elem tensor when HAS_LABEL=0
    beta: ConstFloat,
    inv_n_non_ignore,
    ignore_index: ConstInt,
    n_cols,
    BLOCK_SIZE: ConstInt,
    HAS_LABEL: ConstInt,
):
    """
    cuTile kernel for generalized Jensen-Shannon Divergence.

    JSD(β)(P || Q): loss_i = β·P_i·(log P_i - log M_i) + (1-β)·Q_i·(log Q_i - log M_i)
    where M = β·P + (1-β)·Q

    Inputs x = log Q (student), y = log P (teacher) are log-probabilities.

    Special cases:
      beta == 0.0  →  forward KL:  KL(P || Q)
      beta == 1.0  →  reverse KL:  KL(Q || P)
      else         →  generalized JSD
    """
    row_idx = ct.bid(0)

    if HAS_LABEL:
        lbl = ct.load(label, row_idx, shape=())
        if lbl == ignore_index:
            # Zero out dx for this row; loss stays at 0 (pre-initialized on host).
            num_chunks_early = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            for ci in range(num_chunks_early):
                col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
                ct.scatter(dx, (row_idx, col_indices), ct.full((BLOCK_SIZE,), 0.0, dtype=dx.dtype), check_bounds=True)
            return

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk_idx * BLOCK_SIZE

        # Load x (log Q) and y (log P); out-of-bounds positions are padded with -inf.
        x_tile = ct.gather(x, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
        y_tile = ct.gather(y, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)

        x_f32 = ct.astype(x_tile, ct.float32)
        y_f32 = ct.astype(y_tile, ct.float32)

        # Pre-define outputs before compile-time branches (cuTile compiler requirement).
        loss_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dx_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        if beta == 0.0:  # forward KL: KL(P || Q)
            y_max = ct.max(y_f32, 0, keepdims=True)
            y_prob = ct.exp(y_f32 - y_max) * ct.exp(y_max)
            loss_tile = y_prob * (y_f32 - x_f32)
            dx_tile = -y_prob

        elif beta == 1.0:  # reverse KL: KL(Q || P)
            x_max = ct.max(x_f32, 0, keepdims=True)
            x_prob = ct.exp(x_f32 - x_max) * ct.exp(x_max)
            loss_tile = x_prob * (x_f32 - y_f32)
            # dx = x_prob * (x - y) + x_prob; uses unscaled loss_tile.
            dx_tile = loss_tile + x_prob

        else:  # generalized JSD: M = beta*P + (1-beta)*Q
            x_max = ct.max(x_f32, 0, keepdims=True)
            y_max = ct.max(y_f32, 0, keepdims=True)
            max_val = ct.maximum(x_max, y_max)
            exp_max = ct.exp(max_val)
            q_prob = ct.exp(x_f32 - max_val) * exp_max  # = exp(x)
            p_prob = ct.exp(y_f32 - max_val) * exp_max  # = exp(y)
            beta_p = beta * p_prob
            one_minus_beta_q = (1.0 - beta) * q_prob
            m_prob = beta_p + one_minus_beta_q
            log_m = ct.log(m_prob)
            loss_tile = beta_p * y_f32 + one_minus_beta_q * x_f32 - m_prob * log_m
            dx_tile = one_minus_beta_q * (x_f32 - log_m)

        loss_tile = loss_tile * inv_n_non_ignore
        dx_tile = dx_tile * inv_n_non_ignore

        ct.scatter(loss, (row_idx, col_indices), loss_tile, check_bounds=True)
        ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_tile, dx.dtype), check_bounds=True)


def _jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    num_rows, vocab_size = _input.shape
    BLOCK_SIZE = min(JSD_BLOCK_SIZE, next_power_of_2(vocab_size))

    # loss is pre-initialized to 0: ignored rows contribute 0, partial-chunk pads stay 0.
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dx = torch.empty_like(_input)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = num_rows

    if n_non_ignore == 0:
        return torch.tensor(0.0, device=_input.device, dtype=_input.dtype), torch.zeros_like(_input)

    inv_n_non_ignore = 1.0 / n_non_ignore

    # ct.launch does not accept None; pass a dummy 1-element tensor when no labels.
    label_tensor = shift_labels if has_label else torch.empty(1, device=_input.device, dtype=torch.int64)
    ct.launch(
        torch.cuda.current_stream(),
        (num_rows, 1, 1),
        _jsd_kernel,
        (
            _input,
            target,
            loss,
            dx,
            label_tensor,
            float(beta),
            float(inv_n_non_ignore),
            int(ignore_index),
            int(vocab_size),
            int(BLOCK_SIZE),
            int(has_label),
        ),
    )

    return torch.sum(loss).to(_input.dtype), dx


def _jsd_backward(dx, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dx
    return grad_output * dx


class JSDCuTileFunction(torch.autograd.Function):
    r"""
    cuTile autograd wrapper for the generalized Jensen-Shannon Divergence loss.

    JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M),  M = β*P + (1-β)*Q

    Inputs are expected as log-probabilities (log-space).
    """

    @staticmethod
    @_ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.Tensor],
        beta: float,
        ignore_index: int,
    ) -> torch.Tensor:
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],), (
                f"shift_labels must have shape (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, dx = _jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)
        ctx.save_for_backward(dx)
        return loss

    @staticmethod
    @_ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        (dx,) = ctx.saved_tensors
        dx = _jsd_backward(dx, grad_output)
        return (dx, None, None, None, None)


@register_impl("liger.jsd", backend="cutile")
def jsd(
    input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    return JSDCuTileFunction.apply(input, target, shift_labels, beta, ignore_index)
