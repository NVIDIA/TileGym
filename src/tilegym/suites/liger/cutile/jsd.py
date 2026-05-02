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

_JSD_BLOCK_SIZE = 4096


def ensure_contiguous(fn):
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
    X,  # (BT, V) log Q (student)
    Y,  # (BT, V) log P (teacher)
    loss,  # (BT, V) float32 loss accumulator (pre-initialized to 0)
    dX,  # (BT, V) gradient output (input dtype)
    label,  # (BT,) label tensor, or dummy 1-elem tensor when HAS_LABEL=0
    beta: ct.Constant[float],
    inv_n_non_ignore: ct.Constant[float],
    ignore_index: ct.Constant[int],
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    HAS_LABEL: ct.Constant[int],
):
    """
    cuTile kernel for generalized Jensen-Shannon Divergence.

    JSD(β)(P || Q): loss_i = β·P_i·(log P_i - log M_i) + (1-β)·Q_i·(log Q_i - log M_i)
    where M = β·P + (1-β)·Q

    Inputs X = log Q (student), Y = log P (teacher) are log-probabilities.

    Special cases:
      beta == 0.0  →  forward KL:  KL(P || Q)
      beta == 1.0  →  reverse KL:  KL(Q || P)
      else         →  generalized JSD
    """
    row_idx = ct.bid(0)

    if HAS_LABEL:
        lbl = ct.load(label, row_idx, shape=())
        if lbl == ignore_index:
            # Zero out dX for this row; loss stays at 0 (pre-initialized on host)
            num_chunks_early = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            for ci in range(num_chunks_early):
                col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
                ct.scatter(dX, (row_idx, col_indices), ct.full((BLOCK_SIZE,), 0.0, dtype=dX.dtype), check_bounds=True)
            return

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk_idx * BLOCK_SIZE

        # Load X (log Q) and Y (log P); out-of-bounds positions padded with -inf
        X_tile = ct.gather(X, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
        Y_tile = ct.gather(Y, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)

        X_f32 = ct.astype(X_tile, ct.float32)
        Y_f32 = ct.astype(Y_tile, ct.float32)

        # Pre-define outputs before compile-time branches (cuTile compiler requirement)
        loss_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dX_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        if beta == 0.0:  # forward KL: KL(P || Q)
            Y_max = ct.max(Y_f32, 0, keepdims=True)
            Y_prob = ct.exp(Y_f32 - Y_max) * ct.exp(Y_max)
            loss_tile = Y_prob * (Y_f32 - X_f32)
            dX_tile = -Y_prob

        elif beta == 1.0:  # reverse KL: KL(Q || P)
            X_max = ct.max(X_f32, 0, keepdims=True)
            X_prob = ct.exp(X_f32 - X_max) * ct.exp(X_max)
            loss_tile = X_prob * (X_f32 - Y_f32)
            # dX = X_prob*(X - Y) + X_prob; uses unscaled loss_tile
            dX_tile = loss_tile + X_prob

        else:  # generalized JSD: M = beta*P + (1-beta)*Q
            X_max = ct.max(X_f32, 0, keepdims=True)
            Y_max = ct.max(Y_f32, 0, keepdims=True)
            max_val = ct.maximum(X_max, Y_max)
            exp_max = ct.exp(max_val)
            Q = ct.exp(X_f32 - max_val) * exp_max  # = exp(X)
            P = ct.exp(Y_f32 - max_val) * exp_max  # = exp(Y)
            beta_P = beta * P
            one_minus_beta_Q = (1.0 - beta) * Q
            M = beta_P + one_minus_beta_Q
            log_M = ct.log(M)
            loss_tile = beta_P * Y_f32 + one_minus_beta_Q * X_f32 - M * log_M
            dX_tile = one_minus_beta_Q * (X_f32 - log_M)

        loss_tile = loss_tile * inv_n_non_ignore
        dX_tile = dX_tile * inv_n_non_ignore

        ct.scatter(loss, (row_idx, col_indices), loss_tile, check_bounds=True)
        ct.scatter(dX, (row_idx, col_indices), ct.astype(dX_tile, dX.dtype), check_bounds=True)


def _jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    BT, V = _input.shape
    BLOCK_SIZE = min(_JSD_BLOCK_SIZE, next_power_of_2(V))

    # loss pre-initialized to 0: ignored rows contribute 0, partial-chunk pads stay 0
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    if n_non_ignore == 0:
        return torch.tensor(0.0, device=_input.device, dtype=_input.dtype), torch.zeros_like(_input)

    inv_n_non_ignore = 1.0 / n_non_ignore

    # ct.launch does not accept None; pass a dummy 1-element tensor when no labels
    label_tensor = shift_labels if has_label else torch.empty(1, device=_input.device, dtype=torch.int64)

    ct.launch(
        torch.cuda.current_stream(),
        (BT, 1, 1),
        _jsd_kernel,
        (
            _input,
            target,
            loss,
            dX,
            label_tensor,
            float(beta),
            float(inv_n_non_ignore),
            int(ignore_index),
            int(V),
            int(BLOCK_SIZE),
            int(has_label),
        ),
    )

    return torch.sum(loss).to(_input.dtype), dX


def _jsd_backward(dX, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dX
    else:
        return grad_output * dX


class JSDFunction(torch.autograd.Function):
    r"""
    cuTile autograd wrapper for the generalized Jensen-Shannon Divergence loss.

    JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M),  M = β*P + (1-β)*Q

    Inputs are expected as log-probabilities (log-space).
    """

    @staticmethod
    @ensure_contiguous
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

        loss, dX = _jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)
        ctx.save_for_backward(dX)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        (dX,) = ctx.saved_tensors
        dX = _jsd_backward(dX, grad_output)
        return (dX, None, None, None, None)


@register_impl("liger.jsd", backend="cutile")
def jsd(
    input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    return JSDFunction.apply(input, target, shift_labels, beta, ignore_index)
