# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused linear + Jensen-Shannon Divergence kernel (CuTile backend).

Forward (chunked):
  Processes row-chunks to bound peak memory. Per-chunk: two matmuls + one
  JSD kernel + element-wise softmax-bwd. grad_input/grad_weight accumulated
  across chunks in forward.

Backward:
  Returns pre-computed grad_input and grad_weight scaled by grad_output.
  No matmuls in backward.
"""

import functools
from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .jsd import JSD_BLOCK_SIZE
from .jsd import jsd_kernel_ct
from .utils import next_power_of_2

amp_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
amp_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


def _fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels,
    beta,
    ignore_index,
    has_label,
    temperature,
    compute_grad_input,
    compute_grad_weight,
):
    device = student_input.device
    dtype = student_input.dtype

    num_rows, hidden_size = student_input.shape
    vocab_size = student_weight.shape[0]
    # Cap at JSD_BLOCK_SIZE to avoid register spill in jsd_kernel_ct.
    BLOCK_SIZE = min(JSD_BLOCK_SIZE, next_power_of_2(vocab_size))

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = num_rows

    if n_non_ignore == 0:
        empty_input_grad = (
            torch.zeros(num_rows, hidden_size, dtype=dtype, device=device) if compute_grad_input else None
        )
        empty_weight_grad = (
            torch.zeros(vocab_size, hidden_size, dtype=dtype, device=device) if compute_grad_weight else None
        )
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            empty_input_grad,
            empty_weight_grad,
        )

    inv_n_non_ignore = 1.0 / n_non_ignore
    label_tensor = shift_labels if has_label else torch.empty(1, device=device, dtype=torch.int64)

    inc_factor = (vocab_size + hidden_size - 1) // hidden_size
    chunk_size = next_power_of_2((num_rows + inc_factor - 1) // inc_factor)
    num_chunks = (num_rows + chunk_size - 1) // chunk_size

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    grad_input = torch.zeros(num_rows, hidden_size, dtype=dtype, device=device) if compute_grad_input else None
    grad_weight = torch.zeros_like(student_weight) if compute_grad_weight else None

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, num_rows)

        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32) / temperature
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32) / temperature
        chunk_n_rows = student_logits_chunk.shape[0]

        log_prob_s_chunk = torch.log_softmax(student_logits_chunk, dim=-1).contiguous()
        log_prob_t_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1).contiguous()
        # Use softmax(logits) directly instead of exp(log_softmax(logits))
        # before log_prob_s_chunk is overwritten in-place.
        softmax_s_chunk = torch.softmax(student_logits_chunk, dim=-1)

        loss_chunk = torch.zeros(chunk_n_rows, vocab_size, dtype=torch.float32, device=device)

        label_chunk = label_tensor[start_idx:end_idx] if has_label else label_tensor

        # Pass log_prob_s_chunk as both X and dX in-place. The kernel reads X
        # before writing dX within each row, so this saves one float32 allocation.
        ct.launch(
            torch.cuda.current_stream(),
            (chunk_n_rows, 1, 1),
            jsd_kernel_ct,
            (
                log_prob_s_chunk,
                log_prob_t_chunk,
                loss_chunk,
                log_prob_s_chunk,  # dX in-place
                label_chunk,
                float(beta),
                float(inv_n_non_ignore),
                int(ignore_index),
                int(vocab_size),
                int(BLOCK_SIZE),
                int(has_label),
            ),
        )
        # log_prob_s_chunk now holds dX (written by kernel); softmax_s_chunk holds exp(log_prob_s).
        dX_chunk = log_prob_s_chunk

        grad_chunk = (dX_chunk - softmax_s_chunk * dX_chunk.sum(dim=-1, keepdim=True).expand_as(dX_chunk)) / temperature
        del student_logits_chunk, teacher_logits_chunk, log_prob_s_chunk, log_prob_t_chunk, softmax_s_chunk, dX_chunk

        total_loss = total_loss + loss_chunk.sum()

        grad_chunk_dtype = grad_chunk.to(dtype)
        if compute_grad_input:
            grad_input[start_idx:end_idx] = grad_chunk_dtype @ student_weight
        if compute_grad_weight:
            grad_weight.add_(grad_chunk_dtype.t() @ student_input_chunk)

    return total_loss, grad_input, grad_weight


class FusedLinearJSDCuTileFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor],
        beta: float,
        ignore_index: int,
        temperature: float,
    ):
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (student_input.shape[0],), (
                f"shift_labels must have shape (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        input_requires_grad = student_input.requires_grad
        weight_requires_grad = student_weight.requires_grad

        loss, grad_input, grad_weight = _fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            beta,
            ignore_index,
            has_label,
            temperature,
            compute_grad_input=input_requires_grad,
            compute_grad_weight=weight_requires_grad,
        )

        ctx.save_for_backward(
            grad_input if input_requires_grad else torch.empty(0),
            grad_weight if weight_requires_grad else torch.empty(0),
        )
        ctx.input_requires_grad = input_requires_grad
        ctx.weight_requires_grad = weight_requires_grad
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input_saved, grad_weight_saved = ctx.saved_tensors

        if not ctx.input_requires_grad and not ctx.weight_requires_grad:
            return None, None, None, None, None, None, None, None

        if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            grad_input = grad_input_saved if ctx.input_requires_grad else None
            grad_weight = grad_weight_saved if ctx.weight_requires_grad else None
        else:
            # Multiply on device — avoids .item() host-device sync.
            grad_input = (grad_input_saved * grad_output) if ctx.input_requires_grad else None
            grad_weight = (grad_weight_saved * grad_output) if ctx.weight_requires_grad else None

        return grad_input, grad_weight, None, None, None, None, None, None


@register_impl("liger.fused_linear_jsd", backend="cutile")
def fused_linear_jsd(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    return FusedLinearJSDCuTileFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        beta,
        ignore_index,
        temperature,
    )
