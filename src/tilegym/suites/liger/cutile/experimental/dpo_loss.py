# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused linear + DPO preference loss (CuTile backend).

Computes the Direct Preference Optimization loss over stacked chosen/rejected
sequence pairs without materializing the full (B*T, V) logit tensor, following
the chunked backward-in-forward structure of fused_linear_cross_entropy.py.

The sigmoid DPO loss is elementwise over pairs, so chunks that hold whole
pairs (chosen sequence i with rejected sequence i + n_pairs) contribute
independently to the loss and every gradient. This keeps the chunked path
single-pass: within a chunk the logits buffer stays live between the row-stats
kernel and the in-place d_logits kernel, and no recompute pass is needed.

Two execution paths, selected by B*T*V*sizeof vs _MAX_LOGIT_MEMORY_BYTES:

Single-pass (fits in _MAX_LOGIT_MEMORY_BYTES):
  One GEMM, row stats, pair math, d_logits in place. d_logits is saved and the
  two large gradient GEMMs run in backward, scaled by grad_output.

Chunked (larger, or chunk_size forced):
  Per pair-chunk: GEMM, row stats, pair math, d_logits in place, then
  grad_input / grad_weight_f32 / grad_bias accumulated immediately and the
  chunk logits discarded. Backward is an elementwise scale.

Per chunk the row-wise vocab work runs in one cuTile launch: the liger suite's
tuned cross-entropy kernel (cross_entropy._liger_cross_entropy_kernel) with
reduction="sum" and no weight / smoothing / softcap returns -log p(target) per
row and writes softmax - onehot into the logits buffer in place. The pair math
then gives every row its coefficient dLoss/d(log p), and the buffer is scaled
by it to become d_logits. torch/cuBLAS keeps the GEMMs; the O(n_pairs)
preference math stays device-side torch.

Scope: loss_type="sigmoid" (the original DPO loss), as agreed in
NVIDIA/TileGym#190; the other upstream variants are follow-ups.
"""

from typing import Optional

import cuda.tile as ct
import torch
import torch.nn.functional as F

from tilegym.backend import register_impl
from tilegym.logger import warn_once

from ..cross_entropy import _get_tuned_ce_kernel
from ..cross_entropy import _select_block_size

_EXPERIMENTAL_MESSAGE = (
    "liger.dpo_loss (cutile) is an experimental kernel contributed by external GitHub TileGym "
    "contributors. This kernel has not been fully validated by the core team."
)

# Single-pass threshold: if the full (B*T, V) logit tensor fits within this
# limit, run one chunk and defer the gradient GEMMs to backward.
_MAX_LOGIT_MEMORY_BYTES = 4 * 1024**3  # 4 GB

# Chunked path: largest power-of-2 pair count whose logits fit this budget.
_MAX_CHUNK_LOGIT_BYTES = 1 * 1024**3  # 1 GB


def _mm_f32(a, b):
    """a @ b with an fp32 result. For bf16/fp16 inputs this stays on tensor cores (fp32 accumulate and output)
    instead of upcasting both operands, which would fall back to an fp32 SIMT GEMM; the result matches the
    upcast product to fp32 precision because bf16/fp16 products are exact in fp32."""
    if a.dtype == torch.float32:
        return a @ b
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:  # torch without mm(out_dtype=...)
        return a.float() @ b.float()


def _addmm_f32_(acc, a, b):
    """acc += a @ b in place on an fp32 accumulator, with the accumulation done inside the GEMM (beta=1)."""
    if a.dtype == torch.float32:
        return acc.addmm_(a, b)
    try:
        return torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    except TypeError:  # torch without addmm(out_dtype=...)
        return acc.add_(a.float() @ b.float())


def _row_logp_(logits, target, ignore_index, write_grad, dummies):
    """One cuTile launch over the (R, V) logits rows.

    Returns logp: (R,) float32 log p(target), 0.0 for rows with ignore_index.
    With write_grad, logits is overwritten in place with softmax - onehot
    (zeros for ignored rows), i.e. -d log p / d logits.
    """
    n_rows, V = logits.shape
    loss = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)
    dummy_f32, dummy_i64, dummy_weight = dummies
    block_size = _select_block_size(V, logits.device)
    kernel = _get_tuned_ce_kernel(V, block_size, logits.dtype, logits.device)
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        kernel,
        (
            logits,
            target,
            dummy_weight,
            loss,
            dummy_f32,  # z_loss
            dummy_f32,  # token_accuracy
            dummy_i64,  # predicted_tokens
            int(V),
            1.0,  # inv_n_non_ignore (unused: REDUCTION_MEAN=0)
            1.0,  # sum_non_ignore_weight (unused)
            0.0,  # weight_sum (unused)
            int(ignore_index),
            0.0,  # label_smoothing
            0.0,  # lse_square_scale
            0.0,  # softcap
            int(block_size),
            int(write_grad),  # HAS_GRADIENTS
            0,  # REDUCTION_MEAN
            0,  # HAS_WEIGHT
            0,  # HAS_SOFTCAPPING
            0,  # RETURN_Z_LOSS
            0,  # RETURN_TOKEN_ACCURACY
            0,  # RETURN_PREDICTED_TOKENS
        ),
    )
    return -loss


def _preference_loss_terms(
    chosen_logps,
    rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    n_pairs_total,
    beta,
):
    """Sigmoid DPO loss over one chunk's pairs, normalized by the GLOBAL pair
    count so chunk contributions sum to the full-batch loss.
    Formula follows Liger-Kernel chunked_loss/dpo_loss.py (ead96b618e5c)."""
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    logits_diff = beta * (chosen_logratios - rejected_logratios)
    losses = -F.logsigmoid(logits_diff)

    loss = losses.sum() / n_pairs_total
    return loss, chosen_rewards, rejected_rewards


def _pair_math(
    chosen_logps,
    rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    n_pairs_total,
    beta,
):
    """Preference loss for one chunk plus dLoss/d(seq_logp) via a tiny
    autograd graph over the (chunk_pairs,) log-prob vectors."""
    with torch.enable_grad():
        c = chosen_logps.detach().requires_grad_(True)
        r = rejected_logps.detach().requires_grad_(True)
        loss, chosen_rewards, rejected_rewards = _preference_loss_terms(
            c,
            r,
            ref_chosen_logps,
            ref_rejected_logps,
            n_pairs_total,
            beta,
        )
        g_chosen, g_rejected = torch.autograd.grad(loss, (c, r))
    return loss.detach(), chosen_rewards.detach(), rejected_rewards.detach(), g_chosen, g_rejected


def _gather_pair_chunk(tensor, p0, p1, n_pairs):
    """Rows for pairs [p0, p1): chosen block then the matching rejected block."""
    return torch.cat([tensor[p0:p1], tensor[n_pairs + p0 : n_pairs + p1]], dim=0)


class DPOLossCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias,
        ref_input,
        ref_weight,
        ref_bias,
        ignore_index,
        beta,
        alpha,
        compute_nll_loss,
        use_ref_model,
        average_log_prob,
        chunk_size,
    ):
        B, T, H = _input.shape
        V = weight.shape[0]
        assert B % 2 == 0, "batch must stack chosen then rejected halves, so B must be even"
        n_pairs = B // 2
        device = _input.device
        elsize = _input.element_size()

        if chunk_size is not None:
            single_pass = False
            chunk_pairs = max(1, min(int(chunk_size), n_pairs))
        elif B * T * V * elsize <= _MAX_LOGIT_MEMORY_BYTES:
            single_pass = True
            chunk_pairs = n_pairs
        else:
            single_pass = False
            chunk_pairs = 1
            while 2 * chunk_pairs * 2 * T * V * elsize <= _MAX_CHUNK_LOGIT_BYTES and chunk_pairs < n_pairs:
                chunk_pairs *= 2

        mask = target != ignore_index
        tok_w = mask.to(torch.float32)
        if average_log_prob:
            tok_w = tok_w / mask.sum(dim=-1, keepdim=True).to(torch.float32)
        n_chosen_valid = mask[:n_pairs].sum()

        _input = _input.contiguous()
        target = target.contiguous()
        dummies = (
            torch.zeros(1, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.float32, device=device),
        )

        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        nll_acc = torch.zeros((), device=device, dtype=torch.float32)
        chosen_mean_acc = torch.zeros((), device=device, dtype=torch.float32)
        rejected_mean_acc = torch.zeros((), device=device, dtype=torch.float32)
        chosen_logps_parts = []
        rejected_logps_parts = []
        chosen_rewards_parts = []
        rejected_rewards_parts = []

        if single_pass:
            d_logits_saved = None
        else:
            grad_input = torch.empty_like(_input)
            grad_weight_f32 = torch.zeros(V, H, device=device, dtype=torch.float32)
            grad_bias_f32 = torch.zeros(V, device=device, dtype=torch.float32) if bias is not None else None

        for p0 in range(0, n_pairs, chunk_pairs):
            p1 = min(p0 + chunk_pairs, n_pairs)
            cp = p1 - p0
            rows = 2 * cp * T

            x_c = _gather_pair_chunk(_input, p0, p1, n_pairs)
            x_2d = x_c.reshape(rows, H)
            target_c = _gather_pair_chunk(target, p0, p1, n_pairs)
            target_flat = target_c.reshape(rows)
            tok_w_c = _gather_pair_chunk(tok_w, p0, p1, n_pairs)

            logits_c = x_2d @ weight.t()
            if bias is not None:
                logits_c = logits_c + bias

            # Row sums feed the logits-mean outputs; take them before the kernel overwrites logits.
            rowsum = torch.sum(logits_c, dim=-1, dtype=torch.float32)
            logp = _row_logp_(logits_c, target_flat, ignore_index, True, dummies)

            if use_ref_model:
                with torch.no_grad():
                    ref_x_2d = _gather_pair_chunk(ref_input, p0, p1, n_pairs).reshape(rows, H)
                    ref_logits_c = ref_x_2d @ ref_weight.t()
                    if ref_bias is not None:
                        ref_logits_c = ref_logits_c + ref_bias
                    ref_logp = _row_logp_(ref_logits_c.contiguous(), target_flat, ignore_index, False, dummies)
                    del ref_logits_c
                ref_seq_logp = (ref_logp.view(2 * cp, T) * tok_w_c).sum(dim=-1)
                ref_chosen_logps = ref_seq_logp[:cp]
                ref_rejected_logps = ref_seq_logp[cp:]
            else:
                ref_chosen_logps = torch.zeros(cp, device=device, dtype=torch.float32)
                ref_rejected_logps = torch.zeros(cp, device=device, dtype=torch.float32)

            seq_logp = (logp.view(2 * cp, T) * tok_w_c).sum(dim=-1)
            chosen_logps_c = seq_logp[:cp]
            rejected_logps_c = seq_logp[cp:]

            pref_loss_c, chosen_rewards_c, rejected_rewards_c, g_chosen, g_rejected = _pair_math(
                chosen_logps_c,
                rejected_logps_c,
                ref_chosen_logps,
                ref_rejected_logps,
                n_pairs,
                beta,
            )
            loss_acc += pref_loss_c
            chosen_logps_parts.append(chosen_logps_c)
            rejected_logps_parts.append(rejected_logps_c)
            chosen_rewards_parts.append(chosen_rewards_c)
            rejected_rewards_parts.append(rejected_rewards_c)

            rowsum_2d = rowsum.view(2 * cp, T)
            chosen_mean_acc += rowsum_2d[:cp].sum() / (n_pairs * T * V)
            rejected_mean_acc += rowsum_2d[cp:].sum() / (n_pairs * T * V)

            g_seq = torch.cat([g_chosen, g_rejected], dim=0)
            coeff = -(g_seq.unsqueeze(-1) * tok_w_c)
            if compute_nll_loss:
                logp_2d = logp.view(2 * cp, T)
                nll_acc += -logp_2d[:cp].sum() / n_chosen_valid
                coeff[:cp] += (alpha / n_chosen_valid) * mask[p0:p1].to(torch.float32)

            # logits_c holds softmax - onehot; scale each row by its coefficient to get d_logits.
            logits_c.mul_(coeff.reshape(rows, 1).to(logits_c.dtype))

            if single_pass:
                d_logits_saved = logits_c
            else:
                grad_x_c = (logits_c @ weight).view(2 * cp, T, H)
                grad_input[p0:p1] = grad_x_c[:cp]
                grad_input[n_pairs + p0 : n_pairs + p1] = grad_x_c[cp:]
                _addmm_f32_(grad_weight_f32, logits_c.t(), x_2d)
                if bias is not None:
                    grad_bias_f32 += torch.sum(logits_c, dim=0, dtype=torch.float32)
                del logits_c

        loss = loss_acc + alpha * nll_acc

        chosen_logps = torch.cat(chosen_logps_parts, dim=0)
        rejected_logps = torch.cat(rejected_logps_parts, dim=0)
        chosen_rewards = torch.cat(chosen_rewards_parts, dim=0)
        rejected_rewards = torch.cat(rejected_rewards_parts, dim=0)

        ctx.single_pass = single_pass
        ctx.has_bias = bias is not None
        if single_pass:
            ctx.save_for_backward(d_logits_saved, _input, weight)
            ctx.shape = (B, T, H, n_pairs)
        else:
            ctx.save_for_backward(
                grad_input,
                grad_weight_f32,
                grad_bias_f32 if bias is not None else torch.empty(0),
            )
        ctx.weight_dtype = weight.dtype
        ctx.input_dtype = _input.dtype

        outputs = (
            loss,
            chosen_logps,
            rejected_logps,
            chosen_mean_acc,
            rejected_mean_acc,
            nll_acc,
            chosen_rewards,
            rejected_rewards,
        )
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    def backward(ctx, grad_loss, *aux_grads):
        if ctx.single_pass:
            d_logits, _input, weight = ctx.saved_tensors
            B, T, H, n_pairs = ctx.shape
            # The single-pass chunk spans all pairs, so row order equals input order.
            x_2d = _input.reshape(B * T, H)
            grad_x = (d_logits @ weight).view(B, T, H) * grad_loss
            grad_input = grad_x.to(ctx.input_dtype)
            grad_weight = _mm_f32(d_logits.t(), x_2d) * grad_loss
            grad_weight = grad_weight.to(ctx.weight_dtype)
            grad_bias = (
                (torch.sum(d_logits, dim=0, dtype=torch.float32) * grad_loss).to(ctx.weight_dtype)
                if ctx.has_bias
                else None
            )
        else:
            grad_input_saved, grad_weight_f32, grad_bias_f32 = ctx.saved_tensors
            grad_input = (grad_input_saved.float() * grad_loss).to(ctx.input_dtype)
            grad_weight = (grad_weight_f32 * grad_loss).to(ctx.weight_dtype)
            grad_bias = (grad_bias_f32 * grad_loss).to(ctx.weight_dtype) if ctx.has_bias else None

        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ref_input
            None,  # ref_weight
            None,  # ref_bias
            None,  # ignore_index
            None,  # beta
            None,  # alpha
            None,  # compute_nll_loss
            None,  # use_ref_model
            None,  # average_log_prob
            None,  # chunk_size
        )


@register_impl("liger.dpo_loss", backend="cutile")
def dpo_loss_cutile(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ref_input: Optional[torch.Tensor] = None,
    ref_weight: Optional[torch.Tensor] = None,
    ref_bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    beta: float = 0.1,
    alpha: float = 1.0,
    compute_nll_loss: bool = False,
    use_ref_model: bool = True,
    average_log_prob: bool = False,
    loss_type: str = "sigmoid",
    chunk_size: Optional[int] = None,
):
    if loss_type != "sigmoid":
        raise ValueError(f"dpo_loss (cutile) supports loss_type='sigmoid' only, got {loss_type!r}")
    if use_ref_model and (ref_input is None or ref_weight is None):
        raise ValueError("use_ref_model=True requires ref_input and ref_weight")
    warn_once(_EXPERIMENTAL_MESSAGE, "EXPERIMENTAL")
    return DPOLossCuTileFunction.apply(
        input,
        weight,
        target,
        bias,
        ref_input,
        ref_weight,
        ref_bias,
        ignore_index,
        beta,
        alpha,
        compute_nll_loss,
        use_ref_model,
        average_log_prob,
        chunk_size,
    )
