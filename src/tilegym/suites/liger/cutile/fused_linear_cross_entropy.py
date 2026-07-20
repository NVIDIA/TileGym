# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused linear + cross-entropy kernel (CuTile backend).

Fuses the final linear projection with the cross-entropy loss.

Two execution paths, selected by BT * V * sizeof vs _MAX_LOGIT_MEMORY_BYTES (4 GB):

Single-pass (BT * V * sizeof ≤ 4 GB):
  1. One matmul: logits = input @ weight.T
  2. One CE kernel launch for all BT rows; writes d_logits in-place.
  Save d_logits for backward. Backward does two large matmuls.

Chunked + backward-in-forward (BT * V * sizeof > 4 GB):
  chunk_size = largest pow-2 rows s.t. one chunk logit ≤ _MAX_CHUNK_LOGIT_BYTES (1 GB).
  For V=128256 bf16 this gives chunk_size=4096.
  Each chunk:
    1. logits_chunk = input_chunk @ weight.T            (efficient GEMM)
    2. CE kernel writes d_logit_chunk in-place.
    3. grad_input[start:end] = d_logit_chunk @ weight   (accumulated, large K)
    4. grad_weight_f32      += d_logit_chunk.float().T @ input_chunk.float()
    5. discard logits_chunk — peak logit memory is O(chunk_size × V), not O(BT × V).
  Save grad_input (BT×H) + grad_weight (V×H) instead of d_logits.
  Backward: just element-wise scale by grad_output — nearly free.
"""

from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .cross_entropy import _get_tuned_ce_kernel
from .utils import next_power_of_2

MAX_FUSED_SIZE = 65536 // 2

# Single-pass threshold: if the full (BT, V) logit tensor fits within this limit,
# use one large GEMM + one CE kernel (fastest path).
_MAX_LOGIT_MEMORY_BYTES = 4 * 1024**3  # 4 GB

# Chunked path: target size for one chunk's logit tensor.
# chunk_size = largest pow-2 s.t. chunk_size * V * sizeof ≤ this limit.
# For V=128256, bf16: gives chunk_size = 4096 (≈ 1.05 GB per chunk, acceptable).
_MAX_CHUNK_LOGIT_BYTES = 1024**3  # 1 GB


def _chunk_size_for(V: int, element_size: int) -> int:
    """Largest power-of-2 chunk_size s.t. chunk_size * V * element_size ≤ _MAX_CHUNK_LOGIT_BYTES."""
    max_rows = _MAX_CHUNK_LOGIT_BYTES // (V * element_size)
    if max_rows < 1:
        return 1
    # floor to power of 2
    return 1 << (max_rows.bit_length() - 1)


def _launch_ce(
    logits,
    target,
    ce_weight,
    loss_slice,
    z_loss_slice,
    token_acc_slice,
    pred_slice,
    dummies,
    V,
    BLOCK_SIZE,
    inv_n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    label_smoothing,
    lse_square_scale,
    softcap,
    write_d_logits,
    reduction_mean,
    has_weight,
    has_softcap,
    return_z_loss,
    return_token_accuracy,
    return_predicted_tokens,
):
    """Launch the CE kernel for one chunk, substituting dummy tensors for disabled outputs."""
    dummy_f32, dummy_i64, dummy_weight = dummies
    n_rows = logits.shape[0]
    ce_kernel = _get_tuned_ce_kernel(V, BLOCK_SIZE, logits.dtype, logits.device)
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        ce_kernel,
        (
            logits,
            target,
            ce_weight if has_weight else dummy_weight,
            loss_slice,
            z_loss_slice if return_z_loss else dummy_f32,
            token_acc_slice if return_token_accuracy else dummy_f32,
            pred_slice if return_predicted_tokens else dummy_i64,
            int(V),
            float(inv_n_non_ignore),
            float(sum_non_ignore_weight),
            float(weight_sum),
            int(ignore_index),
            float(label_smoothing),
            float(lse_square_scale),
            float(softcap if softcap is not None else 0.0),
            int(BLOCK_SIZE),
            int(write_d_logits),
            int(reduction_mean),
            int(has_weight),
            int(has_softcap),
            int(return_z_loss),
            int(return_token_accuracy),
            int(return_predicted_tokens),
        ),
    )


def _token_scaling_factors(logits, target, ignore_index, softcap):
    """Predicted-probability scaling factors (before the CE kernel overwrites logits)."""
    logits_for_softmax = logits.detach().clone()
    if softcap is not None:
        logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
    probs = torch.softmax(logits_for_softmax, dim=-1)
    valid_target_mask = target != ignore_index
    valid_targets = target[valid_target_mask]
    pred_probs = torch.zeros_like(target, dtype=probs.dtype, device=logits.device)
    if valid_targets.numel() > 0:
        valid_probs = probs[valid_target_mask]
        pred_probs[valid_target_mask] = torch.gather(valid_probs, -1, valid_targets.unsqueeze(-1)).squeeze(-1)
    return pred_probs.detach()


def _fused_linear_ce_forward_ct(
    _input,
    weight,
    target,
    ce_weight,
    bias,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    accum_dtype,
    use_token_scaling,
    return_token_accuracy,
    return_predicted_tokens,
):
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(V))

    input_requires_grad = _input.requires_grad
    weight_requires_grad = weight.requires_grad
    need_grads = input_requires_grad or weight_requires_grad
    reduction_mean = int(reduction == "mean")

    # Normalization counts.
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    inv_n_non_ignore = 1.0 / max(total_n_non_ignore, 1)

    has_weight = ce_weight is not None
    sum_non_ignore_weight = float(total_n_non_ignore)
    weight_sum = 0.0
    if has_weight:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be of floating point dtype. Got: {ce_weight.dtype}"
        )
        sum_non_ignore_weight = float(
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        weight_sum = float(ce_weight.sum().item())
        ce_weight = ce_weight.contiguous().float()
    has_softcap = softcap is not None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    # Dummy tensors for disabled outputs (CuTile requires valid tensor args).
    dummies = (
        torch.zeros(1, dtype=torch.float32, device=device),  # dummy_f32
        torch.zeros(1, dtype=torch.int64, device=device),  # dummy_i64
        torch.zeros(1, dtype=torch.float32, device=device),  # dummy_weight
    )

    def _ce(logits, target_, loss_slice, z_slice, ta_slice, pred_slice, write_d_logits):
        _launch_ce(
            logits,
            target_,
            ce_weight,
            loss_slice,
            z_slice,
            ta_slice,
            pred_slice,
            dummies,
            V,
            BLOCK_SIZE,
            inv_n_non_ignore,
            sum_non_ignore_weight,
            weight_sum,
            ignore_index,
            label_smoothing,
            lse_square_scale,
            softcap,
            write_d_logits,
            reduction_mean,
            has_weight,
            has_softcap,
            return_z_loss,
            return_token_accuracy,
            return_predicted_tokens,
        )

    grad_bias_saved = None

    if BT * V * _input.element_size() <= _MAX_LOGIT_MEMORY_BYTES:
        # Single-pass: one large GEMM + one CE kernel — fastest path.
        # d_logits is saved for backward; backward does two large matmuls.
        logits = _input @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.contiguous()

        scaling_factors = _token_scaling_factors(logits, target, ignore_index, softcap) if use_token_scaling else None

        _ce(
            logits,
            target.contiguous(),
            loss_1d,
            z_loss_1d,
            token_accuracy_1d,
            predicted_tokens_1d,
            input_requires_grad,
        )

        if use_token_scaling:
            loss_1d = loss_1d * scaling_factors
            if return_z_loss:
                z_loss_1d = z_loss_1d * scaling_factors

        # After the CE kernel, logits holds d_logits = d(loss)/d(logits).
        d_logits = logits if input_requires_grad else None
        if d_logits is not None and use_token_scaling:
            d_logits = d_logits * scaling_factors.unsqueeze(-1)
        grad_input_saved = None
        grad_weight_saved = None
        chunked_bif = False
    else:
        # Chunked + backward-in-forward: per-chunk GEMM → CE → grad_input slice
        # + grad_weight_f32 accumulate. Peak logit memory is O(chunk_size × V),
        # not O(BT × V). Grads are saved directly; backward only scales them.
        chunk_size = min(BT, _chunk_size_for(V, _input.element_size()))
        num_chunks = (BT + chunk_size - 1) // chunk_size

        acc = weight.dtype if accum_dtype is None else accum_dtype
        grad_input_saved = torch.zeros(BT, H, dtype=_input.dtype, device=device) if input_requires_grad else None
        grad_weight_f32 = torch.zeros(V, H, dtype=torch.float32, device=device) if weight_requires_grad else None
        grad_bias_saved = torch.zeros(V, dtype=acc, device=device) if (bias is not None and need_grads) else None

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, BT)
            _input_chunk = _input[start_idx:end_idx]

            logits_chunk = _input_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias
            logits_chunk = logits_chunk.contiguous()

            scaling_factors = (
                _token_scaling_factors(logits_chunk, target[start_idx:end_idx], ignore_index, softcap)
                if use_token_scaling
                else None
            )

            _ce(
                logits_chunk,
                target[start_idx:end_idx].contiguous(),
                loss_1d[start_idx:end_idx],
                z_loss_1d[start_idx:end_idx] if return_z_loss else None,
                token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None,
                predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None,
                need_grads,
            )

            if use_token_scaling:
                loss_1d[start_idx:end_idx] = loss_1d[start_idx:end_idx] * scaling_factors
                if return_z_loss:
                    z_loss_1d[start_idx:end_idx] = z_loss_1d[start_idx:end_idx] * scaling_factors

            # logits_chunk now holds d_logit_chunk. Compute grads immediately,
            # then let logits_chunk go out of scope (freed at end of iteration).
            grad_logits_chunk = logits_chunk
            if use_token_scaling:
                grad_logits_chunk = grad_logits_chunk * scaling_factors.unsqueeze(-1)
            if grad_input_saved is not None:
                grad_input_saved[start_idx:end_idx] = grad_logits_chunk.to(_input.dtype) @ weight
            if grad_weight_f32 is not None:
                grad_weight_f32 += grad_logits_chunk.float().t() @ _input_chunk.float()
            if grad_bias_saved is not None:
                grad_bias_saved += grad_logits_chunk.sum(dim=0).to(grad_bias_saved.dtype)

        d_logits = None
        grad_weight_saved = grad_weight_f32.to(weight.dtype) if grad_weight_f32 is not None else None
        if grad_bias_saved is not None:
            grad_bias_saved = grad_bias_saved.to(bias.dtype)
        chunked_bif = True

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = loss_1d.sum()
        z_loss = z_loss_1d.sum() if return_z_loss else None
        token_accuracy = token_accuracy_1d.sum() / max(total_n_non_ignore, 1) if return_token_accuracy else None
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return (
        loss,
        z_loss,
        token_accuracy,
        predicted_tokens,
        d_logits,
        grad_input_saved,
        grad_weight_saved,
        grad_bias_saved,
        chunked_bif,
    )


class FusedLinearCrossEntropyCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ce_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        (
            loss,
            z_loss,
            token_accuracy,
            predicted_tokens,
            d_logits,
            grad_input_saved,
            grad_weight_saved,
            grad_bias_saved,
            chunked_bif,
        ) = _fused_linear_ce_forward_ct(
            _input,
            weight,
            target,
            ce_weight,
            bias,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            accum_dtype,
            use_token_scaling,
            return_token_accuracy,
            return_predicted_tokens,
        )

        input_requires_grad = _input.requires_grad
        weight_requires_grad = weight.requires_grad
        ctx.input_requires_grad = input_requires_grad
        ctx.weight_requires_grad = weight_requires_grad
        ctx.has_bias = bias is not None
        ctx.chunked_bif = chunked_bif

        _empty = torch.empty(0)
        if chunked_bif:
            # Backward-in-forward path: grads are pre-computed, just save them.
            ctx.save_for_backward(
                grad_input_saved if grad_input_saved is not None else _empty,
                grad_weight_saved if grad_weight_saved is not None else _empty,
                grad_bias_saved if grad_bias_saved is not None else _empty,
                _empty,  # d_logits slot
                _empty,  # _input slot
                _empty,  # weight slot
            )
        else:
            # Single-pass path: save d_logits for backward matmuls.
            need_grads = input_requires_grad or weight_requires_grad
            ctx.save_for_backward(
                _empty,  # grad_input slot
                _empty,  # grad_weight slot
                _empty,  # grad_bias slot
                d_logits if d_logits is not None else _empty,
                _input if need_grads else _empty,
                weight if need_grads else _empty,
            )

        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        grad_input_saved, grad_weight_saved, grad_bias_saved, d_logits_saved, _input, weight = ctx.saved_tensors

        scalar_1 = torch.tensor(1.0, device=grad_output.device)
        scale_by_grad = not torch.equal(grad_output, scalar_1)

        if ctx.chunked_bif:
            # ── BACKWARD-IN-FORWARD PATH ─────────────────────────────────────
            # Grads were computed during forward; just apply grad_output scaling.
            grad_input = grad_input_saved if grad_input_saved.numel() > 0 else None
            grad_weight = grad_weight_saved if grad_weight_saved.numel() > 0 else None
            grad_bias = grad_bias_saved if grad_bias_saved.numel() > 0 else None
            if scale_by_grad:
                if grad_input is not None:
                    grad_input = grad_input * grad_output
                if grad_weight is not None:
                    grad_weight = grad_weight * grad_output
                if grad_bias is not None:
                    grad_bias = grad_bias * grad_output
        else:
            # ── SINGLE-PASS PATH ─────────────────────────────────────────────
            # d_logits was saved; compute grad_input and grad_weight now.
            if not ctx.input_requires_grad:
                return (None,) * 15

            d_logits = d_logits_saved
            if scale_by_grad:
                d_logits = d_logits * grad_output

            # grad_input: (BT, H) — one matmul
            grad_input = d_logits @ weight if ctx.input_requires_grad else None

            # grad_weight: (V, H) — one matmul. Use bf16 operands directly; cuBLAS
            # GEMM bf16 × bf16 → fp32 accumulate internally, so the explicit
            # `.float()` cast on both operands is wasted memory traffic (the
            # BT×V cast alone is up to 16 GB at BT=32768, V=128256). Numerical
            # parity with fp32 inputs (with fp32 accumulate) within atol/rtol=1e-2.
            grad_weight = torch.mm(d_logits.t(), _input) if ctx.weight_requires_grad else None

            # grad_bias: (V,) — cheap row-sum of d_logits
            grad_bias = d_logits.sum(dim=0) if ctx.has_bias else None

        # forward inputs: _input, weight, target, bias, ce_weight, ignore_index,
        # lse_square_scale, label_smoothing, reduction, softcap, return_z_loss,
        # accum_dtype, use_token_scaling, return_token_accuracy, return_predicted_tokens
        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ce_weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # accum_dtype
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )


@register_impl("liger.fused_linear_cross_entropy", backend="cutile")
def fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs,
):
    loss, z_loss, token_accuracy, predicted_tokens = FusedLinearCrossEntropyCuTileFunction.apply(
        input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    # Scalar loss by default; only expose extra outputs when explicitly requested
    if not return_z_loss and not return_token_accuracy and not return_predicted_tokens:
        return loss
    extras = []
    if return_z_loss:
        extras.append(z_loss)
    if return_token_accuracy:
        extras.append(token_accuracy)
    if return_predicted_tokens:
        extras.append(predicted_tokens)
    return (loss, *extras)
