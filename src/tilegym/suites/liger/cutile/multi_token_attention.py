# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Multi-Token Attention kernel (CuTile backend).

Causal masking (CuTile) + softmax/sparsemax (CuTile) + conv2d (PyTorch) + causal zero-mask (CuTile).
sparse=True uses CuTile sparsemax (_sparsemax_forward_ct / _sparsemax_backward_ct); requires fp32.

Masking kernels use a row-parallel grid: (L, N, 1).
Each kernel instance handles one row of the L×L attention matrix.
Column iteration uses a fold trick with BLOCK_SIZE chunks.

TMA NOTE
========
Uses ct.load/ct.store (TMA) for BLOCK_SIZE-aligned column chunks.
OOB reads beyond column L are zero-padded; OOB writes are silently dropped.
This replaces the previous gather/scatter (check_bounds=True) path and
eliminates per-element bounds-check overhead.
"""

import cuda.tile as ct
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from tilegym.backend import register_impl

from .sparsemax import _sparsemax_backward_ct
from .sparsemax import _sparsemax_forward_ct
from .utils import next_power_of_2

_MASK_INF_VAL = -1e9  # large negative; -inf breaks multiply-accumulate pattern ((-inf)*0 = NaN)


def _select_block_size(L: int) -> int:
    bs = next_power_of_2(L)
    # Cap at 128 and use fold trick for larger L
    return min(bs, 128)


@ct.kernel
def _mask_inf_fwd_kernel(
    scores_2d,  # (N*L, L) input scores
    output_2d,  # (N*L, L) output
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """
    Apply causal -inf mask: output[b, r, c] = -1e9 if c > r else scores[b, r, c].

    Grid: (L, N, 1). Each instance handles one row for one batch.
    Columns iterated in BLOCK-sized chunks.
    Uses TMA load/store when possible (aligned BLOCK), with check_bounds=False
    for in-bounds accesses to avoid bounds-check overhead.
    """
    actual_row = ct.bid(0)  # row index within a single batch (0..L-1)
    batch_id = ct.bid(1)

    row_idx = batch_id * L + actual_row  # index in (N*L, L)
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start

        # Use TMA load (row_idx is scalar, col chunk ci is tile index)
        src_tile = ct.load(scores_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        # Future positions: col > row → replace with -1e9
        # OOB elements (col >= L): also future, will be written back as -1e9 but
        # TMA store silently drops OOB writes, so no issue.
        is_future_f = ct.astype(col_idx > actual_row, ct.float32)
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = (
            ct.astype(src_tile, ct.float32) * is_past_f + ct.full((BLOCK,), _MASK_INF_VAL, ct.float32) * is_future_f
        )
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _mask_zero_fwd_kernel(
    scores_2d,  # (N*L, L) input
    output_2d,  # (N*L, L) output
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """
    Apply causal zero mask: output[b, r, c] = 0 if c > r else scores[b, r, c].

    Grid: (L, N, 1). Uses TMA load/store.
    """
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)

    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start

        src_tile = ct.load(scores_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = ct.astype(src_tile, ct.float32) * is_past_f
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _mask_bwd_kernel(
    grad_2d,  # (N*L, L) upstream gradient
    output_2d,  # (N*L, L) output gradient
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """
    Backward mask: zero gradient for future positions (col > row).

    Used for both _mask_inf_backward and _mask_zero_backward.
    Grid: (L, N, 1). Uses TMA load/store.
    """
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)

    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start

        grad_tile = ct.load(grad_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        # Zero out future positions
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = ct.astype(grad_tile, ct.float32) * is_past_f
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _fused_softmax_zeromask_bwd_kernel(
    probs_2d,  # (N*L, L) saved softmax output (float32)
    grad_probs_2d,  # (N*L, L) upstream gradient from conv backward (float32)
    output_2d,  # (N*L, L) output: grad w.r.t. scores_inf, zero-masked
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """Fused: softmax backward + causal zero-mask in one row-wise pass.

    Loads in storage dtype, promotes to fp32 for dot-product accumulation
    (matches PyTorch's internal fp32 accumulation for bf16 inputs), then
    stores back in storage dtype.  Replaces:
      - 4-op manual bwd (mul->sum->sub->mul) in PyTorch
      - separate _mask_backward_ct kernel call

    Grid: (L, N, 1). Two-pass fold:
      Pass 1: dot = sum(p * dp) — accumulated over column chunks
      Pass 2: dx = p * (dp - dot); zero col > row (causal zero mask)
    """
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)
    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    # Pass 1: accumulate dot = sum(p * dp) over column chunks
    dot_tile = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + ci * BLOCK
        p_tile = ct.astype(
            ct.gather(probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dp_tile = ct.astype(
            ct.gather(grad_probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dot_tile = dot_tile + p_tile * dp_tile
    dot = ct.sum(dot_tile, 0, keepdims=False)

    # Pass 2: dx = p * (dp - dot); zero out future positions (causal zero mask)
    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + ci * BLOCK
        p_tile = ct.astype(
            ct.gather(probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dp_tile = ct.astype(
            ct.gather(grad_probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dx_tile = p_tile * (dp_tile - dot)
        # Causal zero mask: zero positions where col > row
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        dx_masked = dx_tile * is_past_f
        ct.scatter(output_2d, (row_idx, col_idx), ct.astype(dx_masked, output_2d.dtype), check_bounds=True)


def _mask_inf_forward_ct(scores: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.reshape(N * L, L).contiguous()
    out = torch.empty_like(scores_f)

    BLOCK = _select_block_size(L)
    grid = (L, N, 1)
    ct.launch(torch.cuda.current_stream(), grid, _mask_inf_fwd_kernel, (scores_f, out, int(L), int(BLOCK)))
    return out.reshape(*batch, L, L)


def _mask_zero_forward_ct(scores: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.reshape(N * L, L).contiguous()
    out = torch.empty_like(scores_f)

    BLOCK = _select_block_size(L)
    grid = (L, N, 1)
    ct.launch(torch.cuda.current_stream(), grid, _mask_zero_fwd_kernel, (scores_f, out, int(L), int(BLOCK)))
    return out.reshape(*batch, L, L)


def _mask_backward_ct(grad: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.reshape(N * L, L).contiguous()
    out = torch.empty_like(grad_f)

    BLOCK = _select_block_size(L)
    grid = (L, N, 1)
    ct.launch(torch.cuda.current_stream(), grid, _mask_bwd_kernel, (grad_f, out, int(L), int(BLOCK)))
    return out.reshape(*batch, L, L)


def _fused_softmax_zeromask_bwd_ct_launch(probs: torch.Tensor, grad_probs: torch.Tensor) -> torch.Tensor:
    """Launch fused softmax backward + causal zero-mask CuTile kernel."""
    *batch, L, _ = probs.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    p_f = probs.reshape(N * L, L).contiguous()
    dp_f = grad_probs.reshape(N * L, L).contiguous()
    out = torch.empty_like(p_f)

    BLOCK = _select_block_size(L)
    grid = (L, N, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _fused_softmax_zeromask_bwd_kernel,
        (p_f, dp_f, out, int(L), int(BLOCK)),
    )
    return out.reshape(*batch, L, L)


def _conv1x1_backward(grad_out: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor):
    """mm-based 1x1 conv backward -- bypasses cuDNN dispatch overhead.

    For a kernel_size=1 conv:
      grad_input[b,cin,h,w]  = sum_cout(W[cout,cin] * dout[b,cout,h,w])
      grad_weight[cout,cin]  = sum_{b,h,w}(dout[b,cout,h,w] * inp[b,cin,h,w])

    Both reduce to matrix multiplications on the (B*H*W, C) reshape, letting
    cuBLAS SGEMM handle the compute.  On B200 for CH=1, L=128 this is ~1.69x
    faster than F.conv_transpose2d + torch.nn.grad.conv2d_weight because it
    bypasses cuDNN's per-call dispatch overhead for this tiny shape.
    """
    B, C_out, H, W = grad_out.shape
    C_in = inp.shape[1]
    N = B * H * W
    go_2d = grad_out.permute(0, 2, 3, 1).reshape(N, C_out)
    in_2d = inp.permute(0, 2, 3, 1).reshape(N, C_in)
    w_2d = weight.view(C_out, C_in)
    grad_input = torch.mm(go_2d, w_2d).reshape(B, H, W, C_in).permute(0, 3, 1, 2).contiguous()
    grad_weight = torch.mm(go_2d.t(), in_2d).view(weight.shape)
    return grad_input, grad_weight


class MultiTokenAttentionCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, sparse=False):
        scores = scores.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        ctx.sparse = sparse

        if sparse:
            # Sparsemax requires float32; raise early for non-fp32 input (mirrors Liger reference).
            if scores.dtype != torch.float32:
                raise RuntimeError(
                    f"CuTile sparse multi-token attention only supports fp32 input scores. Got dtype={scores.dtype}."
                )
            compute_dtype = torch.float32
            weight_c, bias_c = weight, bias

            scores_inf = _mask_inf_forward_ct(scores)
            probs, out_flat_sparse = _sparsemax_forward_ct(scores_inf, dim=-1)

            out_conv = F.conv2d(
                probs,
                weight_c,
                bias_c,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            out = _mask_zero_forward_ct(out_conv)

            # Save tensors needed for backward (out_flat_sparse is the sparsemax support tensor)
            ctx.save_for_backward(scores_inf, probs, out_flat_sparse, weight_c, bias_c)
        else:
            # For fp16 inputs: promote to float32 so that cuDNN conv and softmax
            # run via TF32, matching the reference's float32 path and recovering
            # the fp16 backward regression on small shapes (L≤128, single-channel
            # 1×1 conv). Saves probs/weight in float32 to avoid re-casting in
            # backward. bf16 is unaffected (already 1.17x baseline preserved).
            compute_dtype = scores.dtype
            if compute_dtype == torch.float16:
                scores = scores.float()
                weight_c = weight.float()
                bias_c = bias.float() if bias is not None else None
            else:
                weight_c, bias_c = weight, bias

            scores_inf = _mask_inf_forward_ct(scores)
            probs = torch.softmax(scores_inf, dim=-1)

            out_conv = F.conv2d(
                probs,
                weight_c,
                bias_c,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            out = _mask_zero_forward_ct(out_conv)

            # Save float32 tensors for backward (avoids repeated casts)
            ctx.save_for_backward(scores_inf, probs, weight_c, bias_c)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.compute_dtype = compute_dtype

        return out.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        sparse = ctx.sparse

        if sparse:
            scores_inf, probs, out_flat_sparse, weight, bias = ctx.saved_tensors
        else:
            scores_inf, probs, weight, bias = ctx.saved_tensors

        # .contiguous() is required: PyTorch's sum().backward() passes a broadcast
        # tensor (strides=0), which would cause CuTile gather to read invalid offsets.
        grad_out_c = grad_out.to(probs.dtype).contiguous()

        grad_conv = _mask_backward_ct(grad_out_c)

        # conv backward: mm-based 1x1 shortcut or cuDNN fallback
        if stride == (1, 1) and padding == (0, 0) and dilation == (1, 1) and groups == 1:
            grad_probs, grad_weight = _conv1x1_backward(grad_conv, probs, weight)
        else:
            grad_probs = F.conv_transpose2d(
                grad_conv, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            grad_weight = torch.nn.grad.conv2d_weight(
                input=probs,
                weight_size=weight.shape,
                grad_output=grad_conv,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

        grad_bias = None
        if bias is not None:
            grad_bias = grad_conv.sum(dim=(0, 2, 3))

        if sparse:
            # Sparsemax backward + causal inf-mask backward (mirrors Liger reference).
            grad_scores_inf = _sparsemax_backward_ct(grad_probs.contiguous(), out_flat_sparse, dim=-1)
            grad_scores = _mask_backward_ct(grad_scores_inf.to(probs.dtype).contiguous())
        else:
            # Fused softmax backward + causal zero-mask (single CuTile kernel).
            # Replaces: 4-op manual bwd (mul->sum->sub->mul) + _mask_backward_ct call.
            grad_scores = _fused_softmax_zeromask_bwd_ct_launch(probs, grad_probs)

        # Cast outputs back to original dtype
        orig = ctx.compute_dtype
        return (
            grad_scores.to(orig),
            grad_weight.to(orig),
            grad_bias.to(orig) if grad_bias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


@register_impl("liger.multi_token_attention", backend="cutile")
def multi_token_attention(
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False,
    **kwargs,
) -> torch.Tensor:
    return MultiTokenAttentionCuTileFunction.apply(scores, weight, bias, stride, padding, dilation, groups, sparse)
