# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Self-contained port of `fla.ops.gated_delta_rule.gate`.

import functools

import cuda.tile as ct
import torch

from tilegym.suites.fla.cutile._host_utils import _segmented_arange
from tilegym.suites.fla.cutile._host_utils import input_guard


@functools.lru_cache(maxsize=None)
def _dummy_i32(device, n: int, m: int = 0):
    """Cached zero dummy tensor for non-varlen launch args."""
    if m:
        return torch.zeros(n, m, device=device, dtype=torch.int32)
    return torch.zeros(n, device=device, dtype=torch.int32)


# Sequence metadata


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


# CuTile Kernels


@ct.function
def _gdn_gate_softplus(b_g):
    """softplus(x) = where(x < 20, log(1 + exp(x)), x)."""
    return ct.where(b_g < 20.0, ct.log(1.0 + ct.exp(b_g)), b_g)


@ct.kernel(num_worker_warps=8)
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,  # [B, T, H], or [1, total_T, H] in the varlen layout
    A_log,  # [H]
    dt_bias,  # [H] (dummy [1] when not used)
    o,  # [B, T, H] output
    cu_seqlens,  # [N+1] (dummy [1] when not varlen)
    chunk_indices,  # [NT, 2] (dummy [1, 2] when not varlen)
    scale: ct.Constant[float],
    T,
    H: ct.Constant[int],
    BT: ct.Constant[int],
    BH: ct.Constant[int],
    REVERSE: ct.Constant[int],
    HAS_BIAS: ct.Constant[int],
    HAS_SCALE: ct.Constant[int],
    IS_VARLEN: ct.Constant[int],
):
    """Elementwise gate over a length-BT time slice, then inclusive cumsum
    along time (axis 0, per chunk), optionally scaled.

    grid = (NT, B * cdiv(H, BH)); each block handles a 2D ``[BT, BH]`` tile.
    """
    i_t = ct.bid(0)
    i_bhb = ct.bid(1)
    nhb = ct.cdiv(H, BH)
    i_b = i_bhb // nhb
    i_hb = i_bhb % nhb
    h_offs = i_hb * BH + ct.arange(BH, dtype=ct.int32)
    in_h = h_offs < H

    if IS_VARLEN:
        # chunk_indices is [NT, 2] -> (i_n, intra_chunk_idx); varlen layout is [1, total_T, H].
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
        row = 0
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        glob_t = bos + t_local
    else:
        Tloc = T
        row = i_b
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        glob_t = t_local

    # Mask out positions past the (sub-)sequence length (OOB padding).
    in_t = t_local < Tloc
    mask = in_t[:, None] & in_h[None, :]

    b_g = ct.gather(g, (row, glob_t[:, None], h_offs[None, :]), mask=mask, check_bounds=True, padding_value=0.0)
    b_g = ct.astype(b_g, ct.float32)
    if HAS_BIAS:
        bias = ct.gather(dt_bias, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0)
        b_g = b_g + ct.astype(bias, ct.float32)[None, :]

    b_A = ct.astype(ct.gather(A_log, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0), ct.float32)
    b_gate = -ct.exp(b_A)[None, :] * _gdn_gate_softplus(b_g)

    # Zero the padded tail so it does not pollute the cumsum / sum.
    b_gate = ct.where(in_t[:, None], b_gate, 0.0)

    b_o = ct.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = ct.sum(b_gate, axis=0, keepdims=True)
        b_o = -b_o + b_z + b_gate
    if HAS_SCALE:
        b_o = b_o * scale

    ct.scatter(o, (row, glob_t[:, None], h_offs[None, :]), ct.astype(b_o, o.dtype), mask=mask, check_bounds=True)


@ct.kernel(num_worker_warps=8)
def gdn_gate_fwd_kernel(
    g,  # [T, H]
    A_log,  # [H]
    dt_bias,  # [H] (dummy [1] when not used)
    yg,  # [T, H] output
    T,
    H: ct.Constant[int],
    BT: ct.Constant[int],
    BH: ct.Constant[int],
    HAS_BIAS: ct.Constant[int],
):
    """Elementwise gate, no cumsum. grid = (NT, cdiv(H, BH)); each block handles a ``[BT, BH]`` tile."""
    i_t = ct.bid(0)
    i_hb = ct.bid(1)

    t_offs = i_t * BT + ct.arange(BT, dtype=ct.int32)
    h_offs = i_hb * BH + ct.arange(BH, dtype=ct.int32)
    in_t = t_offs < T
    in_h = h_offs < H
    mask = in_t[:, None] & in_h[None, :]

    b_g = ct.gather(g, (t_offs[:, None], h_offs[None, :]), mask=mask, check_bounds=True, padding_value=0.0)
    b_g = ct.astype(b_g, ct.float32)
    if HAS_BIAS:
        bias = ct.gather(dt_bias, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0)
        b_g = b_g + ct.astype(bias, ct.float32)[None, :]

    b_A = ct.astype(ct.gather(A_log, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0), ct.float32)
    b_yg = -ct.exp(b_A)[None, :] * _gdn_gate_softplus(b_g)

    ct.scatter(yg, (t_offs[:, None], h_offs[None, :]), ct.astype(b_yg, yg.dtype), mask=mask, check_bounds=True)


@ct.kernel(num_worker_warps=8)
def gdn_gate_bwd_kernel(
    g,  # [T, H]
    A_log,  # [H]
    dt_bias,  # [H] (dummy [1] when not used)
    dyg,  # [T, H]
    dg,  # [T, H] output
    dA,  # [NT, H] output (one row per time-tile)
    T,
    H: ct.Constant[int],
    BT: ct.Constant[int],
    BH: ct.Constant[int],
    HAS_BIAS: ct.Constant[int],
):
    """gate = -exp(A_log) * softplus(g + bias)
    d(gate)/d(g)     = -exp(A_log) * sigmoid(g + bias)   (softplus' = sigmoid)
    d(gate)/d(A_log) = gate

    grid = (NT, cdiv(H, BH)); each block handles a 2D ``[BT, BH]`` tile.
    """
    i_t = ct.bid(0)
    i_hb = ct.bid(1)

    t_offs = i_t * BT + ct.arange(BT, dtype=ct.int32)
    h_offs = i_hb * BH + ct.arange(BH, dtype=ct.int32)
    in_t = t_offs < T
    in_h = h_offs < H
    mask = in_t[:, None] & in_h[None, :]

    b_A = ct.astype(ct.gather(A_log, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0), ct.float32)
    b_g = ct.astype(
        ct.gather(g, (t_offs[:, None], h_offs[None, :]), mask=mask, check_bounds=True, padding_value=0.0), ct.float32
    )
    b_dyg = ct.astype(
        ct.gather(dyg, (t_offs[:, None], h_offs[None, :]), mask=mask, check_bounds=True, padding_value=0.0), ct.float32
    )
    if HAS_BIAS:
        bias = ct.gather(dt_bias, (h_offs,), mask=in_h, check_bounds=True, padding_value=0.0)
        b_g = b_g + ct.astype(bias, ct.float32)[None, :]

    b_neg_expA = -ct.exp(b_A)[None, :]
    b_yg = b_neg_expA * _gdn_gate_softplus(b_g)
    # sigmoid(x) = 1 / (1 + exp(-x))  (no ct.sigmoid)
    b_sig = 1.0 / (1.0 + ct.exp(-b_g))
    b_dg = b_neg_expA * (b_dyg * b_sig)

    # zero padded tail so the reduction for dA is clean
    b_prod = ct.where(in_t[:, None], b_dyg * b_yg, 0.0)
    b_dA = ct.sum(b_prod, axis=0)

    ct.scatter(dg, (t_offs[:, None], h_offs[None, :]), ct.astype(b_dg, dg.dtype), mask=mask, check_bounds=True)
    ct.scatter(dA, (i_t, h_offs), b_dA, mask=in_h, check_bounds=True)


# Wrapper Functions


@input_guard
def gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = ct.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    out_dtype = output_dtype or g.dtype
    o = torch.empty_like(g, dtype=out_dtype)

    is_varlen = cu_seqlens is not None
    has_bias = dt_bias is not None
    has_scale = scale is not None
    # Head-wide tile: process all H heads in one [BT, BH] tile.
    BH = H

    # cuTile launch args cannot be None: substitute dummy tensors + flags.
    dt_bias_arg = dt_bias if has_bias else _dummy_i32(g.device, 1).view(torch.float32)
    cu_seqlens_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_i32(g.device, 1)
    chunk_indices_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_i32(g.device, 1, 2)

    grid = (NT, B * ct.cdiv(H, BH))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gdn_gate_chunk_cumsum_scalar_kernel,
        (
            g,
            A_log,
            dt_bias_arg,
            o,
            cu_seqlens_arg,
            chunk_indices_arg,
            float(scale) if has_scale else 1.0,
            T,
            H,
            BT,
            BH,
            0,  # REVERSE (always False from this entry point)
            int(has_bias),
            int(has_scale),
            int(is_varlen),
        ),
    )
    return o


@input_guard
def gdn_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H = g.shape[-1]
    T = g.numel() // H
    BT = 64
    # Head-wide tile: process all H heads in one [BT, BH] tile.
    BH = H

    g2d = g.contiguous().view(T, H)
    yg = torch.empty_like(g2d, dtype=output_dtype)

    has_bias = dt_bias is not None
    dt_bias_arg = dt_bias if has_bias else torch.empty(1, device=g.device, dtype=torch.float32)

    NT = ct.cdiv(T, BT)
    grid = (NT, ct.cdiv(H, BH))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gdn_gate_fwd_kernel,
        (
            g2d,
            A_log,
            dt_bias_arg,
            yg,
            T,
            H,
            BT,
            BH,
            int(has_bias),
        ),
    )
    return yg.view_as(g)


@input_guard
def gdn_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H = g.shape[-1]
    T = g.numel() // H
    BT = 32
    NT = ct.cdiv(T, BT)
    # Head-wide tile: process all H heads in one [BT, BH] tile.
    BH = H

    g2d = g.contiguous().view(T, H)
    dyg2d = dyg.contiguous().view(T, H)
    dg = torch.empty((T, H), device=g.device, dtype=torch.float32)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)

    has_bias = dt_bias is not None
    dt_bias_arg = dt_bias if has_bias else torch.empty(1, device=g.device, dtype=torch.float32)

    grid = (NT, ct.cdiv(H, BH))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gdn_gate_bwd_kernel,
        (
            g2d,
            A_log,
            dt_bias_arg,
            dyg2d,
            dg,
            dA,
            T,
            H,
            BT,
            BH,
            int(has_bias),
        ),
    )

    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H).sum(0).to(dt_bias) if dt_bias is not None else None

    return dg, dA, dbias


class GDNGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = gdn_gate_fwd(g=g, A_log=A_log, dt_bias=dt_bias, output_dtype=output_dtype)
        ctx.save_for_backward(g, A_log, dt_bias)
        return yg

    @staticmethod
    @input_guard
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = gdn_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg)
        return dg, dA, dbias, None


@torch.compiler.disable
@input_guard
def fused_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Fused GDN gate computation with autograd support.

    Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``
    """
    return GDNGateFunction.apply(g, A_log, dt_bias, output_dtype)
