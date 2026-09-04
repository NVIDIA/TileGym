# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
RMS Normalization kernel (CuTile backend).

Y = X / RMS(X) * (W + offset), RMS = sqrt(mean(x^2) + eps).

Forward kernel: row-parallel (one block per row), single pass.
  col_idx = arange(BLOCK_SIZE); gather X (check_bounds=True pads OOB to 0),
  compute rstd, scatter Y.

Backward kernel: SM-count partitioned, single DRAM pass per row (all BLOCK_SIZE).
  - W loaded once per block; dW accumulated in registers, scattered once at end.
  - dW_partial shape: (sm_count, n_cols) instead of (n_rows, n_cols).

Casting modes:
  - "llama" (0): X cast to fp32 for RMS; X*rstd cast BACK to X.dtype before W multiply.
                 RSTD stored as fp32.
  - "gemma" (1): Both X and W cast to fp32; Y cast back to X.dtype.
                 RSTD stored as fp32.
  - "none" (-1): No casting. Everything in X.dtype. RSTD stored in X.dtype.
"""

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1

_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}


def _calculate_settings(n_cols):
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(f"Feature dimension {n_cols} exceeds maximum supported (65536)")
    return BLOCK_SIZE


# ---------------------------------------------------------------------------
# Forward kernel (row-parallel)
# ---------------------------------------------------------------------------


def _apply_modulation(y_f32, scale_val, shift_val, has_shift):
    out = y_f32 * (1.0 + ct.astype(scale_val, ct.float32))
    if has_shift:
        out = out + ct.astype(shift_val, ct.float32)
    return out


@ct.kernel
def _modulated_rms_norm_fwd_ct(
    Y,  # (n_rows, n_cols) output
    X,  # (n_rows, n_cols) input
    W,  # (n_cols,) affine weight (dummy 1-element tensor when elementwise_affine=False)
    Scale,  # (scale_rows, n_cols) modulation scale
    Shift,  # (scale_rows, n_cols) modulation shift
    RSTD,  # (n_rows,) cached rstd
    n_cols,
    eps,  # runtime float
    offset,  # runtime float
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
    elementwise_affine: ct.Constant[bool],
    has_shift: ct.Constant[bool],
    rows_per_modulation: ct.Constant[int],
    aligned: ct.Constant[bool],
):
    """
    RMS norm forward (unified, single pass).

    Row-parallel forward pass:
      col_idx = arange(BLOCK_SIZE)   # BLOCK_SIZE = next_power_of_2(n_cols)
      load X (check_bounds=True → OOB elements zero-padded, harmless for RMS sum)
      compute rstd, store RSTD
      scale X; optionally multiply by (W + offset)
      store Y

    elementwise_affine is a compile-time constant — the W gather/multiply is
    dead-code-eliminated when False, so the no-weight path has zero overhead.

    casting_mode:
      llama (0): X cast to fp32 for RMS; X*rstd cast BACK to X.dtype before W multiply.
      gemma (1): Both X and W cast to fp32; Y cast back to X.dtype.
      none (-1): Compute in X.dtype (no upcast). x*x accumulated in X.dtype;
        division by n_cols promotes to fp32. eps/offset rounded to X.dtype before
        arithmetic.
    """
    row_idx = ct.bid(0)
    mod_row_idx = row_idx // rows_per_modulation
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    # aligned == (BLOCK_SIZE == n_cols): every lane is in-bounds, so gather/scatter
    # need no OOB predication. Non-power-of-2 n_cols keeps check_bounds=True.
    cb = not aligned
    scale_val = ct.gather(Scale, (mod_row_idx, col_idx), check_bounds=cb, padding_value=0.0)
    if has_shift:
        shift_val = ct.gather(Shift, (mod_row_idx, col_idx), check_bounds=cb, padding_value=0.0)
    else:
        shift_val = scale_val

    if casting_mode == _CASTING_MODE_NONE:
        x_val = ct.gather(X, (row_idx, col_idx), check_bounds=cb, padding_value=0.0)
        if elementwise_affine:
            w_val = ct.gather(W, col_idx, check_bounds=cb, padding_value=0.0)
        mean_sq = ct.astype(ct.sum(x_val * x_val, 0, keepdims=False), ct.float32) / n_cols
        eps_rounded = ct.astype(ct.astype(eps, x_val.dtype), ct.float32)
        rstd = ct.rsqrt(mean_sq + eps_rounded)  # fp32
        ct.scatter(RSTD, row_idx, ct.astype(rstd, x_val.dtype), check_bounds=False)
        x_scaled = ct.astype(x_val, ct.float32) * rstd  # fp32 (upcast x for RMS computation)

        if elementwise_affine:
            offset_native = ct.astype(offset, x_val.dtype)  # round offset to X.dtype precision
            w_plus_offset_f32 = ct.astype(w_val + offset_native, ct.float32)
            y_f32 = x_scaled * w_plus_offset_f32
        else:
            y_f32 = x_scaled
        y_f32 = _apply_modulation(y_f32, scale_val, shift_val, has_shift)
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_f32, x_val.dtype), check_bounds=cb)

    elif casting_mode == _CASTING_MODE_LLAMA:
        x_val = ct.gather(X, (row_idx, col_idx), check_bounds=cb, padding_value=0.0)
        if elementwise_affine:
            w_val = ct.gather(W, col_idx, check_bounds=cb, padding_value=0.0)
        x_f32 = ct.astype(x_val, ct.float32)
        mean_sq = ct.sum(ct.mul(x_f32, x_f32, flush_to_zero=True), 0, keepdims=False) / n_cols
        rstd = ct.rsqrt(mean_sq + eps)
        ct.scatter(RSTD, row_idx, rstd, check_bounds=False)
        # Cast X*rstd back to X.dtype before W multiply (llama behaviour)
        x_scaled = ct.astype(x_f32 * rstd, X.dtype)

        if elementwise_affine:
            offset_native = ct.astype(offset, x_val.dtype)  # round offset to X.dtype precision
            w_plus_offset_f32 = ct.astype(w_val + offset_native, ct.float32)
            y_f32 = x_scaled * w_plus_offset_f32
        else:
            y_f32 = x_scaled
        y_f32 = _apply_modulation(y_f32, scale_val, shift_val, has_shift)
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_f32, x_val.dtype), check_bounds=cb)

    else:
        # gemma: both X and W to fp32, Y cast back to X.dtype
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=cb, padding_value=0.0), ct.float32)
        if elementwise_affine:
            w_f32 = ct.astype(ct.gather(W, col_idx, check_bounds=cb, padding_value=0.0), ct.float32)
        mean_sq = ct.sum(x_f32 * x_f32, 0, keepdims=False) / n_cols
        rstd = ct.rsqrt(mean_sq + eps)
        ct.scatter(RSTD, row_idx, rstd, check_bounds=False)
        x_scaled = x_f32 * rstd

        if elementwise_affine:
            y_f32 = x_scaled * (w_f32 + offset)
        else:
            y_f32 = x_scaled
        y_f32 = _apply_modulation(y_f32, scale_val, shift_val, has_shift)
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_f32, Y.dtype), check_bounds=cb)


# Forward is memory-bound and row-parallel (grid = n_rows), so occupancy hints (more
# resident blocks/SM to hide DRAM latency) are the tuning lever; the launch wrapper
# selects these variants by BLOCK_SIZE.
_modulated_rms_norm_fwd_ct_occ4 = _modulated_rms_norm_fwd_ct.replace_hints(occupancy=4)
_modulated_rms_norm_fwd_ct_occ3 = _modulated_rms_norm_fwd_ct.replace_hints(occupancy=3)


# ---------------------------------------------------------------------------
# Backward kernels — SM-count grid, single DRAM pass
# ---------------------------------------------------------------------------


@ct.kernel
def _modulated_rms_norm_bwd_large_ct(
    dY,  # (n_rows, n_cols) upstream gradient
    dX,  # (n_rows, n_cols) output gradient
    X,  # (n_rows, n_cols) saved input
    Scale,  # (scale_rows, n_cols) modulation scale
    RSTD,  # (n_rows,) cached rstd; OOB-safe via bounds-checked gather
    dScale,  # (scale_rows, n_cols) modulation scale gradient
    dShift,  # (scale_rows, n_cols) modulation shift gradient
    n_cols: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
    has_shift: ct.Constant[bool],
    rows_per_modulation: ct.Constant[int],
    single_mod: ct.Constant[bool],
):
    """
    Modulated RMS norm backward without affine weight. SM-count partitioned.

    dRms = dY * (1 + scale) is the gradient flowing back through the norm.
    dScale = dY * rms_output (here rms_output = X * rstd, no weight).
    dShift = dY.

    dScale/dShift selection mirrors the weighted kernel: plain store when
    rows_per_modulation == 1; per-block register accumulation + one atomic flush
    when single_mod (single shared modulation row); per-row atomic otherwise.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    inv_n_cols = 1.0 / n_cols
    if single_mod:
        dScale_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        if has_shift:
            dShift_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri
        mod_row_idx = row_idx // rows_per_modulation

        rstd = ct.astype(ct.gather(RSTD, (row_idx,), padding_value=0.0).item(), ct.float32)
        dy_f32 = ct.astype(ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        scale_f32 = ct.astype(
            ct.gather(Scale, (mod_row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )

        mod_f32 = 1.0 + scale_f32
        drms_f32 = dy_f32 * mod_f32
        rms_out_f32 = x_f32 * rstd

        sum_mX = ct.sum(drms_f32 * x_f32, 0, keepdims=False)
        coeff = rstd * rstd * rstd * inv_n_cols * sum_mX
        dx_f32 = rstd * drms_f32 - coeff * x_f32
        ct.scatter(dX, (row_idx, col_idx), ct.astype(dx_f32, dX.dtype), check_bounds=True)

        dscale_row = dy_f32 * rms_out_f32
        if rows_per_modulation == 1:
            ct.scatter(dScale, (mod_row_idx, col_idx), dscale_row, check_bounds=True)
            if has_shift:
                ct.scatter(dShift, (mod_row_idx, col_idx), dy_f32, check_bounds=True)
        elif single_mod:
            dScale_acc = ct.add(dScale_acc, dscale_row)
            if has_shift:
                dShift_acc = ct.add(dShift_acc, dy_f32)
        else:
            ct.atomic_add(dScale, (mod_row_idx, col_idx), dscale_row, check_bounds=True)
            if has_shift:
                ct.atomic_add(dShift, (mod_row_idx, col_idx), dy_f32, check_bounds=True)

    # Flush the per-block dScale/dShift once into the single shared modulation row.
    if single_mod:
        ct.atomic_add(dScale, (0, col_idx), dScale_acc, check_bounds=True)
        if has_shift:
            ct.atomic_add(dShift, (0, col_idx), dShift_acc, check_bounds=True)


@ct.kernel
def _modulated_rms_norm_bwd_w_large_ct(
    dY,  # (n_rows, n_cols) upstream gradient
    dX,  # (n_rows, n_cols) output gradient
    X,  # (n_rows, n_cols) saved input
    W,  # (n_cols,) affine weight
    Scale,
    RSTD,  # (n_rows,) cached rstd; OOB-safe via bounds-checked gather
    dW_partial,  # (sm_count, n_cols) per-block dW accumulation (host reduces)
    dScale,
    dShift,
    n_cols: ct.Constant[int],
    offset: ct.Constant[float],
    rows_per_program: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
    has_shift: ct.Constant[bool],
    rows_per_modulation: ct.Constant[int],
    single_mod: ct.Constant[bool],
):
    """
    modulated RMS norm backward with affine weight. SM-count partitioned, single DRAM pass.

    Grid: (sm_count,). Block b processes rows [b*rpp, (b+1)*rpp).
    W loaded once per block and reused across all rows.
    dW accumulated in registers throughout the row loop; scattered once at the end.
    dW_partial shape: (sm_count, n_cols) — vastly smaller than (n_rows, n_cols).
    OOB rows return 0 via check_bounds; RSTD zero-padded.

    dScale/dShift selection (unchanged math; only accumulation order differs):
      - rows_per_modulation == 1  → plain store per row (each token owns a modulation row).
      - single_mod (scale_rows == 1, all rows map to modulation row 0) → dScale/dShift
        accumulated in registers across the row loop and atomic_add'd ONCE per block.
        Mirrors the per-block dW register accumulation; collapses the per-row atomic
        contention on the single shared row.
      - otherwise → atomic_add per row into the row's modulation slot.

    casting_mode:
      llama (0): load W in original dtype once; per row: dY in orig dtype,
                 m = (dY*(W+offset)) cast to fp32; dW += dy_orig*(X*rstd cast to X.dtype).
      gemma (1): W loaded in fp32; per row: dY in fp32, m = dy_f32*(w_f32+offset);
                 dW += dy_f32 * x_f32 * rstd.
      none (-1): load W in original dtype once; per row: dY in orig dtype,
                 m = dy_orig*(w_orig+offset) without cast to fp32 (cast for sum);
                 dW += dy_orig * (x_orig * rstd) without extra fp32 cast.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Per-block dW accumulator in registers; scattered to dW_partial once at end
    dW_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    # Per-block dScale/dShift accumulators (single shared modulation row only).
    if single_mod:
        dScale_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        if has_shift:
            dShift_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    inv_n_cols = 1.0 / n_cols

    # Load W once; dtype depends on mode (gemma keeps fp32; llama/none keep original).
    if casting_mode == _CASTING_MODE_GEMMA:
        w_f32 = ct.astype(ct.gather(W, col_idx, check_bounds=True, padding_value=0.0), ct.float32)
    else:
        w_orig = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri
        mod_row_idx = row_idx // rows_per_modulation
        scale_f32 = ct.astype(
            ct.gather(Scale, (mod_row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )
        mod_f32 = 1.0 + scale_f32

        # Bounds-checked scalar read on RSTD (avoids host-side cat-padding).
        rstd = ct.astype(ct.gather(RSTD, (row_idx,), padding_value=0.0).item(), ct.float32)
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if casting_mode == _CASTING_MODE_GEMMA:
            dy_f32 = ct.astype(ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
            drms_f32 = dy_f32 * mod_f32
            rms_out_f32 = x_f32 * rstd * (w_f32 + offset)
            m_f32 = drms_f32 * (w_f32 + offset)
            dW_term_f32 = drms_f32 * x_f32 * rstd

        else:
            dy_orig = ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
            dy_f32 = ct.astype(dy_orig, ct.float32)
            drms_f32 = dy_f32 * mod_f32
            w_off_f32 = ct.astype(w_orig + offset, ct.float32)
            rms_out_f32 = x_f32 * rstd * w_off_f32
            m_f32 = drms_f32 * w_off_f32
            dW_term_f32 = drms_f32 * x_f32 * rstd

        sum_mX = ct.sum(m_f32 * x_f32, 0, keepdims=False)
        coeff = rstd * rstd * rstd * inv_n_cols * sum_mX
        dx_f32 = rstd * m_f32 - coeff * x_f32
        ct.scatter(dX, (row_idx, col_idx), ct.astype(dx_f32, dX.dtype), check_bounds=True)

        dscale_row = dy_f32 * rms_out_f32
        if rows_per_modulation == 1:
            ct.scatter(dScale, (mod_row_idx, col_idx), dscale_row, check_bounds=True)
            if has_shift:
                ct.scatter(dShift, (mod_row_idx, col_idx), dy_f32, check_bounds=True)
        elif single_mod:
            # OOB rows contribute 0 (dy=0 via check_bounds), so accumulating them is harmless.
            dScale_acc = ct.add(dScale_acc, dscale_row)
            if has_shift:
                dShift_acc = ct.add(dShift_acc, dy_f32)
        else:
            ct.atomic_add(dScale, (mod_row_idx, col_idx), dscale_row, check_bounds=True)
            if has_shift:
                ct.atomic_add(dShift, (mod_row_idx, col_idx), dy_f32, check_bounds=True)

        # OOB rows contribute 0 (dy=0, x=0 via check_bounds)
        dW_acc = ct.add(dW_acc, dW_term_f32)

    # Write this block's partial dW once (block_id < sm_count, always in-bounds)
    ct.scatter(dW_partial, (block_id, col_idx), dW_acc, check_bounds=True)

    # Flush the per-block dScale/dShift once into the single shared modulation row.
    if single_mod:
        ct.atomic_add(dScale, (0, col_idx), dScale_acc, check_bounds=True)
        if has_shift:
            ct.atomic_add(dShift, (0, col_idx), dShift_acc, check_bounds=True)


_modulated_rms_norm_bwd_large_ct_nww8 = _modulated_rms_norm_bwd_large_ct.replace_hints(num_worker_warps=8)
_modulated_rms_norm_bwd_w_large_ct_nww8 = _modulated_rms_norm_bwd_w_large_ct.replace_hints(num_worker_warps=8)


# ---------------------------------------------------------------------------
# Python launch wrappers
# ---------------------------------------------------------------------------


def _check_modulation_shape(X, scale, shift):
    dim = X.shape[-1]
    assert scale.numel() % dim == 0, "Scale element count must be a multiple of the hidden size."
    n_rows = X.numel() // dim
    scale_rows = scale.numel() // dim
    assert scale_rows > 0, "Scale must have at least one row."
    assert n_rows % scale_rows == 0, "Scale rows must divide hidden state rows for broadcasting."

    if shift is not None:
        assert shift.numel() == scale_rows * dim, "Shift must use the same broadcast rows as scale."

    return scale_rows, n_rows // scale_rows


def _modulated_rms_norm_forward_ct(X, W, scale, shift, eps, offset, casting_mode_int):
    shape = X.shape
    dim = shape[-1]
    scale_rows, rows_per_modulation = _check_modulation_shape(X, scale, shift)

    X2d = X.view(-1, dim).contiguous()
    n_rows, n_cols = X2d.shape
    BLOCK_SIZE = _calculate_settings(n_cols)

    has_shift = shift is not None
    Scale_tensor = scale.view(scale_rows, dim).contiguous()
    Shift_tensor = shift.view(scale_rows, dim).contiguous() if has_shift else Scale_tensor

    Y = torch.empty_like(X2d)
    # RSTD dtype: fp32 for llama/gemma, X.dtype for none
    rstd_dtype = torch.float32 if casting_mode_int in (_CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
    elementwise_affine = W is not None

    grid = (n_rows, 1, 1)
    # When no weight, pass a 1-element dummy tensor; elementwise_affine=False causes the compiler
    # to dead-code-eliminate every ct.gather(W, ...) so the dummy is never accessed.
    W_tensor = W.contiguous() if elementwise_affine else X2d.new_empty(1)
    if BLOCK_SIZE >= 16384:
        fwd_kernel = _modulated_rms_norm_fwd_ct_occ3
    elif BLOCK_SIZE == 4096:
        fwd_kernel = _modulated_rms_norm_fwd_ct_occ4
    else:
        fwd_kernel = _modulated_rms_norm_fwd_ct
    # When n_cols is a power of 2, BLOCK_SIZE == n_cols so every lane is in-bounds and
    # the kernel can drop bounds checking on all gather/scatter.
    aligned = BLOCK_SIZE == n_cols
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fwd_kernel,
        (
            Y,
            X2d,
            W_tensor,
            Scale_tensor,
            Shift_tensor,
            RSTD,
            int(n_cols),
            float(eps),
            float(offset) if elementwise_affine else 0.0,
            int(BLOCK_SIZE),
            int(casting_mode_int),
            bool(elementwise_affine),
            bool(has_shift),
            int(rows_per_modulation),
            bool(aligned),
        ),
    )

    return Y.view(*shape), X2d, RSTD, int(BLOCK_SIZE), rows_per_modulation


def _modulated_rms_norm_backward_ct(
    dY, X, W, scale, shift, RSTD, offset, BLOCK_SIZE, casting_mode_int, rows_per_modulation, in_place
):
    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim).contiguous()
    n_rows, n_cols = dY2d.shape
    elementwise_affine = W is not None

    # --- modulation setup ---
    scale_shape = scale.shape
    scale_rows = scale.numel() // dim
    has_shift = shift is not None
    Scale_tensor = scale.view(scale_rows, dim).contiguous()

    # atomic_add accumulates onto existing values, so it needs a zeroed buffer.
    # rows_per_modulation == 1 uses a plain scatter, so empty is fine.
    alloc = torch.zeros if rows_per_modulation != 1 else torch.empty
    dScale = alloc(scale_rows, n_cols, dtype=torch.float32, device=scale.device)
    dShift = alloc(scale_rows, n_cols, dtype=torch.float32, device=scale.device) if has_shift else dScale

    # Single shared modulation row (scale_rows == 1, >1 hidden rows): the kernel
    # accumulates dScale/dShift per block and atomic_add's once, collapsing the
    # per-row atomic contention on that one row. rows_per_modulation == 1 keeps the
    # plain-store path; the general (1 < scale_rows < n_rows) case keeps per-row atomics.
    single_mod = scale_rows == 1 and n_rows > 1

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count, 1, 1)

    dX = dY2d if in_place else torch.zeros_like(dY2d)

    if elementwise_affine:
        dW_partial = torch.empty(sm_count, n_cols, dtype=torch.float32, device=W.device)
        bwd_w_kernel = _modulated_rms_norm_bwd_w_large_ct_nww8
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            bwd_w_kernel,
            (
                dY2d,
                dX,
                X.contiguous(),
                W.contiguous(),
                Scale_tensor,
                RSTD,
                dW_partial,
                dScale,
                dShift,
                int(n_cols),
                float(offset),
                int(rows_per_program),
                int(BLOCK_SIZE),
                int(casting_mode_int),
                bool(has_shift),
                int(rows_per_modulation),
                bool(single_mod),
            ),
        )
        dW = dW_partial.sum(dim=0).to(W.dtype)
    else:
        bwd_kernel = _modulated_rms_norm_bwd_large_ct_nww8
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            bwd_kernel,
            (
                dY2d,
                dX,
                X.contiguous(),
                Scale_tensor,
                RSTD,
                dScale,
                dShift,
                int(n_cols),
                int(rows_per_program),
                int(BLOCK_SIZE),
                int(casting_mode_int),
                bool(has_shift),
                int(rows_per_modulation),
                bool(single_mod),
            ),
        )
        dW = None

    dScale_out = dScale.to(scale.dtype).view(*scale_shape)
    dShift_out = dShift.to(shift.dtype).view(*shift.shape) if has_shift else None

    return dX.view(*shape), dW, dScale_out, dShift_out


class ModulatedRMSNormCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, scale, shift, eps, offset=0.0, casting_mode="llama", in_place=True):
        X = X.contiguous()
        if W is not None:
            W = W.contiguous()
        scale = scale.contiguous()
        if shift is not None:
            shift = shift.contiguous()

        if isinstance(casting_mode, int):
            casting_mode_int = casting_mode
        else:
            assert casting_mode in _str_to_casting_mode, f"Invalid casting_mode: {casting_mode}"
            casting_mode_int = _str_to_casting_mode[casting_mode]

        Y, X_saved, RSTD, BLOCK_SIZE, rows_per_modulation = _modulated_rms_norm_forward_ct(
            X, W, scale, shift, eps, offset, casting_mode_int
        )

        ctx.offset = offset
        ctx.casting_mode = casting_mode_int
        ctx.in_place = in_place
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.rows_per_modulation = rows_per_modulation
        ctx.elementwise_affine = W is not None
        ctx.has_shift = shift is not None

        if W is not None and shift is not None:
            ctx.save_for_backward(X_saved, W, scale, shift, RSTD)
        elif W is not None:
            ctx.save_for_backward(X_saved, W, scale, RSTD)
        elif shift is not None:
            ctx.save_for_backward(X_saved, scale, shift, RSTD)
        else:
            ctx.save_for_backward(X_saved, scale, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = dY.contiguous()
        if ctx.elementwise_affine and ctx.has_shift:
            X, W, scale, shift, RSTD = ctx.saved_tensors
        elif ctx.elementwise_affine:
            X, W, scale, RSTD = ctx.saved_tensors
            shift = None
        elif ctx.has_shift:
            X, scale, shift, RSTD = ctx.saved_tensors
            W = None
        else:
            X, scale, RSTD = ctx.saved_tensors
            W = None
            shift = None

        dX, dW, dScale, dShift = _modulated_rms_norm_backward_ct(
            dY,
            X,
            W,
            scale,
            shift,
            RSTD,
            ctx.offset,
            ctx.BLOCK_SIZE,
            ctx.casting_mode,
            ctx.rows_per_modulation,
            ctx.in_place,
        )
        return dX, dW, dScale, dShift, None, None, None, None


@register_impl("liger.modulated_rms_norm", backend="cutile")
def modulated_rms_norm(
    X: torch.Tensor,
    W,
    scale: torch.Tensor,
    shift=None,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
    **kwargs,
) -> torch.Tensor:
    return ModulatedRMSNormCuTileFunction.apply(X, W, scale, shift, eps, offset, casting_mode, in_place)
