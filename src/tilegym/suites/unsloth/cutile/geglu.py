# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
GEGLU exact and approximate forward/backward CuTile kernels.

Kernels:
  - _exact_forward_ct: f = 0.5 * e * (1 + erf(e/sqrt(2))); out = f * up
  - _exact_backward_ct: backward for exact GEGLU
  - _approx_forward_ct: f = 0.5 * e * (1 + tanh(sqrt(2/pi)*e*(1+0.044715*e^2))); out = f * up
  - _approx_backward_ct: backward for approximate GEGLU

Conversion notes:
  - tl.math.erf → erf_ct from ct_ops (Abramowitz & Stegun, shared helper)
  - libdevice.tanh → ct.tanh
  - 1D element-wise: ct.gather/ct.scatter with check_bounds
  - BLOCK_SIZE tuned via sweep benchmark on B200
"""

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import cdiv
from .ct_ops import erf_ct

ConstInt = ct.Constant[int]

# Module-level tune caches: (n_elements, LONG_INDEXING, dtype, device) -> tuned_kernel
_geglu_exact_fwd_tune_cache: dict = {}
_geglu_approx_fwd_tune_cache: dict = {}

# signed int32 max is 2**31-1 so num_elements cannot exceed 2**31
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE_FWD = 1024  # Optimal per BLOCK_SIZE sweep benchmark on B200
BLOCK_SIZE_BWD = 1024  # Optimal per BLOCK_SIZE sweep benchmark on B200
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - max(BLOCK_SIZE_FWD, BLOCK_SIZE_BWD) * SAFE_INT32_BUFFER_MULTIPLIER


# =============================================================================
# Exact GEGLU (using erf via Abramowitz & Stegun approximation)
# =============================================================================


@ct.kernel
def _exact_forward_ct(e, g, h, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """Exact GEGLU forward: h = 0.5 * e * (1 + erf(e/sqrt(2))) * g"""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    # erf(e / sqrt(2)) via shared helper
    sx = e_f32 * 0.7071067811865475  # e / sqrt(2)
    erf_val = erf_ct(sx)

    f_row = 0.5 * e_f32 * (erf_val + 1.0)
    f_row = ct.astype(f_row, g.dtype)
    h_row = f_row * g_row

    ct.scatter(h, offsets, h_row, check_bounds=True)


@ct.kernel
def _exact_backward_ct(DW, e, g, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """Exact GEGLU backward (in-place): DW→h, e→df, g→de"""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    DW_row = ct.gather(DW, offsets, padding_value=0)
    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    # erf(e / sqrt(2)) via shared helper
    sx = e_f32 * 0.7071067811865475
    erf_val = erf_ct(sx)

    # f_partial = 0.5 * (1 + erf(e/sqrt(2)))
    f_partial = 0.5 * (erf_val + 1.0)
    f_row_f32 = f_partial * e_f32

    f_row = ct.astype(f_row_f32, DW.dtype)
    h_row = f_row * g_row  # → DW
    df_row = DW_row * f_row  # → e
    dg_row = DW_row * g_row

    # df/de = f_partial + 1/sqrt(2*pi) * e * exp(-0.5*e^2)
    inv_sqrt_2pi = 0.3989422804014327
    df_de = f_partial + inv_sqrt_2pi * e_f32 * ct.exp(-0.5 * e_f32 * e_f32)

    de_row = ct.astype(dg_row, ct.float32) * df_de
    de_row = ct.astype(de_row, DW.dtype)  # → g

    ct.scatter(DW, offsets, h_row, check_bounds=True)
    ct.scatter(e, offsets, df_row, check_bounds=True)
    ct.scatter(g, offsets, de_row, check_bounds=True)


# =============================================================================
# Approximate GEGLU (using tanh)
# =============================================================================


@ct.kernel
def _approx_forward_ct(e, g, h, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """Approximate GEGLU forward using tanh approximation."""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    s = 0.7978845608028654  # sqrt(2/pi)
    inner = s * e_f32 * (1.0 + 0.044715 * e_f32 * e_f32)
    f_row = 0.5 * e_f32 * (ct.tanh(inner) + 1.0)
    f_row = ct.astype(f_row, g.dtype)
    h_row = f_row * g_row

    ct.scatter(h, offsets, h_row, check_bounds=True)


@ct.kernel
def _approx_backward_ct(DW, e, g, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """Approximate GEGLU backward (in-place): DW→h, e→df, g→de"""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    DW_row = ct.gather(DW, offsets, padding_value=0)
    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    s = 0.7978845608028654  # sqrt(2/pi)
    a = s * e_f32
    b = a * 0.044715 * e_f32 * e_f32
    T = 1.0 + ct.tanh(a + b)
    T2 = 0.5 * T
    Q2 = ct.negative(T2) * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2

    f_row = T2 * e_f32
    f_row = ct.astype(f_row, DW.dtype)
    h_row = f_row * g_row  # → DW
    df_row = DW_row * f_row  # → e
    dg_row = DW_row * g_row

    de_row = ct.astype(dg_row, ct.float32) * df_de
    de_row = ct.astype(de_row, DW.dtype)  # → g

    ct.scatter(DW, offsets, h_row, check_bounds=True)
    ct.scatter(e, offsets, df_row, check_bounds=True)
    ct.scatter(g, offsets, de_row, check_bounds=True)


# =============================================================================
# Dispatch wrappers
# =============================================================================


@register_impl("unsloth.geglu_exact_forward", backend="cutile")
def geglu_exact_forward(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=gate.device)
    stream = torch.cuda.current_stream()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    cache_key = (n_elements, LONG_INDEXING, gate.dtype, str(gate.device))
    if cache_key not in _geglu_exact_fwd_tune_cache:
        result = exhaustive_search(
            list(autotune_configs()),
            stream,
            lambda cfg: (cdiv(n_elements, BLOCK_SIZE_FWD),),
            _exact_forward_ct,
            lambda cfg: (
                gate.reshape(-1),
                up.reshape(-1),
                out.reshape(-1),
                n_elements,
                BLOCK_SIZE_FWD,
                LONG_INDEXING,
            ),
            lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _geglu_exact_fwd_tune_cache[cache_key] = ct.kernel(
            _exact_forward_ct._pyfunc,
            occupancy=best_cfg.occupancy,
        )
    tuned_kernel = _geglu_exact_fwd_tune_cache[cache_key]
    ct.launch(
        stream,
        (cdiv(n_elements, BLOCK_SIZE_FWD),),
        tuned_kernel,
        (
            gate.reshape(-1),
            up.reshape(-1),
            out.reshape(-1),
            n_elements,
            BLOCK_SIZE_FWD,
            LONG_INDEXING,
        ),
    )
    return out


@register_impl("unsloth.geglu_exact_backward", backend="cutile")
def geglu_exact_backward(DW, e, g):
    n_elements = e.numel()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    grid = (cdiv(n_elements, BLOCK_SIZE_BWD),)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _exact_backward_ct,
        (DW.reshape(-1), e.reshape(-1), g.reshape(-1), n_elements, BLOCK_SIZE_BWD, LONG_INDEXING),
    )
    return DW, e, g


@register_impl("unsloth.geglu_approx_forward", backend="cutile")
def geglu_approx_forward(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=gate.device)
    stream = torch.cuda.current_stream()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    cache_key = (n_elements, LONG_INDEXING, gate.dtype, str(gate.device))
    if cache_key not in _geglu_approx_fwd_tune_cache:
        result = exhaustive_search(
            list(autotune_configs()),
            stream,
            lambda cfg: (cdiv(n_elements, BLOCK_SIZE_FWD),),
            _approx_forward_ct,
            lambda cfg: (
                gate.reshape(-1),
                up.reshape(-1),
                out.reshape(-1),
                n_elements,
                BLOCK_SIZE_FWD,
                LONG_INDEXING,
            ),
            lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _geglu_approx_fwd_tune_cache[cache_key] = ct.kernel(
            _approx_forward_ct._pyfunc,
            occupancy=best_cfg.occupancy,
        )
    tuned_kernel = _geglu_approx_fwd_tune_cache[cache_key]
    ct.launch(
        stream,
        (cdiv(n_elements, BLOCK_SIZE_FWD),),
        tuned_kernel,
        (
            gate.reshape(-1),
            up.reshape(-1),
            out.reshape(-1),
            n_elements,
            BLOCK_SIZE_FWD,
            LONG_INDEXING,
        ),
    )
    return out


@register_impl("unsloth.geglu_approx_backward", backend="cutile")
def geglu_approx_backward(DW, e, g):
    n_elements = e.numel()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    grid = (cdiv(n_elements, BLOCK_SIZE_BWD),)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _approx_backward_ct,
        (DW.reshape(-1), e.reshape(-1), g.reshape(-1), n_elements, BLOCK_SIZE_BWD, LONG_INDEXING),
    )
    return DW, e, g
