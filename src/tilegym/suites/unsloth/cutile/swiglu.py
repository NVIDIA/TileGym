# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
SwiGLU forward and backward CuTile kernels.

Kernels:
  - _fg_kernel_ct: h = silu(e) * g  (forward)
  - _DWf_DW_dfg_kernel_ct: backward pass, in-place overwrites DW/e/g
"""

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import cdiv

ConstInt = ct.Constant[int]

# Module-level tune cache: (n_elements, LONG_INDEXING, dtype, device) -> tuned_kernel
_swiglu_fg_tune_cache: dict = {}

# signed int32 max is 2**31-1 so num_elements cannot exceed 2**31
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE_FWD = 1024  # Optimal per BLOCK_SIZE sweep benchmark on B200
BLOCK_SIZE_BWD = 1024  # Optimal per BLOCK_SIZE sweep benchmark on B200
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - max(BLOCK_SIZE_FWD, BLOCK_SIZE_BWD) * SAFE_INT32_BUFFER_MULTIPLIER


def _sigmoid_tanh(x):
    """sigmoid(x) = 0.5 + 0.5 * tanh(0.5 * x)
    Uses ct.tanh which compiles to MUFU.TANH hardware instruction."""
    return 0.5 + 0.5 * ct.tanh(0.5 * x)


@ct.kernel
def _fg_kernel_ct(e, g, h, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """SwiGLU forward: h = silu(e) * g = (e * sigmoid(e)) * g"""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    se = _sigmoid_tanh(e_f32)
    f_row = e_f32 * se
    f_row = ct.astype(f_row, g.dtype)
    h_row = f_row * g_row

    ct.scatter(h, offsets, h_row, check_bounds=True)


@ct.kernel
def _DWf_DW_dfg_kernel_ct(DW, e, g, n_elements: ConstInt, BLOCK_SIZE: ConstInt, LONG_INDEXING: ConstInt):
    """SwiGLU backward (in-place): DW→h, e→df, g→de"""
    bid = ct.bid(0)
    if LONG_INDEXING:
        offsets = ct.astype(ct.arange(BLOCK_SIZE, dtype=ct.int32), ct.int64) + ct.astype(bid, ct.int64) * BLOCK_SIZE
    else:
        offsets = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    DW_row = ct.gather(DW, offsets, padding_value=0)
    e_row = ct.gather(e, offsets, padding_value=0)
    e_f32 = ct.astype(e_row, ct.float32)
    g_row = ct.gather(g, offsets, padding_value=0)

    se = _sigmoid_tanh(e_f32)
    f_row_f32 = se * e_f32
    f_row = ct.astype(f_row_f32, DW.dtype)
    h_row = f_row * g_row  # output stored into DW
    df_row = DW_row * f_row  # stored into e
    dg_row = DW_row * g_row
    # dsilu(e) = se + se*e*(1-se) = se + f_row_f32*(1-se), reuse f_row_f32
    dsilu = se + f_row_f32 * (1.0 - se)
    de_row = ct.astype(dg_row, ct.float32) * dsilu
    de_row = ct.astype(de_row, DW.dtype)  # stored into g

    ct.scatter(DW, offsets, h_row, check_bounds=True)
    ct.scatter(e, offsets, df_row, check_bounds=True)
    ct.scatter(g, offsets, de_row, check_bounds=True)


@register_impl("unsloth.swiglu_fg", backend="cutile")
def swiglu_fg(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    stream = torch.cuda.current_stream()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    cache_key = (n_elements, LONG_INDEXING, e.dtype, str(e.device))
    if cache_key not in _swiglu_fg_tune_cache:
        result = exhaustive_search(
            list(autotune_configs()),
            stream,
            lambda cfg: (cdiv(n_elements, BLOCK_SIZE_FWD),),
            _fg_kernel_ct,
            lambda cfg: (e.reshape(-1), g.reshape(-1), h.reshape(-1), n_elements, BLOCK_SIZE_FWD, LONG_INDEXING),
            lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _swiglu_fg_tune_cache[cache_key] = ct.kernel(
            _fg_kernel_ct._pyfunc,
            occupancy=best_cfg.occupancy,
        )
    tuned_kernel = _swiglu_fg_tune_cache[cache_key]
    ct.launch(
        stream,
        (cdiv(n_elements, BLOCK_SIZE_FWD),),
        tuned_kernel,
        (e.reshape(-1), g.reshape(-1), h.reshape(-1), n_elements, BLOCK_SIZE_FWD, LONG_INDEXING),
    )
    return h


@register_impl("unsloth.swiglu_bwd", backend="cutile")
def swiglu_bwd(DW, e, g):
    n_elements = e.numel()
    LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1
    grid = (cdiv(n_elements, BLOCK_SIZE_BWD),)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _DWf_DW_dfg_kernel_ct,
        (DW.reshape(-1), e.reshape(-1), g.reshape(-1), n_elements, BLOCK_SIZE_BWD, LONG_INDEXING),
    )
    return DW, e, g
