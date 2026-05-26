# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Multi-head Latent Attention (MLA) Implementation

MLA is a specialized attention variant used in DeepSeek models that uses
compressed key-value projections with separate position embeddings.

All dimensions are template parameters for maximum optimization.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_dtype_info
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = get_logger(__name__)


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =============================================================================
# Kernel Definition
# =============================================================================

_prefill_mla_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "mla.cuh",
    kernel_name="prefill_mla_kernel",
)


def _get_cpp_type(dtype: torch.dtype) -> str:
    """Get C++ type string for a torch dtype."""
    type_map = {
        torch.float32: "float",
        torch.float16: "__half",
        torch.bfloat16: "__nv_bfloat16",
    }
    return type_map.get(dtype, "float")


def _launch_prefill_mla_kernel(
    Q: torch.Tensor,
    QPE: torch.Tensor,
    K: torch.Tensor,
    KPE: torch.Tensor,
    V: torch.Tensor,
    Out: torch.Tensor,
    sm_scale: float,
    B: int,
    H: int,
    H_KV: int,
    S_QO: int,
    S_KV: int,
    TILE_D: int,
    TILE_KPE: int,
    TILE_M: int,
    TILE_N: int,
    QUERY_GROUP_SIZE: int,
    IS_CAUSAL: bool,
):
    """Launch the prefill MLA kernel."""
    dtype = Q.dtype
    cpp_type = _get_cpp_type(dtype)

    dump_kernel_types("prefill_mla_kernel", Q, QPE, K, KPE, V)

    # Template params: B, H, H_KV, S_QO, S_KV, TILE_D, TILE_KPE, TILE_M, TILE_N, QUERY_GROUP_SIZE, IS_CAUSAL
    template_params = [B, H, H_KV, S_QO, S_KV, TILE_D, TILE_KPE, TILE_M, TILE_N, QUERY_GROUP_SIZE, IS_CAUSAL]

    # Simplified signature: just pointers + scale (all dims are template params)
    kernel, _, _ = _prefill_mla_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature=f"const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, const {cpp_type}*, {cpp_type}*, float",
    )

    # Grid: (num_query_blocks, B * H)
    grid = (cdiv(S_QO, TILE_M), B * H)

    _prefill_mla_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(QPE.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(KPE.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(Out.data_ptr()),
            np.float32(sm_scale),
        ],
    )


# =============================================================================
# Autograd Function
# =============================================================================


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, qpe, k, kpe, v, sm_scale, IS_CAUSAL, kernel_configs):
        """
        MLA Forward.

        Args:
            q: Query tensor [B, H, S_qo, TILE_D]
            qpe: Query positional embedding [B, H, S_qo, TILE_KPE]
            k: Key tensor [B, H_kv, S_kv, TILE_D]
            kpe: Key positional embedding [B, 1, S_kv, TILE_KPE]
            v: Value tensor [B, H_kv, S_kv, TILE_D]
            sm_scale: Softmax scale
            IS_CAUSAL: Whether to use causal masking
            kernel_configs: Dict with TILE_M, TILE_N settings

        Returns:
            o: Output tensor [B, H, S_qo, TILE_D]
        """

        # Setup stride and shape
        B, H, S_qo, TILE_D = q.shape
        TILE_KPE = kpe.shape[3]
        assert k.shape == v.shape
        H_kv = k.shape[1]
        S_kv = k.shape[2]
        o = torch.empty_like(q)

        if H == H_kv:
            query_group_size = 0
        else:
            assert H % H_kv == 0
            query_group_size = int(H / H_kv)

        TILE_M = kernel_configs.get("TILE_M", 256)
        TILE_N = kernel_configs.get("TILE_N", 128)

        dump_kernel_types("prefill_mla", q, qpe, k, kpe, v)

        _launch_prefill_mla_kernel(
            q,
            qpe,
            k,
            kpe,
            v,
            o,
            sm_scale,
            B,
            H,
            H_kv,
            S_qo,
            S_kv,
            TILE_D,
            TILE_KPE,
            TILE_M,
            TILE_N,
            query_group_size,
            IS_CAUSAL,
        )

        ctx.save_for_backward(q, k, v, o)
        ctx.sm_scale = sm_scale
        ctx.shapes = (B, H, S_qo, S_kv)
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("MLA backward is not implemented yet")


class Attention:
    """Attention wrapper class."""

    def __init__(self, IS_CAUSAL, kernel_configs):
        self.IS_CAUSAL = IS_CAUSAL
        self.kernel_configs = kernel_configs

    def __call__(self, q, k, v, sm_scale, qpe=None, kpe=None):
        return _attention.apply(q, qpe, k, kpe, v, sm_scale, self.IS_CAUSAL, self.kernel_configs)


def tilecpp_mla(q, k, v, qpe, kpe, is_causal, scaling, **kwargs):
    """
    Multi-head Latent Attention (MLA) interface.

    Args:
        q: Query tensor [B, H, S_qo, D]
        k: Key tensor [B, H_kv, S_kv, D]
        v: Value tensor [B, H_kv, S_kv, D]
        qpe: Query positional embedding [B, H, S_qo, KPE_D]
        kpe: Key positional embedding [B, 1, S_kv, KPE_D]
        is_causal: Whether to use causal masking
        scaling: Softmax scale (None for auto)
        **kwargs: Additional arguments including kernel_configs

    Returns:
        o: Output tensor [B, H, S_qo, D]
    """
    if not is_causal:
        raise NotImplementedError("tilecpp MLA only supports is_causal=True; the non-causal path is not validated.")

    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

    defaults = {"TILE_M": 256, "TILE_N": 128}
    user_cfg = kwargs.get("kernel_configs")
    if user_cfg is None:
        kernel_configs = defaults
    else:
        kernel_configs = {**defaults, **user_cfg}

    attention = Attention(is_causal, kernel_configs)
    o = attention(q, k, v, scaling, qpe, kpe)
    return o


register_impl("mla", "tilecpp")(tilecpp_mla)
