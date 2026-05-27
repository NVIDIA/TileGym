# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Gemma attention decode operator.
"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.splitk_reduce import splitk_reduce as _tilecpp_splitk_reduce
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "gemma_attention_decode.cuh",
    kernel_name="gemma_attention_decode_kernel",
)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _launch(
    Q,
    K,
    V,
    Att_Out,
    LSE_Out,
    softmax_scale,
    soft_cap,
    B,
    H_kv,
    S_kv,
    num_q_head_per_kv,
    query_group_block_size,
    kv_len_per_split,
    head_dim,
    block_n,
    num_kv_splits,
    window_size,
    has_soft_cap,
    grid,
):
    dump_kernel_types("gemma_attention_decode_kernel", Q, K, V, Att_Out, LSE_Out)

    occupancy = 1
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        B,
        H_kv,
        S_kv,
        num_q_head_per_kv,
        query_group_block_size,
        kv_len_per_split,
        head_dim,
        block_n,
        num_kv_splits,
        window_size,
        bool_to_str(has_soft_cap),
        occupancy,
    ]

    kernel, _, _ = _kernel.get_kernel(
        dtype=Q.dtype,
        template_params=template_params,
        signature="const {T}*, const {T}*, const {T}*, {T}*, float*, float, float",
    )

    _kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(Att_Out.data_ptr()),
            np.uint64(LSE_Out.data_ptr()),
            np.float32(softmax_scale),
            np.float32(soft_cap),
        ],
    )


class _GemmaAttentionDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, window_size=0, soft_cap=None, kv_len_per_split=None):
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_block_size = max(8, _next_power_of_2(num_q_head_per_kv))

        Q_grouped = Q.view(batch_size, num_kv_heads, num_q_head_per_kv, head_dim).contiguous()
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim).contiguous()
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim).contiguous()

        BLOCK_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            NUM_KV_SPLITS = max(1, NUM_SMS // (batch_size * num_kv_heads))
            BLOCK_SIZE = max(BLOCK_N, _next_power_of_2(_cdiv(seq_len, NUM_KV_SPLITS)))
            NUM_KV_SPLITS = _cdiv(seq_len, BLOCK_SIZE)
        else:
            NUM_KV_SPLITS = _cdiv(seq_len, kv_len_per_split)
            BLOCK_SIZE = kv_len_per_split

        assert BLOCK_SIZE == _next_power_of_2(BLOCK_SIZE)

        HEAD_DIM = _next_power_of_2(head_dim)

        device = Q.device
        Att_Mid_Out_5D = torch.empty(
            (batch_size, num_kv_heads, num_q_head_per_kv, NUM_KV_SPLITS, head_dim),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out_4D = torch.empty(
            (batch_size, num_kv_heads, num_q_head_per_kv, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )

        O = torch.empty(batch_size, num_q_heads, head_dim, device=device, dtype=Q.dtype)

        has_soft_cap = soft_cap is not None
        soft_cap_val = float(soft_cap) if has_soft_cap else 0.0

        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)

        _launch(
            Q_grouped,
            K,
            V,
            Att_Mid_Out_5D,
            LSE_Out_4D,
            softmax_scale,
            soft_cap_val,
            batch_size,
            num_kv_heads,
            seq_len,
            num_q_head_per_kv,
            query_group_block_size,
            BLOCK_SIZE,
            HEAD_DIM,
            BLOCK_N,
            NUM_KV_SPLITS,
            window_size,
            has_soft_cap,
            grid,
        )

        Att_Mid_Out = Att_Mid_Out_5D.view(batch_size, num_q_heads, NUM_KV_SPLITS, head_dim)
        LSE_Out = LSE_Out_4D.view(batch_size, num_q_heads, NUM_KV_SPLITS)

        _tilecpp_splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)
        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Gemma attention decode backward is not implemented yet")


gemma_attention_decode = _GemmaAttentionDecode.apply


@register_impl("gemma_attention_decode", backend="tilecpp")
def gemma_fmha_decode(
    q,
    k,
    v,
    sm_scale=None,
    window_size=0,
    soft_cap=None,
    kv_len_per_split=None,
    **kwargs,
):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return gemma_attention_decode(
        q,
        k,
        v,
        sm_scale,
        window_size,
        soft_cap,
        kv_len_per_split,
    )
