# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ attention_sink_decode (split-KV decode with sink tokens).
"""

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.splitk_reduce import splitk_reduce as _tilecpp_splitk_reduce
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention_sink_decode.cuh",
    kernel_name="attention_sink_decode_kernel",
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
    Sinks,
    Att_Out,
    LSE_Out,
    Start_q,
    softmax_scale,
    B,
    H_kv,
    H_qo,
    S_kv,
    num_q_head_per_kv,
    query_group_block_size,
    kv_len_per_split,
    head_dim,
    block_n,
    num_kv_splits,
    bandwidth,
    has_sinks,
    grid,
):
    dump_kernel_types("attention_sink_decode_kernel", Q, K, V, Att_Out, LSE_Out)

    occupancy = 1
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        B,
        H_kv,
        H_qo,
        S_kv,
        num_q_head_per_kv,
        query_group_block_size,
        kv_len_per_split,
        head_dim,
        block_n,
        num_kv_splits,
        bandwidth,
        bool_to_str(has_sinks),
        occupancy,
    ]

    kernel, _, _ = _kernel.get_kernel(
        dtype=Q.dtype,
        template_params=template_params,
        signature=("const {T}*, const {T}*, const {T}*, const {T}*, {T}*, float*, const int*, float"),
    )

    sinks_ptr = np.uint64(Sinks.data_ptr()) if has_sinks else np.uint64(0)

    _kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            sinks_ptr,
            np.uint64(Att_Out.data_ptr()),
            np.uint64(LSE_Out.data_ptr()),
            np.uint64(Start_q.data_ptr()),
            np.float32(softmax_scale),
        ],
    )


class _AttentionSinkDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q, kv_len_per_split=None):
        # Shapes
        assert start_q.numel() == 1
        bs, n_ctx, n_kv_heads, repeat_kv, head_dim = q.shape
        _, n_kv_ctx, n_kv_heads_k, head_dim_k = k.shape
        _, _, n_kv_heads_v, head_dim_v = v.shape
        n_heads = n_kv_heads * repeat_kv
        assert n_ctx == 1
        assert head_dim == head_dim_k == head_dim_v
        assert n_kv_heads == n_kv_heads_k == n_kv_heads_v
        assert head_dim in {16, 32, 64, 128, 256}

        # Reshape to match kernel expectations
        q_reshaped = q.view(bs, n_heads, head_dim)
        k_reshaped = k.transpose(1, 2).contiguous()
        v_reshaped = v.transpose(1, 2).contiguous()

        BLOCK_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count
            NUM_KV_SPLITS = max(1, NUM_SMS // (bs * n_kv_heads))
            BLOCK_SIZE = max(BLOCK_N, _next_power_of_2(_cdiv(n_kv_ctx, NUM_KV_SPLITS)))
            NUM_KV_SPLITS = _cdiv(n_kv_ctx, BLOCK_SIZE)
        else:
            NUM_KV_SPLITS = _cdiv(n_kv_ctx, kv_len_per_split)
            BLOCK_SIZE = kv_len_per_split
        assert BLOCK_SIZE == _next_power_of_2(BLOCK_SIZE)

        Att_Mid_Out = torch.empty(
            (bs, n_heads, NUM_KV_SPLITS, head_dim),
            device=q.device,
            dtype=q.dtype,
        )
        LSE_Out = torch.empty(
            (bs, n_heads, NUM_KV_SPLITS),
            device=q.device,
            dtype=torch.float32,
        )

        HEAD_DIM = _next_power_of_2(head_dim)
        num_q_head_per_kv = repeat_kv
        query_group_block_size = max(8, _next_power_of_2(num_q_head_per_kv))

        bandwidth_val = bandwidth if bandwidth else 0
        has_sinks = sinks is not None
        sinks_arg = sinks if has_sinks else torch.zeros(1, device=q.device, dtype=q.dtype)

        # Reshape Q into grouped view.
        Q_grouped = q_reshaped.view(bs, n_kv_heads, num_q_head_per_kv, head_dim)
        Att_Mid_Out_5D = Att_Mid_Out.view(
            bs,
            n_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
            head_dim,
        )
        LSE_Out_4D = LSE_Out.view(bs, n_kv_heads, num_q_head_per_kv, NUM_KV_SPLITS)

        grid = (bs, n_kv_heads, NUM_KV_SPLITS)

        # Start_q must be int32 on device.
        if start_q.dtype != torch.int32:
            start_q = start_q.to(torch.int32)
        start_q = start_q.contiguous()

        _launch(
            Q_grouped,
            k_reshaped,
            v_reshaped,
            sinks_arg,
            Att_Mid_Out_5D,
            LSE_Out_4D,
            start_q,
            sm_scale,
            bs,
            n_kv_heads,
            n_heads,
            n_kv_ctx,
            num_q_head_per_kv,
            query_group_block_size,
            BLOCK_SIZE,
            HEAD_DIM,
            BLOCK_N,
            NUM_KV_SPLITS,
            bandwidth_val,
            has_sinks,
            grid,
        )

        # Reduce splits
        o = torch.empty((bs, n_heads, head_dim), device=q.device, dtype=q.dtype)
        _tilecpp_splitk_reduce(Att_Mid_Out, LSE_Out, o, n_kv_ctx)

        o = o.unsqueeze(1).contiguous()
        o = o.view(bs, n_ctx, n_heads * head_dim)
        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass for tilecpp attention_sink_decode is not implemented.")


attention_splitkv = _AttentionSinkDecode.apply


@register_impl("attention_sink_decode", backend="tilecpp")
def attention_sink_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window=None,
    start_q: torch.LongTensor = 0,
    kv_len_per_split=None,
    **kwargs,
):
    return attention_splitkv(query, key, value, sinks, sm_scale, sliding_window, start_q, kv_len_per_split)
