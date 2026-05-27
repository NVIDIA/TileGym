# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Chunk Gated Delta Rule.

Two-kernel flow:
  1. chunk_gated_delta_rule_intra_kernel — per-(b, h, chunk) prepare
  2. chunk_gated_delta_rule_inter_kernel — inter-chunk recurrence over a
     (BLOCK_K, BLOCK_V) running state.
"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_intra_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "chunk_gated_delta_rule.cuh",
    kernel_name="chunk_gated_delta_rule_intra_kernel",
)
_inter_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "chunk_gated_delta_rule.cuh",
    kernel_name="chunk_gated_delta_rule_inter_kernel",
)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _launch_intra(
    Q,
    K,
    V,
    Beta,
    G,
    Q_out,
    K_out,
    V_corr,
    K_cumdecay,
    G_cum_out,
    scale,
    B,
    seq_len,
    num_heads,
    num_chunks,
    K_dim,
    V_dim,
    chunk_size,
    block_k,
    use_qk_l2norm,
):
    dtype = Q.dtype
    dump_kernel_types("chunk_gated_delta_rule_intra_kernel", Q, K, V, Beta, G)

    occupancy = 1
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        chunk_size,
        block_k,
        bool_to_str(use_qk_l2norm),
        occupancy,
    ]
    kernel, _, _ = _intra_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature=(
            "const {T}*, const {T}*, const {T}*, const {T}*, const {T}*, "
            "float*, float*, float*, float*, float*, "
            "float, int, int, int, int, int, int"
        ),
    )
    grid = (B * num_heads, num_chunks, 1)
    _intra_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(Beta.data_ptr()),
            np.uint64(G.data_ptr()),
            np.uint64(Q_out.data_ptr()),
            np.uint64(K_out.data_ptr()),
            np.uint64(V_corr.data_ptr()),
            np.uint64(K_cumdecay.data_ptr()),
            np.uint64(G_cum_out.data_ptr()),
            np.float32(scale),
            np.int32(B),
            np.int32(seq_len),
            np.int32(num_heads),
            np.int32(num_chunks),
            np.int32(K_dim),
            np.int32(V_dim),
        ],
    )


def _launch_inter(
    Q_ch,
    K_ch,
    V_corr,
    K_cumdecay,
    G_cum,
    Output,
    InitState,
    FinalState,
    B,
    num_chunks,
    num_heads,
    K_dim,
    V_dim,
    chunk_size,
    block_k,
    block_v,
    has_initial_state,
    output_final_state,
):
    # The inter kernel takes a float* for all tensors — pick any of them to
    # derive the dtype (torch.float32). We still need a dtype for the
    # TileCppKernel interface even though the template does not parametrize T.
    dump_kernel_types("chunk_gated_delta_rule_inter_kernel", Q_ch)

    occupancy = 1
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        chunk_size,
        block_k,
        block_v,
        bool_to_str(has_initial_state),
        bool_to_str(output_final_state),
        occupancy,
    ]
    # The template T is unused in the signature because all I/O is float, but
    # TileCppKernel.get_kernel always prepends a cpp_type. Pass dtype=float32
    # to produce "float" as the leading template arg, which is harmless since
    # the template signature doesn't use it.
    kernel, _, _ = _inter_kernel.get_kernel(
        dtype=torch.float32,
        template_params=template_params,
        signature=(
            "const float*, const float*, const float*, const float*, const float*, "
            "float*, const float*, float*, "
            "int, int, int, int, int"
        ),
    )
    grid = (B * num_heads, (V_dim + block_v - 1) // block_v, 1)

    init_ptr = np.uint64(InitState.data_ptr()) if has_initial_state else np.uint64(0)
    final_ptr = np.uint64(FinalState.data_ptr()) if output_final_state else np.uint64(0)

    _inter_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q_ch.data_ptr()),
            np.uint64(K_ch.data_ptr()),
            np.uint64(V_corr.data_ptr()),
            np.uint64(K_cumdecay.data_ptr()),
            np.uint64(G_cum.data_ptr()),
            np.uint64(Output.data_ptr()),
            init_ptr,
            final_ptr,
            np.int32(B),
            np.int32(num_chunks),
            np.int32(num_heads),
            np.int32(K_dim),
            np.int32(V_dim),
        ],
    )


class TileCppChunkGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, g, beta, chunk_size, initial_state, output_final_state, use_qk_l2norm_in_kernel
    ):
        initial_dtype = query.dtype
        B, T, H, K = query.shape
        V = value.shape[-1]
        num_chunks = (T + chunk_size - 1) // chunk_size
        scale = 1.0 / math.sqrt(K)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        device = query.device
        BLOCK_K = _next_power_of_2(K)
        if BLOCK_K != K or _next_power_of_2(V) != V:
            raise ValueError(
                f"chunk_gated_delta_rule requires both K and V head dims to be powers of two "
                f"(got K={K}, V={V}); the kernel does unmasked tile loads of size BLOCK_K=next_pow2(K) "
                f"and BLOCK_K-wide stripes over V."
            )

        NC = num_chunks * chunk_size
        if T != NC:
            pad_t = NC - T

            def _pad_time(x, dim=1):
                pad_shape = list(x.shape)
                pad_shape[dim] = pad_t
                pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
                return torch.cat([x, pad], dim=dim)

            query = _pad_time(query)
            key = _pad_time(key)
            value = _pad_time(value)
            g = _pad_time(g)
            beta = _pad_time(beta)

        q_chunked = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=torch.float32)
        k_chunked = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=torch.float32)
        v_corrected = torch.empty(B, H, num_chunks, chunk_size, V, device=device, dtype=torch.float32)
        k_cumdecay = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=torch.float32)
        g_cum = torch.empty(B, H, num_chunks, chunk_size, device=device, dtype=torch.float32)
        output_buf = torch.empty(B, H, num_chunks, chunk_size, V, device=device, dtype=torch.float32)

        _launch_intra(
            query,
            key,
            value,
            beta,
            g,
            q_chunked,
            k_chunked,
            v_corrected,
            k_cumdecay,
            g_cum,
            scale,
            B,
            NC,
            H,
            num_chunks,
            K,
            V,
            chunk_size,
            BLOCK_K,
            use_qk_l2norm_in_kernel,
        )

        has_initial_state = initial_state is not None
        if has_initial_state:
            init_state = initial_state.contiguous().float()
        else:
            init_state = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)

        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, K, V, device=device, dtype=torch.float32)

        BLOCK_V = min(128, _next_power_of_2(V))
        dummy = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)

        _launch_inter(
            q_chunked,
            k_chunked,
            v_corrected,
            k_cumdecay,
            g_cum,
            output_buf,
            init_state if has_initial_state else dummy,
            final_state if output_final_state else dummy,
            B,
            num_chunks,
            H,
            K,
            V,
            chunk_size,
            BLOCK_K,
            BLOCK_V,
            has_initial_state,
            output_final_state,
        )

        # Reshape back to (B, T, H, V).
        output = output_buf.reshape(B, H, NC, V)[:, :, :T, :]
        output = output.transpose(1, 2).contiguous().to(initial_dtype)
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        raise NotImplementedError("TileCpp chunk_gated_delta_rule backward not implemented")


@register_impl("chunk_gated_delta_rule", backend="tilecpp")
def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    return TileCppChunkGatedDeltaRule.apply(
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
