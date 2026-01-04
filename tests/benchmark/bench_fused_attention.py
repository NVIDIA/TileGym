# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

BATCH, N_HEADS = 4, 32


def reference_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float = None,
    is_causal: bool = True,
):
    """Reference implementation using PyTorch"""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=scaling
    )


register_impl("fmha", "torch")(reference_fmha)

# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support"""
    if datatype == torch.float8_e5m2:
        return [p for p in ALL_BACKENDS if p is not None and p[0] != "torch"]  # Torch cannot support FP8
    return [p for p in ALL_BACKENDS if p is not None]  # Filter out None entries


def create_benchmark_config(datatype, HEAD_DIM, mode, causal):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-{dtype_name}-TFLOPS",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": mode,
            "causal": causal,
            "datatype": datatype,
        },
    )


_dtypes = [torch.float16, torch.float8_e5m2]


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, HEAD_DIM, mode, causal)
        for datatype in _dtypes
        for HEAD_DIM in [128]
        for mode in ["fwd"]
        for causal in [True, False]
    ]
)
def bench_fused_attention(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    backend,
    datatype,
    device=DEVICE,
):
    assert mode == "fwd"  # Only forward pass is supported in this benchmark
    dtype = torch.float16

    # Create input tensors
    q = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM),
        dtype=dtype,
        device=device,
    )

    if datatype == torch.float8_e5m2:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.to(torch.float8_e5m2)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)  # Standard scaling factor

    fn = lambda: tilegym.ops.fmha(q, k, v, scaling=sm_scale, is_causal=causal, backend=backend)

    if datatype != torch.float8_e5m2:
        # torch doesn't support FP8 attention
        ref = lambda: reference_fmha(q, k, v, scaling=sm_scale, is_causal=causal)
        torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_fused_attention.run(print_data=True)
