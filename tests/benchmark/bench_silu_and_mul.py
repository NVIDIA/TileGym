#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available, register_impl

# Available backends for benchmarking
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def reference_silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,  # Unused - kept for interface compatibility
):
    """Reference implementation using PyTorch
    Implements: Silu(input[..., :hidden_size]) * input[..., hidden_size:]
    """
    hidden_size = input.shape[-1] // 2
    # Extract the two parts according to the formula
    x1 = input[..., :hidden_size]  # First half for SiLU
    x2 = input[..., hidden_size:]  # Second half for multiplication
    # Apply SiLU (Sigmoid Linear Unit) to x1 and multiply by x2
    return torch.nn.functional.silu(x1) * x2


register_impl("silu_and_mul", "torch")(reference_silu_and_mul)


def create_benchmark_config(datatype, hidden_size):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split('.')[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"silu-and-mul-hidden{hidden_size}-{dtype_name}-GBps",
        args={
            "hidden_size": hidden_size,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, hidden_size)
        for datatype in [torch.float16, torch.float32]
        for hidden_size in [2048]
    ]
)
def bench_silu_and_mul(
    M,
    hidden_size,
    backend,
    datatype,
    device="cuda",
):
    # Create input tensor with shape (M, 2 * hidden_size)
    # Following the formula: Silu(input[..., :hidden_size]) * input[..., hidden_size:]
    input_shape = (M, 2 * hidden_size)
    x = torch.randn(input_shape, dtype=datatype, device=device)

    fn = lambda: tilegym.ops.silu_and_mul(x, backend=backend)
    ref = lambda: reference_silu_and_mul(x)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    # Calculate memory bandwidth in GB/s
    # Total memory: read input tensor + write output tensor
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read full input tensor (M, 2*hidden_size)
    output_bytes = M * hidden_size * bytes_per_element  # Write output tensor (M, hidden_size)

    total_bytes = input_bytes + output_bytes

    # Use triton's cudagraph benchmark for timing
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_silu_and_mul.run(print_data=True)
