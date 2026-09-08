# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

# Available backends for benchmarking
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("tilecpp", "TileCpp", ("purple", "-")) if is_backend_available("tilecpp") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def reference_gelu(
    x: torch.Tensor,
    approximate: str = "none",
):
    """Reference implementation using PyTorch
    Implements: GELU(x) = x * Phi(x), with the tanh approximation when approximate="tanh"
    """
    return torch.nn.functional.gelu(x, approximate=approximate)


register_impl("gelu", "torch")(reference_gelu)


def create_benchmark_config(datatype, approximate, hidden_size):
    """Create a benchmark configuration for given datatype, mode and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"gelu-{approximate}-hidden{hidden_size}-{dtype_name}-GBps",
        args={
            "hidden_size": hidden_size,
            "approximate": approximate,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, approximate, hidden_size)
        for datatype in [torch.float16, torch.float32]
        for approximate in ["none", "tanh"]
        for hidden_size in [2048]
    ]
)
def bench_gelu(
    M,
    hidden_size,
    approximate,
    backend,
    datatype,
    device="cuda",
):
    # Create input tensor with shape (M, hidden_size)
    input_shape = (M, hidden_size)
    x = torch.randn(input_shape, dtype=datatype, device=device)

    fn = lambda: tilegym.ops.activation.gelu(x, approximate=approximate, backend=backend)
    ref = lambda: reference_gelu(x, approximate=approximate)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    # Calculate memory bandwidth in GB/s
    # Total memory: read input tensor + write output tensor
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read full input tensor (M, hidden_size)
    output_bytes = x.numel() * bytes_per_element  # Write output tensor (M, hidden_size)

    total_bytes = input_bytes + output_bytes

    # Use triton's cudagraph benchmark for timing
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_gelu.run(print_data=True)
