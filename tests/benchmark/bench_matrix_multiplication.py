# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = None,  # Unused - kept for interface compatibility
    trans_b: bool = None,  # Unused - kept for interface compatibility
    static_persistent: bool = True,  # Unused - kept for interface compatibility
    use_tma: bool = True,  # Unused - kept for interface compatibility
):
    """Reference implementation using PyTorch"""
    return torch.matmul(a, b)


register_impl("matmul", "torch")(reference_matmul)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    if datatype == torch.float8_e5m2:
        return ALL_BACKENDS[:-1]  # Torch cannot support FP8
    else:
        return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split('.')[-1]  # e.g., 'float16' from 'torch.float16'
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        max_range = 16
    else:
        max_range = 15  # To avoid OOM
    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, max_range)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"matmul-performance-{dtype_name}-TFLOPS",
        args={"datatype": datatype},
    )


@triton.testing.perf_report([create_benchmark_config(datatype) for datatype in [torch.float16, torch.float8_e5m2]])
def benchmark(M, N, K, backend, datatype):
    if datatype == torch.float8_e5m2:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
    else:
        a = torch.randn((M, K), device=DEVICE, dtype=datatype)
        b = torch.randn((K, N), device=DEVICE, dtype=datatype)

    quantiles = [0.5, 0.2, 0.8]

    fn = lambda: tilegym.ops.matmul(a, b, use_tma=True, static_persistent=False, backend=backend)

    if datatype != torch.float8_e5m2:
        # torch doesn't support FP8 matmul
        ref = lambda: reference_matmul(a, b)
        torch.testing.assert_close(fn(), ref())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True)
