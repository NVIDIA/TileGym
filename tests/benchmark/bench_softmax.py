# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_softmax(
    x: torch.Tensor,
    use_tma: bool = False,  # Unused - kept for interface compatibility
    use_online: bool = None,  # Unused - kept for interface compatibility
):
    """Reference implementation of softmax using PyTorch"""
    return torch.nn.functional.softmax(x, dim=-1)


register_impl("softmax", "torch")(reference_softmax)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(M, use_tma=True):
    """Create a benchmark configuration for given parameters"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"softmax-performance-tma-{use_tma}-GBps",
        args={"M": M, "use_tma": use_tma},
    )


@triton.testing.perf_report(
    [create_benchmark_config(M, use_tma) for M in [4096] for use_tma in [True, False]]  # Matrix height
)
def bench_softmax(M, N, backend, use_tma, dtype=torch.float32, device=DEVICE):
    # Create data
    x = torch.randn(M, N, dtype=dtype, device=device)

    fn = lambda: tilegym.ops.softmax(x, use_tma=use_tma, backend=backend)
    ref = lambda: reference_softmax(x)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth (GB/s)
    # Softmax operation: reads input, writes output
    # Memory access: read x + write output = 2 * x.numel() elements
    total_bytes = 2 * x.numel() * x.element_size()

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


bench_softmax.run(print_data=True)
