# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_swiglu(a: torch.Tensor, b: torch.Tensor):
    """Reference implementation of SwiGLU using vanilla PyTorch"""
    return F.silu(a) * b


def get_reference_swiglu():
    """Returns reference swiglu function for dispatch system"""
    return reference_swiglu


register_impl("get_swiglu", "torch")(get_reference_swiglu)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(batch_size, M, dtype):
    """Create a benchmark configuration for given parameters"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)

    dtype_name = str(dtype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # 1024 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"swiglu-batch{batch_size}-M{M}-{dtype_name}-GBps",
        args={
            "batch_size": batch_size,
            "M": M,
            "dtype": dtype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(batch_size, M, dtype)
        for batch_size in [1, 4, 8]  # Different batch sizes
        for M in [128, 4096]  # Different rows
        for dtype in [torch.float16, torch.bfloat16, torch.float32]
    ]
)
def bench_swiglu(
    batch_size,
    M,
    N,
    backend,
    dtype,
    device=DEVICE,
):
    # Generate input data: two tensors for SwiGLU operation
    a = torch.randn(batch_size, M, N, device=device, dtype=dtype)
    b = torch.randn(batch_size, M, N, device=device, dtype=dtype)

    # Use unified dispatch system
    fn = lambda: tilegym.ops.get_swiglu(backend=backend)(a, b)
    if dtype is torch.float32:
        ref = lambda: reference_swiglu(a, b)
        atol, rtol = 1e-5, 1e-5
    else:
        ref = lambda: reference_swiglu(a, b)
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(fn(), ref(), atol=atol, rtol=rtol)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth (GB/s)
    # SwiGLU operation: F.silu(a) * b
    # Memory access: read a, read b, write output

    elements_per_tensor = batch_size * M * N
    bytes_per_element = a.element_size()

    # Total memory access: 2 reads + 1 write = 3 tensor accesses
    total_bytes = 3 * elements_per_tensor * bytes_per_element

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_swiglu.run(print_data=True)
