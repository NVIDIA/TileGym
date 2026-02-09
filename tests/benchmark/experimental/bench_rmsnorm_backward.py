# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Standalone RMSNorm backward benchmark.

This benchmark tests the backward pass in isolation WITHOUT using autograd.
Both implementations receive the same pre-computed rstd values, ensuring
a true comparison of just the backward computation.
"""

import torch
import triton

from tilegym.backend import is_backend_available
from tilegym.ops.cutile.rms_norm import TileRMSNorm
from tilegym.ops.cutile.rms_norm import rms_norm_backward

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# CuTile backward - imported from the actual implementation
rms_norm_backward_cutile = rms_norm_backward

# Torch backward - static method on TileRMSNorm class
rms_norm_backward_torch = TileRMSNorm.rms_norm_backward_torch


# Backend dispatch
BACKWARD_FUNCTIONS = {
    "cutile": rms_norm_backward_cutile,
    "torch": rms_norm_backward_torch,
}

# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype):
    """Create a benchmark configuration for given parameters"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # Hidden size from 1024 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rmsnorm-backward-standalone-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "M": 4096,
        },  # Fixed batch*seq_len
    )


@triton.testing.perf_report(
    [create_benchmark_config(dtype) for dtype in [torch.bfloat16, torch.float16, torch.float32]]
)
def bench_rmsnorm_backward(N, backend, dtype, M, device=DEVICE):
    eps = 1e-5

    # Create input tensors (no autograd needed!)
    x_shape = (M, N)
    w_shape = (N,)

    x = torch.rand(x_shape, dtype=dtype, device=device).mul_(0.5).add_(-2.3)
    weight = torch.randn(w_shape, dtype=dtype, device=device)
    dy = torch.randn(x_shape, dtype=dtype, device=device)

    # Pre-compute rstd (simulating what forward pass would save)
    rstd = TileRMSNorm.compute_rstd_torch(x, eps)

    # Get the backward function for this backend
    backward_fn = BACKWARD_FUNCTIONS[backend]

    # Create the benchmark function
    def run_backward():
        return backward_fn(x, dy, weight, rstd)

    # Compute reference for correctness check
    dx_ref, dw_ref = rms_norm_backward_torch(x, dy, weight, rstd)

    # Run once to verify correctness
    dx, dw = run_backward()
    torch.testing.assert_close(dx, dx_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dw, dw_ref, atol=1e-2, rtol=1e-2)

    # Benchmark ONLY the backward pass (no forward, no autograd overhead)
    ms = triton.testing.do_bench_cudagraph(run_backward)

    # Calculate memory bandwidth (GB/s)
    # RMSNorm backward: read x, read dy, read weight, read rstd, write dx, write dw
    bytes_per_element = x.element_size()

    input_x_bytes = x.numel() * bytes_per_element  # Read input x
    dy_bytes = dy.numel() * bytes_per_element  # Read dy
    weight_bytes = weight.numel() * bytes_per_element  # Read weight
    rstd_bytes = rstd.numel() * 4  # Read rstd (always float32)
    dx_bytes = x.numel() * bytes_per_element  # Write dx
    dw_bytes = weight.numel() * bytes_per_element  # Write dw

    temp_buffer_bytes = x.numel() * 4 * 2  # always write + read float32

    total_bytes = input_x_bytes + dy_bytes + weight_bytes + rstd_bytes + dx_bytes + dw_bytes + temp_buffer_bytes

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rmsnorm_backward.run(print_data=True)
