# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_rms_norm(
    input: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor = None,  # Unused - kept for interface compatibility
    static_persistent: bool = False,  # Unused - kept for interface compatibility
):
    """Reference implementation of RMSNorm"""
    if bias is not None:
        raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # Convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


register_impl("rms_norm", "torch")(reference_rms_norm)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype, static_persistent=True):
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
        plot_name=f"rmsnorm-performance-{dtype_name}-persistent-{static_persistent}-GBps",
        args={
            "dtype": dtype,
            "static_persistent": static_persistent,
            "M": 4096,
        },  # Fixed batch*seq_len
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(dtype, static_persistent)
        for dtype in [torch.float16, torch.bfloat16]
        for static_persistent in [True, False]
    ]
)
def bench_rmsnorm(N, backend, dtype, static_persistent, M, device=DEVICE):
    eps = 1e-5

    # Create input tensors
    x_shape = (M, N)
    w_shape = (N,)

    x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

    fn = lambda: tilegym.ops.rms_norm(x, w_shape, weight, eps, static_persistent=static_persistent, backend=backend)
    ref = lambda: reference_rms_norm(x, w_shape, weight, eps)
    torch.testing.assert_close(fn(), ref(), atol=5e-2, rtol=0.0)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth (GB/s)
    # RMSNorm operation: read input, read weight, write output
    # Memory access: read x + read weight + write output
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read input
    weight_bytes = weight.numel() * bytes_per_element  # Read weight
    output_bytes = x.numel() * bytes_per_element  # Write output

    total_bytes = input_bytes + weight_bytes + output_bytes

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rmsnorm.run(print_data=True)
