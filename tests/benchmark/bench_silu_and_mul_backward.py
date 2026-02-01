#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark for silu_and_mul forward and backward pass.
This tests the concatenated-input version used by PartiallyFusedSwiGLUMLP.
"""

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available


def reference_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    hidden_size = input.shape[-1] // 2
    x1 = input[..., :hidden_size]
    x2 = input[..., hidden_size:]
    return torch.nn.functional.silu(x1) * x2


def get_providers():
    """Get available providers for benchmarking."""
    providers = [("torch", "PyTorch", ("green", "-"))]
    if is_backend_available("cutile"):
        providers.insert(0, ("cutile", "CuTile", ("orange", "-")))
    return providers


def create_benchmark_config(mode, hidden_size, dtype):
    """Create a benchmark configuration."""
    providers = get_providers()
    if not providers:
        return None

    backends, names, styles = zip(*providers)
    dtype_name = str(dtype).split(".")[-1]
    mode_name = mode.replace("_", "-")

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(10, 15)],  # 1K to 16K
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"silu-and-mul-{mode_name}-hidden{hidden_size}-{dtype_name}",
        args={
            "hidden_size": hidden_size,
            "dtype": dtype,
            "mode": mode,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(mode, hidden_size, dtype)
        for mode in ["forward", "backward", "full"]
        for dtype in [torch.float32, torch.bfloat16]
        for hidden_size in [2048, 4096, 11008]  # Common LLaMA sizes
    ]
)
def bench_silu_and_mul(
    M,
    hidden_size,
    backend,
    dtype,
    mode,
    device="cuda",
):
    # Create input tensor (concatenated format)
    input_shape = (M, 2 * hidden_size)
    x = torch.randn(input_shape, dtype=dtype, device=device, requires_grad=True)

    if backend == "cutile":
        tilegym.set_backend("cutile")
        fn = lambda: tilegym.ops.silu_and_mul(x)
    else:
        fn = lambda: reference_silu_and_mul(x)

    # Calculate memory bytes
    bytes_per_element = x.element_size()
    input_bytes = x.numel() * bytes_per_element  # Read: (M, 2*hidden_size)
    output_bytes = M * hidden_size * bytes_per_element  # Write: (M, hidden_size)

    if mode == "forward":
        total_bytes = input_bytes + output_bytes
        ms = triton.testing.do_bench(fn, rep=10)
    elif mode == "backward":
        y = fn()
        dy = torch.randn_like(y)
        # Backward reads: dy, recomputes from input, writes grad_input
        total_bytes = output_bytes + input_bytes + input_bytes  # dy + input + grad_input
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), rep=10)
    else:  # full
        dy = torch.randn(M, hidden_size, dtype=dtype, device=device)
        # Forward + backward
        total_bytes = (input_bytes + output_bytes) + (output_bytes + input_bytes + input_bytes)

        def full():
            y = fn()
            y.backward(dy, retain_graph=True)

        ms = triton.testing.do_bench(full, rep=10)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_silu_and_mul.run(print_data=True)
