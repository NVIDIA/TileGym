#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark comparing SwiGLU forward and backward pass performance.
"""

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available


def reference_swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.nn.functional.silu(a) * b


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
        plot_name=f"swiglu-{mode_name}-hidden{hidden_size}-{dtype_name}",
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
        for hidden_size in [2048, 4096]
    ]
)
def bench_swiglu(
    M,
    hidden_size,
    backend,
    dtype,
    mode,
    device="cuda",
):
    # Create input tensors
    a = torch.randn(M, hidden_size, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(M, hidden_size, dtype=dtype, device=device, requires_grad=True)

    if backend == "cutile":
        tilegym.set_backend("cutile")
        from tilegym.ops.cutile.swiglu import SiLUMulFunction

        def fwd():
            return SiLUMulFunction.apply(a, b)
    else:
        # PyTorch reference - direct function call

        def fwd():
            return reference_swiglu_forward(a, b)

    # Calculate memory bytes
    bytes_per_element = a.element_size()
    # Forward: read a, b, write c -> 3 tensors of size (M, hidden_size)
    fwd_bytes = 3 * M * hidden_size * bytes_per_element
    # Backward: read dc, a, b, write da, db -> 5 tensors
    bwd_bytes = 5 * M * hidden_size * bytes_per_element

    if mode == "forward":
        total_bytes = fwd_bytes
        ms = triton.testing.do_bench(fwd, rep=10)
    elif mode == "backward":
        c = fwd()
        dc = torch.randn_like(c)
        total_bytes = bwd_bytes
        ms = triton.testing.do_bench(lambda: c.backward(dc, retain_graph=True), rep=10)
    else:  # full
        dc = torch.randn(M, hidden_size, dtype=dtype, device=device)
        total_bytes = fwd_bytes + bwd_bytes

        def full():
            c = fwd()
            c.backward(dc, retain_graph=True)

        ms = triton.testing.do_bench(full, rep=10)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_swiglu.run(print_data=True)
