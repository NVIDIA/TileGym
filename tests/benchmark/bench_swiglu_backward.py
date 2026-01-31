#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark for SwiGLU forward and backward pass.
Compares TileGym cutile implementation against PyTorch reference.
"""

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available


def reference_swiglu_forward(a, b):
    """Reference SwiGLU forward using PyTorch."""
    return torch.nn.functional.silu(a) * b


def reference_swiglu_backward(dc, a, b):
    """Reference SwiGLU backward using autograd."""
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    c = reference_swiglu_forward(a, b)
    c.backward(dc)
    return a.grad, b.grad


# Available backends for benchmarking
def get_supported_backends():
    backends = [("torch", "PyTorch", ("green", "-"))]
    if is_backend_available("cutile"):
        backends.insert(0, ("cutile", "CuTile", ("orange", "-")))
    return backends


def create_benchmark_config(mode, hidden_size, dtype):
    """Create a benchmark configuration."""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    mode_name = mode.replace("_", "-")

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="ms",
        plot_name=f"swiglu-{mode_name}-hidden{hidden_size}-{dtype_name}-latency",
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

    if mode == "forward":
        ms = triton.testing.do_bench(fwd, rep=10)
    elif mode == "backward":
        c = fwd()
        dc = torch.randn_like(c)
        ms = triton.testing.do_bench(lambda: c.backward(dc, retain_graph=True), rep=10)
    else:  # full
        dc = torch.randn(M, hidden_size, dtype=dtype, device=device)

        def full():
            c = fwd()
            c.backward(dc, retain_graph=True)

        ms = triton.testing.do_bench(full, rep=10)

    return ms


if __name__ == "__main__":
    bench_swiglu.run(print_data=True)
