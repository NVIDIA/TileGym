#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import itertools

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

# Available backends for benchmarking
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def reference_bmm(a: torch.Tensor, b: torch.Tensor, transpose_a: bool = False, transpose_b: bool = False):
    """Reference implementation using PyTorch"""
    if transpose_a:
        a = torch.transpose(a, 1, 2)
    if transpose_b:
        b = torch.transpose(b, 1, 2)
    return torch.bmm(a, b)


register_impl("bmm", "torch")(reference_bmm)


def create_benchmark_config(batch_size, transpose_a, transpose_b, dtype):
    """Create a benchmark configuration for bmm"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, 13)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"bmm-batch{batch_size}-transpose_a{transpose_a}-transpose_b{transpose_b}-{dtype_name}-TFLOPS",
        args={
            "batch_size": batch_size,
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
            "datatype": dtype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(batch_size, ta, tb, dtype)
        for batch_size in [2, 8]
        for ta, tb in itertools.product([False, True], repeat=2)
        for dtype in [torch.float16]
    ]
)
def bench_bmm(
    M,
    N,
    K,
    batch_size,
    transpose_a,
    transpose_b,
    backend,
    datatype,
    device="cuda",
):
    # Create input tensors with proper shapes
    if transpose_a:
        a_shape = (batch_size, K, M)
    else:
        a_shape = (batch_size, M, K)

    if transpose_b:
        b_shape = (batch_size, N, K)
    else:
        b_shape = (batch_size, K, N)

    a = torch.rand(a_shape, device=device, dtype=datatype)
    b = torch.rand(b_shape, device=device, dtype=datatype)

    fn = lambda: tilegym.ops.bmm(a, b, transpose_a=transpose_a, transpose_b=transpose_b, backend=backend)

    # Run a light sanity check
    ref = lambda: reference_bmm(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    torch.testing.assert_close(fn(), ref(), rtol=1e-2, atol=1e-2)

    # Benchmark the function
    ms = triton.testing.do_bench(fn)

    # Calculate TFLOPS
    # BMM: 2 * batch_size * M * N * K FLOPs
    tflops = 2 * batch_size * M * N * K * 1e-12 / (ms * 1e-3)

    return tflops


if __name__ == "__main__":
    bench_bmm.run(print_data=True)
