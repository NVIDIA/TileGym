#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    if datatype == torch.float8_e5m2:
        return [p for p in ALL_BACKENDS if p is not None and p[0] != "torch"]
    else:
        return [p for p in ALL_BACKENDS if p is not None]


def reference_group_gemm(group_A: list, group_B: list, transpose_b: bool = False):
    """Reference implementation using PyTorch"""
    group_C = []
    for i in range(len(group_A)):
        A = group_A[i]
        B = group_B[i]
        if transpose_b:
            B = B.transpose(-2, -1)
        C = torch.matmul(A, B)
        group_C.append(C)
    return group_C


register_impl("group_gemm", "torch")(reference_group_gemm)


def create_benchmark_config(datatype, num_groups, transpose_b):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, 13)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"group-gemm-num_groups{num_groups}-transpose{transpose_b}-{dtype_name}-TFLOPS",
        args={
            "num_groups": num_groups,
            "transpose_b": transpose_b,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, num_groups, transpose_b)
        for datatype in [torch.float16, torch.float8_e5m2]
        for num_groups in [4, 16]
        for transpose_b in [False, True]
    ]
)
def bench_group_gemm(
    M,
    N,
    K,
    num_groups,
    transpose_b,
    backend,
    datatype,
    device="cuda",
):
    # Create input tensors
    group_A = []
    group_B = []

    for i in range(num_groups):
        A = torch.rand((M, K), device=device, dtype=torch.half).normal_(std=0.3).to(datatype)
        B_shape = (N, K) if transpose_b else (K, N)
        B = torch.rand(B_shape, device=device, dtype=torch.half).normal_(std=0.3).to(datatype)

        group_A.append(A)
        group_B.append(B)

    fn = lambda: tilegym.ops.group_gemm(group_A, group_B, transpose_b=transpose_b, backend=backend)

    if datatype != torch.float8_e5m2:
        # Verify correctness for non-FP8 types because torch doesn't support FP8 matmul
        ref = lambda: reference_group_gemm(group_A, group_B, transpose_b=transpose_b)
        result = fn()
        ref_result = ref()
        for i in range(len(result)):
            torch.testing.assert_close(result[i], ref_result[i], atol=1e-2, rtol=1e-2)

    # Calculate theoretical TFLOPS
    # GEMM operation: C = A @ B
    # For each matrix: 2 * M * N * K FLOPs (multiply-add operations)
    total_flops = num_groups * 2 * M * N * K

    ms = triton.testing.do_bench(fn)

    # Calculate TFLOPS
    tflops = total_flops / (ms * 1e-3) / 1e12

    return tflops


if __name__ == "__main__":
    bench_group_gemm.run(print_data=True)
