# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_persistent_layer_norm(input, normalized_shape, weight, bias, eps, mean=None, rstd=None, **kwargs):
    """Reference implementation for persistent layer norm using PyTorch"""
    y = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    return y, None, None, None, None


# Register torch reference implementation
register_impl("persistent_layer_norm", "torch")(reference_persistent_layer_norm)
register_impl("layer_norm_legacy", "torch")(reference_persistent_layer_norm)


# Available backends for benchmarking
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_persistent_benchmark_config(num_rows, dtype):
    """Create benchmark config for persistent layer norm test"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"persistent-layer-norm-M{num_rows}-{dtype_name}-GBps",
        args={
            "M": num_rows,
            "dtype": dtype,
        },
    )


def create_legacy_benchmark_config(num_rows, dtype):
    """Create benchmark config for legacy layer norm test"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"legacy-layer-norm-M{num_rows}-{dtype_name}-GBps",
        args={
            "M": num_rows,
            "dtype": dtype,
        },
    )


_persistent_num_rows = [30000, 128000]
_persistent_dtype = torch.bfloat16

_legacy_num_rows = [4096]
_legacy_dtype = torch.bfloat16


@triton.testing.perf_report([create_persistent_benchmark_config(M, _persistent_dtype) for M in _persistent_num_rows])
def bench_persistent_layer_norm(
    M,
    N,
    dtype,
    backend,
    device=DEVICE,
):
    """Benchmark for persistent layer norm: (M, N) where M=num_rows, N=feature_dim"""
    eps = 1e-6

    x = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    weight = torch.randn((N,), dtype=dtype, device=device, requires_grad=False)
    bias = torch.randn((N,), dtype=dtype, device=device, requires_grad=False)

    fn = lambda: tilegym.ops.persistent_layer_norm(
        input=x,
        normalized_shape=(N,),
        weight=weight,
        bias=bias,
        eps=eps,
        backend=backend,
    )
    ref = lambda: reference_persistent_layer_norm(x, (N,), weight, bias, eps)

    # Verify correctness
    fn_result = fn()[0]  # Get output tensor from tuple
    ref_result = ref()[0]
    torch.testing.assert_close(fn_result, ref_result, atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench(fn)

    # Calculate bandwidth: (input + output + weight + bias) / time
    num_bytes = x.numel() * x.element_size()  # input
    num_bytes += x.numel() * x.element_size()  # output
    num_bytes += weight.numel() * weight.element_size()  # weight
    num_bytes += bias.numel() * bias.element_size()  # bias

    gb_per_s = num_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


@triton.testing.perf_report([create_legacy_benchmark_config(M, _legacy_dtype) for M in _legacy_num_rows])
def bench_legacy_layer_norm(
    M,
    N,
    dtype,
    backend,
    device=DEVICE,
):
    """Benchmark for legacy layer norm: (M, N) where M=num_rows, N=feature_dim"""
    eps = 1e-6

    x = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    weight = torch.randn((N,), dtype=dtype, device=device, requires_grad=False)
    bias = torch.randn((N,), dtype=dtype, device=device, requires_grad=False)

    fn = lambda: tilegym.ops.layer_norm_legacy(
        input=x,
        normalized_shape=(N,),
        weight=weight,
        bias=bias,
        eps=eps,
        backend=backend,
    )
    ref = lambda: torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)

    # Verify correctness
    fn_result = fn()
    # Handle both tuple and tensor returns
    if isinstance(fn_result, tuple):
        fn_result = fn_result[0]
    ref_result = ref()
    torch.testing.assert_close(fn_result, ref_result, atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench(fn)

    # Calculate bandwidth: (input + output + weight + bias) / time
    num_bytes = x.numel() * x.element_size()  # input
    num_bytes += x.numel() * x.element_size()  # output
    num_bytes += weight.numel() * weight.element_size()  # weight
    num_bytes += bias.numel() * bias.element_size()  # bias

    gb_per_s = num_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_persistent_layer_norm.run(print_data=True)
    bench_legacy_layer_norm.run(print_data=True)
