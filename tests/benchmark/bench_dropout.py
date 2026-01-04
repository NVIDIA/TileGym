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

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)  # For reproducibility


def reference_dropout(
    x: torch.Tensor,
    seed: int,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    **kwargs,
):
    """Reference implementation of dropout using PyTorch"""
    # Seed is accepted for interface compatibility but not used by PyTorch's dropout.
    return torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)


register_impl("dropout", "torch")(reference_dropout)


# Available backends for benchmarking
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype, p: float):
    """Create a benchmark configuration for dropout"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(20, 28, 2)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"dropout-p{p}-{dtype_name}-GBps",
        args={
            "p": p,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [create_benchmark_config(datatype, p) for datatype in [torch.float16, torch.float32] for p in [0.5]]
)
def bench_dropout(
    M,
    backend,
    p,
    datatype,
    device=DEVICE,
):
    seed = torch.random.initial_seed()

    # Create input tensor
    x = torch.rand(M, device=device, dtype=datatype, requires_grad=False)

    training = True
    inplace = False

    fn = lambda: tilegym.ops.dropout(x, seed, p, training, inplace, backend=backend)

    # Run a light sanity check
    out = fn()
    zero_ratio = 1 - torch.count_nonzero(out) / torch.numel(out)
    threshold = 0.04
    assert p - threshold < zero_ratio < p + threshold, f"Unexpected dropout ratio {zero_ratio} for p={p}"

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth
    # Dropout forward pass: read input + write output
    num_elements = x.numel()
    bytes_per_element = x.element_size()
    total_bytes = 2 * num_elements * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_dropout.run(print_data=True)
