# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# Examples of mixing triton and cutile kernels in a single program
# It's a pattern from Llama model: hidden_states = LlamaRMSNorm(residual + hidden_states)
# Here we provide an example of mixing triton vector add kernel and cutile rmsnorm kernel


# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py
@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_vector_add(x, y):
    """Launch triton vector add kernel"""
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def reference_rmsnorm(input, normalized_shape, weight, eps):
    """Reference implementation of RMSNorm"""
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


def reference_add_rmsnorm(x, y, normalized_shape, weight, eps):
    """Reference implementation: add then rmsnorm"""
    added = x + y
    return reference_rmsnorm(added, normalized_shape, weight, eps)


# Available backends
ALL_BACKENDS = [
    ("triton_cutile", "Triton+CuTile", ("red", "-")) if is_backend_available("cutile") else None,
    ("pytorch", "PyTorch", ("green", "-")),
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
    dtype_name = str(dtype).split('.')[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # Hidden size from 1024 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mix-triton-cutile-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "M": 4096,
        },
    )


@triton.testing.perf_report(
    [create_benchmark_config(dtype) for dtype in [torch.float16, torch.bfloat16]]
)
def bench_mix_triton_cutile(N, backend, dtype, M, device=DEVICE):
    eps = 1e-5

    # Create input tensors
    x_shape = (M, N)
    w_shape = (N,)

    x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
    y = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-1.5)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

    ref_implementation = lambda: reference_add_rmsnorm(x, y, w_shape, weight, eps)

    # Setup based on backend
    if backend == "pytorch":
        fn = ref_implementation

    elif backend == "triton_cutile":

        def mixed_fn():
            # Step 1: Triton vector add
            added = triton_vector_add(x, y)
            # Step 2: CuTile rmsnorm
            return tilegym.ops.cutile.rms_norm(added, w_shape, weight, eps, static_persistent=True)

        fn = mixed_fn

        # Validate against reference implementation
        torch.testing.assert_close(fn(), ref_implementation(), atol=5e-2, rtol=0.0)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth (GB/s)
    # Operations: read x, read y, write intermediate, read intermediate, read weight, write output
    # Simplified: read x + read y + read weight + write output
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read x
    input2_bytes = y.numel() * bytes_per_element  # Read y
    weight_bytes = weight.numel() * bytes_per_element  # Read weight
    output_bytes = x.numel() * bytes_per_element  # Write output

    total_bytes = input_bytes + input2_bytes + weight_bytes + output_bytes

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_mix_triton_cutile.run(print_data=True)
