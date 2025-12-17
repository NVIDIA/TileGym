# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_mla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qpe: torch.Tensor,
    kpe: torch.Tensor,
    is_causal: bool,
    scaling: float = None,
):
    """Reference implementation of MLA using PyTorch"""
    qkv_dtype = v.dtype
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

    # Calculate attention scores: Q @ K^T
    qk = torch.matmul(q, k.transpose(2, 3))

    # Add positional encoding attention: QPE @ KPE^T
    if qpe is not None and kpe is not None:
        qk = qk + torch.matmul(qpe, kpe.transpose(2, 3))

    qk = qk.float() * scaling

    # Apply causal mask if needed
    if is_causal:
        seq_len = qk.shape[-2]
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=qk.device), diagonal=1)
            qk = qk.masked_fill(mask.bool(), float("-inf"))

    # Softmax attention
    attn_weights = torch.softmax(qk, dim=-1).to(qkv_dtype)

    # Apply attention to values: Attn @ V
    output = torch.matmul(attn_weights, v)
    return output


register_impl("mla", "torch")(reference_mla)


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
        x_names=["seq_len"],
        x_vals=[2**i for i in range(10, 14)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="FLOPS",
        plot_name=f"mla-performance-{dtype_name}-causal-TFLOPS",
        args={
            "dtype": dtype,
            "batch_size": 4,
            "num_heads": 16,
            "head_dim": 128,
            "d_pe": 64,
        },
    )


@triton.testing.perf_report([create_benchmark_config(dtype) for dtype in [torch.bfloat16]])
def bench_mla(
    seq_len,
    backend,
    dtype,
    batch_size,
    num_heads,
    head_dim,
    d_pe,
    device=DEVICE,
):
    # Create input tensors
    q = torch.empty(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    qpe = torch.empty(batch_size, num_heads, seq_len, d_pe, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    k = torch.empty(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    kpe = torch.empty(batch_size, 1, seq_len, d_pe, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    v = torch.empty(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    # Calculate scaling
    scaling = 1.0 / math.sqrt(head_dim + d_pe)

    # Default to causal attention
    is_causal = True

    # Use unified dispatch system
    fn = lambda: tilegym.ops.mla(q, k, v, qpe, kpe, is_causal, scaling, backend=backend)
    ref = lambda: reference_mla(q, k, v, qpe, kpe, is_causal, scaling)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Main attention computation FLOPs
    flops_per_matmul = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    flops_pe = 2 * batch_size * num_heads * seq_len * seq_len * d_pe
    total_flops = 2 * flops_per_matmul + flops_pe

    # Convert to TFLOPS/s
    tflops_per_s = total_flops * 1e-12 / (ms * 1e-3)

    return tflops_per_s


if __name__ == "__main__":
    bench_mla.run(print_data=True)
