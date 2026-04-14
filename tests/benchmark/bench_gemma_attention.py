# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

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


def reference_gemma_attention(
    q,
    k,
    v,
    scaling=None,
    window_size=0,
    soft_cap=None,
    is_causal=True,
    **kwargs,
):
    """Reference implementation using PyTorch einsum with soft cap support."""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    if num_heads_q != num_heads_kv:
        k = k.expand(-1, num_heads_q, -1, -1)
        v = v.expand(-1, num_heads_q, -1, -1)
    p = torch.einsum("bnid,bnjd->bnij", q.float(), k.float()) * scaling
    if soft_cap is not None:
        p = torch.tanh(p / soft_cap) * soft_cap
    if is_causal:
        seq = p.shape[-1]
        causal_mask = torch.triu(torch.ones(seq, seq, device=q.device, dtype=torch.bool), diagonal=1)
        p = p.masked_fill(causal_mask, float("-inf"))
    p = torch.softmax(p, dim=-1)
    return torch.einsum("bnij,bnjd->bnid", p, v.float()).to(q.dtype)


register_impl("gemma_attention", "torch")(reference_gemma_attention)


def create_benchmark_config(batch_size, num_heads, num_kv_heads, head_dim, soft_cap, dtype):
    """Create a benchmark configuration for gemma attention"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    cap_str = f"cap{soft_cap}" if soft_cap is not None else "nocap"

    return triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(9, 14)],  # 512 .. 8192
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="sequence length",
        ylabel="TFLOPS",
        plot_name=(
            f"gemma-attn-batch{batch_size}-h{num_heads}-kvh{num_kv_heads}"
            f"-d{head_dim}-{cap_str}-causal-{dtype_name}-TFLOPS"
        ),
        args={
            "batch_size": batch_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "soft_cap": soft_cap,
            "datatype": dtype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(batch_size, num_heads, num_kv_heads, head_dim, soft_cap, dtype)
        for batch_size in [2]
        for num_heads, num_kv_heads in [(8, 1)]
        for head_dim in [256]
        for soft_cap in [50.0]
        for dtype in [torch.bfloat16]
    ]
)
def bench_gemma_attention(
    batch_size,
    num_heads,
    num_kv_heads,
    head_dim,
    soft_cap,
    N_CTX,
    backend,
    datatype,
    device="cuda",
):
    scaling = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch_size, num_heads, N_CTX, head_dim, dtype=datatype, device=device)
    k = torch.randn(batch_size, num_kv_heads, N_CTX, head_dim, dtype=datatype, device=device)
    v = torch.randn(batch_size, num_kv_heads, N_CTX, head_dim, dtype=datatype, device=device)

    fn = lambda: tilegym.ops.gemma_attention(
        q, k, v, scaling=scaling, soft_cap=soft_cap, is_causal=True, backend=backend
    )

    if backend != "torch":
        ref = lambda: reference_gemma_attention(q, k, v, scaling=scaling, soft_cap=soft_cap, is_causal=True)
        torch.testing.assert_close(fn(), ref(), rtol=1e-2, atol=1e-2)

    ms = triton.testing.do_bench(fn)

    # FLOPs: 2 matmuls (QK^T and PV), causal halves the count
    flops = 2 * 2.0 * batch_size * num_heads * N_CTX * N_CTX * head_dim * 0.5
    return flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_gemma_attention.run(print_data=True)
