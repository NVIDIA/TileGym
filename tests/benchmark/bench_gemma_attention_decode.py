# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ("tilecpp", "TileCpp", ("purple", "-")) if is_backend_available("tilecpp") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def reference_gemma_attention_decode(
    q,
    k,
    v,
    scaling=None,
    window_size=0,
    soft_cap=None,
    **kwargs,
):
    """Reference implementation using PyTorch einsum with soft cap and sliding window support."""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])
    dtype = q.dtype
    batch, num_heads_q = q.shape[0], q.shape[1]
    num_heads_kv = k.shape[1]
    seq_len = k.shape[2]

    if num_heads_q != num_heads_kv and num_heads_kv != 1:
        num_head_groups = num_heads_q // num_heads_kv
        k = torch.repeat_interleave(k, num_head_groups, dim=1)
        v = torch.repeat_interleave(v, num_head_groups, dim=1)

    p = torch.einsum("bnid,bnjd->bnij", q, k) * scaling

    if soft_cap is not None:
        p = torch.tanh(p / soft_cap) * soft_cap

    if window_size > 0:
        kv_positions = torch.arange(seq_len, device=q.device)
        query_pos = seq_len - 1
        mask = kv_positions < (query_pos - window_size)
        mask = mask.view(1, 1, 1, seq_len).expand(batch, num_heads_q, 1, -1)
        p = p.masked_fill(mask, torch.finfo(p.dtype).min)

    p = torch.softmax(p, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.einsum("bnij,bnjd->bnid", p, v).to(dtype)


register_impl("gemma_attention_decode", "torch")(reference_gemma_attention_decode)


def create_benchmark_config(batch_size, num_heads, num_kv_heads, head_dim, window_size, soft_cap, dtype):
    """Create a benchmark configuration for gemma attention decode"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    cap_str = f"cap{soft_cap}" if soft_cap is not None else "nocap"
    win_str = f"win{window_size}" if window_size > 0 else "nowin"

    return triton.testing.Benchmark(
        x_names=["kv_seq_len"],
        x_vals=[2**i for i in range(8, 15)] + [10019],  # KV cache length from 256 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="KV sequence length",
        ylabel="GB/s",
        plot_name=(
            f"gemma-attn-decode-batch{batch_size}-h{num_heads}-kvh{num_kv_heads}"
            f"-d{head_dim}-{cap_str}-{win_str}-{dtype_name}-GBps"
        ),
        args={
            "batch_size": batch_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "window_size": window_size,
            "soft_cap": soft_cap,
            "datatype": dtype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(batch_size, num_heads, num_kv_heads, head_dim, window_size, soft_cap, dtype)
        for batch_size in [1]
        for num_heads, num_kv_heads in [(16, 8)]
        for head_dim in [128]
        for window_size, soft_cap in [(0, None), (0, 50.0), (4096, 50.0)]
        for dtype in [torch.bfloat16]
    ]
)
def bench_gemma_attention_decode(
    batch_size,
    num_heads,
    num_kv_heads,
    head_dim,
    window_size,
    soft_cap,
    kv_seq_len,
    backend,
    datatype,
    device="cuda",
):
    scaling = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=datatype, device=device)
    k = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim, dtype=datatype, device=device)
    v = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim, dtype=datatype, device=device)

    fn = lambda: tilegym.ops.gemma_attention_decode(
        q, k, v, scaling=scaling, window_size=window_size, soft_cap=soft_cap, backend=backend
    )

    if backend != "torch":
        ref = lambda: reference_gemma_attention_decode(
            q, k, v, scaling=scaling, window_size=window_size, soft_cap=soft_cap
        )
        torch.testing.assert_close(fn(), ref(), rtol=1e-2, atol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth in GB/s
    bytes_per_element = q.element_size()
    q_bytes = q.numel() * bytes_per_element
    k_bytes = k.numel() * bytes_per_element
    v_bytes = v.numel() * bytes_per_element
    output_bytes = q.numel() * bytes_per_element

    total_bytes = q_bytes + k_bytes + v_bytes + output_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_gemma_attention_decode.run(print_data=True)
