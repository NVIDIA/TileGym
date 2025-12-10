# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def reference_mla_decoding(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: float = None,
):
    """Reference implementation using PyTorch"""
    if sm_scale is None:
        sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    qk = torch.matmul(q, kv.transpose(1, 2)).float()
    if kpe.numel() > 0:
        qk = qk + torch.matmul(qpe, kpe.transpose(1, 2)).float()
    qk = qk * sm_scale

    m = torch.max(qk, dim=-1)[0]
    p = torch.exp(qk - m.unsqueeze(-1))
    l = torch.sum(p, dim=-1)
    p = p / (l.unsqueeze(-1))
    o = torch.matmul(p.to(q.dtype), kv)
    return o.to(q.dtype)


register_impl("mla_decoding", "torch")(reference_mla_decoding)
register_impl("mla_decoding_split_kv", "torch")(reference_mla_decoding)


def create_benchmark_config(head_dim, use_split_kv, dtype):
    """Create a benchmark configuration for MLA decoding scenarios"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split('.')[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["kv_seq_len"],
        x_vals=[2**i for i in range(8, 15)] + [11043],  # KV cache length from 256 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mla-decoding-performance-{dtype_name}-head_dim={head_dim}-split_kv={use_split_kv}-GBps",
        args={
            "dtype": dtype,
            "batch_size": 1,
            "num_heads": 16,
            "head_dim": head_dim,
            "d_pe": 64,
            "use_split_kv": use_split_kv,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(head_dim, use_split_kv, dtype)
        for head_dim in [128, 512]
        for use_split_kv in [True]
        for dtype in [torch.float16]
    ]
)
def bench_mla_decoding(
    kv_seq_len,
    backend,
    dtype,
    batch_size,
    num_heads,
    head_dim,
    d_pe,
    use_split_kv,
    device=DEVICE,
):
    q = torch.empty(batch_size, num_heads, head_dim, device=device, dtype=dtype).normal_(
        mean=0.3, std=0.2
    )

    qpe = torch.empty(batch_size, num_heads, d_pe, device=device, dtype=dtype).normal_(
        mean=0.3, std=0.1
    )

    kv = torch.empty(batch_size, kv_seq_len, head_dim, device=device, dtype=dtype).normal_(
        mean=0.3, std=0.2
    )

    kpe = torch.empty(batch_size, kv_seq_len, d_pe, device=device, dtype=dtype).normal_(
        mean=0.3, std=0.1
    )

    scaling = 1.0 / math.sqrt(head_dim + d_pe)

    # Use unified dispatch system
    if use_split_kv:
        fn_raw = lambda: tilegym.ops.mla_decoding_split_kv(
            q, qpe, kv, kpe, scaling, backend=backend
        )
    else:
        fn_raw = lambda: tilegym.ops.mla_decoding(q, qpe, kv, kpe, scaling, backend=backend)

    fn = lambda: (lambda _o: _o[0] if isinstance(_o, tuple) else _o)(fn_raw())
    ref = lambda: reference_mla_decoding(q, qpe, kv, kpe, scaling)

    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth in GB/s
    bytes_per_element = q.element_size()

    q_bytes = q.numel() * bytes_per_element
    qpe_bytes = qpe.numel() * bytes_per_element
    kv_bytes = kv.numel() * bytes_per_element
    kpe_bytes = kpe.numel() * bytes_per_element
    output_bytes = q.numel() * bytes_per_element

    total_bytes = q_bytes + qpe_bytes + kv_bytes + kpe_bytes + output_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_mla_decoding.run(print_data=True)
