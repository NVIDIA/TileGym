# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,  # Unused - kept for interface compatibility
    unsqueeze_dim: int = 1,
    use_tma: bool = False,  # Unused - kept for interface compatibility
):
    """torch implementation of RoPE (Rotary Position Embedding)."""
    # cos and sin have shape (bsz, seq_len, head_dim)
    # q and k have shape (bsz, num_heads, seq_len, head_dim)

    # Add head dimension to cos and sin: (bsz, seq_len, head_dim) -> (bsz, 1, seq_len, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply RoPE using the rotate_half function
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


register_impl("apply_rope_base", "torch")(apply_rope_torch)


def create_rotary_embeddings(seq_len, head_dim, dtype, device, base=10000.0):
    """Create cos and sin tensors for rotary embeddings."""
    # Create frequency tensor
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))

    # Create position tensor
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute outer product to get all position-frequency combinations
    freqs = torch.outer(t, freqs)

    # Compute cos and sin (shape: seq_len, head_dim//2)
    cos_half = torch.cos(freqs).to(dtype)
    sin_half = torch.sin(freqs).to(dtype)

    # Repeat to create full head_dim: (seq_len, head_dim//2) -> (seq_len, head_dim)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)

    return cos, sin


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    if datatype == torch.float8_e5m2:
        return ALL_BACKENDS[:-1]  # Torch cannot support FP8
    else:
        return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype, BSZ, NUM_HEADS, HEAD_DIM):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[2**i for i in range(12, 16)],  # 4096 to 32768
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rope-benchmark-bsz{BSZ}-heads{NUM_HEADS}-d{HEAD_DIM}-{dtype_name}-GBps",
        args={
            "BSZ": BSZ,
            "NUM_HEADS": NUM_HEADS,
            "HEAD_DIM": HEAD_DIM,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, BSZ, NUM_HEADS, HEAD_DIM)
        for datatype in [torch.float16]
        for BSZ in [1]
        for NUM_HEADS in [16]
        for HEAD_DIM in [64]
    ]
)
def bench_rope(
    BSZ,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    backend,
    datatype,
    device=DEVICE,
):
    dtype = datatype
    q = torch.randn(
        (BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    k = torch.randn(
        (BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    # Create position ids
    pos_ids = torch.arange(SEQ_LEN, device=device, dtype=torch.long).unsqueeze(0)
    pos_ids = pos_ids.expand(BSZ, -1)

    # Create rotary embeddings
    cos, sin = create_rotary_embeddings(
        SEQ_LEN,
        HEAD_DIM,
        dtype if dtype != torch.float8_e5m2 else torch.float16,
        device,
    )
    cos = cos.unsqueeze(0).expand(BSZ, -1, -1)
    sin = sin.unsqueeze(0).expand(BSZ, -1, -1)

    fn = lambda: tilegym.ops.apply_rope_base(
        q.clone(), k.clone(), cos, sin, pos_ids, backend=backend
    )  # Use clone because of in-place modification
    ref = lambda: apply_rope_torch(q, k, cos, sin, pos_ids)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate memory bandwidth
    # Total memory: read q, k, cos, sin + write q_out, k_out
    bytes_per_element = q.element_size()
    q_bytes = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM * bytes_per_element
    k_bytes = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM * bytes_per_element
    cos_sin_bytes = 2 * BSZ * SEQ_LEN * HEAD_DIM * bytes_per_element
    total_bytes = 2 * q_bytes + 2 * k_bytes + cos_sin_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rope.run(print_data=True)
