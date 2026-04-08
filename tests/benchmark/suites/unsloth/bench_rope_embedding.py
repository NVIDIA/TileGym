# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: RoPE Embedding QK joint, forward + backward (standalone)."""

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import rope_embedding_qk

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch references
# ---------------------------------------------------------------------------


def reference_qk(Q, K, cos, sin):
    cos = cos.squeeze()
    sin = sin.squeeze()
    batch, n_heads_Q, seq_len, head_dim = Q.shape
    half = head_dim // 2
    cos_exp = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)
    sin_exp = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)

    Q0, Q1 = Q[..., :half], Q[..., half:]
    Q_out = torch.cat([Q0 * cos_exp - Q1 * sin_exp, Q1 * cos_exp + Q0 * sin_exp], dim=-1)
    K0, K1 = K[..., :half], K[..., half:]
    K_out = torch.cat([K0 * cos_exp - K1 * sin_exp, K1 * cos_exp + K0 * sin_exp], dim=-1)
    return Q_out.to(Q.dtype), K_out.to(K.dtype)


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("pytorch", "PyTorch", ("green", "-")),
]


def _supported():
    return [b for b in ALL_BACKENDS if b is not None]


def _cfg(plot_name, x_name, x_vals, **extra):
    avail = _supported()
    if not avail:
        return None
    backends, names, styles = zip(*avail)
    return triton.testing.Benchmark(
        x_names=[x_name],
        x_vals=x_vals,
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=plot_name,
        args=extra,
    )


# ---------------------------------------------------------------------------
# QK Joint RoPE forward: sweep seq_len
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg(
            "rope-qk-fwd-B2-HQ8-HK8-D64-bf16-GBps",
            "seq_len",
            [64, 128, 256, 512, 1024],
            batch=2,
            n_heads_Q=8,
            n_heads_K=8,
            head_dim=64,
        ),
    ]
)
def bench_rope_qk_fwd(batch, n_heads_Q, n_heads_K, seq_len, head_dim, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    Q = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=device)
    half = head_dim // 2
    cos = torch.randn(seq_len, half, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference_qk(Q.clone(), K.clone(), cos, sin)
    else:
        tilegym.set_backend(backend)
        fn = lambda: rope_embedding_qk(Q.clone(), K.clone(), cos, sin)

    # Correctness check
    ref = lambda: reference_qk(Q.clone(), K.clone(), cos, sin)
    fn_out = fn()
    ref_out = ref()
    torch.testing.assert_close(fn_out[0], ref_out[0], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(fn_out[1], ref_out[1], atol=5e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read Q + read K + read cos + read sin + write Q_out + write K_out
    bytes_per_element = Q.element_size()
    q_bytes = batch * n_heads_Q * seq_len * head_dim * bytes_per_element
    k_bytes = batch * n_heads_K * seq_len * head_dim * bytes_per_element
    cos_sin_bytes = 2 * seq_len * half * bytes_per_element
    total_bytes = 2 * q_bytes + 2 * k_bytes + cos_sin_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# QK Joint RoPE backward: sweep seq_len
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg(
            "rope-qk-bwd-B2-HQ8-HK8-D64-bf16-GBps",
            "seq_len",
            [64, 128, 256, 512, 1024],
            batch=2,
            n_heads_Q=8,
            n_heads_K=8,
            head_dim=64,
        ),
    ]
)
def bench_rope_qk_bwd(batch, n_heads_Q, n_heads_K, seq_len, head_dim, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    half = head_dim // 2
    cos = torch.randn(seq_len, half, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half, dtype=dtype, device=device)
    dQ_out = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=device)
    dK_out = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=device)

    if backend == "pytorch":
        Q = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        Q_out, K_out = reference_qk(Q, K, cos, sin)

        def fn():
            Q.grad = None
            K.grad = None
            torch.autograd.backward([Q_out, K_out], [dQ_out, dK_out], retain_graph=True)
    else:
        tilegym.set_backend(backend)
        Q = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        Q_out, K_out = rope_embedding_qk(Q, K, cos, sin)

        def fn():
            Q.grad = None
            K.grad = None
            torch.autograd.backward([Q_out, K_out], [dQ_out, dK_out], retain_graph=True)

    ms = triton.testing.do_bench(fn)

    # Memory: read dQ_out + read dK_out + read cos + read sin + write dQ + write dK
    bytes_per_element = dQ_out.element_size()
    q_bytes = batch * n_heads_Q * seq_len * head_dim * bytes_per_element
    k_bytes = batch * n_heads_K * seq_len * head_dim * bytes_per_element
    cos_sin_bytes = 2 * seq_len * half * bytes_per_element
    total_bytes = 2 * q_bytes + 2 * k_bytes + cos_sin_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_rope_qk_fwd.run(print_data=True)
    bench_rope_qk_bwd.run(print_data=True)
