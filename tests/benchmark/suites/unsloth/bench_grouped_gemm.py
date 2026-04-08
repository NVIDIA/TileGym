# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: Grouped GEMM (MoE) forward + backward (standalone)."""

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import grouped_gemm

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend, **kwargs):
    """Run grouped_gemm with the correct backend-specific kwargs."""
    return grouped_gemm(X, W, m_sizes, topk, gather_indices=gather_indices, **kwargs)


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def reference_forward(X, W, m_sizes, topk):
    """PyTorch reference for grouped GEMM forward: Y = X @ W^T per expert."""
    num_experts = W.shape[0]
    N = W.shape[1]
    total_tokens = X.shape[0]
    Y = torch.zeros(total_tokens, N, device=X.device, dtype=X.dtype)

    offset = 0
    for e in range(num_experts):
        m = m_sizes[e].item()
        if m > 0:
            X_e = X[offset : offset + m]
            W_e = W[e]
            Y[offset : offset + m] = X_e @ W_e.T
        offset += m
    return Y


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
# Forward: sweep tokens_per_expert with fixed (num_experts=8, N=2048, K=1024)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg(
            "grouped-gemm-fwd-E8-N2048-K1024-GBps",
            "tokens_per_expert",
            [32, 64, 128, 256],
            num_experts=8,
            N=2048,
            K=1024,
            topk=1,
        ),
    ]
)
def bench_grouped_gemm_fwd(num_experts, tokens_per_expert, N, K, topk, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    total_tokens = num_experts * tokens_per_expert * topk
    X = torch.randn(total_tokens, K, dtype=dtype, device=device) * 0.1
    W = torch.randn(num_experts, N, K, dtype=dtype, device=device) * 0.1
    m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=device)
    gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

    if backend == "pytorch":
        fn = lambda: reference_forward(X, W, m_sizes, topk)
    else:
        tilegym.set_backend(backend)
        fn = lambda: _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend)

    # Correctness check
    ref = lambda: reference_forward(X, W, m_sizes, topk)
    torch.testing.assert_close(fn(), ref(), atol=1e-1, rtol=1e-1)

    ms = triton.testing.do_bench(fn)

    # Memory: read X (total_tokens * K) + read W (E * N * K) + write Y (total_tokens * N)
    bytes_per_element = X.element_size()
    total_bytes = (total_tokens * K + num_experts * N * K + total_tokens * N) * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Backward: sweep tokens_per_expert with fixed (num_experts=8, N=2048, K=1024)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg(
            "grouped-gemm-bwd-E8-N2048-K1024-GBps",
            "tokens_per_expert",
            [32, 64, 128, 256],
            num_experts=8,
            N=2048,
            K=1024,
            topk=1,
        ),
    ]
)
def bench_grouped_gemm_bwd(num_experts, tokens_per_expert, N, K, topk, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    total_tokens = num_experts * tokens_per_expert * topk
    m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=device)
    gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    dY = torch.randn(total_tokens, N, dtype=dtype, device=device) * 0.1

    if backend == "pytorch":
        X = (torch.randn(total_tokens, K, dtype=dtype, device=device) * 0.1).requires_grad_(True)
        W = (torch.randn(num_experts, N, K, dtype=dtype, device=device) * 0.1).requires_grad_(True)
        Y = reference_forward(X, W, m_sizes, topk)

        def fn():
            X.grad = None
            W.grad = None
            Y.backward(dY, retain_graph=True)
    else:
        tilegym.set_backend(backend)
        X = (torch.randn(total_tokens, K, dtype=dtype, device=device) * 0.1).requires_grad_(True)
        W = (torch.randn(num_experts, N, K, dtype=dtype, device=device) * 0.1).requires_grad_(True)
        Y = _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend)

        def fn():
            X.grad = None
            W.grad = None
            Y.backward(dY, retain_graph=True)

    ms = triton.testing.do_bench(fn)

    # Memory: read dY + read X + read W + write dX + write dW
    bytes_per_element = dY.element_size()
    total_bytes = (
        total_tokens * N + total_tokens * K + num_experts * N * K + total_tokens * K + num_experts * N * K
    ) * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_grouped_gemm_fwd.run(print_data=True)
    bench_grouped_gemm_bwd.run(print_data=True)
