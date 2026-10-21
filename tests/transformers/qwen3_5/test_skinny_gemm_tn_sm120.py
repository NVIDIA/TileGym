# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Correctness and concurrency-safety tests for the skinny_gemm_tn SM120 cuTile kernel."""

import pytest
import torch


def _is_supported_arch() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(not _is_supported_arch(), reason="SM120 CUDA GPU required")

MOD = "tilegym.transformers.qwen3_5.kernels.skinny_gemm_tn_sm120"

# (M, N, K) model shapes — a representative subset of the workload rows,
# covering GEMV, tensor-core, and split-K config-table paths.
SHAPES = [
    (1, 12288, 5120),
    (1, 1024, 5120),
    (1, 5120, 6144),
    (8, 17408, 5120),
    (32, 1024, 5120),  # split-K (tag 13)
    (32, 6144, 5120),  # staggered tensor-core (tag 8)
    (32, 248320, 5120),  # wide-N bnatural (tag 4)
]


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shape_numerics(M, N, K):
    import importlib

    m = importlib.import_module(MOD)
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    c = m.skinny_gemm_tn_sm120_cutile(a, b)
    ref = torch.nn.functional.linear(a, b)
    assert torch.allclose(c.float(), ref.float(), atol=2e-2, rtol=2e-2), (
        f"M={M} N={N} K={K}: {(c.float() - ref.float()).abs().max().item():.2e}"
    )


def test_generic_fallback_shape():
    """Shapes outside the tuned table must stay correct via _pick()."""
    import importlib

    m = importlib.import_module(MOD)
    torch.manual_seed(0)
    for M, N, K in [(1, 768, 2048), (16, 640, 1024), (48, 256, 512)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        c = m.skinny_gemm_tn_sm120_cutile(a, b)
        ref = torch.nn.functional.linear(a, b)
        assert torch.allclose(c.float(), ref.float(), atol=2e-2, rtol=2e-2), (
            f"M={M} N={N} K={K}: {(c.float() - ref.float()).abs().max().item():.2e}"
        )


def test_same_shape_concurrent_streams_splitk():
    """Same-shape split-K calls on two streams must not share mutable workspace.

    The split-K path allocates fp32 partials (+ fused-path counters) scratch. Caching
    it across calls would let concurrent same-shape calls on different streams
    write/reduce the same buffer and silently corrupt each other; scratch must be
    allocated per call. This shape (M=32, N=1024, K=5120) selects the split-K path.
    """
    import importlib

    m = importlib.import_module(MOD)
    M, N, K = 32, 1024, 5120
    torch.manual_seed(0)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    a1 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    a2 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    ref1 = torch.nn.functional.linear(a1, b1)
    ref2 = torch.nn.functional.linear(a2, b2)

    for _ in range(20):  # interleave long enough to expose shared-workspace races
        with torch.cuda.stream(s1):
            out1 = m.skinny_gemm_tn_sm120_cutile(a1, b1)
        with torch.cuda.stream(s2):
            out2 = m.skinny_gemm_tn_sm120_cutile(a2, b2)
    torch.cuda.synchronize()

    assert torch.allclose(out1.float(), ref1.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(out2.float(), ref2.float(), atol=2e-2, rtol=2e-2)
