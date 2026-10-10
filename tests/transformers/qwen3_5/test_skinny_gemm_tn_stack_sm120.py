# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Correctness and concurrency-safety tests for the skinny_gemm_tn_stack SM120 cuTile kernel."""

import importlib

import pytest
import torch


def _is_supported_arch() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(not _is_supported_arch(), reason="SM120 CUDA GPU required")

MOD = "tilegym.transformers.qwen3_5.kernels.skinny_gemm_tn_stack_sm120"

# (M, G, N, K) — the kv-pair ++ workload shapes from the definition.
SHAPES = [
    (1, 2, 1024, 5120),  # attn k/v pair decode (tuned mode 10)
    (8, 2, 1024, 5120),  # kv pair chunk (tuned mode 3, split-K S=5)
    (32, 2, 1024, 5120),  # kv pair chunk (tuned mode 1, split-K S=4)
    (1, 2, 3584, 1024),  # GDN ba decode (tuned mode 8)
    (32, 2, 3584, 1024),  # GDN ba chunk
    (1, 4, 1024, 4096),  # 4-way uniform group decode
    (32, 2, 17408, 5120),  # wide-N chunk (tuned mode 6)
]


def _run(m, M, G, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    c = m.skinny_gemm_tn_stack_sm120_cutile(a, b)
    return a, b, c


@pytest.mark.parametrize("M,G,N,K", SHAPES)
def test_shape_numerics(M, G, N, K):
    m = importlib.import_module(MOD)
    torch.manual_seed(0)
    a, b, c = _run(m, M, G, N, K)
    ref = torch.stack([torch.nn.functional.linear(a, b[g]) for g in range(b.shape[0])])
    assert torch.allclose(c.float(), ref.float(), atol=2e-2, rtol=2e-2), (
        f"M={M} G={G} N={N} K={K}: {(c.float() - ref.float()).abs().max().item():.2e}"
    )


def test_generic_fallback_shape():
    """Shapes outside the tuned table must stay correct via _heuristic()."""
    m = importlib.import_module(MOD)
    torch.manual_seed(0)
    for M, G, N, K in [(1, 2, 768, 2048), (16, 3, 640, 1024)]:
        a, b, c = _run(m, M, G, N, K)
        ref = torch.stack([torch.nn.functional.linear(a, b[g]) for g in range(b.shape[0])])
        assert torch.allclose(c.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_same_shape_concurrent_streams_splitk():
    """Same-shape split-K calls on two streams must not share mutable workspace.

    The split-K schedules allocate an fp32 partials buffer. Caching it across
    calls would let concurrent same-shape calls on different streams corrupt each
    other; scratch must be allocated per call. (M=8, G=2, N=1024, K=5120) selects a
    split-K schedule (S=5).
    """
    m = importlib.import_module(MOD)
    M, G, N, K = 8, 2, 1024, 5120
    torch.manual_seed(0)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    a1 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    a2 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    ref1 = torch.stack([torch.nn.functional.linear(a1, b1[g]) for g in range(G)])
    ref2 = torch.stack([torch.nn.functional.linear(a2, b2[g]) for g in range(G)])

    for _ in range(20):  # interleave long enough to expose shared-workspace races
        with torch.cuda.stream(s1):
            out1 = m.skinny_gemm_tn_stack_sm120_cutile(a1, b1)
        with torch.cuda.stream(s2):
            out2 = m.skinny_gemm_tn_stack_sm120_cutile(a2, b2)
    torch.cuda.synchronize()

    assert torch.allclose(out1.float(), ref1.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(out2.float(), ref2.float(), atol=2e-2, rtol=2e-2)
