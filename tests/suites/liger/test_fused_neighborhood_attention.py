# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc
import math

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import fused_neighborhood_attention

# (batch, heads, seq_len, head_dim)
SHAPES = [
    (1, 1, 16, 32),
    (2, 4, 32, 64),
    (1, 2, 64, 32),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

KERNEL_SIZES = [3, 7]
DILATIONS = [1, 2]


def _ref_neighborhood_attention(query, key, value, kernel_size=7, dilation=1, scale=None):
    """Pure-PyTorch reference implementation."""
    batch, heads, seq_len, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q = query.float()
    k = key.float()
    v = value.float()

    # Build neighborhood mask [seq_len, seq_len]
    half = kernel_size // 2
    mask = torch.zeros(seq_len, seq_len, device=query.device, dtype=torch.float32)
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(i - j)
            if dilation == 1:
                if dist <= half:
                    mask[i, j] = 1.0
            else:
                if dist <= half * dilation and (i - j) % dilation == 0:
                    mask[i, j] = 1.0

    # scores: [B, H, S, S]
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    scores = scores.masked_fill(mask[None, None] == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # rows with all -inf become 0
    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    return out.to(query.dtype)


class Test_Liger_FusedNeighborhoodAttention(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, kernel_size, dtype, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        batch, heads, seq_len, head_dim = shape
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")

        atol = 2e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2

        framework_fn = lambda: fused_neighborhood_attention(q, k, v, kernel_size=kernel_size)
        ref_fn = lambda: _ref_neighborhood_attention(q, k, v, kernel_size=kernel_size)

        self.assertCorrectness(
            framework_fn,
            ref_fn,
            kwargs={},
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dilation", DILATIONS)
    @pytest.mark.parametrize("backend", _backends)
    def test_op_dilation(self, shape, dilation, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        batch, heads, seq_len, head_dim = shape
        dtype = torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")

        framework_fn = lambda: fused_neighborhood_attention(q, k, v, kernel_size=7, dilation=dilation)
        ref_fn = lambda: _ref_neighborhood_attention(q, k, v, kernel_size=7, dilation=dilation)

        self.assertCorrectness(
            framework_fn,
            ref_fn,
            kwargs={},
            atol=5e-3,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, backend, monkeypatch):
        self.setUp()
        if backend == "cutile":
            pytest.skip("cutile backward for FusedNeighborhoodAttention hangs in CI; skip until fixed")
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        batch, heads, seq_len, head_dim = shape
        dtype = torch.float32

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda", requires_grad=True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        out = fused_neighborhood_attention(q, k, v, kernel_size=7)
        out_ref = _ref_neighborhood_attention(q_ref, k_ref, v_ref, kernel_size=7)

        grad = torch.randn_like(out)
        out.backward(grad)
        out_ref.backward(grad.clone())

        torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-2, rtol=1e-1)
        torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-2, rtol=1e-1)
        torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-2, rtol=1e-1)
