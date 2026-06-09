# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import rope


def _make_cos_sin(bsz, seq_len, head_dim, device, dtype=torch.float32, broadcast_batch=False):
    """
    Build cos/sin tensors of shape (cos_bsz, seq_len, head_dim).

    cos_bsz = 1 if broadcast_batch else bsz.
    Only the first head_dim//2 columns carry real rotation values; the rest are zeros.
    """
    cos_bsz = 1 if broadcast_batch else bsz
    half = head_dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)  # (seq_len, half)
    cos_1d = freqs.cos()  # (seq_len, half)
    sin_1d = freqs.sin()

    # Pad right half with zeros (only first half is used by RoPE)
    cos_full = torch.cat([cos_1d, torch.zeros_like(cos_1d)], dim=-1)  # (seq_len, head_dim)
    sin_full = torch.cat([sin_1d, torch.zeros_like(sin_1d)], dim=-1)

    cos = cos_full.unsqueeze(0).expand(cos_bsz, seq_len, head_dim).contiguous().to(dtype)
    sin = sin_full.unsqueeze(0).expand(cos_bsz, seq_len, head_dim).contiguous().to(dtype)
    return cos, sin


def _reference_rope(q, k, cos, sin):
    """
    PyTorch float32 reference for RoPE (HuggingFace Llama/Mistral half-split variant).

    q: (bsz, n_q_heads, seq_len, head_dim)
    k: (bsz, n_k_heads, seq_len, head_dim)
    cos/sin: (1_or_bsz, seq_len, head_dim)
    """
    head_dim = q.shape[-1]
    hd_half = head_dim // 2

    # cos/sin: use first half of head_dim; expand batch dim if needed
    cos_h = cos[..., :hd_half].float()  # (cos_bsz, seq_len, hd_half)
    sin_h = sin[..., :hd_half].float()

    def apply(x):
        x_f = x.float()
        x_r = x_f[..., :hd_half]  # (bsz, n_heads, seq_len, hd_half)
        x_i = x_f[..., hd_half:]

        # cos/sin: (cos_bsz, seq_len, hd_half) → (cos_bsz, 1, seq_len, hd_half)
        c = cos_h.unsqueeze(1)
        s = sin_h.unsqueeze(1)
        new_r = x_r * c - x_i * s
        new_i = x_i * c + x_r * s
        return torch.cat([new_r, new_i], dim=-1).to(x.dtype)

    return apply(q), apply(k)


class Test_Liger_Rope(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 16, 64), torch.float32),
            ((1, 4, 8, 128), torch.float32),
            ((2, 8, 16, 64), torch.float16),
            ((2, 8, 16, 64), torch.bfloat16),
            ((1, 2, 4, 48), torch.float32),  # non-power-of-2 head_dim
            ((3, 4, 7, 64), torch.float32),  # non-power-of-2 heads
        ],
    )
    @pytest.mark.parametrize("broadcast_batch", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, broadcast_batch, backend, monkeypatch):
        """Test RoPE forward pass."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        bsz, n_q_heads, seq_len, head_dim = shape
        n_k_heads = max(1, n_q_heads // 2)

        q = torch.randn(bsz, n_q_heads, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.randn(bsz, n_k_heads, seq_len, head_dim, dtype=dtype, device=device)
        cos, sin = _make_cos_sin(bsz, seq_len, head_dim, device, broadcast_batch=broadcast_batch)

        q_ref, k_ref = _reference_rope(q, k, cos, sin)
        q_out, k_out = rope(q.clone(), k.clone(), cos, sin)

        assert torch.allclose(q_out.float(), q_ref.float(), atol=1e-2, rtol=1e-2), (
            f"q mismatch: max_diff={(q_out.float() - q_ref.float()).abs().max()}"
        )
        assert torch.allclose(k_out.float(), k_ref.float(), atol=1e-2, rtol=1e-2), (
            f"k mismatch: max_diff={(k_out.float() - k_ref.float()).abs().max()}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 16, 64), torch.float32),
            ((2, 8, 16, 64), torch.float16),
            ((2, 8, 16, 64), torch.bfloat16),
            ((1, 4, 8, 128), torch.float32),
            ((1, 2, 4, 48), torch.float32),  # non-power-of-2 head_dim
            ((3, 4, 7, 64), torch.float32),  # non-power-of-2 heads
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradient flows through RoPE)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        bsz, n_q_heads, seq_len, head_dim = shape
        n_k_heads = n_q_heads

        q = torch.randn(bsz, n_q_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(bsz, n_k_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        cos, sin = _make_cos_sin(bsz, seq_len, head_dim, device)

        q_out, k_out = rope(q, k, cos, sin)
        (q_out.sum() + k_out.sum()).backward()

        assert q.grad is not None
        assert k.grad is not None
