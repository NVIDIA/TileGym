# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth RoPE (single-tensor and joint Q+K) with autograd."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import rope_embedding
from tilegym.suites.unsloth.ops import rope_embedding_qk

DEVICE = "cuda"

_backends = ["cutile"]


def _apply_rope_reference(x, cos, sin):
    """
    Apply rotary embedding to a tensor.

    x: (..., head_dim)
    cos, sin: (seq_len, head_dim//2)

    Splits x into first half and second half along last dim, then:
      x0_out = x0 * cos - x1 * sin
      x1_out = x1 * cos + x0 * sin
    """
    half = x.shape[-1] // 2
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)


class Test_Unsloth_RoPE_Embedding(common.PyTestCase):
    @staticmethod
    def reference(Q, cos, sin):
        """
        PyTorch reference for single-tensor RoPE.
        Q: (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim//2) — may have extra dims
        sin: (seq_len, head_dim//2) — may have extra dims
        """
        cos = cos.squeeze()
        sin = sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        half = head_dim // 2
        # cos/sin: (seq_len, half) → broadcast over batch and heads
        cos_exp = cos[:seq_len, :half].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, half)
        sin_exp = sin[:seq_len, :half].unsqueeze(0).unsqueeze(2)
        Q0 = Q[..., :half]
        Q1 = Q[..., half:]
        out = torch.cat([Q0 * cos_exp - Q1 * sin_exp, Q1 * cos_exp + Q0 * sin_exp], dim=-1)
        return out.to(Q.dtype)

    @pytest.mark.parametrize(
        "batch, seq_len, n_heads, head_dim",
        [
            (2, 64, 8, 64),
            (1, 128, 4, 128),
            (4, 32, 16, 64),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, batch, seq_len, n_heads, head_dim, dtype, framework):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        Q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=DEVICE)
        half = head_dim // 2
        cos = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        sin = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rope_embedding,
            self.reference,
            {"Q": Q.clone(), "cos": cos, "sin": sin},
            rtol=5e-2,
            atol=2e-2,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "batch, seq_len, n_heads, head_dim",
        [
            (2, 64, 8, 64),
            (1, 128, 4, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward(self, batch, seq_len, n_heads, head_dim, dtype, framework):
        """Test backward pass: verify Q gradient VALUES against PyTorch reference."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        half = head_dim // 2
        cos = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        sin = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        dout = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=DEVICE)

        Q_ref = (torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=DEVICE)).requires_grad_(True)
        out_ref = self.reference(Q_ref, cos, sin)
        out_ref.backward(dout)
        grad_ref = Q_ref.grad.clone()

        Q_test = Q_ref.detach().clone().requires_grad_(True)
        out_test = rope_embedding(Q_test, cos, sin)
        out_test.backward(dout)

        assert Q_test.grad is not None, "Gradient should flow back to Q"
        torch.testing.assert_close(Q_test.grad, grad_ref, rtol=5e-2, atol=2e-2)


class Test_Unsloth_RoPE_Embedding_QK(common.PyTestCase):
    @staticmethod
    def reference(Q, K, cos, sin, rope_indices=None):
        """
        PyTorch reference for joint Q+K RoPE.
        Q: (batch, n_heads_Q, seq_len, head_dim)
        K: (batch, n_heads_K, seq_len, head_dim)
        cos: (seq_len, head_dim//2)
        sin: (seq_len, head_dim//2)
        """
        cos = cos.squeeze()
        sin = sin.squeeze()
        batch, n_heads_Q, seq_len, head_dim = Q.shape
        _, n_heads_K, _, _ = K.shape
        half = head_dim // 2

        Q_out = Q.clone()
        K_out = K.clone()

        # cos/sin: (seq_len, half) → (1, 1, seq, half)
        cos_exp = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)
        sin_exp = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)

        Q0 = Q_out[..., :half]
        Q1 = Q_out[..., half:]
        Q_out = torch.cat([Q0 * cos_exp - Q1 * sin_exp, Q1 * cos_exp + Q0 * sin_exp], dim=-1)

        K0 = K_out[..., :half]
        K1 = K_out[..., half:]
        K_out = torch.cat([K0 * cos_exp - K1 * sin_exp, K1 * cos_exp + K0 * sin_exp], dim=-1)

        return Q_out.to(Q.dtype), K_out.to(K.dtype)

    @pytest.mark.parametrize(
        "batch, n_heads_Q, n_heads_K, seq_len, head_dim",
        [
            (2, 8, 8, 64, 64),
            (1, 32, 8, 128, 128),  # GQA: n_heads_Q > n_heads_K
            (4, 16, 4, 32, 64),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, batch, n_heads_Q, n_heads_K, seq_len, head_dim, dtype, framework):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        Q = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=DEVICE)
        K = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=DEVICE)
        half = head_dim // 2
        cos = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        sin = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rope_embedding_qk,
            self.reference,
            {"Q": Q.clone(), "K": K.clone(), "cos": cos, "sin": sin},
            rtol=5e-2,
            atol=2e-2,
            multiple_outputs=True,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "batch, n_heads_Q, n_heads_K, seq_len, head_dim",
        [
            (2, 8, 8, 64, 64),
            (1, 32, 8, 128, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward(self, batch, n_heads_Q, n_heads_K, seq_len, head_dim, dtype, framework):
        """Test backward pass: verify Q and K gradient VALUES against PyTorch reference."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        half = head_dim // 2
        cos = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        sin = torch.randn(seq_len, half, dtype=dtype, device=DEVICE)
        dQ_out = torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=DEVICE)
        dK_out = torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=DEVICE)

        # --- Reference backward ---
        Q_ref = (torch.randn(batch, n_heads_Q, seq_len, head_dim, dtype=dtype, device=DEVICE)).requires_grad_(True)
        K_ref = (torch.randn(batch, n_heads_K, seq_len, head_dim, dtype=dtype, device=DEVICE)).requires_grad_(True)
        Q_ref_out, K_ref_out = self.reference(Q_ref, K_ref, cos, sin)
        torch.autograd.backward([Q_ref_out, K_ref_out], [dQ_out, dK_out])
        Q_grad_ref = Q_ref.grad.clone()
        K_grad_ref = K_ref.grad.clone()

        # --- Kernel backward ---
        Q_test = Q_ref.detach().clone().requires_grad_(True)
        K_test = K_ref.detach().clone().requires_grad_(True)
        Q_test_out, K_test_out = rope_embedding_qk(Q_test, K_test, cos, sin)
        torch.autograd.backward([Q_test_out, K_test_out], [dQ_out, dK_out])

        assert Q_test.grad is not None, "Gradient should flow back to Q"
        assert K_test.grad is not None, "Gradient should flow back to K"
        torch.testing.assert_close(Q_test.grad, Q_grad_ref, rtol=5e-2, atol=2e-2, msg="Q gradient mismatch")
        torch.testing.assert_close(K_test.grad, K_grad_ref, rtol=5e-2, atol=2e-2, msg="K gradient mismatch")
