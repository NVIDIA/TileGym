# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth Cross-Entropy Loss (standard + chunked large-vocab)."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import cross_entropy_loss

DEVICE = "cuda"

_backends = ["cutile"]


class Test_Unsloth_CrossEntropyLoss(common.PyTestCase):
    @staticmethod
    def reference(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
        """PyTorch reference for cross-entropy loss."""
        batch, seq_len, vocab_size = logits.shape
        logits_f32 = logits.float().view(batch * seq_len, vocab_size)
        labels_flat = labels.view(-1)

        if logit_scaling != 0:
            logits_f32 = logit_scaling * logits_f32
        if logit_softcapping != 0:
            logits_f32 = logit_softcapping * torch.tanh(logits_f32 / logit_softcapping)

        loss = torch.nn.functional.cross_entropy(logits_f32, labels_flat, ignore_index=-100, reduction="sum")
        if n_items is None:
            n_items = (labels_flat != -100).sum()
        return loss / n_items

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 64, 1024),
            (4, 32, 32000),  # LLaMA-like vocab
            (1, 128, 50257),  # GPT-2 vocab
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_basic(self, batch, seq_len, vocab_size, dtype, framework):
        """Test basic cross-entropy (vocab <= 65536)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
        labels[0, :4] = -100

        result = cross_entropy_loss(logits, labels)
        expected = self.reference(logits, labels)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (1, 16, 100000),
            (2, 8, 128256),  # Llama-3 vocab
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_large_vocab(self, batch, seq_len, vocab_size, dtype, framework):
        """Test chunked large-vocab path (vocab > 65536)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)

        result = cross_entropy_loss(logits, labels)
        expected = self.reference(logits, labels)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 16, 1024),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_softcapping(self, batch, seq_len, vocab_size, dtype, framework):
        """Test cross-entropy with logit softcapping (Gemma-2 style)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)

        result = cross_entropy_loss(logits, labels, logit_softcapping=30.0)
        expected = self.reference(logits, labels, logit_softcapping=30.0)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 16, 1024),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_logit_scaling(self, batch, seq_len, vocab_size, dtype, framework):
        """Test cross-entropy with logit scaling (Cohere style)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)

        result = cross_entropy_loss(logits, labels, logit_scaling=0.5)
        expected = self.reference(logits, labels, logit_scaling=0.5)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("framework", _backends)
    def test_op_all_ignored(self, framework):
        """Test with all labels set to -100 (all ignored).

        Two sub-checks:
          1. With default n_items (auto-count): kernel should not crash.
             Result may be NaN/Inf/0 since n_items=0 → division by zero.
          2. With explicit n_items=1: loss should be 0.0 since all CE terms
             are masked out (sum of 0 masked losses / 1 = 0).
        """
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        logits = torch.randn(1, 8, 1024, dtype=torch.bfloat16, device=DEVICE)
        labels = torch.full((1, 8), -100, dtype=torch.long, device=DEVICE)

        # Sub-check 1: default n_items (auto-count → 0). Should not crash.
        # Result is undefined (NaN/Inf/0) but must be a valid scalar tensor.
        result_auto = cross_entropy_loss(logits, labels)
        assert result_auto.ndim == 0, "Should return scalar tensor"
        assert torch.isfinite(result_auto) or torch.isnan(result_auto) or torch.isinf(result_auto), (
            "Should return a valid scalar (finite, NaN, or Inf)"
        )

        # Sub-check 2: explicit n_items=1 → masked sum / 1 = 0.0
        result_explicit = cross_entropy_loss(logits, labels, n_items=1)
        assert result_explicit.item() == 0.0, f"All-ignored with n_items=1 should be 0.0, got {result_explicit.item()}"

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 32, 1024),
            (1, 64, 32000),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward(self, batch, seq_len, vocab_size, dtype, framework):
        """Test backward pass: verify logits gradient against PyTorch reference."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
        labels[0, :4] = -100

        logits_ref = (torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)).requires_grad_(True)
        loss_ref = self.reference(logits_ref, labels)
        loss_ref.backward()
        grad_ref = logits_ref.grad.clone()

        logits_test = logits_ref.detach().clone().requires_grad_(True)
        loss_test = cross_entropy_loss(logits_test, labels)
        loss_test.backward()

        assert logits_test.grad is not None, "Gradient should flow back to logits"
        torch.testing.assert_close(logits_test.grad, grad_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 16, 1024),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward_softcapping(self, batch, seq_len, vocab_size, dtype, framework):
        """Test backward pass with logit_softcapping (Gemma-2 style)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
        labels[0, :2] = -100

        logits_ref = (torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)).requires_grad_(True)
        loss_ref = self.reference(logits_ref, labels, logit_softcapping=30.0)
        loss_ref.backward()
        grad_ref = logits_ref.grad.clone()

        logits_test = logits_ref.detach().clone().requires_grad_(True)
        loss_test = cross_entropy_loss(logits_test, labels, logit_softcapping=30.0)
        loss_test.backward()

        assert logits_test.grad is not None, "Gradient should flow back to logits"
        torch.testing.assert_close(logits_test.grad, grad_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (2, 16, 1024),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward_logit_scaling(self, batch, seq_len, vocab_size, dtype, framework):
        """Test backward pass with logit_scaling (Cohere style)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
        labels[0, :2] = -100

        logits_ref = (torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)).requires_grad_(True)
        loss_ref = self.reference(logits_ref, labels, logit_scaling=0.5)
        loss_ref.backward()
        grad_ref = logits_ref.grad.clone()

        logits_test = logits_ref.detach().clone().requires_grad_(True)
        loss_test = cross_entropy_loss(logits_test, labels, logit_scaling=0.5)
        loss_test.backward()

        assert logits_test.grad is not None, "Gradient should flow back to logits"
        torch.testing.assert_close(logits_test.grad, grad_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "batch, seq_len, vocab_size",
        [
            (1, 16, 100000),
            (2, 8, 128256),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward_large_vocab(self, batch, seq_len, vocab_size, dtype, framework):
        """Test backward pass for large-vocab: verify logits gradient against PyTorch reference."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
        labels[0, :4] = -100

        logits_ref = (torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=DEVICE)).requires_grad_(True)
        loss_ref = self.reference(logits_ref, labels)
        loss_ref.backward()
        grad_ref = logits_ref.grad.clone()

        logits_test = logits_ref.detach().clone().requires_grad_(True)
        loss_test = cross_entropy_loss(logits_test, labels)
        loss_test.backward()

        assert logits_test.grad is not None, "Gradient should flow back to logits"
        torch.testing.assert_close(logits_test.grad, grad_ref, rtol=1e-2, atol=1e-2)
