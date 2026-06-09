# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import group_norm


class Test_Liger_GroupNorm(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(X, num_channels, num_groups, W, B, eps=1e-5):
        """PyTorch float32 reference for group normalization."""
        return F.group_norm(X.float(), num_groups, W.float(), B.float(), eps=eps).to(X.dtype)

    @pytest.mark.parametrize(
        "batch_size, num_channels, num_groups, hidden_size, dtype",
        [
            (2, 4, 2, 8, torch.float32),
            (4, 8, 4, 16, torch.float32),
            (2, 4, 2, 8, torch.float16),
            (2, 4, 2, 8, torch.bfloat16),
            (2, 6, 3, 10, torch.float32),  # non-power-of-2
            # Shapes from Liger test/transformers/test_group_norm.py
            (2, 63, 21, 2163, torch.float32),
            (16, 32, 1, 4096, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, batch_size, num_channels, hidden_size, num_groups, dtype, backend, monkeypatch):
        """Test forward output matches PyTorch F.group_norm reference."""
        monkeypatch.setenv("DISABLE_AUTOTUNE", "1")
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        X = torch.randn(batch_size, num_channels, hidden_size, dtype=dtype, device=device)
        W = torch.randn(num_channels, dtype=dtype, device=device)
        B = torch.randn(num_channels, dtype=dtype, device=device)

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        Y_test = group_norm(X.clone(), num_channels, num_groups, W, B)
        Y_ref = self.reference(X, num_channels, num_groups, W, B)

        assert torch.allclose(Y_test.float(), Y_ref.float(), atol=atol, rtol=rtol), (
            f"Forward mismatch: max_diff={((Y_test.float() - Y_ref.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "batch_size, num_channels, num_groups, hidden_size, dtype",
        [
            (2, 4, 2, 8, torch.float32),
            (4, 8, 4, 16, torch.float32),
            (2, 4, 2, 8, torch.float16),
            (2, 4, 2, 8, torch.bfloat16),
            (2, 63, 21, 2163, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, batch_size, num_channels, hidden_size, num_groups, dtype, backend, monkeypatch):
        """Test backward gradients (dX, dW, dB) match PyTorch reference."""
        monkeypatch.setenv("DISABLE_AUTOTUNE", "1")
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        X_data = torch.randn(batch_size, num_channels, hidden_size, dtype=dtype, device=device)
        W_data = torch.randn(num_channels, dtype=dtype, device=device)
        B_data = torch.randn(num_channels, dtype=dtype, device=device)

        atol = 1e-1
        rtol = 1e-1

        # Test implementation
        X_test = X_data.clone().requires_grad_(True)
        W_test = W_data.clone().requires_grad_(True)
        B_test = B_data.clone().requires_grad_(True)
        Y_test = group_norm(X_test, num_channels, num_groups, W_test, B_test)
        Y_test.backward(torch.ones_like(Y_test))

        # Reference (float32)
        X_ref = X_data.clone().float().requires_grad_(True)
        W_ref = W_data.clone().float().requires_grad_(True)
        B_ref = B_data.clone().float().requires_grad_(True)
        Y_ref = F.group_norm(X_ref, num_groups, W_ref, B_ref)
        Y_ref.backward(torch.ones_like(Y_ref))

        assert torch.allclose(X_test.grad.float(), X_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dX mismatch: max_diff={((X_test.grad.float() - X_ref.grad.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(W_test.grad.float(), W_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dW mismatch: max_diff={((W_test.grad.float() - W_ref.grad.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(B_test.grad.float(), B_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dB mismatch: max_diff={((B_test.grad.float() - B_ref.grad.float()).abs().max()).item():.6f}"
        )
