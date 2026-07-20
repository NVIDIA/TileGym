# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import fused_add_rms_norm


class Test_Liger_FusedAddRMSNorm(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(X, R, W, eps=1e-6, offset=0.0, casting_mode="llama"):
        """
        PyTorch float32 reference for fused residual-add + RMSNorm.

        Computes:
          S = X + R
          Y = S / RMS(S) * (W + offset)
        where RMS(S) = sqrt(mean(S^2) + eps).
        """
        S = X.float() + R.float()
        rms = S.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        W_shifted = W.float() + offset
        Y = (S * rms * W_shifted).to(X.dtype)
        return Y, S.to(X.dtype)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((16, 1024), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((2, 4, 512), torch.float32),  # multi-dimensional input
            ((4, 300), torch.float32),  # non-power-of-2 hidden dim
            # Shapes from Liger test/transformers/test_fused_add_rms_norm.py
            ((9, 7, 41), torch.float32),
            ((2, 128, 512), torch.float32),
        ],
    )
    @pytest.mark.parametrize("offset", [0.0, 1.0])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, shape, dtype, offset, backend, monkeypatch):
        """Test forward output (Y, S) matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]

        X = torch.randn(*shape, dtype=dtype, device=device)
        R = torch.randn(*shape, dtype=dtype, device=device)
        W = torch.randn(H, dtype=dtype, device=device)

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        Y_test, S_test = fused_add_rms_norm(X, R, W, eps=1e-6, offset=offset, casting_mode="llama")
        Y_ref, S_ref = self.reference(X, R, W, eps=1e-6, offset=offset)

        assert torch.allclose(Y_test.float(), Y_ref.float(), atol=atol, rtol=rtol), (
            f"Y mismatch: max_diff={((Y_test.float() - Y_ref.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(S_test.float(), S_ref.float(), atol=1e-5, rtol=1e-5), (
            f"S mismatch: max_diff={((S_test.float() - S_ref.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((9, 7, 41), torch.float32),
            ((2, 128, 512), torch.float32),
        ],
    )
    @pytest.mark.parametrize("offset", [0.0, 1.0])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, offset, backend, monkeypatch):
        """Test backward gradients (dX, dR, dW) match PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]

        X_data = torch.randn(*shape, dtype=dtype, device=device)
        R_data = torch.randn(*shape, dtype=dtype, device=device)
        W_data = torch.randn(H, dtype=dtype, device=device)

        atol = 1e-1
        rtol = 1e-1

        # Test implementation
        X_test = X_data.clone().requires_grad_(True)
        R_test = R_data.clone().requires_grad_(True)
        W_test = W_data.clone().requires_grad_(True)

        Y_test, S_test = fused_add_rms_norm(X_test, R_test, W_test, eps=1e-6, offset=offset)
        # Backward through both outputs
        dout_Y = torch.ones_like(Y_test)
        dout_S = torch.ones_like(S_test)
        torch.autograd.backward([Y_test, S_test], [dout_Y, dout_S])

        # Reference (float32 for stability)
        X_ref = X_data.clone().float().requires_grad_(True)
        R_ref = R_data.clone().float().requires_grad_(True)
        W_ref = W_data.clone().float().requires_grad_(True)

        S_ref = X_ref + R_ref
        rms = S_ref.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        Y_ref = S_ref * rms * (W_ref + offset)
        # Backward through both outputs
        torch.autograd.backward(
            [Y_ref, S_ref],
            [torch.ones_like(Y_ref), torch.ones_like(S_ref)],
        )

        assert torch.allclose(X_test.grad.float(), X_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dX mismatch: max_diff={((X_test.grad.float() - X_ref.grad.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(R_test.grad.float(), R_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dR mismatch: max_diff={((R_test.grad.float() - R_ref.grad.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(W_test.grad.float(), W_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dW mismatch: max_diff={((W_test.grad.float() - W_ref.grad.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "casting_mode",
        ["llama", "gemma", "none"],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_casting_modes(self, casting_mode, backend, monkeypatch):
        """Test that all casting modes produce numerically close results."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        X = torch.randn(4, 256, dtype=torch.float32, device=device)
        R = torch.randn(4, 256, dtype=torch.float32, device=device)
        W = torch.randn(256, dtype=torch.float32, device=device)

        Y_test, S_test = fused_add_rms_norm(X, R, W, eps=1e-6, offset=0.0, casting_mode=casting_mode)
        Y_ref, S_ref = self.reference(X, R, W, eps=1e-6, offset=0.0)

        assert torch.allclose(Y_test, Y_ref, atol=1e-3, rtol=1e-3), f"casting_mode={casting_mode}: Y mismatch"
        assert torch.allclose(S_test, S_ref, atol=1e-5, rtol=1e-5), f"casting_mode={casting_mode}: S mismatch"
