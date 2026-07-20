# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import fused_linear_cross_entropy


class Test_Liger_FusedLinearCrossEntropy(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(input, weight, target, bias=None, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        """PyTorch float32 reference for fused linear + cross-entropy."""
        logits = input.float() @ weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        return F.cross_entropy(
            logits,
            target,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (8, 128, 256, torch.float32),
            (4, 64, 128, torch.float16),
            (4, 64, 128, torch.bfloat16),
            (6, 64, 150, torch.float32),  # non-power-of-2 vocab
            # Shapes from Liger test/transformers/test_fused_linear_cross_entropy.py
            (63, 41, 41, torch.float32),
            (1024, 1024, 4096, torch.float32),
        ],
    )
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, BT, H, V, dtype, reduction, backend, monkeypatch):
        """Test forward loss matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_data = torch.randn(BT, H, dtype=dtype, device=device)
        weight = torch.randn(V, H, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        loss_test = fused_linear_cross_entropy(
            input_data.clone(), weight, target, ignore_index=-100, reduction=reduction
        )
        loss_ref = self.reference(input_data, weight, target, ignore_index=-100, reduction=reduction)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"Loss mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (8, 128, 256, torch.float32),
            (4, 64, 128, torch.float16),
            (4, 64, 128, torch.bfloat16),
            (63, 41, 41, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, BT, H, V, dtype, backend, monkeypatch):
        """Test backward gradient w.r.t. input matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_data = torch.randn(BT, H, dtype=dtype, device=device)
        weight = torch.randn(V, H, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        atol = 1e-1
        rtol = 1e-1

        # Test implementation
        x_test = input_data.clone().requires_grad_(True)
        loss_test = fused_linear_cross_entropy(x_test, weight, target, ignore_index=-100)
        loss_test.backward()

        # Reference (float32)
        x_ref = input_data.clone().float().requires_grad_(True)
        w_ref = weight.float()
        logits_ref = x_ref @ w_ref.t()
        loss_ref = F.cross_entropy(logits_ref, target, ignore_index=-100)
        loss_ref.backward()

        assert torch.allclose(x_test.grad.float(), x_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dInput mismatch: max_diff={((x_test.grad.float() - x_ref.grad.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (4, 64, 128, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_bias(self, BT, H, V, dtype, backend, monkeypatch):
        """Test that bias is correctly applied."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_data = torch.randn(BT, H, dtype=dtype, device=device)
        weight = torch.randn(V, H, dtype=dtype, device=device)
        bias = torch.randn(V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        loss_test = fused_linear_cross_entropy(input_data.clone(), weight, target, bias=bias)
        loss_ref = self.reference(input_data, weight, target, bias=bias)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"Bias test loss mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize("backend", _backends)
    def test_op_features(self, backend, monkeypatch):
        """Test the extended features: ce_weight, softcap, z_loss, token accuracy/predicted tokens."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        BT, H, V = 32, 64, 128
        atol = rtol = 1e-3
        x = torch.randn(BT, H, device=device)
        w = torch.randn(V, H, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100
        logits = x.float() @ w.float().t()
        valid = target != -100

        # z-loss regularizer: lse_square_scale * logsumexp(logits)^2
        lss = 1e-4
        loss_z, z_loss = fused_linear_cross_entropy(
            x.clone(), w, target, return_z_loss=True, lse_square_scale=lss, reduction="sum"
        )
        z_ref = (lss * torch.logsumexp(logits, -1)[valid] ** 2).sum()
        assert torch.allclose(z_loss.float(), z_ref.float(), atol=atol, rtol=rtol), (
            f"z_loss mismatch: {z_loss.item():.6f} vs {z_ref.item():.6f}"
        )

        # softcap: logits -> softcap * tanh(logits / softcap)
        sc = 30.0
        loss_sc = fused_linear_cross_entropy(x.clone(), w, target, softcap=sc, reduction="sum")
        loss_sc_ref = F.cross_entropy(sc * torch.tanh(logits / sc), target, ignore_index=-100, reduction="sum")
        assert torch.allclose(loss_sc.float(), loss_sc_ref.float(), atol=atol, rtol=rtol)

        # per-class weight (weighted CE)
        cw = torch.rand(V, device=device) + 0.5
        loss_w = fused_linear_cross_entropy(x.clone(), w, target, ce_weight=cw, reduction="mean")
        loss_w_ref = F.cross_entropy(logits, target, weight=cw, ignore_index=-100, reduction="mean")
        assert torch.allclose(loss_w.float(), loss_w_ref.float(), atol=atol, rtol=rtol)

        # token accuracy + predicted tokens
        _, ta, pt = fused_linear_cross_entropy(
            x.clone(), w, target, return_token_accuracy=True, return_predicted_tokens=True
        )
        pred = logits.argmax(-1)
        acc_ref = (pred[valid] == target[valid]).float().mean()
        assert torch.allclose(ta.float(), acc_ref.float(), atol=atol, rtol=rtol)
        assert (pt[valid] == pred[valid]).all(), "predicted tokens must match argmax on valid rows"
