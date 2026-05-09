# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import cross_entropy


class Test_Liger_CrossEntropy(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(input, target, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        """
        PyTorch float32 reference for cross-entropy loss.
        """
        return F.cross_entropy(
            input.float(),
            target,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        ).to(input.dtype)

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (4, 256, torch.float32),
            (4, 128, torch.float16),
            (4, 128, torch.bfloat16),
            (6, 300, torch.float32),  # non-power-of-2 vocab
            (8, 1024, torch.float32),
            (63, 41, torch.float32),
            (1269, 32000, torch.float32),
        ],
    )
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, BT, V, dtype, reduction, backend, monkeypatch):
        """Test forward loss value matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        # Use no requires_grad so input is not modified in-place
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        # Set some tokens to ignore_index
        target[: BT // 4] = -100

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        loss_test = cross_entropy(x.clone(), target, ignore_index=-100, reduction=reduction)
        loss_ref = self.reference(x, target, ignore_index=-100, reduction=reduction)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"Forward loss mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 128, torch.float16),
            (4, 128, torch.bfloat16),
            (63, 41, torch.float32),
            (6, 300, torch.float32),  # non-power-of-2 vocab
        ],
    )
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, BT, V, dtype, label_smoothing, backend, monkeypatch):
        """Test backward gradient matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x_data = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        atol = 1e-2
        rtol = 1e-2

        # Test implementation
        x_test = x_data.clone().requires_grad_(True)
        loss_test = cross_entropy(x_test, target, ignore_index=-100, label_smoothing=label_smoothing, reduction="mean")
        loss_test.backward()

        # Reference (float32 for stability)
        x_ref = x_data.clone().float().requires_grad_(True)
        loss_ref = F.cross_entropy(x_ref, target, ignore_index=-100, label_smoothing=label_smoothing, reduction="mean")
        loss_ref.backward()

        assert torch.allclose(x_test.grad.float(), x_ref.grad.float(), atol=atol, rtol=rtol), (
            f"Gradient mismatch (label_smoothing={label_smoothing})"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 256, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_ignore_index(self, BT, V, dtype, backend, monkeypatch):
        """Test that ignored tokens produce zero gradient and zero loss contribution."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        # All tokens ignored
        target = torch.full((BT,), -100, dtype=torch.long, device=device)

        x_test = x.clone().requires_grad_(True)
        loss_test = cross_entropy(x_test, target, ignore_index=-100, reduction="mean")
        loss_test.backward()

        assert loss_test.item() == 0.0, "Loss should be 0 when all tokens ignored"
        assert torch.all(x_test.grad == 0), "Gradient should be zero for all ignored tokens"

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 128, torch.float16),
            (4, 128, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_reduction_none(self, BT, V, dtype, backend, monkeypatch):
        """Test reduction='none' returns per-token loss matching PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        loss_test = cross_entropy(x.clone(), target, ignore_index=-100, reduction="none")
        loss_ref = F.cross_entropy(x.float(), target, ignore_index=-100, reduction="none").to(dtype)

        assert loss_test.shape == loss_ref.shape, f"Shape mismatch: {loss_test.shape} vs {loss_ref.shape}"
        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"Per-token loss mismatch (reduction=none)"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 128, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_weight(self, BT, V, dtype, backend, monkeypatch):
        """Test per-class weight matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100
        weight = torch.rand(V, dtype=torch.float32, device=device)

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        loss_test = cross_entropy(x.clone(), target, ignore_index=-100, weight=weight, reduction="mean")
        loss_ref = F.cross_entropy(x.float(), target, ignore_index=-100, weight=weight, reduction="mean").to(dtype)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"Weighted loss mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 128, torch.float32),
        ],
    )
    @pytest.mark.parametrize("softcap", [30.0, 50.0])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_softcap(self, BT, V, dtype, softcap, backend, monkeypatch):
        """Test softcap: logits are capped via softcap*tanh(x/softcap) before cross-entropy."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device) * 10  # large values to exercise softcap
        target = torch.randint(0, V, (BT,), device=device)

        loss_test = cross_entropy(x.clone(), target, ignore_index=-100, softcap=softcap, reduction="mean")
        # Reference: apply softcap manually then use PyTorch CE
        x_capped = softcap * torch.tanh(x.float() / softcap)
        loss_ref = F.cross_entropy(x_capped, target, ignore_index=-100, reduction="mean").to(dtype)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=1e-2, rtol=1e-2), (
            f"Softcap loss mismatch (softcap={softcap}): test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 512, torch.float32),
            (4, 128, torch.float32),
        ],
    )
    @pytest.mark.parametrize("lse_square_scale", [1e-4, 1e-2])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_z_loss(self, BT, V, dtype, lse_square_scale, backend, monkeypatch):
        """Test return_z_loss: z_loss = sum(lse_square_scale * logsumexp(x_i)^2) / n_non_ignore."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        loss_test, z_loss_test, *_ = cross_entropy(
            x.clone(), target, ignore_index=-100, lse_square_scale=lse_square_scale, return_z_loss=True
        )

        # Reference z_loss: per-token lse^2 * scale / n_non_ignore, summed over non-ignored
        mask = target != -100
        n_non_ignore = mask.sum().item()
        lse = torch.logsumexp(x.float(), dim=-1)  # (BT,)
        z_loss_ref = (lse_square_scale * lse * lse)[mask].sum() / max(n_non_ignore, 1)

        assert z_loss_test is not None, "z_loss should not be None when return_z_loss=True"
        assert torch.allclose(z_loss_test.float(), z_loss_ref, atol=1e-3, rtol=1e-3), (
            f"z_loss mismatch: test={z_loss_test.item():.6f}, ref={z_loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 256, torch.float32),
            (4, 128, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_token_accuracy(self, BT, V, dtype, backend, monkeypatch):
        """Test return_token_accuracy: fraction of non-ignored tokens where argmax == target."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        _, _, token_accuracy_test, _ = cross_entropy(x.clone(), target, ignore_index=-100, return_token_accuracy=True)

        # Reference: accuracy = correct / n_non_ignore
        mask = target != -100
        n_non_ignore = mask.sum().item()
        predicted = x.float().argmax(dim=-1)
        correct = (predicted[mask] == target[mask]).sum().item()
        token_accuracy_ref = correct / max(n_non_ignore, 1)

        assert token_accuracy_test is not None, "token_accuracy should not be None when return_token_accuracy=True"
        assert abs(token_accuracy_test.item() - token_accuracy_ref) < 1e-5, (
            f"token_accuracy mismatch: test={token_accuracy_test.item():.6f}, ref={token_accuracy_ref:.6f}"
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (8, 256, torch.float32),
            (4, 128, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_predicted_tokens(self, BT, V, dtype, backend, monkeypatch):
        """Test return_predicted_tokens: returned indices match argmax of logits."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100

        _, _, _, predicted_tokens_test = cross_entropy(
            x.clone(), target, ignore_index=-100, return_predicted_tokens=True
        )

        # Reference: argmax per row; ignored rows get -1
        mask = target != -100
        predicted_ref = x.float().argmax(dim=-1)
        predicted_ref[~mask] = -1

        assert predicted_tokens_test is not None, (
            "predicted_tokens should not be None when return_predicted_tokens=True"
        )
        assert torch.all(predicted_tokens_test == predicted_ref), (
            f"predicted_tokens mismatch:\ntest={predicted_tokens_test}\nref={predicted_ref}"
        )
