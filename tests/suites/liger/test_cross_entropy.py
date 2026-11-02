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
from tilegym.suites.liger.ops import fused_linear_cross_entropy


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
            (4, 128, torch.float16),
            (4, 128, torch.bfloat16),
            (63, 41, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_reduction_none_backward(self, BT, V, dtype, backend, monkeypatch):
        """Test reduction='none' backward scales each row by its own (non-uniform) upstream grad."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x_data = torch.randn(BT, V, dtype=dtype, device=device)
        target = torch.randint(0, V, (BT,), device=device)
        target[: BT // 4] = -100
        row_weight = torch.rand(BT, dtype=torch.float32, device=device) * 2.0 + 0.1

        x_test = x_data.clone().requires_grad_(True)
        loss_test = cross_entropy(x_test, target, ignore_index=-100, reduction="none")
        (loss_test.float() * row_weight).sum().backward()

        x_ref = x_data.clone().float().requires_grad_(True)
        loss_ref = F.cross_entropy(x_ref, target, ignore_index=-100, reduction="none")
        (loss_ref * row_weight).sum().backward()

        torch.testing.assert_close(x_test.grad.float(), x_ref.grad, atol=1e-2, rtol=1e-2)

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

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("fused", [False, True])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("rows,vocab", [(16, 64), (2053, 513), (16, 2053)])
    def test_op_graph_replay(self, backend, fused, weighted, rows, vocab, monkeypatch):
        self.setUp()
        monkeypatch.setenv("DISABLE_FALLBACK", "1")
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as error:
            pytest.skip(f"Backend is not supported: {error}")

        hidden = 32
        data = torch.randn(rows, hidden if fused else vocab, device="cuda") * 0.1
        projection = torch.randn(vocab, hidden, device="cuda") * 0.1
        target = torch.arange(rows, device="cuda", dtype=torch.int64).remainder(vocab)
        weights = (torch.rand(vocab * 2, device="cuda") + 0.5)[::2] if weighted else None
        scale = torch.ones((), device="cuda")

        def run():
            x = data.clone().requires_grad_(True)
            if fused:
                loss = fused_linear_cross_entropy(x, projection, target, ce_weight=weights, label_smoothing=0.1)
            else:
                loss = cross_entropy(x, target, weight=weights, label_smoothing=0.1)
            gradient = torch.autograd.grad(loss, x, grad_outputs=scale)[0]
            return loss, gradient

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual_loss, actual_gradient = run()

        for ignored, multiplier in [(0, 2.5), (5, 0.5), (rows, -0.25)]:
            target.copy_(torch.arange(rows, device="cuda", dtype=torch.int64).remainder(vocab))
            target[:ignored] = -100
            scale.fill_(multiplier)
            if weighted:
                weights.uniform_(0.001, 0.005)
            graph.replay()

            if ignored == rows:
                torch.testing.assert_close(actual_loss, torch.zeros_like(actual_loss))
                torch.testing.assert_close(actual_gradient, torch.zeros_like(actual_gradient))
                continue
            ref_input = data.clone().requires_grad_(True)
            logits = ref_input @ projection.t() if fused else ref_input
            expected_loss = F.cross_entropy(logits, target, weight=weights, label_smoothing=0.1)
            expected_gradient = torch.autograd.grad(expected_loss, ref_input, grad_outputs=scale)[0]
            torch.testing.assert_close(actual_loss, expected_loss, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-4, rtol=2e-4)
