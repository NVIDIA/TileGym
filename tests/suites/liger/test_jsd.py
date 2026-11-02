# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import fused_linear_jsd
from tilegym.suites.liger.ops import jsd
from tilegym.suites.liger.ops import tvd


class Test_Liger_JSD(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(input, target, shift_labels=None, beta=0.5, ignore_index=-100):
        """
        PyTorch reference implementation of generalized JSD loss.

        JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M), M = β*P + (1-β)*Q
        where Q = exp(input), P = exp(target).
        """
        Q = torch.exp(input.float())
        P = torch.exp(target.float())

        if beta == 0.0:  # forward KL: KL(P || Q)
            loss = P * (target.float() - input.float())
        elif beta == 1.0:  # reverse KL: KL(Q || P)
            loss = Q * (input.float() - target.float())
        else:
            M = beta * P + (1 - beta) * Q
            log_M = torch.log(M)
            loss = beta * P * target.float() + (1 - beta) * Q * input.float() - M * log_M

        if shift_labels is not None:
            mask = (shift_labels != ignore_index).float()
            n_non_ignore = mask.sum()
            loss = loss * mask.unsqueeze(-1)
        else:
            n_non_ignore = input.shape[0]

        if n_non_ignore == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        return (loss.sum() / n_non_ignore).to(input.dtype)

    @pytest.mark.parametrize(
        "BT, V, beta, dtype",
        [
            (4, 256, 0.5, torch.float32),
            (8, 512, 0.5, torch.float32),
            (16, 1024, 0.5, torch.float32),
            (4, 256, 0.0, torch.float32),
            (4, 256, 1.0, torch.float32),
            (8, 512, 0.3, torch.float32),
            (4, 256, 0.5, torch.float16),
            (4, 256, 0.5, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, BT, V, beta, dtype, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_logits = torch.randn(BT, V, device=device, dtype=dtype, requires_grad=True)
        target_logits = torch.randn(BT, V, device=device, dtype=dtype, requires_grad=False)
        input_log_prob = torch.nn.functional.log_softmax(input_logits, dim=-1).detach().requires_grad_(True)
        target_log_prob = torch.nn.functional.log_softmax(target_logits, dim=-1)

        dout = torch.tensor(1.0, device=device, dtype=dtype)

        self.assertCorrectness(
            jsd,
            self.reference,
            {
                "input": input_log_prob,
                "target": target_log_prob,
                "beta": beta,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "BT, V, beta, dtype",
        [
            (8, 512, 0.5, torch.float32),
            (16, 1024, 0.5, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_labels(self, BT, V, beta, dtype, backend, monkeypatch):
        """Test JSD with shift_labels (masking ignored tokens)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_log_prob = (
            torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        target_log_prob = torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
        # Set half the labels to ignore_index=-100
        shift_labels = torch.randint(0, V, (BT,), device=device)
        shift_labels[: BT // 2] = -100

        dout = torch.tensor(1.0, device=device, dtype=dtype)

        self.assertCorrectness(
            jsd,
            self.reference,
            {
                "input": input_log_prob,
                "target": target_log_prob,
                "shift_labels": shift_labels,
                "beta": beta,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "BT, V, beta, dtype",
        [
            (2048, 3200, 0.5, torch.float32),
            (2048, 3200, 0.5, torch.bfloat16),
            (16441, 1271, 0.5, torch.float32),
            (8, 512, 0.5, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_not_last_layer(self, BT, V, beta, dtype, backend, monkeypatch):
        """JSD gradients are correct when loss is not the final layer (non-unit upstream grad)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_log_prob = (
            torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        target_log_prob = torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)

        # Simulate a scaling factor of 2.0 upstream (output * 2.0 before backward)
        dout = torch.tensor(2.0, device=device, dtype=dtype)

        self.assertCorrectness(
            jsd,
            self.reference,
            {
                "input": input_log_prob,
                "target": target_log_prob,
                "beta": beta,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (2048, 3200, torch.float32),
            (2048, 3200, torch.bfloat16),
            (16441, 1271, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_symmetry(self, BT, V, dtype, backend, monkeypatch):
        """JSD is symmetric at beta=0.5: jsd(P, Q) == jsd(Q, P)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_log_prob = torch.nn.functional.log_softmax(
            torch.randn(BT, V, device=device, dtype=dtype), dim=-1
        ).detach()
        target_log_prob = torch.nn.functional.log_softmax(
            torch.randn(BT, V, device=device, dtype=dtype), dim=-1
        ).detach()

        out1 = jsd(input_log_prob, target_log_prob, beta=0.5)
        out2 = jsd(target_log_prob, input_log_prob, beta=0.5)

        torch.testing.assert_close(out1, out2, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "BT, V, beta, ignore_index, dtype",
        [
            (8, 512, 0.5, 2, torch.float32),
            (8, 512, 0.5, 42, torch.float32),
            (16, 1024, 0.3, 2, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_nondefault_ignore_index(self, BT, V, beta, ignore_index, dtype, backend, monkeypatch):
        """JSD with non-default ignore_index values (2, 42) in shift_labels."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        input_log_prob = (
            torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        target_log_prob = torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
        shift_labels = torch.randint(0, V, (BT,), device=device)
        # Randomly assign ~half to ignore_index
        num_ignore = max(1, BT // 2)
        shift_labels[torch.randperm(BT)[:num_ignore]] = ignore_index

        dout = torch.tensor(1.0, device=device, dtype=dtype)

        self.assertCorrectness(
            jsd,
            self.reference,
            {
                "input": input_log_prob,
                "target": target_log_prob,
                "shift_labels": shift_labels,
                "beta": beta,
                "ignore_index": ignore_index,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "BT, V, dtype",
        [
            (20, 32, torch.float32),
            (20, 32, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_all_ignored(self, BT, V, dtype, backend, monkeypatch):
        """JSD returns 0 and has zero gradient when all tokens are masked by ignore_index."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        ignore_index = -100
        input_log_prob = (
            torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        target_log_prob = torch.nn.functional.log_softmax(torch.randn(BT, V, device=device, dtype=dtype), dim=-1)
        shift_labels = torch.full((BT,), ignore_index, device=device, dtype=torch.long)

        out = jsd(input_log_prob, target_log_prob, shift_labels=shift_labels, beta=0.5)

        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0.0)

        out.backward(torch.tensor(1.0, device=device, dtype=dtype))
        torch.testing.assert_close(input_log_prob.grad, torch.zeros_like(input_log_prob.grad), atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize(
        "operator,reduction,vocab,dtype",
        [("jsd", "batchmean", 64, torch.float32), ("fused_jsd", "batchmean", 64, torch.float32)]
        + [
            ("tvd", reduction, vocab, torch.float32)
            for reduction in ["batchmean", "mean", "sum", "none"]
            for vocab in [63, 64, 68]
        ]
        + [
            ("tvd", reduction, 64, dtype)
            for reduction in ["batchmean", "mean"]
            for dtype in [torch.float16, torch.bfloat16]
        ],
    )
    def test_op_graph_replay(self, backend, operator, reduction, vocab, dtype, monkeypatch):
        self.setUp()
        monkeypatch.setenv("DISABLE_FALLBACK", "1")
        torch.backends.cuda.matmul.allow_tf32 = False
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as error:
            pytest.skip(f"Backend is not supported: {error}")
        rows, hidden = 8, 16
        data = torch.randn(rows, hidden if operator == "fused_jsd" else vocab, dtype=dtype, device="cuda") * 0.1
        teacher = torch.randn_like(data) * 0.1
        projection = torch.randn(vocab, hidden, device="cuda") * 0.1
        labels = torch.arange(rows, device="cuda")
        scale = torch.ones((), device="cuda")
        if operator != "fused_jsd":
            data = data.log_softmax(-1) if operator == "jsd" else data.softmax(-1)
            teacher = teacher.log_softmax(-1) if operator == "jsd" else teacher.softmax(-1)
        if dtype != torch.float32:
            data.copy_(torch.arange(vocab, device="cuda").remainder(2) * (2.0 / vocab))
            teacher.copy_(2.0 / vocab - data)

        def run(reference=False):
            x = data.clone().requires_grad_(True)
            if reference:
                valid = (labels != -100).float().unsqueeze(1)
                count = valid.sum().clamp_min(1)
                if operator == "tvd":
                    loss = ((x.float() - teacher.float()).abs() * 0.5 * valid).sum()
                    if reduction == "batchmean":
                        loss = loss / count
                    elif reduction == "mean":
                        loss = loss / (count * vocab)
                else:
                    log_q = (x @ projection.t()).log_softmax(-1) if operator == "fused_jsd" else x
                    log_p = (teacher @ projection.t()).log_softmax(-1) if operator == "fused_jsd" else teacher
                    q, p = log_q.exp(), log_p.exp()
                    mixture = 0.3 * p + 0.7 * q
                    loss = ((0.3 * p * log_p + 0.7 * q * log_q - mixture * mixture.log()) * valid).sum() / count
            elif operator == "jsd":
                loss = jsd(x, teacher, shift_labels=labels, beta=0.3)
            elif operator == "fused_jsd":
                loss = fused_linear_jsd(x, projection, teacher, projection, shift_labels=labels, beta=0.3)
            else:
                loss = tvd(x, teacher, shift_labels=labels, reduction=reduction)
                assert loss.grad_fn.saved_tensors[0].dtype == x.dtype
                loss = loss.sum()
            return loss, torch.autograd.grad(loss, x, grad_outputs=scale)[0]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual_loss, actual_gradient = run()
        for ignored, multiplier in [(0, 2.5), (3, -0.5), (rows, 0.75)]:
            labels.copy_(torch.arange(rows, device="cuda"))
            labels[:ignored] = -100
            scale.fill_(multiplier)
            graph.replay()
            expected_loss, expected_gradient = run(reference=True)
            rtol = {torch.float32: 2e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
            torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=rtol)
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=rtol)
