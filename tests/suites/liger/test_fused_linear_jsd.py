# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc
import math

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import fused_linear_jsd


class Test_Liger_FusedLinearJSD(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference_jsd(student_logits, teacher_logits, shift_labels=None, beta=0.5, ignore_index=-100, temperature=1.0):
        """PyTorch float32 reference for JSD.

        beta=0 → forward KL: KL(teacher || student)
        beta=1 → reverse KL: KL(student || teacher)
        else   → generalized JSD(beta)
        Matches the jsd kernel's special-casing for boundary betas.
        """
        student_logits = student_logits.float() / temperature
        teacher_logits = teacher_logits.float() / temperature

        student_log_prob = F.log_softmax(student_logits, dim=-1)
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_prob = torch.exp(student_log_prob)
        teacher_prob = torch.exp(teacher_log_prob)

        if beta == 0.0:  # forward KL: KL(teacher || student)
            loss = teacher_prob * (teacher_log_prob - student_log_prob)
        elif beta == 1.0:  # reverse KL: KL(student || teacher)
            loss = student_prob * (student_log_prob - teacher_log_prob)
        else:
            M = beta * teacher_prob + (1 - beta) * student_prob
            log_M = torch.log(M.clamp(min=1e-40))
            loss = beta * teacher_prob * (teacher_log_prob - log_M) + (1 - beta) * student_prob * (
                student_log_prob - log_M
            )  # (BT, V)

        if shift_labels is not None:
            mask = (shift_labels != ignore_index).float().unsqueeze(-1)  # (BT, 1)
            loss = loss * mask
            n_non_ignore = (shift_labels != ignore_index).sum().clamp(min=1)
            return loss.sum() / n_non_ignore
        else:
            return loss.sum() / student_logits.shape[0]

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (8, 128, 256, torch.float32),
            (4, 64, 128, torch.float16),
            (4, 64, 128, torch.bfloat16),
            # Shapes from Liger test/transformers/test_fused_linear_jsd.py
            (63, 41, 41, torch.float32),
            (188, 31, 123, torch.float32),
            # Corner cases
            (1, 64, 128, torch.float32),  # single-token batch (normalization edge)
            (4, 64, 32768, torch.float32),  # large V aligned (power-of-2 path)
            (3, 7, 11, torch.float32),  # all-prime tiny dims (non-aligned worst case)
        ],
    )
    @pytest.mark.parametrize("beta", [0.5, 0.1])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, BT, H, V, dtype, beta, backend, monkeypatch):
        """Test forward loss matches reference JSD."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        student_input = torch.randn(BT, H, dtype=dtype, device=device)
        student_weight = torch.randn(V, H, dtype=dtype, device=device)
        teacher_input = torch.randn(BT, H, dtype=dtype, device=device)
        teacher_weight = torch.randn(V, H, dtype=dtype, device=device)

        atol = 3e-2 if dtype != torch.float32 else 1e-2
        rtol = 3e-2 if dtype != torch.float32 else 1e-2

        loss_test = fused_linear_jsd(
            student_input.clone(),
            student_weight,
            teacher_input.clone(),
            teacher_weight,
            beta=beta,
        )

        # Reference: compute full logits first
        with torch.no_grad():
            student_logits_ref = student_input.float() @ student_weight.float().t()
            teacher_logits_ref = teacher_input.float() @ teacher_weight.float().t()
        loss_ref = self.reference_jsd(student_logits_ref, teacher_logits_ref, beta=beta)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=atol, rtol=rtol), (
            f"JSD loss mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}, "
            f"diff={abs(loss_test.item() - loss_ref.item()):.6f}"
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (8, 128, 256, torch.float32),
            (63, 41, 41, torch.float32),
            (188, 31, 123, torch.float32),
            # Corner cases
            (1, 64, 128, torch.float32),  # single-token batch
            (4, 64, 128, torch.float16),  # half precision backward
            (4, 64, 128, torch.bfloat16),  # bfloat16 backward
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, BT, H, V, dtype, backend, monkeypatch):
        """Test backward gradient w.r.t. student_input."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        student_input_data = torch.randn(BT, H, dtype=dtype, device=device)
        student_weight = torch.randn(V, H, dtype=dtype, device=device)
        teacher_input = torch.randn(BT, H, dtype=dtype, device=device)
        teacher_weight = torch.randn(V, H, dtype=dtype, device=device)

        atol = 1e-1
        rtol = 1e-1

        # Test implementation
        s_input_test = student_input_data.clone().requires_grad_(True)
        loss_test = fused_linear_jsd(s_input_test, student_weight, teacher_input, teacher_weight)
        loss_test.backward()

        # Reference (float32)
        s_input_ref = student_input_data.clone().float().requires_grad_(True)
        s_logits = s_input_ref @ student_weight.float().t()
        t_logits = (teacher_input.float() @ teacher_weight.float().t()).detach()
        loss_ref = self.reference_jsd(s_logits, t_logits)
        loss_ref.backward()

        assert torch.allclose(s_input_test.grad.float(), s_input_ref.grad.float(), atol=atol, rtol=rtol), (
            f"dInput mismatch: max_diff="
            f"{((s_input_test.grad.float() - s_input_ref.grad.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (4, 64, 128, torch.float32),
            (8, 128, 256, torch.float32),  # larger shape
            (16, 64, 128, torch.float32),  # more tokens to ignore
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_labels(self, BT, H, V, dtype, backend, monkeypatch):
        """Test with shift_labels (some tokens ignored)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        student_input = torch.randn(BT, H, dtype=dtype, device=device)
        student_weight = torch.randn(V, H, dtype=dtype, device=device)
        teacher_input = torch.randn(BT, H, dtype=dtype, device=device)
        teacher_weight = torch.randn(V, H, dtype=dtype, device=device)
        shift_labels = torch.randint(0, V, (BT,), device=device)
        shift_labels[: BT // 4] = -100

        loss_test = fused_linear_jsd(
            student_input.clone(),
            student_weight,
            teacher_input.clone(),
            teacher_weight,
            shift_labels=shift_labels,
        )

        with torch.no_grad():
            student_logits_ref = student_input.float() @ student_weight.float().t()
            teacher_logits_ref = teacher_input.float() @ teacher_weight.float().t()
        loss_ref = self.reference_jsd(student_logits_ref, teacher_logits_ref, shift_labels=shift_labels)

        assert torch.allclose(loss_test.float(), loss_ref.float(), atol=1e-2, rtol=1e-2), (
            f"JSD with labels mismatch: test={loss_test.item():.6f}, ref={loss_ref.item():.6f}"
        )

    @pytest.mark.parametrize(
        "BT, H, V, dtype",
        [
            (1024, 1024, 4096, torch.bfloat16),  # (B=8, T=128) from Liger chunked_loss test
            (1024, 1024, 4096, torch.float32),
            (141, 31, 123, torch.bfloat16),  # (B=3, T=47) random shape from Liger chunked_loss test
            (141, 31, 123, torch.float32),
        ],
    )
    @pytest.mark.parametrize(
        "temperature, beta",
        [
            (1.0, 0.5),
            (2.0, 0.8),
            (0.5, 0.2),
        ],
    )
    @pytest.mark.parametrize("ignore_index", [-100, 42])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_liger_shapes(self, BT, H, V, dtype, temperature, beta, ignore_index, backend, monkeypatch):
        """Liger chunked_loss shapes: student H//2, teacher H (different hidden dims).

        Mirrors upstream test/chunked_loss/test_jsd_loss.py: TorchLMHeadJSD uses
        H//2 for student and H for teacher.  Checks loss + dInput + dWeight.
        """
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)

        atol = 5e-2 if dtype != torch.float32 else 1e-5
        rtol = 5e-1 if dtype != torch.float32 else 5e-4

        torch.manual_seed(42)
        device = torch.device("cuda")
        Hs = H // 2  # student hidden size (mirrors Liger upstream)

        # Scale weights with kaiming-like init (std = 1/sqrt(fan_in)) so logit std ≈ 1,
        # avoiding float32 underflow that causes M → 0 → log(M) = -inf → NaN in JSD.
        student_weight = torch.randn(V, Hs, dtype=dtype, device=device) / math.sqrt(Hs)
        teacher_weight = torch.randn(V, H, dtype=dtype, device=device) / math.sqrt(H)

        _tensor = torch.randn(BT, Hs, device=device, dtype=dtype)
        teacher_input = torch.randn(BT, H, device=device, dtype=dtype)

        target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)
        num_ignore = torch.randint(1, max(2, BT // 2), (1,)).item()
        target[torch.randperm(BT)[:num_ignore]] = ignore_index

        # Reference: full matmul → float32 logits → reference_jsd (mirrors TorchLMHeadJSD)
        student_input_ref = _tensor.detach().clone().requires_grad_(True)
        student_weight_ref = student_weight.detach().clone().requires_grad_(True)
        s_logits_ref = (student_input_ref.float() @ student_weight_ref.float().t()) / temperature
        t_logits_ref = (teacher_input.float() @ teacher_weight.float().t()).detach() / temperature
        loss_ref = self.reference_jsd(
            s_logits_ref,
            t_logits_ref,
            shift_labels=target,
            beta=beta,
            ignore_index=ignore_index,
            temperature=1.0,  # pre-divided above
        )
        loss_ref.backward()

        # TileGym fused kernel
        student_input_tg = _tensor.detach().clone().requires_grad_(True)
        student_weight_tg = student_weight.detach().clone().requires_grad_(True)
        loss_tg = fused_linear_jsd(
            student_input_tg,
            student_weight_tg,
            teacher_input,
            teacher_weight,
            shift_labels=target,
            beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
        )
        loss_tg.backward()

        assert torch.allclose(loss_ref.float(), loss_tg.float(), atol=atol, rtol=rtol), (
            f"[{dtype} T={temperature} β={beta} ig={ignore_index}] Loss mismatch: "
            f"ref={loss_ref.item():.6f}, tg={loss_tg.item():.6f}, "
            f"diff={abs(loss_ref.item() - loss_tg.item()):.6f}"
        )

        assert torch.allclose(student_input_ref.grad.float(), student_input_tg.grad.float(), atol=atol, rtol=rtol), (
            f"[{dtype} T={temperature} β={beta}] dInput mismatch: max_diff="
            f"{(student_input_ref.grad.float() - student_input_tg.grad.float()).abs().max().item():.6f}"
        )

        assert torch.allclose(student_weight_ref.grad.float(), student_weight_tg.grad.float(), atol=atol, rtol=rtol), (
            f"[{dtype} T={temperature} β={beta}] dWeight mismatch: max_diff="
            f"{(student_weight_ref.grad.float() - student_weight_tg.grad.float()).abs().max().item():.6f}"
        )
