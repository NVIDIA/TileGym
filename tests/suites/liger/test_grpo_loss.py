# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import grpo_loss


def _reference_grpo_loss(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask=None,
    temperature=0.9,
    beta=0.0,
    eps_low=0.2,
    eps_high=0.2,
    loss_type="grpo",
):
    """PyTorch float32 reference for GRPO forward pass."""
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    logits_f = logits.float()
    advantages_f = advantages.float()

    # Compute log-softmax / logsumexp for each (b, l)
    log_probs = torch.log_softmax(logits_f[:, :L, :] / temperature, dim=-1)  # (B, L, N)
    logp = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)

    if old_logp is not None:
        old_logp_f = old_logp.float()
    else:
        old_logp_f = logp.detach()

    coef_1 = torch.exp(logp - old_logp_f)
    adv = advantages_f.unsqueeze(1).expand_as(logp)  # (B, L)

    if loss_type in ("grpo", "dapo", "bnpo", "dr_grpo"):
        coef_2 = coef_1.clamp(1.0 - eps_low, 1.0 + eps_high)
        loss1 = coef_1 * adv
        loss2 = coef_2 * adv
        per_token_loss = -torch.minimum(loss1, loss2)
        is_low_clipped = (coef_1 < (1.0 - eps_low)) & (adv < 0)
        is_high_clipped = (coef_1 > (1.0 + eps_high)) & (adv > 0)
        is_clipped = (is_low_clipped | is_high_clipped).float()
    elif loss_type == "cispo":
        coef_2 = torch.clamp(coef_1, max=eps_high)
        per_token_loss = -coef_2 * adv * logp
        is_clipped = (coef_1 > eps_high) & (adv > 0)
        is_clipped = is_clipped.float()
    else:
        is_clipped = torch.zeros_like(logp)
        per_token_loss = torch.zeros_like(logp)

    kl = None
    if beta != 0.0 and ref_logp is not None:
        ref_logp_f = ref_logp.float()
        kl = torch.exp(ref_logp_f - logp) - (ref_logp_f - logp) - 1.0
        per_token_loss = per_token_loss + beta * kl

    if completion_mask is not None:
        mask_f = completion_mask.float()
        per_token_loss = per_token_loss * mask_f

    return per_token_loss, kl, is_clipped


class Test_Liger_GrpoLoss(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "B, L, N",
        [
            (2, 8, 256),
            (4, 16, 512),
            (1, 4, 128),
        ],
    )
    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, B, L, N, loss_type, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        temperature = 0.9
        eps_low = 0.2
        eps_high = 0.2

        logits = torch.randn(B, L + 1, N, dtype=torch.float32, device=device, requires_grad=True)
        completion_ids = torch.randint(0, N, (B, L), dtype=torch.int64, device=device)
        advantages = torch.randn(B, dtype=torch.float32, device=device)
        old_logp = torch.randn(B, L, dtype=torch.float32, device=device)

        loss, kl, is_clipped = grpo_loss(
            logits,
            old_logp,
            None,  # ref_logp
            completion_ids,
            advantages,
            temperature=temperature,
            beta=0.0,
            eps_low=eps_low,
            eps_high=eps_high,
            inplace=False,
            loss_type=loss_type,
        )

        ref_loss, _, ref_is_clipped = _reference_grpo_loss(
            logits.detach(),
            old_logp,
            None,
            completion_ids,
            advantages,
            temperature=temperature,
            eps_low=eps_low,
            eps_high=eps_high,
            loss_type=loss_type,
        )

        assert torch.allclose(loss.float(), ref_loss.float(), atol=1e-2, rtol=1e-2), (
            f"loss mismatch: max={(loss.float() - ref_loss.float()).abs().max()}"
        )

    @pytest.mark.parametrize(
        "B, L, N",
        [
            (2, 8, 256),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_mask(self, B, L, N, backend, monkeypatch):
        """Test with completion_mask."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")

        logits = torch.randn(B, L + 1, N, dtype=torch.float32, device=device)
        completion_ids = torch.randint(0, N, (B, L), dtype=torch.int64, device=device)
        advantages = torch.randn(B, dtype=torch.float32, device=device)
        old_logp = torch.randn(B, L, dtype=torch.float32, device=device)
        completion_mask = torch.randint(0, 2, (B, L), dtype=torch.bool, device=device)

        loss, kl, is_clipped = grpo_loss(
            logits,
            old_logp,
            None,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            beta=0.0,
            inplace=False,
        )

        ref_loss, _, _ = _reference_grpo_loss(
            logits,
            old_logp,
            None,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
        )

        assert torch.allclose(loss.float(), ref_loss.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "B, L, N",
        [
            (2, 8, 256),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_kl(self, B, L, N, backend, monkeypatch):
        """Test with KL penalty (beta > 0)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        beta = 0.1

        logits = torch.randn(B, L + 1, N, dtype=torch.float32, device=device)
        completion_ids = torch.randint(0, N, (B, L), dtype=torch.int64, device=device)
        advantages = torch.randn(B, dtype=torch.float32, device=device)
        old_logp = torch.randn(B, L, dtype=torch.float32, device=device)
        ref_logp = torch.randn(B, L, dtype=torch.float32, device=device)

        loss, kl, is_clipped = grpo_loss(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            beta=beta,
            inplace=False,
        )
        assert kl is not None

        ref_loss, ref_kl, _ = _reference_grpo_loss(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            beta=beta,
        )

        assert torch.allclose(loss.float(), ref_loss.float(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(kl.float(), ref_kl.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, backend, monkeypatch):
        """Test backward pass (gradient w.r.t. logits)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        B, L, N = 2, 8, 128

        logits = torch.randn(B, L + 1, N, dtype=torch.float32, device=device, requires_grad=True)
        completion_ids = torch.randint(0, N, (B, L), dtype=torch.int64, device=device)
        advantages = torch.randn(B, dtype=torch.float32, device=device)
        old_logp = torch.randn(B, L, dtype=torch.float32, device=device)

        loss, kl, is_clipped = grpo_loss(
            logits,
            old_logp,
            None,
            completion_ids,
            advantages,
            beta=0.0,
            inplace=False,
        )
        loss.sum().backward()
        assert logits.grad is not None, "logits.grad should not be None after backward"
