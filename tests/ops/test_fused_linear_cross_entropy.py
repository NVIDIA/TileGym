# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from tests import common
from tilegym import set_backend
from tilegym.backend import is_backend_available
from tilegym.ops import fused_linear_cross_entropy


class TestFusedLinearCrossEntropy(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def _reference(hidden_states, weight, target, ignore_index, reduction):
        logits = F.linear(hidden_states, weight)
        if hidden_states.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
        return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize(
        "batch,seq_len,hidden_size,vocab_size,dtype,reduction",
        [
            (2, 128, 256, 2048, torch.float16, "mean"),
            (2, 64, 256, 2048, torch.bfloat16, "mean"),
            (1, 256, 384, 4096, torch.float16, "sum"),
        ],
    )
    def test_forward_backward_matches_pytorch(
        self,
        backend,
        batch,
        seq_len,
        hidden_size,
        vocab_size,
        dtype,
        reduction,
        arch,
    ):
        if not torch.cuda.is_available() or not is_backend_available("cutile"):
            pytest.skip("CUDA + cuTile backend required")

        self.setUp()
        set_backend(backend)

        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        target = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.long)
        target[:, 0] = -100  # exercise ignore_index path

        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)

        loss = fused_linear_cross_entropy(
            x,
            w,
            target,
            ignore_index=-100,
            chunk_size=128,
            reduction=reduction,
        )
        ref_loss = self._reference(x_ref, w_ref, target, ignore_index=-100, reduction=reduction)

        atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

        torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=rtol, atol=atol)

        loss.backward()
        ref_loss.backward()

        torch.testing.assert_close(x.grad, x_ref.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(w.grad, w_ref.grad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("backend", _backends)
    def test_chunk_size_consistency(self, backend, arch):
        if not torch.cuda.is_available() or not is_backend_available("cutile"):
            pytest.skip("CUDA + cuTile backend required")

        self.setUp()
        set_backend(backend)

        batch, seq_len, hidden_size, vocab_size = 2, 257, 192, 3072
        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.long)

        x2 = x.detach().clone().requires_grad_(True)
        w2 = w.detach().clone().requires_grad_(True)

        loss_c64 = fused_linear_cross_entropy(
            x,
            w,
            target,
            ignore_index=-100,
            chunk_size=64,
            reduction="mean",
        )
        loss_c512 = fused_linear_cross_entropy(
            x2,
            w2,
            target,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
        )

        torch.testing.assert_close(loss_c64, loss_c512, rtol=2e-2, atol=2e-2)

        loss_c64.backward()
        loss_c512.backward()

        torch.testing.assert_close(x.grad, x2.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(w.grad, w2.grad, rtol=3e-2, atol=3e-2)

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _backends)
    def test_peak_memory_less_than_pytorch(self, backend, arch):
        if not torch.cuda.is_available() or not is_backend_available("cutile"):
            pytest.skip("CUDA + cuTile backend required")

        self.setUp()
        set_backend(backend)

        batch, seq_len, hidden_size, vocab_size = 2, 1024, 1024, 16384
        dtype = torch.bfloat16

        def measure_peak(fn):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype)
        t = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.long)

        def run_fused():
            xf = x.detach().clone().requires_grad_(True)
            wf = w.detach().clone().requires_grad_(True)
            loss = fused_linear_cross_entropy(xf, wf, t, chunk_size=256, reduction="mean")
            loss.backward()

        def run_torch():
            xt = x.detach().clone().requires_grad_(True)
            wt = w.detach().clone().requires_grad_(True)
            logits = F.linear(xt, wt)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="mean")
            loss.backward()

        fused_peak = measure_peak(run_fused)
        torch_peak = measure_peak(run_torch)

        assert fused_peak <= torch_peak, f"Expected fused peak <= torch peak, got {fused_peak} > {torch_peak}"
