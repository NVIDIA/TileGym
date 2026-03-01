# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from tests import common
from tilegym.backend import is_backend_available
from tilegym.ops.cutile.fused_linear_cross_entropy import fused_linear_cross_entropy_forward_only


class TestFusedLinearCrossEntropy(common.PyTestCase):
    @staticmethod
    def _reference(hidden_states, weight, target, ignore_index, reduction):
        logits = F.linear(hidden_states, weight)
        if hidden_states.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
        return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)

    @pytest.mark.parametrize(
        "batch,seq_len,hidden_size,vocab_size,dtype,reduction",
        [
            (2, 128, 256, 2048, torch.float16, "mean"),
            (2, 64, 256, 2048, torch.bfloat16, "mean"),
            (1, 256, 384, 4096, torch.float16, "sum"),
        ],
    )
    def test_forward_matches_pytorch(
        self,
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

        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype)
        target = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.long)
        target[:, 0] = -100

        loss = fused_linear_cross_entropy_forward_only(
            x,
            w,
            target,
            ignore_index=-100,
            chunk_size=128,
            reduction=reduction,
        )
        ref_loss = self._reference(x, w, target, ignore_index=-100, reduction=reduction)

        atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=rtol, atol=atol)

    def test_chunk_size_consistency(self, arch):
        if not torch.cuda.is_available() or not is_backend_available("cutile"):
            pytest.skip("CUDA + cuTile backend required")

        self.setUp()

        batch, seq_len, hidden_size, vocab_size = 2, 257, 192, 3072
        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        w = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float16)
        target = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.long)

        loss_c64 = fused_linear_cross_entropy_forward_only(
            x,
            w,
            target,
            ignore_index=-100,
            chunk_size=64,
            reduction="mean",
        )
        loss_c512 = fused_linear_cross_entropy_forward_only(
            x,
            w,
            target,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
        )

        torch.testing.assert_close(loss_c64, loss_c512, rtol=2e-2, atol=2e-2)

    @pytest.mark.slow
    def test_peak_memory_less_than_pytorch(self, arch):
        if not torch.cuda.is_available() or not is_backend_available("cutile"):
            pytest.skip("CUDA + cuTile backend required")

        self.setUp()

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
            _ = fused_linear_cross_entropy_forward_only(x, w, t, chunk_size=256, reduction="mean")

        def run_torch():
            logits = F.linear(x, w)
            _ = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="mean")

        fused_peak = measure_peak(run_fused)
        torch_peak = measure_peak(run_torch)

        assert fused_peak <= torch_peak, f"Expected fused peak <= torch peak, got {fused_peak} > {torch_peak}"
