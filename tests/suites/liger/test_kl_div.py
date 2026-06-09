# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import kl_div


class Test_Liger_KLDiv(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(y_pred, y_true, reduction="batchmean", log_target=False, eps=1e-10):
        """PyTorch float32 reference for KL divergence."""
        y_pred_f = y_pred.float()
        y_true_f = y_true.float()
        if not log_target:
            loss = y_true_f * (torch.log(torch.clamp(y_true_f, min=eps)) - y_pred_f)
        else:
            loss = torch.exp(y_true_f) * (y_true_f - y_pred_f)

        if reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.sum() / (loss.shape[0] * loss.shape[1])
        else:  # batchmean
            return loss.sum() / loss.shape[0]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((16, 1024), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((4, 300), torch.float32),  # non-power-of-2
        ],
    )
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean", "batchmean"])
    @pytest.mark.parametrize("log_target", [False, True])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, reduction, log_target, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        # y_pred: log-probs (log-softmax ensures valid log-probs)
        y_pred = torch.log_softmax(torch.randn(*shape, dtype=dtype, device=device), dim=-1)
        # y_true: probs (softmax) or log-probs
        y_true_raw = torch.softmax(torch.randn(*shape, dtype=dtype, device=device), dim=-1)
        y_true = torch.log(y_true_raw.float().clamp(min=1e-10)).to(dtype) if log_target else y_true_raw

        self.assertCorrectness(
            kl_div,
            self.reference,
            {
                "y_pred": y_pred,
                "y_true": y_true,
                "reduction": reduction,
                "log_target": log_target,
            },
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((16, 1024), torch.float32),
            ((4, 300), torch.float32),  # non-power-of-2
        ],
    )
    @pytest.mark.parametrize("reduction", ["sum", "mean", "batchmean"])
    @pytest.mark.parametrize("log_target", [False, True])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, reduction, log_target, backend, monkeypatch):
        """Test backward pass (gradient w.r.t. y_pred)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        y_pred = torch.log_softmax(torch.randn(*shape, dtype=dtype, device=device), dim=-1).requires_grad_(True)
        y_true_raw = torch.softmax(torch.randn(*shape, dtype=dtype, device=device), dim=-1)
        y_true = torch.log(y_true_raw.float().clamp(min=1e-10)).to(dtype) if log_target else y_true_raw

        dout = torch.ones((), dtype=dtype, device=device)

        self.assertCorrectness(
            kl_div,
            self.reference,
            {
                "y_pred": y_pred,
                "y_true": y_true,
                "reduction": reduction,
                "log_target": log_target,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )
