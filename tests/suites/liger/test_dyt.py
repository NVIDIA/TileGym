# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import dyt


class Test_Liger_DyT(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(x, alpha, gamma, beta=None):
        """
        PyTorch reference implementation of DyT.

        Formula: y = tanh(alpha * x) * gamma + beta
        """
        x_f = x.float()
        alpha_f = alpha.float()
        gamma_f = gamma.float()
        tanh_x = torch.tanh(alpha_f * x_f)
        y = tanh_x * gamma_f
        if beta is not None:
            y = y + beta.float()
        return y.to(x.dtype)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((2, 4, 512), torch.float32),  # multi-dimensional
            ((4, 300), torch.float32),  # non-power-of-2 hidden dim
            # Shapes from Liger test/transformers/test_dyt.py
            ((2, 8, 4096), torch.float32),
            ((4, 16, 2048), torch.float32),
            ((1, 1, 1023), torch.float32),  # non-power-of-2
            ((3, 7, 256), torch.float32),
        ],
    )
    @pytest.mark.parametrize("have_beta", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, have_beta, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        N = shape[-1]

        x = torch.randn(*shape, dtype=dtype, device=device)
        alpha = torch.randn(1, dtype=dtype, device=device)
        gamma = torch.randn(N, dtype=dtype, device=device)
        beta = torch.randn(N, dtype=dtype, device=device) if have_beta else None

        self.assertCorrectness(
            dyt,
            self.reference,
            {
                "x": x,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
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
            # Shapes from Liger test/transformers/test_dyt.py
            ((2, 8, 4096), torch.float32),
            ((4, 16, 2048), torch.float32),
            ((1, 1, 1023), torch.float32),  # non-power-of-2
            ((3, 7, 256), torch.float32),
        ],
    )
    @pytest.mark.parametrize("have_beta", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, have_beta, backend, monkeypatch):
        """Test backward pass (gradients for x, alpha, gamma, and optional beta)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        N = shape[-1]

        x = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        alpha = torch.randn(1, dtype=dtype, device=device, requires_grad=True)
        gamma = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
        beta = torch.randn(N, dtype=dtype, device=device, requires_grad=True) if have_beta else None

        dout = torch.ones(*shape, dtype=dtype, device=device)

        self.assertCorrectness(
            dyt,
            self.reference,
            {
                "x": x,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )
