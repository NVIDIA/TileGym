# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import poly_norm


class Test_Liger_PolyNorm(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(input, weight, bias, eps=1e-6):
        """
        PyTorch reference implementation of PolyNorm.

        Formula: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)
        """
        x = input.float()
        shape = x.shape
        dim = shape[-1]
        x_2d = x.view(-1, dim)

        x3 = x_2d * x_2d * x_2d
        x2 = x_2d * x_2d

        rstd_3 = torch.rsqrt((x3 * x3).mean(dim=-1, keepdim=True) + eps)
        rstd_2 = torch.rsqrt((x2 * x2).mean(dim=-1, keepdim=True) + eps)
        rstd_1 = torch.rsqrt((x_2d * x_2d).mean(dim=-1, keepdim=True) + eps)

        w0, w1, w2 = weight[0].float(), weight[1].float(), weight[2].float()
        b = bias[0].float()

        y = w0 * x3 * rstd_3 + w1 * x2 * rstd_2 + w2 * x_2d * rstd_1 + b
        return y.view(*shape).to(input.dtype)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((16, 1024), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((2, 4, 512), torch.float32),
            ((4, 300), torch.float32),  # non-power-of-2 hidden dim
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        eps = 1e-6

        x = torch.randn(*shape, dtype=dtype, device=device)
        weight = torch.randn(3, dtype=dtype, device=device)
        bias = torch.randn(1, dtype=dtype, device=device)

        self.assertCorrectness(
            poly_norm,
            self.reference,
            {
                "input": x,
                "weight": weight,
                "bias": bias,
                "eps": eps,
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
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradient computation)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        eps = 1e-6

        x = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(3, dtype=dtype, device=device, requires_grad=True)
        bias = torch.randn(1, dtype=dtype, device=device, requires_grad=True)

        dout = torch.ones(*shape, dtype=dtype, device=device)

        self.assertCorrectness(
            poly_norm,
            self.reference,
            {
                "input": x,
                "weight": weight,
                "bias": bias,
                "eps": eps,
            },
            gradient=dout,
            atol=1e-2,
            rtol=1e-2,
        )
