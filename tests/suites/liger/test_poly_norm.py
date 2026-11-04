# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import importlib

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
            ((64, 4096), torch.float32),
            ((128, 2048), torch.bfloat16),
            ((33, 8192), torch.bfloat16),
            ((8, 16384), torch.float32),
            ((16, 3000), torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_fwd_schedules(self, shape, dtype, backend, monkeypatch):
        """Check outputs and gradients for every cuTile forward schedule."""
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        module = importlib.import_module("tilegym.suites.liger.cutile.poly_norm")
        monkeypatch.setattr(module, "is_autotune_disabled", lambda: False)

        device = torch.device("cuda")
        eps = 1e-6
        n_rows, n_cols = shape
        x = torch.randn(*shape, dtype=dtype, device=device)
        weight = torch.randn(3, dtype=dtype, device=device)
        bias = torch.randn(1, dtype=dtype, device=device)
        key = (n_rows, n_cols, dtype, str(x.device))

        for body_key, blk, occ, nww in module._fwd_configs(n_cols):
            kernel = module._FWD_BODY[body_key].replace_hints(occupancy=occ, num_worker_warps=nww)
            monkeypatch.setattr(module, "_fwd_tune_cache", {key: (kernel, blk)})
            self.assertCorrectness(
                poly_norm,
                self.reference,
                {
                    "input": x.clone().requires_grad_(True),
                    "weight": weight.clone().requires_grad_(True),
                    "bias": bias.clone().requires_grad_(True),
                    "eps": eps,
                },
                gradient=torch.randn(*shape, dtype=dtype, device=device),
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
