# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth full LayerNorm (with bias) with autograd."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import layernorm

DEVICE = "cuda"

_backends = ["cutile"]


class Test_Unsloth_Layernorm(common.PyTestCase):
    @staticmethod
    def reference(X, W, b, eps=1e-6):
        """PyTorch reference for full LayerNorm with bias."""
        x_f32 = X.float()
        shape = x_f32.shape
        dim = shape[-1]
        x_2d = x_f32.view(-1, dim)
        mean = x_2d.mean(dim=-1, keepdim=True)
        var = ((x_2d - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x_2d - mean) * torch.rsqrt(var + eps)
        output = normed * W.float() + b.float()
        return output.view(*shape).to(X.dtype)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 256),
            (8, 512),
            (16, 1024),
            (2, 4, 512),
            (4, 300),  # non-power-of-2 hidden dim
            (32, 4096),
            (21, 349, 1024),  # upstream unsloth: bsz=21, seqlen=349, dim=1024
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, shape, dtype, framework):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        b = torch.randn(dim, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            layernorm,
            self.reference,
            {"X": X, "W": W, "b": b, "eps": 1e-6},
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float16),
            ((8, 512), torch.bfloat16),
            ((4, 256), torch.float32),
            ((2, 4, 512), torch.bfloat16),
            ((4, 2048), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward(self, shape, dtype, framework):
        """Test backward pass via autograd."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        b = torch.randn(dim, dtype=dtype, device=DEVICE)
        dout = torch.ones(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            layernorm,
            self.reference,
            {"X": X, "W": W, "b": b, "eps": 1e-6},
            gradient=dout,
            rtol=1e-2,
            atol=1e-2,
        )
