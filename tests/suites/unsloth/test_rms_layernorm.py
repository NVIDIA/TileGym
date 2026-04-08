# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth RMS LayerNorm (standard + Gemma variant) with autograd."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import rms_layernorm

DEVICE = "cuda"

_backends = ["cutile"]


class Test_Unsloth_RMSLayernorm(common.PyTestCase):
    @staticmethod
    def reference(X, W, eps=1e-6, gemma=False):
        """PyTorch reference for RMS LayerNorm."""
        x_f32 = X.float()
        shape = x_f32.shape
        dim = shape[-1]
        x_2d = x_f32.view(-1, dim)
        var = (x_2d * x_2d).mean(dim=-1, keepdim=True)
        inv_var = torch.rsqrt(var + eps)
        normed = x_2d * inv_var
        if gemma:
            output = normed * (W.float() + 1.0)
        else:
            normed = normed.to(W.dtype)
            output = normed * W
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
    def test_op_standard(self, shape, dtype, framework):
        """Test standard RMS LayerNorm (gemma=False)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rms_layernorm,
            self.reference,
            {"X": X, "W": W, "eps": 1e-6, "gemma": False},
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 256),
            (8, 512),
            (16, 1024),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_gemma(self, shape, dtype, framework):
        """Test Gemma variant: (W + 1) * norm(X)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rms_layernorm,
            self.reference,
            {"X": X, "W": W, "eps": 1e-6, "gemma": True},
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
        dout = torch.ones(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rms_layernorm,
            self.reference,
            {"X": X, "W": W, "eps": 1e-6, "gemma": False},
            gradient=dout,
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float16),
            ((8, 512), torch.bfloat16),
            ((16, 1024), torch.bfloat16),
            ((2, 4, 512), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward_gemma(self, shape, dtype, framework):
        """Test backward pass for Gemma variant: (W + 1) * norm(X)."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        dout = torch.ones(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            rms_layernorm,
            self.reference,
            {"X": X, "W": W, "eps": 1e-6, "gemma": True},
            gradient=dout,
            rtol=1e-2,
            atol=1e-2,
        )
