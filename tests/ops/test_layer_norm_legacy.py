# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.backend import set_backend

from .. import common

_backends = ["cutile"]


class Test_LayerNorm(common.PyTestCase):
    @staticmethod
    def reference(input, normalized_shape, weight, bias, eps, weight_shift):
        weight = weight + weight_shift
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)

    @pytest.mark.parametrize(
        "m,n,weight_shift,dtype",
        [
            (256, 256, 0.0, torch.float16),
            (256, 256, 0.0, torch.float32),
            (256, 256, 1.0, torch.float16),
            (9, 9, 0.0, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, weight_shift, dtype, arch, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        if m == 9 and backend == "cutile":
            pytest.skip("Skip due to cutile kernel can only support m % BLOCK == 0")

        device = torch.device("cuda")
        eps = 1e-5

        x_shape = (m, n)
        w_shape = (n,)

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
        x = x.detach().requires_grad_(True)

        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

        dy = 0.1 * torch.randn_like(x)
        with torch.no_grad():
            self.assertCorrectness(
                tilegym.ops.layer_norm_legacy,
                self.reference,
                {
                    "input": x,
                    "normalized_shape": w_shape,
                    "weight": weight,
                    "bias": bias,
                    "eps": eps,
                    "weight_shift": weight_shift,
                },
                gradient=dy,
                rtol=0.0,
                atol=1e-2,
            )


class Test_PersistentLayerNorm(common.PyTestCase):
    """Test class for persistent layer norm with TMA support."""

    @staticmethod
    def reference(input, weight, bias, eps):
        """Reference implementation using PyTorch's layer_norm."""
        normalized_shape = (input.shape[-1],)
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)

    @pytest.mark.parametrize(
        "m,n,dtype",
        [
            (256, 256, torch.bfloat16),
            (1024, 1024, torch.bfloat16),
            (4096, 512, torch.bfloat16),
            (30000, 1024, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()

        device = torch.device("cuda")
        eps = 1e-6

        x_shape = (m, n)
        w_shape = (n,)

        x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=False)
        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

        # Get output from persistent_layer_norm
        y, mean, rstd, _, _ = tilegym.ops.persistent_layer_norm(
            input=x,
            normalized_shape=w_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )

        # Get reference output
        y_ref = self.reference(x, weight, bias, eps)

        # Compute reference mean and rstd
        x_2d = x.reshape(-1, n)
        mean_ref = x_2d.mean(dim=-1)
        var_ref = x_2d.var(dim=-1, unbiased=False)
        rstd_ref = 1.0 / torch.sqrt(var_ref + eps)

        # Verify outputs
        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mean, mean_ref.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(rstd, rstd_ref.float(), atol=1e-2, rtol=1e-2)
