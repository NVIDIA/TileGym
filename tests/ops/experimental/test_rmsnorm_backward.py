# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.ops import get_rms_norm_module

from ... import common
from ...common import markif


class Test_RMSNormBackward(common.PyTestCase):
    @staticmethod
    def reference(x, dy, weight, rstd):
        """
        Reference implementation for RMSNorm backward pass using PyTorch.
        Uses the shared torch reference implementation.
        """
        return get_rms_norm_module().rms_norm_backward_torch(x, dy, weight, rstd)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m, n, dtype",
        [
            (256, 256, torch.float16),
            (4096, 2**8, torch.bfloat16),
            (31072, 4096, torch.bfloat16),
            (256, 256, torch.float32),
            (2003, 2001, torch.float16),  # testing when dims are not multiples of 2
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")
        eps = 1e-5

        x_shape = (m, n)
        w_shape = (n,)

        # Create input tensors
        x = torch.rand(x_shape, dtype=dtype, device=device).mul_(0.5).add_(-2.3)
        weight = torch.randn(w_shape, dtype=dtype, device=device)
        dy = torch.randn(x_shape, dtype=dtype, device=device)

        # Compute rstd (simulating what forward pass would save)
        RMSNormModule = get_rms_norm_module()
        rstd = RMSNormModule.compute_rstd_torch(x, eps)

        # Test the backend backward function against PyTorch reference
        self.assertCorrectness(
            RMSNormModule.rms_norm_backward,
            self.reference,
            {
                "x": x,
                "dy": dy,
                "weight": weight,
                "rstd": rstd,
            },
            rtol=0.0,
            atol=5e-2,
            multiple_outputs=True,
        )


class Test_RMSNormAutogradBackward(common.PyTestCase):
    @staticmethod
    def reference(input, weight, eps):
        x_fp32 = input.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + eps)
        out = x_norm * weight.to(torch.float32)
        return out.to(input.dtype)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m, n, dtype",
        [
            (256, 256, torch.float16),
            (4096, 256, torch.bfloat16),
            (256, 256, torch.float32),
        ],
    )
    @pytest.mark.parametrize("static_persistent", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, static_persistent, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")
        eps = 1e-5

        x = torch.rand((m, n), dtype=dtype, device=device).mul_(0.5).add_(-2.3)
        w = torch.randn((n,), dtype=dtype, device=device)
        dy = torch.randn((m, n), dtype=dtype, device=device)

        x_cutile = x.clone().detach().requires_grad_(True)
        w_cutile = w.clone().detach().requires_grad_(True)
        y_cutile = tilegym.ops.rms_norm(
            x_cutile,
            (n,),
            w_cutile,
            eps,
            static_persistent=static_persistent,
        )
        y_cutile.backward(dy)

        x_ref = x.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)
        y_ref = self.reference(x_ref, w_ref, eps)
        y_ref.backward(dy)

        torch.testing.assert_close(x_cutile.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w_cutile.grad, w_ref.grad, rtol=1e-2, atol=1e-2)
