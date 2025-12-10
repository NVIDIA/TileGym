# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common
from ..common import markif


class Test_RMSNorm(common.PyTestCase):
    @staticmethod
    # Reference implementation from Huggingface
    def reference(input, normalized_shape, weight, eps):
        # Layer norm should always be calculated in float32
        dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
        variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
        input = input * torch.rsqrt(variance + eps)

        if weight is None:
            return input

        # Convert into half-precision if necessary
        if weight.dtype in [torch.float16, torch.bfloat16]:
            input = input.to(weight.dtype)

        return weight * input

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m, n, dtype",
        [
            (256, 256, torch.float16),
            (4096, 2**8, torch.bfloat16),
            (31072, 4096, torch.bfloat16),
            (256, 256, torch.float32),
        ],
    )
    @pytest.mark.parametrize("static_persistent", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    @markif(
        lambda arch, m, n: arch in ["sm120", "sm121"] and m == 31072 and n == 4096,
        mark=pytest.mark.slow,
    )
    def test_op(self, m, n, dtype, static_persistent, backend, arch):

        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device('cuda')
        eps = 1e-5

        x_shape = (m, n)
        w_shape = (n,)

        x = (
            torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False)
            .mul_(0.5)
            .add_(-2.3)
        )
        x = x.detach().requires_grad_(True)

        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        with torch.no_grad():
            self.assertCorrectness(
                tilegym.ops.rms_norm,
                self.reference,
                {
                    'input': x,
                    'normalized_shape': w_shape,
                    'weight': weight,
                    'eps': eps,
                },
                extra_test_kwargs={
                    'static_persistent': static_persistent,
                },
                rtol=0.0,
                atol=5e-2,
            )
