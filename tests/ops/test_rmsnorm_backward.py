# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.ops import get_rms_norm_module

from .. import common
from ..common import markif


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
