# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from ... import common


class Test_GeLU(common.PyTestCase):
    @staticmethod
    def reference(input, approximate):
        return torch.nn.functional.gelu(input, approximate=approximate)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m,n,approximate,dtype",
        [
            (256, 2048, "none", torch.float32),
            (256, 2048, "tanh", torch.float32),
            (256, 2048, "none", torch.float16),
            (256, 2048, "tanh", torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, approximate, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")

        x_shape = (m, n)

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)

        self.assertCorrectness(
            tilegym.ops.activation.gelu,
            self.reference,
            {"input": x, "approximate": approximate},
            rtol=0.0,
            atol=1e-2,
        )
