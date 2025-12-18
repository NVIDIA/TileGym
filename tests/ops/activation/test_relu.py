# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from ... import common


class Test_ReLU(common.PyTestCase):
    @staticmethod
    def reference(x):
        return torch.nn.functional.relu(x)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m,n,dtype",
        [
            (256, 64, torch.float16),
            (256, 256, torch.float16),
            (256, 2048, torch.float32),
            (256, 1024 * 32, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        self.setUp()

        x = torch.rand(m, n, device=device, dtype=dtype) - 0.5
        self.assertCorrectness(
            tilegym.ops.activation.relu,
            self.reference,
            {"x": x},
            rtol=1e-5,
            atol=1e-8,
        )
