# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import relu_squared


def _reference_relu_squared(x):
    return torch.square(torch.relu(x))


class Test_Liger_ReLUSquared(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(input):
        """
        PyTorch reference implementation of ReLU-squared.

        Formula: y = relu(x)^2 = max(x, 0)^2
        """
        x_f = input.float()
        relu_x = torch.relu(x_f)
        y = relu_x * relu_x
        return y.to(input.dtype)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((2, 4, 512), torch.float32),  # multi-dimensional
            ((4, 300), torch.float32),  # non-power-of-2 hidden dim
            ((2, 8, 4096), torch.float32),
            ((4, 16, 2048), torch.float32),
            ((1, 1, 1023), torch.float32),  # non-power-of-2
            ((3, 7, 256), torch.float32),
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
        input = torch.randn(*shape, dtype=dtype, device=device)

        self.assertCorrectness(
            relu_squared,
            self.reference,
            {"input": input},
            atol=1e-2,
            rtol=1e-2,
        )
