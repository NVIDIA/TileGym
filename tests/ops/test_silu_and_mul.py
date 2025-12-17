# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common


class Test_SiLUAndMul(common.PyTestCase):
    @staticmethod
    # Reference implementation using PyTorch
    def reference(input):
        hidden_size = input.shape[-1] // 2
        x1 = input[..., :hidden_size]
        x2 = input[..., hidden_size:]
        return torch.nn.functional.silu(x1) * x2

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size, seq_len, hidden_size, dtype",
        [
            (32, 1024, 512, torch.float16),
            (32, 1024, 1024, torch.float32),
            (32, 1024, 4096, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, batch_size, seq_len, hidden_size, dtype, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")

        # Create input tensor
        input_shape = (batch_size, seq_len, 2 * hidden_size)
        torch.manual_seed(0)

        x = torch.randn(input_shape, dtype=dtype, device=device, requires_grad=True)
        dy = 0.1 * torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device)

        self.assertCorrectness(
            tilegym.ops.silu_and_mul,
            self.reference,
            {
                "input": x,
            },
            gradient=dy,
            rtol=0.0,
            atol=1e-2,
        )
