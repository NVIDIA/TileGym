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

    # Regular shapes with power-of-2 dimensions
    @pytest.mark.parametrize(
        "batch_size, seq_len, hidden_size, dtype",
        [
            # Original test cases
            (32, 1024, 512, torch.float16),
            (32, 1024, 1024, torch.float32),
            # Expanded combinations: bs={8, 16, 32}, seq_len={512, 1024}, hidden={512, 1024, 2048, 4096}
            (8, 512, 512, torch.float16),
            (8, 512, 1024, torch.float16),
            (8, 512, 2048, torch.float32),
            (8, 1024, 512, torch.float16),
            (8, 1024, 1024, torch.float32),
            (8, 1024, 2048, torch.float32),
            (16, 512, 512, torch.float16),
            (16, 512, 1024, torch.float32),
            (16, 512, 2048, torch.float16),
            (16, 1024, 512, torch.float32),
            (16, 1024, 1024, torch.float16),
            (16, 1024, 2048, torch.float32),
            (32, 512, 512, torch.float32),
            (32, 512, 1024, torch.float16),
            (32, 512, 2048, torch.float32),
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

    # Irregular shapes: non-power-of-2, odd numbers, prime numbers
    @pytest.mark.parametrize(
        "batch_size, seq_len, hidden_size, dtype",
        [
            # Prime batch sizes
            (7, 512, 512, torch.float32),
            (13, 256, 1024, torch.float32),
            # Odd seq_len
            (8, 100, 512, torch.float32),
            (8, 333, 1024, torch.float32),
            # Non-power-of-2 hidden sizes
            (8, 256, 1000, torch.float32),
            (8, 256, 1500, torch.float32),
            (8, 256, 3000, torch.float32),
            # Combined irregular
            (7, 100, 1000, torch.float32),
            (13, 333, 1500, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_irregular_shapes(self, batch_size, seq_len, hidden_size, dtype, backend, arch):
        """Test with irregular (non-power-of-2) shapes."""
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")

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

    # Large hidden size test (may OOM on some GPUs)
    @pytest.mark.parametrize(
        "batch_size, seq_len, hidden_size, dtype",
        [
            (32, 1024, 4096, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_large(self, batch_size, seq_len, hidden_size, dtype, backend, arch):
        """Large hidden size test - may OOM on some GPUs."""
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")

        input_shape = (batch_size, seq_len, 2 * hidden_size)
        torch.manual_seed(0)

        try:
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
        except torch.cuda.OutOfMemoryError:
            pytest.skip("Not enough GPU memory for this test case")
