# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tilegym import set_backend
from tilegym.ops.fused_mlp import PartiallyFusedSwiGLUMLP

from .. import common


class MockConfig:
    """Mock config for PartiallyFusedSwiGLUMLP."""

    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"


class Test_FusedSwiGLUBackward(common.PyTestCase):
    """Test backward pass of PartiallyFusedSwiGLUMLP."""

    @staticmethod
    def reference_forward(x, gate_proj_weight, up_proj_weight, down_proj_weight):
        """Reference implementation using vanilla PyTorch."""
        gate = F.linear(x, gate_proj_weight)
        up = F.linear(x, up_proj_weight)
        glu_out = F.silu(gate) * up
        return F.linear(glu_out, down_proj_weight)

    _backends = ["cutile"]

    # Regular shapes (power-of-2)
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,intermediate_size",
        [
            (2, 64, 256, 512),
            (4, 128, 512, 1024),
            (8, 256, 512, 2048),
            (2, 512, 1024, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, batch_size, seq_len, hidden_size, intermediate_size, dtype, backend, arch):
        """Test backward pass of PartiallyFusedSwiGLUMLP."""
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()
        device = torch.device("cuda")

        # Create config and module
        config = MockConfig(hidden_size, intermediate_size)
        mlp = PartiallyFusedSwiGLUMLP(config).to(device).to(dtype)

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)

        # Forward with TileGym
        out = mlp(x)

        # Forward with PyTorch reference
        out_ref = self.reference_forward(
            x_ref,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
        )

        # Check forward correctness
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)

        # Backward
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        out_ref.backward(grad_out)

        # Check gradient correctness
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)

    # Irregular shapes: prime batch, odd seq_len, non-power-of-2 dimensions
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,intermediate_size",
        [
            # Prime batch sizes
            (7, 64, 256, 512),
            (13, 128, 512, 1024),
            # Odd seq_len
            (4, 100, 256, 512),
            (4, 333, 512, 1024),
            # Non-power-of-2 hidden sizes
            (4, 128, 300, 600),
            (4, 128, 500, 1000),
            (4, 128, 750, 1500),
            # Combined irregular
            (7, 100, 300, 600),
            (13, 333, 500, 1000),
            (11, 127, 750, 1500),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_irregular(self, batch_size, seq_len, hidden_size, intermediate_size, dtype, backend, arch):
        """Test backward pass with irregular (non-power-of-2) shapes."""
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()
        device = torch.device("cuda")
        torch.manual_seed(0)

        config = MockConfig(hidden_size, intermediate_size)
        mlp = PartiallyFusedSwiGLUMLP(config).to(device).to(dtype)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)

        out = mlp(x)
        out_ref = self.reference_forward(
            x_ref,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
        )

        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)

        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        out_ref.backward(grad_out)

        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
