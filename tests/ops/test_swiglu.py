# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from tests import common
from tilegym import set_backend
from tilegym.ops import get_swiglu
from tilegym.ops.cutile.swiglu import SiLUMulFunction


class Test_SwiGLU(common.PyTestCase):
    @staticmethod
    def reference(a, b):
        """Reference implementation of SwiGLU using vanilla PyTorch"""
        return F.silu(a) * b

    _backends = ["cutile"]

    # Regular shapes (power-of-2)
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,intermediate_size",
        [
            (8, 2048, 4096, 14336),
            (4, 1024, 2048, 8192),
            (2, 512, 1024, 4096),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, batch_size, seq_len, hidden_size, intermediate_size, backend, arch):
        """Test for functional correctness of SwiGLU forward pass."""
        self.setUp()
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        # Generate input data
        a = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        b = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        with torch.no_grad():
            self.assertCorrectness(
                lambda a, b: get_swiglu()(a, b),
                lambda a, b: self.reference(a, b),
                {"a": a, "b": b},
                rtol=1e-2,
                atol=1e-2,
            )

    # Forward with irregular shapes
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size",
        [
            # Prime batch sizes
            (7, 512, 1024),
            (13, 256, 2048),
            # Odd seq_len
            (8, 100, 1024),
            (8, 333, 512),
            # Non-power-of-2 hidden sizes
            (8, 256, 1000),
            (8, 256, 1500),
            (8, 256, 3000),
            # Combined irregular
            (7, 100, 1000),
            (13, 333, 1500),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward_irregular(self, batch_size, seq_len, hidden_size, backend, arch):
        """Test forward pass with irregular (non-power-of-2) shapes."""
        self.setUp()
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        a = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        b = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        with torch.no_grad():
            self.assertCorrectness(
                lambda a, b: get_swiglu()(a, b),
                lambda a, b: self.reference(a, b),
                {"a": a, "b": b},
                rtol=1e-2,
                atol=1e-2,
            )

    # Backward tests (regular shapes)
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,dtype",
        [
            (4, 512, 1024, torch.float32),
            (8, 256, 2048, torch.float32),
            (2, 1024, 512, torch.float32),
            (8, 512, 512, torch.float16),
            (8, 512, 1024, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, batch_size, seq_len, hidden_size, dtype, backend, arch):
        """Test backward pass of SwiGLU (SiLUMulFunction)."""
        self.setUp()
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        device = torch.device("cuda")
        torch.manual_seed(0)

        # Create inputs with grad
        a = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
        a_ref = a.clone().detach().requires_grad_(True)
        b_ref = b.clone().detach().requires_grad_(True)

        # Forward
        out = SiLUMulFunction.apply(a, b)
        out_ref = self.reference(a_ref, b_ref)

        # Check forward correctness
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)

        # Backward
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        out_ref.backward(grad_out)

        # Check gradient correctness
        torch.testing.assert_close(a.grad, a_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-2, atol=1e-2)

    # Backward with irregular shapes
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size",
        [
            # Prime batch sizes
            (7, 256, 512),
            (13, 128, 1024),
            # Odd seq_len
            (8, 100, 512),
            (8, 333, 256),
            # Non-power-of-2 hidden sizes
            (4, 256, 1000),
            (4, 256, 1500),
            (4, 128, 3000),
            # Combined irregular
            (7, 100, 1000),
            (13, 333, 1500),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_irregular(self, batch_size, seq_len, hidden_size, backend, arch):
        """Test backward pass with irregular (non-power-of-2) shapes."""
        self.setUp()
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        device = torch.device("cuda")
        dtype = torch.float32
        torch.manual_seed(0)

        a = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
        a_ref = a.clone().detach().requires_grad_(True)
        b_ref = b.clone().detach().requires_grad_(True)

        # Forward
        out = SiLUMulFunction.apply(a, b)
        out_ref = self.reference(a_ref, b_ref)

        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)

        # Backward
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        out_ref.backward(grad_out)

        torch.testing.assert_close(a.grad, a_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-2, atol=1e-2)
