# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import swiglu


def _reference_swiglu(a, b):
    """PyTorch float32 reference: silu(a) * b."""
    a_f = a.float()
    b_f = b.float()
    silu_a = a_f * torch.sigmoid(a_f)
    return (silu_a * b_f).to(a.dtype)


class Test_Liger_SwiGLU(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            # functional test shapes from Liger test_swiglu.py
            ((2, 8, 8), torch.float32),
            ((9, 7, 41), torch.float32),
            # transformer shapes from Liger test/transformers/test_swiglu.py
            ((2, 256, 256), torch.float32),
            ((6, 42, 123), torch.float32),
            # standard shapes
            ((4, 256), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            # non-power-of-2
            ((4, 300), torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, shape, dtype, backend, monkeypatch):
        """Test forward output matches PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        a = torch.randn(*shape, dtype=dtype, device=device)
        b = torch.randn(*shape, dtype=dtype, device=device)

        atol = 1e-2 if dtype != torch.float32 else 5e-3
        rtol = 1e-2 if dtype != torch.float32 else 5e-3

        c_test = swiglu(a.clone(), b.clone())
        c_ref = _reference_swiglu(a, b)

        assert torch.allclose(c_test.float(), c_ref.float(), atol=atol, rtol=rtol), (
            f"Forward mismatch: max_diff={((c_test.float() - c_ref.float()).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 8), torch.float32),
            ((9, 7, 41), torch.float32),
            ((2, 256, 256), torch.float32),
            ((6, 42, 123), torch.float32),
            ((4, 256), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward gradients (da, db) match PyTorch reference."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        a_data = torch.randn(*shape, dtype=dtype, device=device)
        b_data = torch.randn(*shape, dtype=dtype, device=device)

        atol = 1e-1
        rtol = 1e-1

        # Test implementation
        a_test = a_data.clone().requires_grad_(True)
        b_test = b_data.clone().requires_grad_(True)
        c_test = swiglu(a_test, b_test)
        c_test.backward(torch.ones_like(c_test))

        # Reference (float32)
        a_ref = a_data.clone().float().requires_grad_(True)
        b_ref = b_data.clone().float().requires_grad_(True)
        c_ref = _reference_swiglu(a_ref, b_ref.to(a_ref.dtype))
        c_ref.backward(torch.ones_like(c_ref))

        assert torch.allclose(a_test.grad.float(), a_ref.grad.float(), atol=atol, rtol=rtol), (
            f"da mismatch: max_diff={((a_test.grad.float() - a_ref.grad.float()).abs().max()).item():.6f}"
        )
        assert torch.allclose(b_test.grad.float(), b_ref.grad.float(), atol=atol, rtol=rtol), (
            f"db mismatch: max_diff={((b_test.grad.float() - b_ref.grad.float()).abs().max()).item():.6f}"
        )
