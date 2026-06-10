# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import sparsemax


def _reference_sparsemax(x, dim=-1):
    """
    PyTorch float32 reference for sparsemax.

    Projects input onto the probability simplex along `dim`.
    Algorithm: sort descending → cumsum → find support → compute tau → clip.
    """
    x_f = x.float()
    input_dims = x_f.dim()
    if dim < 0:
        dim = input_dims + dim

    x_sorted, _ = torch.sort(x_f, dim=dim, descending=True)
    cumsum = torch.cumsum(x_sorted, dim=dim)
    input_size = x_f.size(dim)
    r = torch.arange(1, input_size + 1, device=x.device, dtype=torch.float32)
    shape = [1] * input_dims
    shape[dim] = input_size
    r = r.view(shape)
    k_bound = 1 + r * x_sorted
    support = k_bound > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (x_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(x_f - tau, min=0).to(x.dtype)


class Test_Liger_Sparsemax(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 128, 512), torch.float32),
            ((5, 123, 123), torch.float32),
            ((4, 256), torch.float32),
        ],
    )
    @pytest.mark.parametrize("dim", [-1, 1])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, dim, backend, monkeypatch):
        """Test sparsemax forward pass."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        # Skip trivial cases (dim with size 1 or same as last dim special handling)
        if dim >= len(shape) or dim < -len(shape):
            pytest.skip("invalid dim")
        actual_dim = dim if dim >= 0 else len(shape) + dim
        if shape[actual_dim] <= 1:
            pytest.skip("trivial dim")

        device = torch.device("cuda")
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype, device=device)

        def fw():
            return sparsemax(x.clone(), dim=dim)

        def ref():
            return _reference_sparsemax(x, dim=dim)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 128, 512), torch.float32),
            ((4, 256), torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradient flows through sparsemax)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)

        y = sparsemax(x, dim=-1)
        y.sum().backward()

        assert x.grad is not None
