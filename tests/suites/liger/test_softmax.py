# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import softmax


def _reference_softmax(x):
    """PyTorch float32 reference for softmax on the last dimension."""
    return torch.nn.functional.softmax(x.float(), dim=-1).to(x.dtype)


class Test_Liger_Softmax(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8), torch.float32),
            ((4, 16), torch.float32),
            ((1, 1023), torch.float32),  # non-power-of-2 n_cols, single block
            ((3, 7, 256), torch.float32),  # 3D input
            ((2, 8), torch.float16),
            ((2, 8), torch.bfloat16),
            ((1, 4096), torch.float32),  # multi-block dispatch
            ((1, 2, 4096), torch.float32),  # 3D multi-block dispatch
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend, monkeypatch):
        """Test softmax forward pass."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(*shape, dtype=dtype, device=device)

        def fw():
            return softmax(x.clone())

        def ref():
            return _reference_softmax(x)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8), torch.float32),
            ((4, 16), torch.float32),
            ((2, 8), torch.float16),
            ((2, 8), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradient flows through softmax)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)

        y = softmax(x)
        y.sum().backward()

        assert x.grad is not None
