# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F
from tilegym import set_backend
from tilegym.ops import get_swiglu
from tests import common


class Test_SwiGLU(common.PyTestCase):
    @staticmethod
    def reference(a, b):
        """Reference implementation of SwiGLU using vanilla PyTorch"""
        return F.silu(a) * b

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,intermediate_size",
        [
            # (1, 128, 1024, 4096),
            # (2, 256, 2048, 8192),
            (8, 2048, 4096, 14336)
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, batch_size, seq_len, hidden_size, intermediate_size, backend, arch):
        """Test for functional correctness of SwiGLU implementation"""
        self.setUp()
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        # Generate input data
        a = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        b = torch.randn(batch_size, seq_len, hidden_size, device='cuda')

        with torch.no_grad():
            self.assertCorrectness(
                lambda a, b: get_swiglu()(a, b),
                lambda a, b: self.reference(a, b),
                {'a': a, 'b': b},
                rtol=1e-2,
                atol=1e-2,
            )
