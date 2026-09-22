# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available

from ... import common


class Test_GeLU(common.PyTestCase):
    @staticmethod
    def reference(input, approximate):
        return torch.nn.functional.gelu(input, approximate=approximate)

    _tolerances = {
        torch.float32: dict(rtol=1e-6, atol=1e-6),
        torch.float16: dict(rtol=2e-3, atol=1e-5),
        torch.bfloat16: dict(rtol=1.6e-2, atol=1e-4),
    }

    _backends = ["cutile"]
    if is_backend_available("tilecpp"):
        _backends = _backends + ["tilecpp"]
    _perf_backends = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "m,n,approximate,dtype",
        [
            (256, 2048, "none", torch.float32),
            (256, 2048, "tanh", torch.float32),
            (256, 2048, "none", torch.float16),
            (256, 2048, "tanh", torch.float16),
            (256, 2048, "none", torch.bfloat16),
            (256, 2048, "tanh", torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, approximate, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
            self.setUp()
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")

        x_shape = (m, n)

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(12.0).add_(-6.0)
        x = x.detach().requires_grad_(True)

        dy = 0.1 * torch.randn_like(x)

        self.assertCorrectness(
            tilegym.ops.activation.gelu,
            self.reference,
            {"input": x, "approximate": approximate},
            gradient=dy,
            rtol=self._tolerances[dtype]["rtol"],
            atol=self._tolerances[dtype]["atol"],
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_unaligned(self, approximate, dtype, backend):
        # 255 * 2047 = 521985 is not a multiple of BLOCK_SIZE (1024), so this
        # exercises the masked tail of the last tile.
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
            self.setUp()
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")

        x = torch.rand((255, 2047), dtype=dtype, device=device, requires_grad=False).mul_(12.0).add_(-6.0)
        x = x.detach().requires_grad_(True)

        dy = 0.1 * torch.randn_like(x)

        self.assertCorrectness(
            tilegym.ops.activation.gelu,
            self.reference,
            {"input": x, "approximate": approximate},
            gradient=dy,
            rtol=self._tolerances[dtype]["rtol"],
            atol=self._tolerances[dtype]["atol"],
        )
