# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available

from ... import common


class Test_ReLU(common.PyTestCase):
    @staticmethod
    def reference(x):
        return torch.nn.functional.relu(x)

    _backends = ["cutile"]
    if is_backend_available("tilecpp"):
        _backends = _backends + ["tilecpp"]
    _perf_backends = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "m,n,dtype,contiguous_grad",
        [
            (256, 64, torch.float16, True),
            (256, 256, torch.float16, True),
            (256, 2048, torch.float32, False),
            (256, 1024 * 32, torch.float16, True),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, contiguous_grad, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
            self.setUp()
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        self.setUp()

        x = torch.rand(m, n, device=device, dtype=dtype) - 0.5
        x.requires_grad = True
        if contiguous_grad:
            dout = torch.rand_like(x)
        else:
            dout = (torch.rand(n, m, device=device, dtype=dtype) - 0.5).t()
            assert not dout.is_contiguous()
        self.assertCorrectness(
            tilegym.ops.activation.relu,
            self.reference,
            {"x": x},
            gradient=dout,
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.parametrize(
        "m,n,dtype",
        [
            (256, 1024, torch.float16),
            (256, 2048, torch.float16),
            (256, 1024 * 4, torch.float16),
            (256, 1024 * 8, torch.float16),
            (256, 1024 * 16, torch.float16),
            (256, 1024 * 32, torch.float16),
            (256, 1024 * 64, torch.float16),
        ],
        ids=lambda x: str(x),
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, m, n, dtype, backend, record_property):
        if backend in self._backends:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        x = torch.rand(m, n, device=device, dtype=dtype) - 0.5
        x.requires_grad = True

        if backend in self._backends:
            backend_fn = lambda: tilegym.ops.activation.relu(x)
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(x)
        else:
            pytest.skip(f"Backend {backend} not supported")
        res = common.benchmark_framework(backend, backend_fn)
        record_property("benchmark", res)
