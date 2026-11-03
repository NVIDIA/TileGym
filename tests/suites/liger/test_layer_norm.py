# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import layer_norm


class Test_Liger_LayerNorm(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @staticmethod
    def reference(X, W, B, eps=1e-5):
        """PyTorch float32 reference for layer normalization."""
        return torch.nn.functional.layer_norm(X.float(), [X.shape[-1]], W.float(), B.float(), eps).to(X.dtype)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((16, 1024), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((2, 4, 512), torch.float32),  # multi-dimensional
            ((4, 300), torch.float32),  # non-power-of-2 hidden dim
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]

        X = torch.randn(*shape, dtype=dtype, device=device)
        W = torch.ones(H, dtype=dtype, device=device)
        B = torch.zeros(H, dtype=dtype, device=device)

        self.assertCorrectness(
            layer_norm,
            self.reference,
            {"X": X, "W": W, "B": B},
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
            ((16, 1024), torch.float32),
            ((2, 4, 512), torch.float32),
            ((4, 300), torch.float32),  # non-power-of-2
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradients for X, W, B)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]

        X = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        W = torch.ones(H, dtype=dtype, device=device, requires_grad=True)
        B = torch.zeros(H, dtype=dtype, device=device, requires_grad=True)

        dout = torch.ones(*shape, dtype=dtype, device=device)

        self.assertCorrectness(
            layer_norm,
            self.reference,
            {"X": X, "W": W, "B": B},
            gradient=dout,
            atol=1e-2,
            rtol=1e-1,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4096, 1024), torch.float32),
            ((4096, 2048), torch.float32),
            ((4096, 4096), torch.float32),
            ((4096, 8192), torch.float32),
            ((4096, 16384), torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, dtype, backend, record_property, monkeypatch):
        self.setUp()
        device = torch.device("cuda")
        H = shape[-1]

        X = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        W = torch.ones(H, dtype=dtype, device=device)
        B = torch.zeros(H, dtype=dtype, device=device)

        if backend in self._backends:
            if not tilegym.is_backend_available(backend):
                pytest.skip(f"Backend {backend} not available")
            tilegym.set_backend(backend)
            framework_fn = lambda: layer_norm(X, W, B)
        elif backend == "pytorch":
            framework_fn = lambda: self.reference(X, W, B)
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(X, W, B),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
            )

        result = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", result)

        del X, W, B, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
