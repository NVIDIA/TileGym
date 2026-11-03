# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import rms_norm


def _reference_rms_norm(X, W, eps, offset=0.0):
    """PyTorch float32 reference for RMS normalization."""
    X_f = X.float()
    rms = torch.sqrt(torch.mean(X_f**2, dim=-1, keepdim=True) + eps)
    Y = X_f / rms
    if W is not None:
        Y = Y * (W.float() + offset)
    return Y.to(X.dtype)


class Test_Liger_RMSNorm(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

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
        """Test RMS norm forward with affine weight (W)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device)
        W = torch.ones(H, dtype=dtype, device=device)

        def fw():
            return rms_norm(X.clone(), W, eps)

        def ref():
            return _reference_rms_norm(X, W, eps)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_no_affine(self, shape, dtype, backend, monkeypatch):
        """Test RMS norm forward without affine weight (W=None)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device)

        def fw():
            return rms_norm(X.clone(), None, eps)

        def ref():
            return _reference_rms_norm(X, None, eps)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma", "none"])
    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((4, 256), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_casting_modes(self, shape, dtype, casting_mode, backend, monkeypatch):
        """All three casting modes must compile and produce near-reference output.

        Regression guard: NONE mode in the CuTile fwd kernel previously referenced
        an undefined `x_sq` variable and would crash at compile time if exercised.
        """
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device)
        W = torch.ones(H, dtype=dtype, device=device)

        def fw():
            return rms_norm(X.clone(), W, eps, casting_mode=casting_mode)

        def ref():
            return _reference_rms_norm(X, W, eps)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("offset", [0.0, 1.0])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_offset(self, shape, dtype, offset, backend, monkeypatch):
        """Test RMS norm with non-zero offset (Gemma-style W+1 shift)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device)
        W = torch.ones(H, dtype=dtype, device=device)

        def fw():
            return rms_norm(X.clone(), W, eps, offset=offset)

        def ref():
            return _reference_rms_norm(X, W, eps, offset=offset)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float32),
            ((8, 512), torch.float32),
            ((4, 256), torch.float16),
            ((4, 256), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradients for X and W)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        H = shape[-1]
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        W = torch.ones(H, dtype=dtype, device=device, requires_grad=True)
        dout = torch.ones(*shape, dtype=dtype, device=device)

        self.assertCorrectness(
            rms_norm,
            _reference_rms_norm,
            {"X": X, "W": W, "eps": eps},
            gradient=dout,
            atol=1e-2,
            rtol=1e-1,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2048, 1024), torch.bfloat16),
            ((2048, 2048), torch.bfloat16),
            ((2048, 4096), torch.bfloat16),
            ((2048, 8192), torch.bfloat16),
            ((2048, 16384), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, dtype, backend, record_property, monkeypatch):
        self.setUp()
        device = torch.device("cuda")
        H = shape[-1]
        eps = 1e-6

        X = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        W = torch.ones(H, dtype=dtype, device=device)

        if backend in self._backends:
            if not tilegym.is_backend_available(backend):
                pytest.skip(f"Backend {backend} not available")
            tilegym.set_backend(backend)
            framework_fn = lambda: rms_norm(X.clone(), W, eps)
        elif backend == "pytorch":
            framework_fn = lambda: _reference_rms_norm(X, W, eps)
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            self.assertCorrectness(
                framework_fn,
                lambda: _reference_rms_norm(X, W, eps),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
            )

        result = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", result)

        del X, W, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
