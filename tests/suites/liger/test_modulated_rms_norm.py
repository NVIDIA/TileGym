# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import modulated_rms_norm


def _reference(X, W, scale, shift, eps=1e-6, offset=0.0):
    dtype = X.dtype
    xf = X.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xn = (xf * rstd).to(dtype).float()  # llama: round to input dtype, then keep fp32
    out = xn * (W.float() + offset) if W is not None else xn
    out = out * (1.0 + scale.float())
    if shift is not None:
        out = out + shift.float()
    return out.to(dtype)


class Test_Liger_ModulatedRMSNorm(common.PyTestCase):
    _backends = ["cutile"]

    # "shared" -> one modulation row for all tokens (atomic_add path)
    # "per_token" -> one modulation row per token (plain store path)
    @pytest.mark.parametrize("mod_mode", ["shared", "per_token"])
    @pytest.mark.parametrize("shape", [(8, 128), (64, 512)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, mod_mode, shape, dtype, framework, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        n_rows, n_cols = shape
        mod_shape = (n_cols,) if mod_mode == "shared" else (n_rows, n_cols)

        X = torch.randn(shape, dtype=dtype, device="cuda")
        W = torch.randn(n_cols, dtype=dtype, device="cuda")
        scale = torch.randn(mod_shape, dtype=dtype, device="cuda")
        shift = torch.randn(mod_shape, dtype=dtype, device="cuda")

        self.assertCorrectness(
            lambda: modulated_rms_norm(X, W, scale, shift, 1e-6, in_place=False),
            lambda: _reference(X, W, scale, shift),
            kwargs={},
            atol=2e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("mod_mode", ["shared", "per_token"])
    @pytest.mark.parametrize("shape", [(8, 128), (64, 512)])
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_backward(self, mod_mode, shape, dtype, framework, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        n_rows, n_cols = shape
        mod_shape = (n_cols,) if mod_mode == "shared" else (n_rows, n_cols)

        def make(s):
            return torch.randn(s, dtype=dtype, device="cuda", requires_grad=True)

        X, W = make(shape), make((n_cols,))
        scale, shift = make(mod_shape), make(mod_shape)
        Xr, Wr, sr, shr = [t.detach().clone().requires_grad_(True) for t in (X, W, scale, shift)]

        modulated_rms_norm(X, W, scale, shift, 1e-6, in_place=False).sum().backward()
        _reference(Xr, Wr, sr, shr).sum().backward()

        for a, b in [(X, Xr), (W, Wr), (scale, sr), (shift, shr)]:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("framework", _backends)
    def test_op_no_shift(self, framework, monkeypatch):
        """shift=None — exercises the dummy-tensor path in forward and backward."""
        self.setUp()
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        X = torch.randn(64, 512, dtype=torch.float32, device="cuda", requires_grad=True)
        W, scale = [torch.randn(512, dtype=torch.float32, device="cuda", requires_grad=True) for _ in range(2)]
        Xr, Wr, sr = [t.detach().clone().requires_grad_(True) for t in (X, W, scale)]

        modulated_rms_norm(X, W, scale, None, 1e-6, in_place=False).sum().backward()
        _reference(Xr, Wr, sr, None).sum().backward()

        for a, b in [(X, Xr), (W, Wr), (scale, sr)]:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("casting_mode", ["none", "llama", "gemma"])
    @pytest.mark.parametrize("framework", _backends)
    def test_op_casting_modes(self, casting_mode, framework, monkeypatch):
        """All three casting branches. Other tests only exercise the llama default."""
        self.setUp()
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        X = torch.randn(64, 512, dtype=torch.float32, device="cuda")
        W, scale, shift = [torch.randn(512, dtype=torch.float32, device="cuda") for _ in range(3)]

        self.assertCorrectness(
            lambda: modulated_rms_norm(X, W, scale, shift, 1e-6, casting_mode=casting_mode, in_place=False),
            lambda: _reference(X, W, scale, shift),
            kwargs={},
            atol=1e-3,
            rtol=1e-3,
        )
