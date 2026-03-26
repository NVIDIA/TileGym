# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth GEGLU (exact and approximate) forward and backward kernels."""

import math

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import geglu_approx_backward
from tilegym.suites.unsloth.ops import geglu_approx_forward
from tilegym.suites.unsloth.ops import geglu_exact_backward
from tilegym.suites.unsloth.ops import geglu_exact_forward

DEVICE = "cuda"


# =============================================================================
# Exact GEGLU
# =============================================================================


class Test_Unsloth_GEGLU_Exact_Forward(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(gate, up):
        """PyTorch reference: f = 0.5 * gate * (1 + erf(gate / sqrt(2))); out = f * up"""
        g_f32 = gate.float()
        f = 0.5 * g_f32 * (1.0 + torch.erf(g_f32 / math.sqrt(2.0)))
        f = f.to(up.dtype)
        return f * up

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128, 256),
            (4, 64, 512),
            (1, 1, 1024),
            (2, 256, 4096),
            # Production-like
            (4, 512, 5120),
            (2, 1024, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        gate = torch.randn(*shape, dtype=dtype, device=DEVICE)
        up = torch.randn(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            geglu_exact_forward,
            self.reference,
            {"gate": gate, "up": up},
            rtol=1e-2,
            atol=1e-3,
        )


class Test_Unsloth_GEGLU_Exact_Backward(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(DW, e, g):
        """PyTorch reference for GEGLU exact backward (in-place)."""
        e_f32 = e.float()
        f_partial = 0.5 * (torch.erf(e_f32 / math.sqrt(2.0)) + 1.0)
        f = f_partial * e_f32
        f = f.to(DW.dtype)
        h = f * g
        df = DW * f
        dg = DW * g
        t = 1.0 / math.sqrt(2.0 * math.pi)
        df_de = f_partial + t * e_f32 * torch.exp(-0.5 * e_f32 * e_f32)
        de = dg.float() * df_de
        de = de.to(DW.dtype)
        return h, df, de

    @pytest.mark.parametrize(
        "M, N",
        [
            (256, 512),
            (1024, 1024),
            (2048, 5120),
            (4096, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, M, N, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        DW = torch.randn(M, N, dtype=dtype, device=DEVICE)
        e = torch.randn(M, N, dtype=dtype, device=DEVICE)
        g = torch.randn(M, N, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            geglu_exact_backward,
            self.reference,
            {"DW": DW.clone(), "e": e.clone(), "g": g.clone()},
            rtol=1e-2,
            atol=1e-3,
            multiple_outputs=True,
            check_stride=False,
        )


# =============================================================================
# Approximate GEGLU
# =============================================================================


class Test_Unsloth_GEGLU_Approx_Forward(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(gate, up):
        """PyTorch reference: GELU approximate * up"""
        g_f32 = gate.float()
        s = math.sqrt(2.0 / math.pi)
        f = 0.5 * g_f32 * (torch.tanh(s * g_f32 * (1.0 + 0.044715 * g_f32 * g_f32)) + 1.0)
        f = f.to(up.dtype)
        return f * up

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128, 256),
            (4, 64, 512),
            (1, 1, 1024),
            (2, 256, 4096),
            # Production-like
            (4, 512, 5120),
            (2, 1024, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        gate = torch.randn(*shape, dtype=dtype, device=DEVICE)
        up = torch.randn(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            geglu_approx_forward,
            self.reference,
            {"gate": gate, "up": up},
            rtol=1e-2,
            atol=1e-3,
        )


class Test_Unsloth_GEGLU_Approx_Backward(common.PyTestCase):
    _backends = ["cutile"]

    @staticmethod
    def reference(DW, e, g):
        """PyTorch reference for GEGLU approximate backward (in-place)."""
        e_f32 = e.float()
        s = math.sqrt(2.0 / math.pi)
        a = s * e_f32
        b = a * 0.044715 * e_f32 * e_f32
        T = 1.0 + torch.tanh(a + b)
        T2 = 0.5 * T
        Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
        df_de = T2 + Q2

        f = T2 * e_f32
        f = f.to(DW.dtype)
        h = f * g
        df = DW * f
        dg = DW * g
        de = dg.float() * df_de
        de = de.to(DW.dtype)
        return h, df, de

    @pytest.mark.parametrize(
        "M, N",
        [
            (256, 512),
            (1024, 1024),
            (2048, 5120),
            (4096, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, M, N, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        DW = torch.randn(M, N, dtype=dtype, device=DEVICE)
        e = torch.randn(M, N, dtype=dtype, device=DEVICE)
        g = torch.randn(M, N, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            geglu_approx_backward,
            self.reference,
            {"DW": DW.clone(), "e": e.clone(), "g": g.clone()},
            rtol=1e-2,
            atol=1e-3,
            multiple_outputs=True,
            check_stride=False,
        )
