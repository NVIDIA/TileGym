# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth SwiGLU forward and backward kernels."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import swiglu_bwd
from tilegym.suites.unsloth.ops import swiglu_fg

DEVICE = "cuda"

_backends = ["cutile"]


class Test_Unsloth_SwiGLU_Forward(common.PyTestCase):
    @staticmethod
    def reference(e, g):
        """PyTorch reference: h = silu(e) * g = (e * sigmoid(e)) * g"""
        e_f32 = e.float()
        f = e_f32 * torch.sigmoid(e_f32)
        f = f.to(g.dtype)
        return f * g

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128, 256),
            (4, 64, 512),
            (1, 1, 1024),
            (2, 256, 4096),
            # Production-like: batch*seq x intermediate_size
            (4, 512, 5120),  # Llama-like intermediate
            (2, 1024, 8192),  # Large intermediate
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend):
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        torch.manual_seed(42)
        e = torch.randn(*shape, dtype=dtype, device=DEVICE)
        g = torch.randn(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            swiglu_fg,
            self.reference,
            {"e": e, "g": g},
            rtol=1e-2,
            atol=1e-3,
        )


class Test_Unsloth_SwiGLU_Backward(common.PyTestCase):
    @staticmethod
    def reference(DW, e, g):
        """
        PyTorch reference for SwiGLU backward (in-place semantics).

        Returns (h, df, de) where:
          h = silu(e) * g  (forward output, stored into DW)
          df = DW * f       (stored into e)
          de = DW * g * sigmoid(e) * (1 + e*(1-sigmoid(e)))  (stored into g)
        """
        e_f32 = e.float()
        se = torch.sigmoid(e_f32)
        f = se * e_f32
        f = f.to(DW.dtype)
        h = f * g
        df = DW * f
        dg = DW * g
        de = dg.float() * se * (1.0 + e_f32 * (1.0 - se))
        de = de.to(DW.dtype)
        return h, df, de

    @pytest.mark.parametrize(
        "M, N",
        [
            (256, 512),
            (1024, 1024),
            (512, 4096),
            (2048, 5120),
            (4096, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, M, N, dtype, backend):
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        torch.manual_seed(42)
        DW = torch.randn(M, N, dtype=dtype, device=DEVICE)
        e = torch.randn(M, N, dtype=dtype, device=DEVICE)
        g = torch.randn(M, N, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            swiglu_bwd,
            self.reference,
            {"DW": DW.clone(), "e": e.clone(), "g": g.clone()},
            rtol=1e-2,
            atol=1e-3,
            multiple_outputs=True,
            check_stride=False,
        )
