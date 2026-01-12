# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import itertools
import os

import pytest
import torch

import tilegym

from .. import common


class Test_BMM_FWD(common.PyTestCase):
    @staticmethod
    def reference(a, b, transpose_a=False, transpose_b=False):
        if transpose_a:
            a = torch.transpose(a, 1, 2)
        if transpose_b:
            b = torch.transpose(b, 1, 2)
        return torch.bmm(a, b)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size, m, n, k, transpose_a, transpose_b, dtype",
        [
            (batch_size, m, n, k, transpose_a, transpose_b, dtype)
            for batch_size in [
                4,
            ]
            for m in [
                128,
                1024,
            ]
            for n in [
                256,
                512,
            ]
            for k in [511, 512, 1023, 1024]
            for transpose_a in [True, False]
            for transpose_b in [True, False]
            for dtype in [torch.float16]
        ],
    )
    @pytest.mark.parametrize("static_persistent", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        dtype,
        static_persistent,
        backend,
    ):
        device = torch.device("cuda")
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        if backend == "cutile" and not static_persistent and (transpose_a or transpose_b):
            pytest.skip("CuTile non-persistent kernel doesn't support transpose")

        if transpose_a:
            a_shape = (batch_size, k, m)
        else:
            a_shape = (batch_size, m, k)

        if transpose_b:
            b_shape = (batch_size, n, k)
        else:
            b_shape = (batch_size, k, n)

        a = torch.rand(a_shape, device=device, dtype=dtype)
        b = torch.rand(b_shape, device=device, dtype=dtype)
        self.assertCorrectness(
            tilegym.ops.bmm,
            self.reference,
            {
                "a": a,
                "b": b,
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
            },
            extra_test_kwargs={
                "static_persistent": static_persistent,
            },
            rtol=1e-3,
            atol=1e-8,
        )
