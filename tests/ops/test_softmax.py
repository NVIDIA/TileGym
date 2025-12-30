# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common


class Test_Softmax(common.PyTestCase):
    @staticmethod
    def reference(x):
        return torch.nn.functional.softmax(x, dim=-1)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m,n,dtype",
        [
            (256, 256, torch.float32),
            (256, 2048, torch.float32),
            (256, 1024 * 32, torch.float32),
            (256, 256, torch.float16),
            (256, 2048, torch.float16),
            (256, 9, torch.float32),
            (256, 1009, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("use_tma", [True, False], ids=["use_tma=True", "use_tma=False"])
    def test_op(self, m, n, dtype, arch, backend, use_tma):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x = torch.rand(
            m,
            n,
            device=device,
            dtype=dtype,
        )
        dout = torch.rand_like(x)

        if dtype == torch.float16:
            rtol, atol = 1e-3, 1e-5
        else:
            rtol, atol = 1e-5, 1e-7

        self.assertCorrectness(
            tilegym.ops.softmax,
            self.reference,
            {"x": x},
            extra_test_kwargs={"use_tma": use_tma},
            gradient=dout,
            rtol=rtol,
            atol=atol,
        )
