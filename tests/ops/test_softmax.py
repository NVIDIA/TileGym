# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available

from .. import common


class Test_Softmax(common.PyTestCase):
    @staticmethod
    def reference(x):
        return torch.nn.functional.softmax(x, dim=-1)

    _backends = ["cutile"]
    if is_backend_available("tilecpp"):
        _backends = _backends + ["tilecpp"]
    _perf_frameworks = _backends + ["pytorch"]

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
    @pytest.mark.parametrize(
        "use_tma,use_chunked,use_multi_wave",
        [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
        ids=["baseline", "use_tma", "use_chunked", "use_multi_wave"],
    )
    def test_op(self, m, n, dtype, arch, backend, use_tma, use_chunked, use_multi_wave):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
            self.setUp()
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
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
            extra_test_kwargs={"use_tma": use_tma, "use_chunked": use_chunked, "use_multi_wave": use_multi_wave},
            gradient=dout,
            rtol=rtol,
            atol=atol,
        )
