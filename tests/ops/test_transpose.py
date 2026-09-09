# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common
from ..common import markif


class Test_Transpose(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @staticmethod
    def reference(inp):
        return inp.t().contiguous()

    @markif(lambda arch: arch in ["sm120", "sm121"], mark=pytest.mark.slow)
    @pytest.mark.parametrize(
        "m, n, dtype",
        [(2**s, 2**s, dt) for dt in [torch.float16, torch.float32] for s in range(1, 14, 1)],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        self.setUp()
        x = torch.rand(m, n, device=device, dtype=dtype, requires_grad=True)

        self.assertCorrectness(
            tilegym.ops.transpose,
            self.reference,
            {
                "inp": x,
            },
            gradient=torch.rand_like,
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "m, n, dtype",
        [(2**s, 2**s, dt) for dt in [torch.float16, torch.float32] for s in range(6, 15, 2)],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, m, n, dtype, backend, record_property):
        self.setUp()
        device = torch.device("cuda")
        x = torch.rand(m, n, device=device, dtype=dtype, requires_grad=True)
        if backend in self._backends:
            tilegym.set_backend(backend)
            backend_fn = lambda: tilegym.ops.transpose(x)
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(x)
        else:
            pytest.skip(f"Backend {backend} not supported")

        res = common.benchmark_framework(backend, backend_fn, use_cudagraph=False)
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del x, backend_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
