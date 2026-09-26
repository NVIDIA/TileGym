# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import importlib
import itertools
from types import SimpleNamespace

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available

from .. import common


class Test_BMM_FWD(common.PyTestCase):
    @staticmethod
    def reference(a, b, transpose_a=False, transpose_b=False):
        if transpose_a:
            a = torch.transpose(a, 1, 2)
        if transpose_b:
            b = torch.transpose(b, 1, 2)
        return torch.bmm(a, b)

    @pytest.mark.parametrize("backend", ["cutile"])
    @pytest.mark.parametrize("m,n,k", [(128, 256, 511), (129, 257, 65), (1024, 512, 1023)])
    @pytest.mark.parametrize("transpose_a,transpose_b", [(False, True), (True, False)])
    def test_op_static_persistent_cudagraph(self, backend, m, n, k, transpose_a, transpose_b, monkeypatch):
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("This multi-CTA regression targets SM100-family GPUs")

        module = importlib.import_module("tilegym.ops.cutile.bmm")
        config = SimpleNamespace(TILE_M=128, TILE_N=256, TILE_K=64, GROUP_SIZE_M=8, LATENCY=3, occupancy=2, num_ctas=2)
        a = torch.rand((4, k, m) if transpose_a else (4, m, k), dtype=torch.float16, device="cuda")
        b = torch.rand((4, n, k) if transpose_b else (4, k, n), dtype=torch.float16, device="cuda")
        key = (4, m, n, k, transpose_a, transpose_b, a.dtype, str(a.device))
        kernel = module._static_persistent_bmm_kernel.replace_hints(num_ctas=2, occupancy=2)
        monkeypatch.setattr(module, "_bmm_tune_cache", {key: (config, kernel)})

        def run():
            return tilegym.ops.bmm(a, b, transpose_a=transpose_a, transpose_b=transpose_b, static_persistent=True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        torch.cuda.current_stream().wait_stream(stream)

        for _ in range(3):
            a.uniform_()
            b.uniform_()
            expected = self.reference(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-8)

    _backends = ["cutile"]
    if is_backend_available("tilecpp"):
        _backends = _backends + ["tilecpp"]
    if is_backend_available("cutile-rs"):
        _backends = _backends + ["cutile-rs"]
    _perf_frameworks = _backends + ["pytorch"]

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

        if backend == "tilecpp" and static_persistent:
            pytest.skip("tilecpp static_persistent is under investigation")

        if backend == "cutile" and not static_persistent and (transpose_a or transpose_b):
            pytest.skip("CuTile non-persistent kernel doesn't support transpose")
        if backend == "cutile-rs" and not static_persistent and (transpose_a or transpose_b):
            pytest.skip("cutile-rs bmm non-persistent variant does not support transpose")

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

    @pytest.mark.parametrize(
        "q, m, n, k, transpose_a, transpose_b, dtype",
        [
            (q, 2**i, 2**i, 2**i, ta, tb, torch.float16)
            for q in [2, 8]
            for i in range(11, 14)
            for ta, tb in itertools.product([True, False], repeat=2)
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, q, m, n, k, transpose_a, transpose_b, dtype, framework, record_property):
        self.setUp()
        device = torch.device("cuda")

        if transpose_a:
            a_shape = (q, k, m)
        else:
            a_shape = (q, m, k)

        if transpose_b:
            b_shape = (q, n, k)
        else:
            b_shape = (q, k, n)

        a = torch.rand(a_shape, device=device, dtype=dtype)
        b = torch.rand(b_shape, device=device, dtype=dtype)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(a, b, transpose_a, transpose_b)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: tilegym.ops.bmm(
                a,
                b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
        else:
            pytest.skip(f"Framework {framework} is not available")

        if framework != "pytorch":
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(a, b, transpose_a, transpose_b),
                kwargs={},
                rtol=1e-3,
                atol=1e-8,
            )

        res = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del a, b, framework_fn
        if "dout" in locals():
            del dout
        torch.cuda.empty_cache()
        import gc

        gc.collect()
