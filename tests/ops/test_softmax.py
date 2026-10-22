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
            requires_grad=True,
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

    @pytest.mark.parametrize("transposed", [False, True], ids=["contiguous", "transposed"])
    @pytest.mark.parametrize("use_tma", [False, True], ids=["use_tma=False", "use_tma=True"])
    @pytest.mark.parametrize("n", [1000, 1024, 3000])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_fully_written(self, n, dtype, use_tma, transposed, backend):
        """Forward/backward on non-power-of-2 widths and transposed inputs.

        ``assertCorrectness`` runs the reference backward first and frees its
        gradient buffer, so an op that never writes its ``grad_input`` can
        return the reference gradient by recycling that buffer and still pass.
        Here the op's backward runs first, after the caching allocator has been
        filled with NaN, so any element the kernels leave unwritten shows up as
        a mismatch.
        """
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        if backend != "cutile" and transposed:
            pytest.skip("Transposed input is only supported by the cutile softmax")

        self.setUp()
        torch.manual_seed(8601 + n)
        device = torch.device("cuda")
        m = 64
        if transposed:
            x = torch.randn(n, m, device=device, dtype=dtype).t()
        else:
            x = torch.randn(m, n, device=device, dtype=dtype)
        dout = torch.randn(m, n, device=device, dtype=dtype)
        x_test = x.detach().requires_grad_(True)

        # Leave NaN in the allocator's free blocks so unwritten outputs are visible.
        torch.full((4 * m * n,), float("nan"), device=device, dtype=dtype)
        y = tilegym.ops.softmax(x_test, use_tma=use_tma, use_online=False)
        y.backward(dout)
        torch.cuda.synchronize()

        x_ref = x.detach().float().requires_grad_(True)
        y_ref = self.reference(x_ref)
        y_ref.backward(dout.float())

        tol = 1e-1 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(y.float(), y_ref, rtol=tol, atol=tol)
        torch.testing.assert_close(x_test.grad.float(), x_ref.grad, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "m,n,dtype",
        [(4096, 2**i, torch.float32) for i in range(10, 15)],
        ids=lambda x: str(x),
    )
    @pytest.mark.parametrize("use_tma", [True, False], ids=["use_tma=True", "use_tma=False"])
    @pytest.mark.parametrize("backend", _perf_frameworks)
    def test_perf(self, m, n, dtype, use_tma, backend, record_property):
        if backend in self._backends:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")

        if torch.cuda.get_device_capability() == (12, 0) and n >= 16384 and dtype == torch.float32:
            pytest.skip("Skip OOM on sm120: large float32 softmax exceeds 32 GiB VRAM")

        self.setUp()
        device = torch.device("cuda")
        x = torch.rand(m, n, device=device, dtype=dtype, requires_grad=True)

        if backend in self._backends:
            backend_fn = lambda: tilegym.ops.softmax(x, use_tma=use_tma, use_online=False)
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(x)
        else:
            pytest.skip(f"Backend {backend} not supported")

        res = common.benchmark_framework(backend, backend_fn)
        if backend != "pytorch":
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(x),
                {},
                gradient=None,
            )
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del x, backend_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @pytest.mark.parametrize(
        "m,n,dtype",
        [(4096, 2**i, torch.float32) for i in range(14, 16)],
        ids=lambda x: str(x),
    )
    @pytest.mark.parametrize("use_tma", [True, False], ids=["use_tma=True", "use_tma=False"])
    @pytest.mark.parametrize("backend", _backends + ["pytorch"])
    def test_perf_online_softmax(self, m, n, dtype, use_tma, backend, record_property):
        if backend in self._backends:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")

        if torch.cuda.get_device_capability() == (12, 0) and n >= 16384 and dtype == torch.float32:
            pytest.skip("Skip OOM on sm120: large float32 softmax exceeds 32 GiB VRAM")

        # Skip if single buffer exceeds 1 GiB (known OOM in full suite due to memory accumulation)
        if m * n * 4 > 1 * 1024**3:
            pytest.skip(
                f"Skipping {m}x{n} float32: single buffer > 1 GiB, "
                "known OOM in full test suite due to memory accumulation."
            )

        self.setUp()
        device = torch.device("cuda")
        x = torch.rand(m, n, device=device, dtype=dtype, requires_grad=True)

        if backend in self._backends:
            backend_fn = lambda: tilegym.ops.softmax(x, use_tma=use_tma, use_online=True)
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(x)
        else:
            pytest.skip(f"Backend {backend} not supported")

        res = common.benchmark_framework(backend, backend_fn)
        if backend != "pytorch":
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(x),
                {},
                gradient=None,
            )
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del x, backend_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
