# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym

from .. import common


class Test_Matmul(common.PyTestCase):
    @staticmethod
    def reference(a, b, trans_a=False, trans_b=False):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        if a.dtype == torch.float8_e4m3fn:
            # NOTE: float8_e4m3fn is not supported in pytorch, so we convert it to float16 and then convert it back to float8_e4m3fn
            # This is a workaround to avoid torch error
            a_fp16 = a.to(torch.float16)
            b_fp16 = b.to(torch.float16)
            return (a_fp16 @ b_fp16).to(torch.float8_e4m3fn)
        else:
            return a @ b

    @staticmethod
    def prepare_data(m, n, k, trans_a, trans_b, offset_a, offset_b, dtype):
        device = torch.device("cuda")

        assert offset_a <= 64
        assert offset_b <= 64

        a_size = m * k + offset_a
        b_size = k * n + offset_b
        if dtype == torch.float8_e4m3fn:
            a = torch.rand(a_size, device=device, dtype=torch.float16, requires_grad=False).normal_(std=0.3).to(dtype)
            b = torch.rand(b_size, device=device, dtype=torch.float16, requires_grad=False).normal_(std=0.3).to(dtype)
        else:
            a = torch.rand(a_size, device=device, dtype=dtype, requires_grad=True)
            b = torch.rand(b_size, device=device, dtype=dtype, requires_grad=True)

        if trans_a:
            a = a[offset_a:].view(k, m).detach().contiguous().requires_grad_()
        else:
            a = a[offset_a:].view(m, k).detach().contiguous().requires_grad_()
        if trans_b:
            b = b[offset_b:].view(n, k).detach().contiguous().requires_grad_()
        else:
            b = b[offset_b:].view(k, n).detach().contiguous().requires_grad_()

        alignment_a = common.get_tensor_alignment(a) % 64
        alignment_b = common.get_tensor_alignment(b) % 64

        assert alignment_a == offset_a * a.element_size()
        assert alignment_b == offset_b * b.element_size()
        return a, b

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "m, n, k, offset_a, offset_b, dtype",
        [
            (1024, 1024, 1024, 0, 0, torch.bfloat16),
            (1024, 1024, 1023, 0, 0, torch.bfloat16),
            (16384, 16384, 16384, 0, 0, torch.bfloat16),
            (8, 8, 8, 0, 0, torch.bfloat16),
            (3072, 6144, 2720, 0, 0, torch.bfloat16),
        ],
        ids=lambda x: (
            str(x) if isinstance(x, list) else f"{x.__module__}.{x.__name__}" if hasattr(x, "__name__") else str(x)
        ),
    )
    @pytest.mark.parametrize(
        "static_persistent",
        [True, False],
        ids=["static_persistent=True", "static_persistent=False"],
    )
    @pytest.mark.parametrize("use_tma", [True, False], ids=["use_tma=True", "use_tma=False"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        m,
        n,
        k,
        offset_a,
        offset_b,
        dtype,
        static_persistent,
        use_tma,
        backend,
        arch,
        request,
    ):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        if arch in ["sm120", "sm121"] and n >= 6144:
            pytest.skip("Skip due to global memory OOM")
        if k == 1023:
            pytest.skip("Skip matmul due to result mismatch when cannot divide BLOCK")
        self.setUp()
        a, b = self.prepare_data(m, n, k, False, False, offset_a, offset_b, dtype)
        self.assertCorrectness(
            tilegym.ops.matmul,
            self.reference,
            {
                "a": a,
                "b": b,
                "trans_a": False,
                "trans_b": False,
            },
            extra_test_kwargs={
                "static_persistent": static_persistent,
                "use_tma": use_tma,
            },
            gradient=torch.rand_like,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "m,n,k,offset_a,offset_b,dtype",
        [
            (2**i, 2**i, 2**i, 0, 0, dtype)
            for i in list(range(11, 16)) + [6, 8]
            for dtype in ([torch.float16, torch.float32, torch.float8_e4m3fn])
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("transpose_a", [False, True])
    @pytest.mark.parametrize("transpose_b", [False, True])
    @pytest.mark.parametrize("static_persistent", [False, True])
    @pytest.mark.parametrize("use_tma", [False] if torch.cuda.get_device_capability()[0] == 8 else [True])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(
        self,
        m,
        n,
        k,
        offset_a,
        offset_b,
        transpose_a,
        transpose_b,
        static_persistent,
        use_tma,
        dtype,
        framework,
        record_property,
    ):
        self.setUp()
        # Enforce SM80 restrictions: use_tma=False, static_persistent=False, dtype=float16 only
        if torch.cuda.get_device_capability()[0] == 8:
            if use_tma != False or static_persistent != False or dtype != torch.float16:
                pytest.skip(
                    "SM80 restriction: use_tma must be False, static_persistent must be False, and dtype must be float16"
                )

        # Skip FP8 for pytorch reference (no native support)
        if dtype == torch.float8_e4m3fn and framework == "pytorch":
            pytest.skip("Skip float8_e4m3fn because pytorch reference doesn't support it")
        # xfail on sm121 for 32768x32768 matmul due to performance
        if torch.cuda.get_device_capability() == (12, 1) and m == 32768:
            pytest.skip("32768x32768 matmul takes too long on sm121")
        if torch.cuda.get_device_capability() == (12, 0) and m == 32768:
            pytest.skip("Skip OOM on B20X (sm120): 32768³ matmul exceeds 32 GiB VRAM")
        if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip due to sm80 not support fp8 type")
        if framework == "cutile" and not static_persistent and transpose_a:
            pytest.skip("Cutile transpose_a is not supported when static_persistent is False")

        a, b = self.prepare_data(m, n, k, transpose_a, transpose_b, offset_a, offset_b, dtype)
        kernel_kwargs = {
            "trans_a": transpose_a,
            "trans_b": transpose_b,
            "static_persistent": static_persistent,
            "use_tma": use_tma,
        }
        if framework == "pytorch":
            framework_fn = lambda: self.reference(a, b, transpose_a, transpose_b)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            if framework == "cutile" and transpose_b:
                pytest.skip("[matmul] cutile transpose_b is not supported")
            framework_fn = lambda: tilegym.ops.matmul(a, b, **kernel_kwargs)
        else:
            pytest.skip(f"Framework {framework} is not available")
        skip_correctness = framework == "pytorch"
        if not skip_correctness:
            if dtype == torch.float8_e4m3fn:
                atol = 1
                rtol = 1
                # float8 doesn't support autograd, disable requires_grad for correctness check
                a = a.detach()
                b = b.detach()
            else:
                atol = 1e-2
                rtol = 1e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(a, b, trans_a=transpose_a, trans_b=transpose_b),
                kwargs={},
                atol=atol,
                rtol=rtol,
            )
        try:
            res = common.benchmark_framework(framework, framework_fn, use_cudagraph=False)
        except torch.OutOfMemoryError as e:
            pytest.skip(f"OOM during benchmark: {e}")
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del a, b, framework_fn
        torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.parametrize(
        "model,m,n,k,offset_a,offset_b,dtype",
        [
            ("gpt3-40b", 4096, 8192, 2728, 0, 0, torch.float16),
            ("gpt3-7b", 4096, 4096, 5440, 0, 0, torch.float16),
            ("t5-3b", 12288, 2048, 2560, 0, 0, torch.float16),
            ("t5-11b", 12288, 4096, 2560, 0, 0, torch.float16),
            ("t5-23b", 4096, 5120, 2720, 0, 0, torch.float16),
            ("t5-41b", 3072, 6144, 2720, 0, 0, torch.float16),
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("static_persistent", [True, False])
    @pytest.mark.parametrize("use_tma", [True])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf_llm(
        self,
        model,
        m,
        n,
        k,
        offset_a,
        offset_b,
        dtype,
        static_persistent,
        use_tma,
        framework,
        record_property,
    ):
        self.setUp()
        if torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip on sm80")

        a, b = self.prepare_data(m, n, k, False, False, offset_a, offset_b, dtype)
        if framework == "pytorch":
            framework_fn = lambda: self.reference(a, b)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            kernel_kwargs = {
                "trans_a": False,
                "trans_b": False,
                "static_persistent": static_persistent,
                "use_tma": use_tma,
            }
            framework_fn = lambda: tilegym.ops.matmul(a, b, **kernel_kwargs)
        else:
            pytest.skip(f"Framework {framework} is not available")
        skip_correctness = framework == "pytorch"
        if not skip_correctness:
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(a, b),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
            )
        res = common.benchmark_framework(framework, framework_fn, use_cudagraph=False)
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del a, b, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
