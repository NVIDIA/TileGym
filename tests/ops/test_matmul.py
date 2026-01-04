# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

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
