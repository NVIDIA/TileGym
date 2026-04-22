# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Tuple

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.flashinfer import ops as tilegym_flashinfer_ops


def _sota_impl_sgl_kernel_only(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    dst_dtype: torch.dtype,
    column_major_scales: bool,
    scale_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Thin wrapper around sgl_kernel (no sglang dependency). Only supports FP8, row-major scales."""
    from sgl_kernel import sgl_per_token_group_quant_8bit

    assert dst_dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "sgl_kernel v1 only supports FP8"
    assert not column_major_scales, "sgl_kernel v1 only supports row-major scales"
    finfo = torch.finfo(dst_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    sgl_per_token_group_quant_8bit(x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0, enable_v2=False)
    return x_q, x_s


def native_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of per-token group 8-bit quantization using PyTorch."""
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    if dst_dtype == torch.int8:
        bit8_min = float(torch.iinfo(dst_dtype).min)
        bit8_max = float(torch.iinfo(dst_dtype).max)
    else:
        bit8_min = torch.finfo(dst_dtype).min
        bit8_max = torch.finfo(dst_dtype).max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / bit8_max
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=bit8_min, max=bit8_max).to(dst_dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


class Test_Per_Token_Group_Quant_8bit(common.PyTestCase):
    @pytest.mark.parametrize("num_tokens", [512, 513])
    @pytest.mark.parametrize("hidden_dim", [2048])
    @pytest.mark.parametrize("group_size", [16, 32, 64, 128])
    @pytest.mark.parametrize("dst_dtype", [torch.float8_e4m3fn, torch.int8])
    @pytest.mark.parametrize("column_major_scales", [False, True])
    @pytest.mark.parametrize("scale_tma_aligned", [False, True])
    @pytest.mark.parametrize("scale_ue8m0", [False, True])
    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    def test_op(
        self,
        num_tokens,
        hidden_dim,
        group_size,
        dst_dtype,
        column_major_scales,
        scale_tma_aligned,
        scale_ue8m0,
        framework,
        arch,
    ):
        if framework == "cutile":
            if tilegym.is_backend_available("cutile"):
                tilegym.set_backend("cutile")
            else:
                pytest.skip("CuTile backend is not available")
        else:
            pytest.skip(f"Framework {framework} not supported")

        # scale_ue8m0 is a Blackwell-only feature (arch > 9); skip when testing it on older archs.
        arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        if scale_ue8m0 and arch_major <= 9:
            pytest.skip("scale_ue8m0 only relevant on Blackwell (arch > 9)")
        if scale_ue8m0 and not column_major_scales:
            pytest.skip("scale_ue8m0 requires column_major_scales=True")
        if scale_tma_aligned and not column_major_scales:
            pytest.skip("scale_tma_aligned requires column_major_scales=True")
        # Ampere (sm80) has no native FP8 support; kernel/backend may fail or be unsupported.
        if arch == "sm80" and "float8" in dst_dtype.__repr__():
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        device = "cuda:0"
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)

        # Reference (torch)
        x_q_ref, x_s_ref = native_per_token_group_quant_8bit(
            x,
            group_size=group_size,
            eps=1e-10,
            dst_dtype=dst_dtype,
            scale_ue8m0=scale_ue8m0,
        )

        # Implementation under test
        x_q, x_s = tilegym_flashinfer_ops.per_token_group_quant_8bit(
            x,
            group_size=group_size,
            eps=1e-10,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
        )

        # Compare quantized values (allow tiny diff for fp8/int8 rounding)
        torch.testing.assert_close(x_q_ref.float(), x_q.float(), atol=1e-2, rtol=2e-1)
        torch.testing.assert_close(
            x_s_ref.contiguous(),
            x_s.contiguous(),
            rtol=1e-3,
            atol=1e-5,
        )
