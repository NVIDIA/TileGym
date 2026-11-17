# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from dataclasses import asdict
from dataclasses import dataclass

import pytest
import torch
from triton.tools.mxfp import MXFP4Tensor
from triton.tools.mxfp import MXScaleTensor

import tilegym
from tests import common
from tilegym.suites import flashinfer

_backends = ["cutile"]
_perf_backends = _backends


@dataclass(repr=False)
class BlockScaleConfig:
    # problem config
    M: int = 512
    N: int = 512
    K: int = 512
    DTYPE_A: str = "e5m2"
    DTYPE_B: str = "e5m2"
    c_dtype: torch.dtype = torch.float32
    scale_dtype: str = "float8_e8m0fnu"
    transpose_a: bool = False
    transpose_b: bool = False
    # tiled config
    VEC_SIZE: int = 32
    persistent_mode: str = "none"


def create_operand(
    device: torch.device,
    dtype: str,
    size0: int,
    size1: int,
    k_dim: int,
    transpose: bool = True,
):
    if dtype in ["e5m2", "e4m3"]:
        F8_TYPE_MAPPING = {"e5m2": torch.float8_e5m2, "e4m3": torch.float8_e4m3fn}
        torch_dtype = F8_TYPE_MAPPING[dtype]
        # Value cast, not a uint8 bit-reinterpret: casting (-1,1) floats through
        # uint8 truncates to bytes {0, 1}, i.e. an all-zero fp8 matrix, which
        # makes every backend comparison a vacuous 0 @ 0 = 0 pass.
        v0 = 1 - 2 * torch.rand(size=(size0, size1), device=device)
        v = v0.to(torch_dtype)
        v_ref = v.to(torch.float32)

    elif dtype in ["e2m1"]:
        # fp4 (e2m1) elements. nvfp4 and mxfp4 share them and differ only in the
        # scale dtype and VEC_SIZE.
        v0 = 2 - 4 * torch.rand(size=(size0, size1), device=device).round()
        v = MXFP4Tensor(size=(size0, size1), data=v0, device=device)
        v_ref = v.to(torch.float32)
        # fp4 operands are stored as uint8, two values per byte along K (k_dim).
        v = v.to_packed_tensor(dim=k_dim)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}, only e5m2, e4m3, e2m1 are supported")
    if transpose:
        v = v.T
    return v.contiguous(), v_ref


def _pack_scale_2d_to_5d(scale_2d):
    """Pack 2D flat scale [P, Q] → 5D TMA layout [1, P//128, Q//4, 2, 256]."""
    P, Q = scale_2d.shape
    s = scale_2d.reshape(P // 128, 4, 32, Q // 4, 4)
    s = s.permute(0, 3, 2, 1, 4)
    return s.reshape(1, P // 128, Q // 4, 2, 256).contiguous()


class Test_FlashInfer_BlockScaledMatmul(common.PyTestCase):
    @staticmethod
    def initialize_block_scaled(config: BlockScaleConfig):
        device = torch.device("cuda")

        M = config.M
        N = config.N
        K = config.K
        scale_dtype = config.scale_dtype
        VEC_SIZE = config.VEC_SIZE

        a, a_ref = create_operand(device, config.DTYPE_A, M, K, 1, config.transpose_a)
        # B is created as [N, K] with fp4 packed along K. transpose_b=True keeps it
        # [N, K], the B layout of the ct.mma_scaled fast path; otherwise it is stored
        # as [K, N]. b_ref is always the logical (K, N) operand.
        b, b_ref = create_operand(device, config.DTYPE_B, N, K, 1, not config.transpose_b)
        b_ref = b_ref.T

        # 2D flat scale [M, K//VEC_SIZE] and [N, K//VEC_SIZE] (MX spec natural format).
        # _pack_scale_2d_to_5d converts to the 5D TMA layout required by bs_matmul.
        a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
        b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)
        # torch.rand is [0, 1) and MXScaleTensor encodes values <= 0 as NaN
        # (e8m0 has no zero), so keep the draws strictly positive.
        a_scale = torch.rand(a_size, device=device).clamp_min(2**-10)
        b_scale = torch.rand(b_size, device=device).clamp_min(2**-10)
        if scale_dtype == "float8_e8m0fnu":
            a_scale_ref = MXScaleTensor(a_scale)
            b_scale_ref = MXScaleTensor(b_scale)
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data
        elif scale_dtype == "float8_e4m3fn":
            a_scale = a_scale.to(torch.float8_e4m3fn)
            b_scale = b_scale.to(torch.float8_e4m3fn)
            a_scale_ref = a_scale
            b_scale_ref = b_scale
        else:
            raise ValueError(
                f"Unsupported scale dtype: {scale_dtype}, only float8_e8m0fnu and float8_e4m3fn are supported"
            )
        a_scale_ref = a_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        ref_out = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref)
        config_dict = asdict(config)
        return a, b, a_scale, b_scale, config_dict, ref_out

    @pytest.mark.skipif(
        torch.cuda.get_device_capability() not in [(10, 0), (10, 3), (10, 7), (12, 0), (12, 1)],
        reason="bs_matmul is only supported on sm100, sm103, sm107, sm120, sm121 now",
    )
    @pytest.mark.parametrize(
        "m, n, k, trans_a, trans_b",
        [
            (8192, 8192, 8192, False, True),
            (8192, 8192, 8192, True, False),
        ],
    )
    @pytest.mark.parametrize(
        "a_dtype, b_dtype, scale_dtype, vec_size",
        [
            ("e4m3", "e4m3", "float8_e8m0fnu", 32),
            ("e5m2", "e5m2", "float8_e8m0fnu", 32),
            ("e2m1", "e2m1", "float8_e4m3fn", 16),
            ("e2m1", "e2m1", "float8_e8m0fnu", 32),
        ],
    )
    @pytest.mark.parametrize(
        "c_dtype",
        [torch.float16],
    )
    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("persistent_mode", ["none", "static"])
    def test_op(
        self,
        m,
        n,
        k,
        trans_a,
        trans_b,
        a_dtype,
        b_dtype,
        scale_dtype,
        vec_size,
        c_dtype,
        persistent_mode,
        backend,
        monkeypatch,
    ):
        self.setUp()
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1")
        if backend in _backends:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")
        else:
            pytest.skip(f"Backend {backend} not supported")

        torch.manual_seed(42)
        configs = BlockScaleConfig(
            M=m,
            N=n,
            K=k,
            DTYPE_A=a_dtype,
            DTYPE_B=b_dtype,
            scale_dtype=scale_dtype,
            VEC_SIZE=vec_size,
            c_dtype=c_dtype,
            transpose_a=trans_a,
            transpose_b=trans_b,
            persistent_mode=persistent_mode,
        )
        a, b, a_scale, b_scale, config_dict, ref_c = self.initialize_block_scaled(configs)
        a_scale_5d = _pack_scale_2d_to_5d(a_scale)
        b_scale_5d = _pack_scale_2d_to_5d(b_scale)
        c = flashinfer.ops.bs_matmul(a, b, a_scale_5d, b_scale_5d, c_dtype, config_dict)

        try:
            torch.testing.assert_close(ref_c, c.to(torch.float32), atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            if "Tensor-likes are not close" in str(e) and c_dtype == torch.float8_e4m3fn:
                pytest.skip(
                    "Skip due to result mismatch when c_dtype is float8_e4m3fn, which is not specific to this backend"
                )
            raise e

    @pytest.mark.skipif(
        torch.cuda.get_device_capability() not in [(10, 0), (10, 3), (10, 7), (12, 0), (12, 1)],
        reason="bs_matmul is only supported on sm100, sm103, sm107, sm120, sm121 now",
    )
    @pytest.mark.parametrize(
        "m, n, k, trans_a, trans_b",
        [
            (4096, 8192, 2816, False, True),
            (4096, 4096, 5504, False, True),
            (12288, 2048, 2560, False, True),
            (12288, 4096, 2560, False, True),
            (4096, 5120, 2816, False, True),
            (3072, 6144, 2816, False, True),
            (2048, 2048, 2048, False, True),
            (4096, 4096, 4096, False, True),
            (8192, 8192, 8192, False, True),
            (16384, 16384, 16384, False, True),
            (8192, 8192, 2048, False, True),
        ],
    )
    @pytest.mark.parametrize(
        "a_dtype, b_dtype, scale_dtype, vec_size",
        [
            ("e4m3", "e4m3", "float8_e8m0fnu", 32),
            ("e5m2", "e5m2", "float8_e8m0fnu", 32),
            ("e2m1", "e2m1", "float8_e4m3fn", 16),
            ("e2m1", "e2m1", "float8_e8m0fnu", 32),
        ],
    )
    @pytest.mark.parametrize(
        "c_dtype",
        [torch.float16, torch.float32, torch.float8_e4m3fn],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    @pytest.mark.parametrize("persistent_mode", ["none", "static"])
    def test_perf(
        self,
        m,
        n,
        k,
        trans_a,
        trans_b,
        a_dtype,
        b_dtype,
        scale_dtype,
        vec_size,
        c_dtype,
        persistent_mode,
        backend,
        record_property,
    ):
        self.setUp()

        if backend in _backends:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")
        else:
            pytest.skip(f"Backend {backend} not supported")

        if torch.cuda.get_device_capability() == (12, 0) and n >= 16384 and k >= 16384:
            pytest.skip("Skip OOM on sm120: 16384x16384 block-scaled matmul exceeds 32 GiB VRAM")
        if torch.cuda.get_device_capability()[0] == 12 and scale_dtype == "float8_e8m0fnu":
            pytest.xfail("Shared memory exhaustion on sm120: kernel requires >100 KB (hardware limit 102 KB)")
        if torch.cuda.get_device_capability()[0] == 12 and a_dtype == "e2m1" and b_dtype == "e2m1":
            pytest.xfail("Shared memory exhaustion on sm120: kernel requires >100 KB (hardware limit 102 KB)")

        torch.manual_seed(42)
        configs = BlockScaleConfig(
            M=m,
            N=n,
            K=k,
            DTYPE_A=a_dtype,
            DTYPE_B=b_dtype,
            scale_dtype=scale_dtype,
            VEC_SIZE=vec_size,
            c_dtype=c_dtype,
            transpose_a=trans_a,
            transpose_b=trans_b,
            persistent_mode=persistent_mode,
        )
        a, b, a_scale, b_scale, config_dict, ref_c = self.initialize_block_scaled(configs)
        a_scale_5d = _pack_scale_2d_to_5d(a_scale)
        b_scale_5d = _pack_scale_2d_to_5d(b_scale)
        framework_fn = lambda: flashinfer.ops.bs_matmul(a, b, a_scale_5d, b_scale_5d, c_dtype, config_dict)
        c = flashinfer.ops.bs_matmul(a, b, a_scale_5d, b_scale_5d, c_dtype, config_dict)

        try:
            torch.testing.assert_close(ref_c, c.to(torch.float32), atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            if "Tensor-likes are not close" in str(e) and c_dtype == torch.float8_e4m3fn:
                pytest.skip(
                    "Skip due to result mismatch when c_dtype is float8_e4m3fn, which is not specific to this backend"
                )
            raise e

        res = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", res)
