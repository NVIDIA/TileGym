# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import math
import tempfile

import pytest
import torch

import tilegym
import tilegym.ops
from tests import common
from tilegym.backend import set_backend

FP4_MAX = 6.0
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _is_tileir_v13_3_available() -> bool:
    """Return True if the tileiras compiler supports V_13_3 bytecode (required for fp4)."""
    try:
        from cuda.tile._bytecode.version import BytecodeVersion
        from cuda.tile._cext import dev_features_enabled
        from cuda.tile._compile import _get_max_supported_bytecode_version

        return (
            dev_features_enabled()
            and _get_max_supported_bytecode_version(tempfile.gettempdir(), allow_dev=True) >= BytecodeVersion.V_13_3
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_tileir_v13_3_available(),
    reason="Requires TileIR bytecode V_13_3+ (fp4_e2m1fn / pack_to_bytes)",
)

_backends = ["cutile"]


class TestNVFP4Quantize(common.PyTestCase):
    # ------------------------------------------------------------------ #
    # Output shapes                                                        #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "rows,cols",
        [
            (128, 64),  # single tile, rows%128==0, cols%64==0
            (512, 256),  # multiple tiles, rows%128==0, cols%64==0
            (128, 80),  # partial col (80%64==16)
            (256, 48),  # partial col (48%64!=0)
            (64, 64),  # partial row (64%128!=0)
            (192, 64),  # partial row (192%128==64)
            (160, 80),  # partial row AND partial col
            (17, 64),  # rows not divisible by 16
            (33, 48),  # rows not divisible by 16, partial col
            (1, 64),  # single row (extreme partial row block)
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_output_shapes(self, rows, cols, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
        s_enc = (FP4_MAX * FP8_E4M3_MAX) / (x.abs().max().item() + 1e-12)
        packed, scales = tilegym.ops.nvfp4_quantize(x, s_enc=s_enc)
        num_n_blocks = math.ceil(cols / 64)
        num_m_blocks = math.ceil(rows / 128)
        num_tiles = num_m_blocks * num_n_blocks
        assert packed.shape == (rows, cols // 2), f"packed shape: {packed.shape}"
        assert scales.shape == (num_tiles * 512,), f"scales shape: {scales.shape}"
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    # ------------------------------------------------------------------ #
    # Input dtypes (smoke test)                                            #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_input_dtypes(self, dtype, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.randn(128, 64, dtype=dtype, device="cuda")
        s_enc = (FP4_MAX * FP8_E4M3_MAX) / (x.abs().max().item() + 1e-12)
        packed, scales = tilegym.ops.nvfp4_quantize(x, s_enc=s_enc)
        assert packed.shape == (128, 32)
        assert scales.shape == (512,)

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("backend", _backends)
    def test_op_invalid_inputs(self, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with pytest.raises(ValueError):
            tilegym.ops.nvfp4_quantize(torch.randn(4, 128, 64, dtype=torch.bfloat16, device="cuda"))
        with pytest.raises(ValueError):
            tilegym.ops.nvfp4_quantize(torch.randint(0, 127, (128, 64), dtype=torch.int8, device="cuda"))
        with pytest.raises(ValueError):
            tilegym.ops.nvfp4_quantize(torch.randn(128, 63, dtype=torch.bfloat16, device="cuda"))
