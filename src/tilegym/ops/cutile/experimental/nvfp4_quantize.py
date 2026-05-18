# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from functools import lru_cache
from typing import Optional

import cuda.tile as ct
import numpy as np
import torch
from cuda.tile._datatype import float4_e2m1fn
from cuda.tile._datatype import float8_e4m3fn
from cuda.tile._stub import pack_to_bytes

from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

ct.pack_to_bytes = pack_to_bytes

FP4_MAX = 6.0
E4M3_EPS = 1.5258789e-05

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# Currently uses a fixed 128x64 tile (128 rows x 64 cols per CTA).
# Future work: support additional tile sizes (e.g. 64x64, 256x64) to enable
# auto-tuning across tile configurations for different matrix shapes.
@lru_cache(maxsize=None)
def _make_nvfp4_kernel(occupancy: int):
    @experimental_kernel
    @ct.kernel(occupancy=occupancy)
    def _nvfp4_quant_kernel(
        x,  # bf16/fp16/fp32 [rows, cols]
        scales_out,  # uint8 [num_tiles * 512]  — swizzled fp8 e4m3fn
        packed_out,  # uint8 [rows, cols//2]  — packed fp4
        s_enc,  # float32 [1]  — global encoding scale (runtime)
        NUM_N_BLOCKS: ConstInt,
        NUM_M_BLOCKS: ConstInt,
        NUM_COLS: ConstInt,  # actual unpadded col count
        NUM_ROWS: ConstInt,  # actual unpadded row count
        USE_COL_MASK: ConstBool,  # True when cols % 64 != 0
        USE_ROW_MASK: ConstBool,  # True when rows % 128 != 0
    ):
        blk_n = ct.bid(0)
        blk_m = ct.bid(1)

        # TMA zero-pads OOB rows and cols in hardware, so ct.load works for all
        # blocks including partial last row/col blocks.
        block = ct.load(x, index=(blk_m, blk_n), shape=(128, 64))  # 128x64 tile
        block_3d = ct.reshape(ct.astype(block, ct.float32), (128, 4, 16))
        block_max = ct.max(ct.abs(block_3d), axis=2, keepdims=True)

        s_enc_3d = ct.reshape(ct.load(s_enc, index=(0,), shape=(1,)), (1, 1, 1))
        scale_pre = ct.maximum((block_max / FP4_MAX) * s_enc_3d, E4M3_EPS)

        # Zero scales for OOB col groups in the last partial col block
        if USE_COL_MASK and blk_n == NUM_N_BLOCKS - 1:
            valid_col_grps = (NUM_COLS % 64) // 16
            grp_idx = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 4, 1))
            scale_pre = ct.where(grp_idx < valid_col_grps, scale_pre, 0.0)

        # Zero scales for OOB rows in the last partial row block
        if USE_ROW_MASK and blk_m == NUM_M_BLOCKS - 1:
            valid_rows = NUM_ROWS % 128
            row_idx = ct.reshape(ct.arange(128, dtype=ct.int32), (128, 1, 1))
            scale_pre = ct.where(row_idx < valid_rows, scale_pre, 0.0)

        scale_e4m3 = ct.astype(scale_pre, float8_e4m3fn)
        scale_u8 = ct.bitcast(scale_e4m3, np.uint8)

        # Swizzle the 128x4 logical scale grid into a 512-byte block.
        #
        # Step 1 — split the 128 rows into four 32-row groups:
        #
        #    group 0        group 1        group 2        group 3
        #   (rows 0-31)   (rows 32-63)  (rows 64-95)  (rows 96-127)
        #   +---------+   +---------+   +---------+   +---------+
        #   |r0  c0-3 |   |r32 c0-3 |   |r64 c0-3 |   |r96 c0-3 |
        #   |r1  c0-3 |   |r33 c0-3 |   |r65 c0-3 |   |r97 c0-3 |
        #   |   ...   |   |   ...   |   |   ...   |   |   ...   |
        #   |r31 c0-3 |   |r63 c0-3 |   |r95 c0-3 |   |r127 c0-3|
        #   +---------+   +---------+   +---------+   +---------+
        #
        # Step 2 — interleave: sub-row s takes row index s from each group,
        # yielding 32 sub-rows of 16 bytes (4 groups x 4 bytes):
        #
        #     s= 0: [r0 c0-3 | r32 c0-3 | r64 c0-3 | r96 c0-3]  <- bytes   0-15
        #     s= 1: [r1 c0-3 | r33 c0-3 | r65 c0-3 | r97 c0-3]  <- bytes  16-31
        #      ...
        #     s=31: [r31 c0-3| r63 c0-3 | r95 c0-3 |r127 c0-3]  <- bytes 496-511
        #           ^-4 bytes^
        #
        # Expanded, each sub-row s (s = 0..31) holds one scale per col-group for the
        # same relative row index from each group:
        #
        #                group 0          group 1          group 2          group 3
        #               (rows 0-31)      (rows 32-63)     (rows 64-95)    (rows 96-127)
        #     byte:   0   1   2   3  |  4   5   6   7  |  8   9  10  11  | 12  13  14  15
        #            +---------------+----------------+----------------+----------------+
        #     s= 0   | r0  c0 c1 c2 c3| r32 c0 c1 c2 c3| r64 c0 c1 c2 c3| r96 c0 c1 c2 c3|
        #     s= 1   | r1  c0 c1 c2 c3| r33 c0 c1 c2 c3| r65 c0 c1 c2 c3| r97 c0 c1 c2 c3|
        #     s= 2   | r2  c0 c1 c2 c3| r34 c0 c1 c2 c3| r66 c0 c1 c2 c3| r98 c0 c1 c2 c3|
        #      ...   |      ...       |      ...       |      ...       |      ...       |
        #     s=31   |r31  c0 c1 c2 c3| r63 c0 c1 c2 c3| r95 c0 c1 c2 c3|r127 c0 c1 c2 c3|
        #            +---------------+----------------+----------------+----------------+
        #
        # Each byte is one fp8 e4m3fn scale. Col-groups cover the input columns:
        #     c0 = cols  0-15    c1 = cols 16-31    c2 = cols 32-47    c3 = cols 48-63
        #
        # byte offset of scale[r, c]:  (r % 32) * 16  +  (r // 32) * 4  +  c
        #
        # Implemented as: [128, 4, 1] -> reshape [4, 32, 1, 4] -> permute(2,1,0,3) -> [1, 32, 4, 4] -> reshape [512]
        scale_4d = ct.reshape(scale_u8, (4, 32, 1, 4))
        scale_swizzled = ct.reshape(ct.permute(scale_4d, (2, 1, 0, 3)), (512,))

        # Store 512-byte swizzle block to its slot in the 1D scales buffer
        tile_idx = blk_m * NUM_N_BLOCKS + blk_n
        ct.store(scales_out, index=(tile_idx,), tile=scale_swizzled)

        enc_scale = s_enc_3d / ct.astype(scale_e4m3, ct.float32)
        fp4 = ct.astype(block_3d * enc_scale, float4_e2m1fn)
        packed = ct.reshape(ct.pack_to_bytes(fp4), (128, 32))

        ct.store(packed_out, index=(blk_m, blk_n), tile=packed)

    return _nvfp4_quant_kernel


def tile_nvfp4_quantize(
    x: torch.Tensor,
    s_enc: float = 1.0,
    occupancy: Optional[int] = None,
    **kwargs,
):
    """
    Host launcher for NVFP4 quantization (cuTile backend).

    Args:
        x:          bf16, fp16, or fp32 input tensor, shape (rows, cols).
                    cols must be divisible by 16.
        s_enc:      Global encoding scale (float, default 1.0).
        occupancy:  CTA occupancy hint for the kernel (default: None, uses ct.kernel default).
                    Different values compile to separate kernel variants.

    Returns:
        packed_out: uint8 tensor of shape (rows, cols // 2) -- two fp4 values per byte.
        scales_out: uint8 tensor of shape (num_tiles * 512,) -- swizzled fp8 e4m3fn scales,
                    where num_tiles = ceil(rows / tile_m) * ceil(cols / tile_n).
    """
    if x.dim() != 2:
        raise ValueError(f"nvfp4_quantize expects a 2D input, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"nvfp4_quantize requires bf16, fp16, or fp32 input, got {x.dtype}")
    rows, cols = x.shape
    if cols % 16 != 0:
        raise ValueError(f"cols must be divisible by 16, got {cols}")

    x = x.contiguous()
    tile_m, tile_n = 128, 64
    num_m_blocks = (rows + tile_m - 1) // tile_m
    num_n_blocks = (cols + tile_n - 1) // tile_n
    use_row_mask = rows % tile_m != 0
    use_col_mask = cols % tile_n != 0
    num_tiles = num_m_blocks * num_n_blocks

    scales_out = torch.zeros(num_tiles * 512, dtype=torch.uint8, device=x.device)
    packed_out = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=x.device)
    s_enc_tensor = torch.tensor([s_enc], dtype=torch.float32, device=x.device)

    kernel = _make_nvfp4_kernel(occupancy)
    ct.launch(
        torch.cuda.current_stream(),
        (num_n_blocks, num_m_blocks, 1),
        kernel,
        (x, scales_out, packed_out, s_enc_tensor, num_n_blocks, num_m_blocks, cols, rows, use_col_mask, use_row_mask),
    )
    return packed_out, scales_out


register_impl("nvfp4_quantize", backend="cutile")(tile_nvfp4_quantize)
