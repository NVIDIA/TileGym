---
id: ragged-tma-fixed-descriptor
kind: reference
title: Fixed-descriptor ragged row access (code snapshot)
summary: BF16 virtual metadata and cuTile coordinates for fixed-descriptor ragged access.
used_by: [tech-ragged-tma-fixed-descriptor]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->


# Fixed-descriptor ragged row access

This minimal copy example shows the Python-to-kernel interface. `RaggedTensor`
provides virtual shape and byte strides; the annotation specializes those values,
it does not create them. The same wrapper applies to matmul inputs and outputs.

```python
from typing import Annotated
import cuda.tile as ct
import torch

ROW_BOUND = 1 << 30
RaggedArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(index_dtype=ct.int64,
                       static_shape_dims=(0, 1, 2, 3),
                       static_stride_dims=(0, 1, 2, 3)),
]

class RaggedTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        row_stride = tensor.stride(0)
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (0x7FFF0000, 0x7FFF0000, ROW_BOUND, tensor.shape[1]),
            "strides": (2 * ((1 << 34) - row_stride),
                        2 * row_stride, 2 * row_stride, 2),
            "typestr": "<V2",  # cuTile BF16; strides are in bytes.
            "data": (tensor.data_ptr(), False),
            "stream": torch.cuda.current_stream().cuda_stream or 1,
        }

@ct.kernel
def copy_ragged(input_view: RaggedArray, output_view: RaggedArray, chunk_bounds,
                BLOCK_M: ct.Constant[int], WIDTH: ct.Constant[int]):
    start = ct.load(chunk_bounds, index=(ct.bid(0), 0), shape=(), allow_tma=False)
    end = ct.load(chunk_bounds, index=(ct.bid(0), 1), shape=(), allow_tma=False)
    coords = (ROW_BOUND, end, ROW_BOUND - (end - start), 0)
    input_rows = input_view.tiled_view(
        (1, 1, BLOCK_M, WIDTH), padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, 1, 1, WIDTH))
    output_rows = output_view.tiled_view(
        (1, 1, BLOCK_M, WIDTH), padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, 1, 1, WIDTH))
    output_rows.store(coords, input_rows.load(coords, allow_tma=True), allow_tma=True)

def launch_copy(input_tensor, output_tensor, chunk_bounds, block_m=64):
    if chunk_bounds.shape[0] == 0:
        return
    stream = torch.cuda.current_stream()
    for tensor in (input_tensor, output_tensor, chunk_bounds):
        tensor.record_stream(stream)
    kernel_args = (RaggedTensor(input_tensor), RaggedTensor(output_tensor),
                   chunk_bounds, block_m, input_tensor.shape[1])
    ct.launch(stream, (chunk_bounds.shape[0],), copy_ragged, kernel_args)
```

Pass real tensors to `launch_copy`; it passes the wrappers to `ct.launch`.
Each wrapper keeps its backing tensor, uses its own row stride, and supplies
BF16 `"<V2"`, not FP16 `"<f2"`. No storage of the virtual shape is allocated.

The caller supplies equal-width CUDA BF16 matrices on the current device, with
unit inner strides, 16-byte-aligned bases and positive row strides divisible by
eight elements. The copy loads each row as one full-width tile: width and tile height must be
legal power-of-two tile sizes, with at least eight BF16 elements per row (16 bytes).
The tile must also satisfy the selected TMA layout and per-load size limits; see
`tech-tma-load`. Larger widths require a column-tiled kernel.
Each backing matrix must have at most `ROW_BOUND` rows. Each int32 bounds pair
must satisfy `0 <= start <= end <= min(real input rows, real output rows)`; owned intervals
`[start, min(start + block_m, end))` must not overlap. The caller orders tensor
production before this launch; `record_stream` protects lifetime, not producer
synchronization. Untouched output rows retain their prior contents.

This copy example has not been GPU-validated. See `tech-ragged-tma-fixed-descriptor`
for the address proof, remaining stride/TMA constraints and the historical
matmul validation scope. In particular, inspect the selected TMA lowering before
relying on the virtual bounds. Adapted from
[Triton's ragged TMA construction](https://github.com/triton-lang/triton/blob/main/python/triton/tools/ragged_tma.py).
