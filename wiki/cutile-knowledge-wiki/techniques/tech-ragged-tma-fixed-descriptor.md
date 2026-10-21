---
id: tech-ragged-tma-fixed-descriptor
kind: technique
title: Fixed-descriptor TMA for ragged row bounds
summary: Encode a ragged row end in synthetic tensor coordinates so one launch descriptor can serve multiple bounded segments.
basis: single-observation
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Fixed-descriptor TMA for ragged row bounds

## What it is

Encode each segment end in the coordinates of a fixed virtual tensor view.
TMA bounds checks zero-fill invalid load lanes and suppress invalid store lanes;
64-bit wrapping address generation maps valid lanes to the real allocation.
The descriptor stays fixed across chunks sharing an allocation/layout.

An earlier full implementation of this BF16 construction was validated on B200
(see Evidence). The address argument requires TMA bounds checks and wrapping
address generation; correctness of a non-TMA fallback is not established.
`ragged-tma-fixed-descriptor` supplies the wrapper and an unexecuted copy example;
ordinary TMA is described in `tech-tma-load`.

## Pattern

For a BF16 matrix with row stride `row_stride` in **elements**, use:

```python
B = 1 << 30
H = 1 << 34
M = 0x7fff0000
shape = (M, M, B, width)
byte_strides = (2 * (H - row_stride), 2 * row_stride, 2 * row_stride, 2)
# Tile rows advance coordinate 2.
coords = (B, end, B - (end - start), 0)
```

Use `ct.ArrayAnnotation(index_dtype=ct.int64,
static_shape_dims=(0, 1, 2, 3), static_stride_dims=(0, 1, 2, 3))` to specialize
metadata while retaining int64 indexing. The annotation does not supply shape or
strides: wrap the real tensors with `RaggedTensor`, which exposes them through
`__cuda_array_interface__`, and pass those wrappers to `ct.launch`. The
`ragged-tma-fixed-descriptor` reference includes the class, annotated kernel
arguments and launch wrapper. Here `input_view` is a wrapped kernel argument,
`width` is its real column count, and `BLOCK_M` and `width` are compile-time
tile dimensions. Load through:

```python
view = input_view.tiled_view((1, 1, BLOCK_M, width), padding_mode=ct.PaddingMode.ZERO,
                    traversal_steps=(1, 1, 1, width))
tile = view.load(coords, allow_tma=True)
input_tile = ct.reshape(tile, (BLOCK_M, width))
```

For output-column tile `column_tile_id`, the same construction uses a tile shape of
`(1, 1, BLOCK_M, BLOCK_N)`, traversal steps `(1, 1, 1, BLOCK_N)`, and coordinates
`(B, end, B - (end - start), column_tile_id)`. The last coordinate is a **tile index**
under that traversal, whereas the first three traversal steps are one.

For a local row `r >= 0` and column `c`, the byte displacement is

```text
2 * [B*(H-row_stride) + end*row_stride + (B-(end-start)+r)*row_stride + c]
= 2*B*H + 2*((start+r)*row_stride + c).
```

The caller must enforce `0 <= start <= end <= real_rows` for every backing
matrix, with each real matrix having at most `B` rows. The synthetic view replaces the
real allocation's row bound with `B`: violating the real-row precondition can
therefore cause out-of-allocation reads or writes without a TMA bounds failure.

Here `B*H = 2**64`, so the constant displacement vanishes modulo `2**64`
for any integer item size; the displayed BF16 byte displacement is `2**65`.
With the real-row precondition above, the third coordinate is inside `[0, B)` exactly when
`r < end-start`. The first coordinate is valid because `B < M`; the second
is valid because `end <= B < M`. Thus an in-bounds element addresses the real row
`start+r`; a tile lane beyond the segment end is rejected by the synthetic bound.
An empty segment starts exactly at the third dimension's upper bound.

## When to use

- Chunks share one allocation/layout, but each has a different row start and end.
- A slice-based implementation emits per-chunk descriptor construction/publication
  on the critical path, and a fixed launch descriptor is a candidate replacement.
- Invalid rows must be zero-filled on input or suppressed on output; fetching
  neighboring sequence rows would violate the operation's intended boundary.

## Caveats

- The evidence scope is Blackwell B200, BF16, cuda-tile 1.6.0 and tileiras 13.4.29.
  The modular-address construction is not established here for other architectures,
  dtypes, or alternative descriptor lowerings.
- Require a unit inner stride, 16-byte-aligned bases, and positive row strides divisible
  by eight BF16 elements. In this representation `row_stride < H`, the outer byte strides
  must fit the TMA descriptor, and the actual matrices have at most `B` rows.
  Tile sizes and swizzles must also be legal for the selected TMA path. Synthetic
  dimensions do not grant permission to access memory outside the real allocation.
- The constants jointly satisfy coordinate range, descriptor shape/stride encoding,
  representability in the specialized int64 metadata, and modular cancellation.
  Increasing `B` alone is not justified by the hardware's unsigned shape field: the TMA coordinates have
  a signed representation, and cancellation also depends on `H`. Item size affects
  byte-stride alignment and encoding; this snapshot specifically hard-codes BF16
  through the factor `2` and `<V2`.
- `static_shape_dims` and `static_stride_dims` expose metadata to specialization;
  they let the compiler see the finite virtual shapes and strides. The earlier validated
  implementation used this specialization; no ablation without it is included, so this
  evidence does not establish that it is necessary for every TMA lowering.
  They do not check allocation or chunk bounds at runtime. Distinct specialized
  shapes/strides can require separate compiled variants.
  The earlier implementation used the public interface in cuda-tile 1.6.0; the
  selected toolkit must support `tiled_view(..., traversal_steps=...)`.
- Each chunk owns `[start, min(start + BLOCK_M, end))`. A longer segment needs
  successive chunk starts separated by `BLOCK_M`; owned intervals must not overlap.
  A row-wise matmul built on this view, as in the earlier validated implementation,
  additionally requires `INPUT_WIDTH == input_a.shape[1] == input_b.shape[1] == weights.shape[0]` with power-of-two
  `INPUT_WIDTH` loaded as one tile, and `output.shape[1] == weights.shape[1]`.
  A supplied output preserves untouched rows;
  uncovered rows of a newly allocated output are uninitialized. The earlier matmul
  used shared `weights`; grouped weights require an additional mapping.
- Zero is suitable padding for such a row-wise matmul, but does not automatically
  implement masked softmax, max reductions, or normalization. Those operations
  need their own masking values and reduction-domain semantics.
- `allow_tma=True` permits TMA; the generated path remains a compiler decision.
  Confirm the expected `UTMALDG.4D` / `UTMASTG.4D` instructions for a new tile
  configuration or toolchain before relying on this TMA-specific argument.
  A persistent reuse of the view is a source-level extension, not demonstrated
  by the copy example; no overlap or speedup is implied.

## Evidence

- `ragged-tma-fixed-descriptor` provides an unexecuted copy example showing the wrapper, virtual metadata, coordinates and bounded load/store operations used in the address proof.
- Previous measurement: validation of an earlier full implementation of this construction (not the copy example) on B200 passed empty/tail-segment loads, guarded stores, padded layouts, nondefault streams, random shared-weight matmuls and CUDA Graph replay; all 13 inspected compiled variants used `UTMALDG.4D` and `UTMASTG.4D`, with no `UTMACCTL.IV` observed. Absence of that opcode alone is not a general proof that descriptor work is absent. This is one implementation's validation, not a portability claim. [2026-10, B200, cuda-tile 1.6.0, toolkit="Cuda compilation tools, release 13.4, V13.4.29", N=1 kernel]
