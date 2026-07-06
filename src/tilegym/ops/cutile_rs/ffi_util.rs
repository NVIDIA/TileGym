// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared FFI marshalling for every cutile-rs op (the Rust "unpacker").
//
// Each op's ffi.rs used to hand-unpack ptr + dims + strides + elem_size + dtype
// for every tensor and then `mem::forget` each borrowed wrapper. This centralizes
// that: ops receive `*const TensorDesc` and call `borrow_tensor::<E>(desc)`.
// Mirrors how tilecpp centralizes arg packing in `make_kernel_args`, but on the
// Rust side. Keep `TensorDesc` in sync with the cffi typedef in
// backend/cutile_rs/utils.py (`_TENSORDESC_CDEF`) and the Python packer
// `make_tensor_desc`.

use core::mem::ManuallyDrop;
use cutile::prelude::*;

/// Max tensor rank carried by a [`TensorDesc`]. Bump (here + in the Python cdef)
/// if an op needs higher-rank tensors.
pub const MAX_DIMS: usize = 4;

/// C-ABI view of a device tensor borrowed from PyTorch. `#[repr(C)]` layout MUST
/// match `_TENSORDESC_CDEF` in backend/cutile_rs/utils.py. Strides are in ELEMENTS
/// (not bytes); `shape`/`strides` entries beyond `ndim` are ignored.
#[repr(C)]
pub struct TensorDesc {
    pub ptr: u64,
    pub ndim: i32,
    pub shape: [i64; MAX_DIMS],
    pub strides: [i64; MAX_DIMS],
    /// dtype code: 0 = f32, 1 = f16, 2 = bf16.
    pub dtype: i32,
}

impl TensorDesc {
    /// Logical extent of dimension `i` as i32 (cutile's `from_raw_parts` wants i32).
    pub fn dim(&self, i: usize) -> i32 {
        self.shape[i] as i32
    }

    fn shape_i32(&self) -> Vec<i32> {
        (0..self.ndim as usize)
            .map(|i| self.shape[i] as i32)
            .collect()
    }

    fn strides_i32(&self) -> Vec<i32> {
        (0..self.ndim as usize)
            .map(|i| self.strides[i] as i32)
            .collect()
    }

    fn nelem(&self) -> usize {
        (0..self.ndim as usize)
            .map(|i| self.shape[i] as usize)
            .product()
    }

    /// Total byte length of the (contiguous) allocation this view spans.
    pub fn nbytes(&self) -> usize {
        self.nelem() * dtype_elem_size(self.dtype)
    }
}

/// dtype code -> cutile type-name string used in `.generics(...)`. `None` if unknown.
/// Code 3 (i32) is for integer index tensors (e.g. attention start offsets); it
/// is never an element-type generic, but is included for completeness.
pub fn dtype_str(code: i32) -> Option<&'static str> {
    match code {
        0 => Some("f32"),
        1 => Some("f16"),
        2 => Some("bf16"),
        3 => Some("i32"),
        _ => None,
    }
}

/// dtype code -> element size in bytes. `0` if unknown.
pub fn dtype_elem_size(code: i32) -> usize {
    match code {
        0 | 3 => 4,
        1 | 2 => 2,
        _ => 0,
    }
}

/// `CAST_TF32` generic value: 1 iff the dtype is f32, else 0.
pub fn cast_tf32(code: i32) -> i32 {
    i32::from(code == 0)
}

/// Borrow a PyTorch tensor as a cutile `Tensor<E>` WITHOUT taking ownership.
///
/// The returned [`ManuallyDrop`] never runs `Tensor::Drop`, so the underlying
/// PyTorch device memory is never freed when it goes out of scope — the FFI
/// ownership gate, enforced by construction (no explicit `mem::forget` needed).
/// Access the tensor via `&*` / `&**`.
///
/// # Safety
/// `d.ptr` must point to a live device allocation of at least `d.nbytes()` bytes,
/// holding elements of type `E` laid out per `d.shape`/`d.strides`, and must stay
/// valid for the duration of the kernel launch.
pub unsafe fn borrow_tensor<E: DType>(d: &TensorDesc) -> ManuallyDrop<Tensor<E>> {
    ManuallyDrop::new(unsafe {
        Tensor::<E>::from_raw_parts(d.ptr, d.nbytes(), 0, d.shape_i32(), d.strides_i32())
    })
}
