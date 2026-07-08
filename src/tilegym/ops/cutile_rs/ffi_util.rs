// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

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
pub const MAX_DIMS: usize = 5;

/// C-ABI view of a device tensor borrowed from PyTorch. `#[repr(C)]` layout MUST
/// match `_TENSORDESC_CDEF` in backend/cutile_rs/utils.py. Strides are in ELEMENTS
/// (not bytes); `shape`/`strides` entries beyond `ndim` are ignored.
#[repr(C)]
pub struct TensorDesc {
    pub ptr: u64,
    pub ndim: i32,
    pub shape: [i64; MAX_DIMS],
    pub strides: [i64; MAX_DIMS],
    /// dtype code: 0 = f32, 1 = f16, 2 = bf16, 3 = i32, 4 = i64, 5 = f8e5m2.
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
/// Codes 3 (i32) / 4 (i64) are integer tensors (index / descriptor-table
/// entries), never element-type generics; included for completeness.
pub fn dtype_str(code: i32) -> Option<&'static str> {
    match code {
        0 => Some("f32"),
        1 => Some("f16"),
        2 => Some("bf16"),
        3 => Some("i32"),
        4 => Some("i64"),
        5 => Some("f8e5m2"),
        _ => None,
    }
}

/// dtype code -> element size in bytes. `0` if unknown.
pub fn dtype_elem_size(code: i32) -> usize {
    match code {
        0 | 3 => 4,
        1 | 2 => 2,
        4 => 8,
        5 => 1,
        _ => 0,
    }
}

/// `CAST_TF32` generic value: 1 iff the dtype is f32, else 0.
pub fn cast_tf32(code: i32) -> i32 {
    i32::from(code == 0)
}

/// C-ABI return codes shared by every op's `ffi.rs`. `0` = success; negatives are
/// errors. Keep in sync with `_RC_MESSAGES` in backend/cutile_rs/utils.py.
///
/// NOTE: `-1` is intentionally NOT a return code — it is the "auto / compiler
/// default" sentinel for the `num_cta_in_cga` / `occupancy` inputs
/// (`_AUTO_COMPILE_OPTION` in the Python wrappers), so it must not be overloaded.
pub mod rc {
    /// Kernel launched and synced successfully.
    pub const OK: i32 = 0;
    /// A `TensorDesc.dtype` code has no cutile element type (see `dtype_str`).
    pub const UNSUPPORTED_DTYPE: i32 = -2;
    /// The kernel launch / stream sync returned an error.
    pub const LAUNCH_FAILED: i32 = -3;
    /// `Device::new(device_id)` failed.
    pub const DEVICE_INIT_FAILED: i32 = -4;
    /// A required `*const TensorDesc` argument was null.
    pub const NULL_PTR: i32 = -5;
    /// A scalar / shape / launch parameter was invalid (e.g. a non-positive
    /// dimension, block, or grid size, or an out-of-range variant selector).
    pub const INVALID_ARGS: i32 = -6;
}

/// Human-readable message for an FFI return code (diagnostics / `eprintln!`).
pub fn rc_message(code: i32) -> &'static str {
    match code {
        rc::OK => "ok",
        rc::UNSUPPORTED_DTYPE => "unsupported dtype",
        rc::LAUNCH_FAILED => "kernel launch failed",
        rc::DEVICE_INIT_FAILED => "device init failed",
        rc::NULL_PTR => "null tensor pointer",
        rc::INVALID_ARGS => "invalid arguments",
        _ => "unknown error",
    }
}

/// Null-check each `*const TensorDesc` (returning `rc::NULL_PTR` from the calling
/// FFI fn on any null) then deref them all to `&TensorDesc` in one `unsafe` block.
/// Replaces the per-op `if a.is_null() || ... { return -5; } let (..) = unsafe {..};`
/// boilerplate. Every op passes >= 2 descriptors, so this always yields a tuple.
#[macro_export]
macro_rules! deref_descs {
    ($($p:ident),+ $(,)?) => {{
        $( if $p.is_null() { return $crate::ffi_util::rc::NULL_PTR; } )+
        unsafe { ($(&*$p),+) }
    }};
}

/// Resolve a `TensorDesc`'s dtype code to its cutile element-type string, or
/// `return rc::UNSUPPORTED_DTYPE` from the calling FFI fn. Evaluates to `&'static str`.
#[macro_export]
macro_rules! resolve_dtype {
    ($d:expr) => {
        match $crate::ffi_util::dtype_str($d.dtype) {
            Some(s) => s,
            None => return $crate::ffi_util::rc::UNSUPPORTED_DTYPE,
        }
    };
}

/// Bind `$dev` to a cuda-core `Device` for `$device_id` (must be >= 0; a
/// negative id is rejected rather than coerced to GPU 0) and `$strm` to a
/// *borrowed* `Stream` over the caller's raw CUDA stream, in the caller's
/// scope. On an invalid id or device-init failure it `return rc::DEVICE_INIT_FAILED`s
/// from the calling FFI fn. Replaces the per-op device+stream setup boilerplate.
///
/// Resolves `Device` / `Stream` / `c_void` at the call site, so the ffi.rs must
/// have `use cuda_core::{Device, Stream};` and `use core::ffi::c_void;` in scope
/// (every op already does). The borrow ties `$strm` to `$dev`, so both are bound
/// in the caller (not returned as a tuple).
#[macro_export]
macro_rules! setup_device_stream {
    ($dev:ident, $strm:ident, $device_id:expr, $raw_stream:expr) => {
        if ($device_id) < 0 {
            eprintln!("cutile-rs: invalid device id: {}", $device_id);
            return $crate::ffi_util::rc::DEVICE_INIT_FAILED;
        }
        let $dev = match Device::new(($device_id) as usize) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("cutile-rs: Device::new failed: {e:?}");
                return $crate::ffi_util::rc::DEVICE_INIT_FAILED;
            }
        };
        let $strm = unsafe { Stream::borrow_raw(($raw_stream) as *mut c_void, &$dev) };
    };
}

/// Dispatch on a dtype string `$dty` to the op-local `$dispatch!(<E>)` macro, one
/// arm per supported element type, falling back to `rc::UNSUPPORTED_DTYPE`. The
/// arm string is `stringify!($ty)`, which matches `dtype_str` (e.g. `f16`, `bf16`,
/// `f8e5m2`). Pass only the dtypes an op actually supports:
///   `dispatch_by_dtype!(dty, dispatch, f32, f16, bf16)`         // most ops
///   `dispatch_by_dtype!(dty, dispatch, f32, f16, bf16, f8e5m2)` // + fp8 ops
#[macro_export]
macro_rules! dispatch_by_dtype {
    ($dty:expr, $dispatch:ident, $($ty:ident),+ $(,)?) => {
        match $dty {
            $( stringify!($ty) => $dispatch!($ty), )+
            _ => $crate::ffi_util::rc::UNSUPPORTED_DTYPE,
        }
    };
}

/// Build a `CompileOptions` from the FFI compile-option ints, applying only
/// positive values — `<= 0` means "auto / compiler default" (the
/// `_AUTO_COMPILE_OPTION` = -1 sentinel). Evaluates to the `CompileOptions`.
/// Resolves `CompileOptions` at the call site (every ffi.rs imports it).
#[macro_export]
macro_rules! compile_options {
    ($occupancy:expr, $num_cta_in_cga:expr) => {{
        let mut opts = CompileOptions::default();
        if $occupancy > 0 {
            opts = opts.occupancy($occupancy);
        }
        if $num_cta_in_cga > 0 {
            opts = opts.num_cta_in_cga($num_cta_in_cga);
        }
        opts
    }};
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
