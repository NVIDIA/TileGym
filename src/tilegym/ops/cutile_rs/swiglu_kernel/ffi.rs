// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// FFI export for the swiglu kernel — one C-ABI symbol `cutile_swiglu`.
//
// c = silu(a) * b, with a/b/c the same 2-D shape [n_rows, n_cols]. Raw-pointer
// (elementwise) kernel: it takes `*mut E` pointers + per-tensor shapes/strides
// (NOT `&Tensor`), so the FFI reads ptr/shape/strides from the `TensorDesc`s
// (crate::ffi_util) and wraps the pointers as `DevicePointer<E>`. The grid is
// one tile-block per row (n_rows). TILE_SIZE = next_power_of_2(n_cols).

use core::ffi::c_void;
use cuda_async::device_buffer::DevicePointer;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, dtype_str};
use swiglu_module::swiglu_forward_kernel;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_swiglu(
    // tensors (dtype + shapes + strides carried in the descriptors), all [n_rows, n_cols]:
    c: *const TensorDesc,
    a: *const TensorDesc,
    b: *const TensorDesc,
    // TILE_SIZE const generic = next_power_of_2(n_cols)
    tile_size: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // CUDA stream
    // CUDA device ordinal of the tensors/stream (multi-GPU correctness)
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return -5;
    }
    let (a_d, b_d, c_d) = unsafe { (&*a, &*b, &*c) };

    let dty: &'static str = match dtype_str(a_d.dtype) {
        Some(s) => s,
        None => return -2,
    };
    let n_rows = a_d.dim(0); // grid: one tile-block per row

    let device = match Device::new(device_id.max(0) as usize) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cutile_swiglu: Device::new failed: {e:?}");
            return -4;
        }
    };
    let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };

    macro_rules! dispatch {
        ($E:ty) => {{
            let a_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(a_d.ptr) };
            let b_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(b_d.ptr) };
            let c_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(c_d.ptr) };

            // generics: <E, TILE_SIZE>
            let generics = vec![dty.to_string(), tile_size.to_string()];

            let mut opts = CompileOptions::default();
            if occupancy > 0 {
                opts = opts.occupancy(occupancy);
            }
            if num_cta_in_cga > 0 {
                opts = opts.num_cta_in_cga(num_cta_in_cga);
            }

            let op = unsafe {
                swiglu_forward_kernel(
                    a_dp,
                    a_d.dim(0),
                    a_d.dim(1),
                    a_d.strides[0] as i32,
                    a_d.strides[1] as i32,
                    b_dp,
                    b_d.dim(0),
                    b_d.dim(1),
                    b_d.strides[0] as i32,
                    b_d.strides[1] as i32,
                    c_dp,
                    c_d.dim(0),
                    c_d.dim(1),
                    c_d.strides[0] as i32,
                    c_d.strides[1] as i32,
                )
            }
            .generics(generics)
            .grid((n_rows as u32, 1, 1))
            .compile_options(opts);

            match op.sync_on(&stream) {
                Ok(_) => 0,
                Err(e) => {
                    eprintln!("cutile_swiglu error: {e:?}");
                    -3
                }
            }
        }};
    }

    match dty {
        "f32" => dispatch!(f32),
        "f16" => dispatch!(f16),
        "bf16" => dispatch!(bf16),
        _ => -2,
    }
}
