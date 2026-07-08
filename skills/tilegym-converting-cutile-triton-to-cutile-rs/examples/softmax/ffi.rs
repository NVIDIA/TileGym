// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

//
// FFI export for the softmax kernel — one C-ABI symbol `cutile_softmax`.
//
// y[m, :] = softmax(x[m, :]) over the last dim. This is a raw-pointer kernel:
// the entry takes `DevicePointer<E>` + dims/strides (NOT `&Tensor`), so the FFI
// reads ptr/dims/strides from the `TensorDesc`s (crate::ffi_util) and wraps the
// pointers as `DevicePointer<E>` — no borrow_tensor / ownership gate needed (the
// kernel never holds a Tensor wrapper, so nothing can free PyTorch memory).
//
// Anatomy of a cutile-rs FFI function:
//   1. Null-check the descriptors; read dtype via `dtype_str(desc.dtype)`.
//   2. Read logical dims / strides off the descriptors (`.dim(i)`, `.strides[i]`).
//   3. Build a Device from `device_id` and borrow the caller's stream.
//   4. Build `.generics()` in the SAME ORDER as the entry's `<E, BM, BN, LATENCY>`.
//   5. Call the entry (op-builder), chain `.generics()` `.grid()` `.compile_options()`,
//      then `.sync_on(&stream)`.
// dtype dispatch: a `match` on the dtype string picks which `<E>` the macro shim
// instantiates.

use core::ffi::c_void;
use cuda_async::device_buffer::DevicePointer;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, dtype_str};
use softmax_module::softmax_kernel;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_softmax(
    // tensors (dtype + shapes + strides carried in the descriptors):
    //   y: [m, n] (output)   x: [m, n] (input)
    y: *const TensorDesc,
    x: *const TensorDesc,
    // tile sizes — autotuner-tunable
    bm: i32,
    bn: i32,
    // latency — autotuner-tunable (1:1 map from Triton-TileIR num_stages)
    latency: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // launch grid (number of programs / CTAs) for the persistent grid-stride loop
    grid_size: i32,
    // CUDA device ordinal of the tensors/stream (multi-GPU correctness)
    device_id: i32,
    // CUDA stream (cuStream_t cast to u64)
    raw_stream: u64,
) -> i32 {
    if y.is_null() || x.is_null() {
        return -5;
    }
    let (y_d, x_d) = unsafe { (&*y, &*x) };

    let dty: &'static str = match dtype_str(x_d.dtype) {
        Some(s) => s,
        None => return -2,
    };
    // Logical dims + row/col strides from the input descriptor (y matches).
    let m = x_d.dim(0);
    let n = x_d.dim(1);
    let (s_m, s_n) = match (i32::try_from(x_d.strides[0]), i32::try_from(x_d.strides[1])) {
        (Ok(a), Ok(b)) => (a, b),
        _ => return -6,
    };

    let device = match Device::new(device_id.max(0) as usize) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cutile_softmax: Device::new failed: {e:?}");
            return -4;
        }
    };
    let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };

    macro_rules! dispatch {
        ($E:ty) => {{
            let y_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(y_d.ptr) };
            let x_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(x_d.ptr) };

            // Generics in the SAME ORDER as kernel.rs:
            //   <E: ElementType, const BM, const BN, const LATENCY>
            let generics = vec![
                dty.to_string(),
                bm.to_string(),
                bn.to_string(),
                latency.to_string(),
            ];

            // Honor the wrapper's sentinel: only call a setter when the value is
            // > 0; a value <= 0 leaves that field at the compiler default.
            let mut opts = CompileOptions::default();
            if occupancy > 0 {
                opts = opts.occupancy(occupancy);
            }
            if num_cta_in_cga > 0 {
                opts = opts.num_cta_in_cga(num_cta_in_cga);
            }

            let op = unsafe { softmax_kernel(y_dp, x_dp, m, n, s_m, s_n) }
                .generics(generics)
                .grid((grid_size as u32, 1, 1))
                .compile_options(opts);

            match op.sync_on(&stream) {
                Ok(_) => 0,
                Err(e) => {
                    eprintln!("cutile_softmax: launch failed: {e:?}");
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
