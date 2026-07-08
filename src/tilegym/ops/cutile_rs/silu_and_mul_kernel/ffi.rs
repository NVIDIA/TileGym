// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// FFI export for the silu_and_mul kernel — one C-ABI symbol `cutile_silu_and_mul`.
//
// out[r, :] = silu(in[r, :hidden]) * in[r, hidden:], i.e. the input's last dim is
// split in half. This is a raw-pointer (elementwise) kernel: it takes `*mut E`
// device pointers + hidden_size + row strides (NOT `&Tensor`), so the FFI extracts
// the pointer/dims/strides from the `TensorDesc`s (crate::ffi_util) and wraps them
// as `DevicePointer<E>` — no borrow_tensor / ownership gate needed (the kernel
// never holds a Tensor wrapper). The grid is one tile-block per row (n_rows).
//
// const-generic order: <E, BLOCK_SIZE>   (BLOCK_SIZE = next_power_of_2(hidden_size))

use core::ffi::c_void;
use cuda_async::device_buffer::DevicePointer;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, rc};
use silu_and_mul_module::silu_and_mul_kernel;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_silu_and_mul(
    // tensors (dtype + shapes + strides carried in the descriptors):
    //   out: [n_rows, hidden]   in: [n_rows, 2*hidden]
    out: *const TensorDesc,
    inp: *const TensorDesc,
    // BLOCK_SIZE const generic = next_power_of_2(hidden_size)
    block_size: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // CUDA stream
    // CUDA device ordinal of the tensors/stream (multi-GPU correctness)
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    let (out_d, in_d) = crate::deref_descs!(out, inp);
    let dty: &'static str = crate::resolve_dtype!(out_d);
    let hidden_size = out_d.dim(1);
    let n_rows = out_d.dim(0); // grid: one tile-block per row
    let in_row_stride = in_d.strides[0] as i32;
    let out_row_stride = out_d.strides[0] as i32;

    crate::setup_device_stream!(device, stream, device_id, raw_stream);

    macro_rules! dispatch {
        ($E:ty) => {{
            let out_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(out_d.ptr) };
            let in_dp: DevicePointer<$E> = unsafe { DevicePointer::from_cu_deviceptr(in_d.ptr) };

            // generics: <E, BLOCK_SIZE>
            let generics = vec![dty.to_string(), block_size.to_string()];

            let opts = crate::compile_options!(occupancy, num_cta_in_cga);

            let op = unsafe {
                silu_and_mul_kernel(out_dp, in_dp, hidden_size, in_row_stride, out_row_stride)
            }
            .generics(generics)
            .grid((n_rows as u32, 1, 1))
            .compile_options(opts);
            match op.sync_on(&stream) {
                Ok(_) => rc::OK,
                Err(e) => {
                    eprintln!("cutile_silu_and_mul: launch failed: {e:?}");
                    rc::LAUNCH_FAILED
                }
            }
        }};
    }

    crate::dispatch_by_dtype!(dty, dispatch, f32, f16, bf16)
}
