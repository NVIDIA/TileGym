// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

//
// FFI export for the matmul kernel — one C-ABI symbol `cutile_matmul`.
//
// matmul is a TILED-output tensor-core kernel. Both structural variants declare
// A, B AND the output C as read-only `&Tensor<E, {[-1, -1]}>` (C is written
// in-body via `c.partition_full_mut(...)` + `store_view_tko_mut`). Tensors cross
// the boundary as `TensorDesc` (see crate::ffi_util); `borrow_tensor` rebuilds
// borrowed host `Tensor`s over the PyTorch device pointers and never frees them
// (ManuallyDrop = FFI ownership gate). m/n/k come from the descriptors' shapes.
//
// const-generic order (MUST match kernel.rs .generics()):
//   non_persistent:    <E, BM, BN, BK, CAST_TF32>           (no runtime scalars)
//   static_persistent: <E, BM, BN, BK, GROUP_SIZE_M, CAST_TF32> (runtime m,n,k)
// CAST_TF32 = 1 iff dtype == "f32", else 0.

use core::ffi::c_void;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, borrow_tensor, cast_tf32, rc};
use matmul_module::{non_persistent_matmul_kernel, static_persistent_matmul_kernel};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_matmul(
    // tensors (dtype + shapes + strides carried in the descriptors):
    //   a: [m, k]   b: [k, n]   c: [m, n]
    c: *const TensorDesc,
    a: *const TensorDesc,
    b: *const TensorDesc,
    // tile sizes
    bm: i32,
    bn: i32,
    bk: i32,
    // swizzle group (static_persistent only)
    group_size_m: i32,
    // launch grid (number of programs / CTAs)
    num_programs: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // variant select: 0 = non_persistent, 1 = static_persistent
    persistent: i32,
    // CUDA stream
    // CUDA device ordinal of the tensors/stream (multi-GPU correctness)
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    let (a_d, b_d, c_d) = crate::deref_descs!(a, b, c);
    let dty: &'static str = crate::resolve_dtype!(a_d);
    let cast = cast_tf32(a_d.dtype);
    // logical dims from shapes: A[m, k], B[k, n], C[m, n].
    let (m, k, n) = (a_d.dim(0), a_d.dim(1), b_d.dim(1));

    crate::setup_device_stream!(device, stream, device_id, raw_stream);

    macro_rules! dispatch {
        ($E:ty) => {{
            // Borrowed host tensors over PyTorch memory (ManuallyDrop = never freed).
            let a_t = unsafe { borrow_tensor::<$E>(a_d) };
            let b_t = unsafe { borrow_tensor::<$E>(b_d) };
            let c_t = unsafe { borrow_tensor::<$E>(c_d) };

            let opts = crate::compile_options!(occupancy, num_cta_in_cga);

            if persistent != 0 {
                // generics: <E, BM, BN, BK, GROUP_SIZE_M, CAST_TF32>
                let generics = vec![
                    dty.to_string(),
                    bm.to_string(),
                    bn.to_string(),
                    bk.to_string(),
                    group_size_m.to_string(),
                    cast.to_string(),
                ];
                let op = unsafe { static_persistent_matmul_kernel(&*a_t, &*b_t, &*c_t, m, n, k) }
                    .generics(generics)
                    .grid((num_programs as u32, 1, 1))
                    .compile_options(opts);
                match op.sync_on(&stream) {
                    Ok(_) => rc::OK,
                    Err(e) => {
                        eprintln!("cutile_matmul static_persistent launch failed: {e:?}");
                        rc::LAUNCH_FAILED
                    }
                }
            } else {
                // generics: <E, BM, BN, BK, CAST_TF32>
                let generics = vec![
                    dty.to_string(),
                    bm.to_string(),
                    bn.to_string(),
                    bk.to_string(),
                    cast.to_string(),
                ];
                let op = unsafe { non_persistent_matmul_kernel(&*a_t, &*b_t, &*c_t) }
                    .generics(generics)
                    .grid((num_programs as u32, 1, 1))
                    .compile_options(opts);
                match op.sync_on(&stream) {
                    Ok(_) => rc::OK,
                    Err(e) => {
                        eprintln!("cutile_matmul non_persistent launch failed: {e:?}");
                        rc::LAUNCH_FAILED
                    }
                }
            }
            // a_t/b_t/c_t are ManuallyDrop<Tensor> -> dropped here as no-ops,
            // so PyTorch memory is never freed.
        }};
    }

    crate::dispatch_by_dtype!(dty, dispatch, f32, f16, bf16)
}
