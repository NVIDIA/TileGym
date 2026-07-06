// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// FFI export for the bmm (batched matmul) kernel — one C-ABI symbol `cutile_bmm`.
//
// C[Q, M, N] = A[Q, M, K] @ B[Q, K, N], optionally transposing A and/or B. Both
// structural variants declare A, B AND output C as read-only `&Tensor<E,{[-1,-1,-1]}>`
// (C is written in-body via `partition_full_mut`). Tensors cross the boundary as
// `TensorDesc` (crate::ffi_util); `borrow_tensor` rebuilds borrowed host `Tensor`s
// over the PyTorch device pointers and never frees them (ManuallyDrop = FFI
// ownership gate). Logical dims (rt_q/m/n/k) are derived from the descriptors'
// shapes + the transpose flags for the grid math.
//
// const-generic order (from the kernels' #[cutile::entry]):
//   non_persistent:    <E, BM, BN, BK>                                          (3-D grid, no transpose)
//   static_persistent: <E, BM, BN, BK, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B>  (1-D grid-stride)

use core::ffi::c_void;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, borrow_tensor, dtype_str};
use bmm_module::{non_persistent_bmm_kernel, static_persistent_bmm_kernel};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_bmm(
    // tensors (dtype + shapes + strides carried in the descriptors):
    //   a: [Q,M,K] (or [Q,K,M] if trans_a)  b: [Q,K,N] (or [Q,N,K] if trans_b)  c: [Q,M,N]
    c: *const TensorDesc,
    a: *const TensorDesc,
    b: *const TensorDesc,
    // tile sizes
    bm: i32,
    bn: i32,
    bk: i32,
    // swizzle group (static_persistent only)
    group_size_m: i32,
    // transpose flags (0 = no, 1 = yes); static_persistent only
    trans_a: i32,
    trans_b: i32,
    // variant select: 0 = non_persistent, 1 = static_persistent
    persistent: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // launch grid for the persistent variant (number of programs / CTAs)
    num_programs: i32,
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
    // Logical (post-transpose) dims from the A/B physical shapes.
    //   A physical: trans_a==0 -> [Q,M,K]; trans_a==1 -> [Q,K,M]
    //   B physical: trans_b==0 -> [Q,K,N]; trans_b==1 -> [Q,N,K]
    let rt_q = a_d.dim(0);
    let rt_m = if trans_a != 0 { a_d.dim(2) } else { a_d.dim(1) };
    let rt_k = if trans_a != 0 { a_d.dim(1) } else { a_d.dim(2) };
    let rt_n = if trans_b != 0 { b_d.dim(1) } else { b_d.dim(2) };

    let device = match Device::new(device_id.max(0) as usize) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cutile_bmm: Device::new failed: {e:?}");
            return -4;
        }
    };
    let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };

    macro_rules! dispatch {
        ($E:ty) => {{
            // Borrowed host tensors over PyTorch memory (ManuallyDrop = never freed).
            let a_t = unsafe { borrow_tensor::<$E>(a_d) };
            let b_t = unsafe { borrow_tensor::<$E>(b_d) };
            let c_t = unsafe { borrow_tensor::<$E>(c_d) };

            let mut opts = CompileOptions::default();
            if occupancy > 0 {
                opts = opts.occupancy(occupancy);
            }
            if num_cta_in_cga > 0 {
                opts = opts.num_cta_in_cga(num_cta_in_cga);
            }

            if persistent != 0 {
                // generics: <E, BM, BN, BK, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B>
                let generics = vec![
                    dty.to_string(),
                    bm.to_string(),
                    bn.to_string(),
                    bk.to_string(),
                    group_size_m.to_string(),
                    trans_a.to_string(),
                    trans_b.to_string(),
                ];
                let op = unsafe {
                    static_persistent_bmm_kernel(&*a_t, &*b_t, &*c_t, rt_q, rt_m, rt_n, rt_k)
                }
                .generics(generics)
                .grid((num_programs as u32, 1, 1))
                .compile_options(opts);
                match op.sync_on(&stream) {
                    Ok(_) => 0,
                    Err(e) => {
                        eprintln!("cutile_bmm static_persistent launch failed: {e:?}");
                        -3
                    }
                }
            } else {
                // generics: <E, BM, BN, BK>; 3-D grid (cdiv(M,BM), cdiv(N,BN), Q).
                let generics = vec![
                    dty.to_string(),
                    bm.to_string(),
                    bn.to_string(),
                    bk.to_string(),
                ];
                let grid_m = ((rt_m + bm - 1) / bm) as u32;
                let grid_n = ((rt_n + bn - 1) / bn) as u32;
                let op = unsafe { non_persistent_bmm_kernel(&*a_t, &*b_t, &*c_t) }
                    .generics(generics)
                    .grid((grid_m, grid_n, rt_q as u32))
                    .compile_options(opts);
                match op.sync_on(&stream) {
                    Ok(_) => 0,
                    Err(e) => {
                        eprintln!("cutile_bmm non_persistent launch failed: {e:?}");
                        -3
                    }
                }
            }
            // a_t/b_t/c_t are ManuallyDrop<Tensor> -> dropped here as no-ops,
            // so PyTorch memory is never freed.
        }};
    }

    match dty {
        "f32" => dispatch!(f32),
        "f16" => dispatch!(f16),
        "bf16" => dispatch!(bf16),
        _ => -2,
    }
}
