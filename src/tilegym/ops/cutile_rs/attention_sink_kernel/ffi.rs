// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// FFI export for the attention_sink kernel — one C-ABI symbol `cutile_attention_sink`.
//
// Forward-only flash-attention with attention-sink tokens. q/k/v/out are 4-D
// [bs, heads, seq, head_dim], sinks is 1-D [n_heads] (E), start_q is 1-D [*] (i32).
// All cross as `TensorDesc` (crate::ffi_util); `borrow_tensor` rebuilds borrowed
// host `Tensor`s over the PyTorch device pointers (ManuallyDrop = FFI ownership
// gate). The element-type generic E comes from q/k/v/out/sinks; start_q is i32.
//
// const-generic order (from the kernel's #[cutile::entry]):
//   <E, TILE_D, H, N_KV_CTX, TILE_M, TILE_N, QUERY_GROUP_SIZE, BANDWIDTH>
// qk_scale (raw sm_scale) is a runtime f32 scalar.

use core::ffi::c_void;
use cuda_core::{Device, Stream};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

use crate::ffi_util::{TensorDesc, borrow_tensor, dtype_str};
use attention_sink_module::attention_sink_kernel;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_attention_sink(
    // tensors (dtype + shapes + strides carried in the descriptors):
    //   out/q/k/v: [bs, heads, seq, head_dim]   sinks: [n_heads]   start_q: [*] (i32)
    out: *const TensorDesc,
    q: *const TensorDesc,
    k: *const TensorDesc,
    v: *const TensorDesc,
    sinks: *const TensorDesc,
    start_q: *const TensorDesc,
    // raw sm_scale
    qk_scale: f32,
    // const generics (computed by the wrapper from shapes)
    tile_d: i32,
    h: i32,
    n_kv_ctx: i32,
    tile_m: i32,
    tile_n: i32,
    query_group_size: i32,
    bandwidth: i32,
    // 2-D grid (ceil(n_ctx/TILE_M), bs*n_heads)
    grid_x: i32,
    grid_y: i32,
    // compile options: <=0 means auto/default
    num_cta_in_cga: i32,
    occupancy: i32,
    // CUDA stream
    // CUDA device ordinal of the tensors/stream (multi-GPU correctness)
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    if out.is_null()
        || q.is_null()
        || k.is_null()
        || v.is_null()
        || sinks.is_null()
        || start_q.is_null()
    {
        return -5;
    }
    let (out_d, q_d, k_d, v_d, sinks_d, start_q_d) =
        unsafe { (&*out, &*q, &*k, &*v, &*sinks, &*start_q) };

    let dty: &'static str = match dtype_str(q_d.dtype) {
        Some(s) => s,
        None => return -2,
    };

    let device = match Device::new(device_id.max(0) as usize) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cutile_attention_sink: Device::new failed: {e:?}");
            return -4;
        }
    };
    let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };

    macro_rules! dispatch {
        ($E:ty) => {{
            // Borrowed host tensors over PyTorch memory (ManuallyDrop = never freed).
            let q_t = unsafe { borrow_tensor::<$E>(q_d) };
            let k_t = unsafe { borrow_tensor::<$E>(k_d) };
            let v_t = unsafe { borrow_tensor::<$E>(v_d) };
            let sinks_t = unsafe { borrow_tensor::<$E>(sinks_d) };
            let out_t = unsafe { borrow_tensor::<$E>(out_d) };
            let start_q_t = unsafe { borrow_tensor::<i32>(start_q_d) };

            // generics: <E, TILE_D, H, N_KV_CTX, TILE_M, TILE_N, QUERY_GROUP_SIZE, BANDWIDTH>
            let generics = vec![
                dty.to_string(),
                tile_d.to_string(),
                h.to_string(),
                n_kv_ctx.to_string(),
                tile_m.to_string(),
                tile_n.to_string(),
                query_group_size.to_string(),
                bandwidth.to_string(),
            ];

            let mut opts = CompileOptions::default();
            if occupancy > 0 {
                opts = opts.occupancy(occupancy);
            }
            if num_cta_in_cga > 0 {
                opts = opts.num_cta_in_cga(num_cta_in_cga);
            }

            let op = unsafe {
                attention_sink_kernel(
                    &*q_t,
                    &*k_t,
                    &*v_t,
                    &*sinks_t,
                    &*out_t,
                    &*start_q_t,
                    qk_scale,
                )
            }
            .generics(generics)
            .grid((grid_x as u32, grid_y as u32, 1))
            .compile_options(opts);

            match op.sync_on(&stream) {
                Ok(_) => 0,
                Err(e) => {
                    eprintln!("cutile_attention_sink: launch failed: {e:?}");
                    -3
                }
            }
            // borrowed ManuallyDrop tensors -> never free PyTorch memory.
        }};
    }

    match dty {
        "f32" => dispatch!(f32),
        "f16" => dispatch!(f16),
        "bf16" => dispatch!(bf16),
        _ => -2,
    }
}
