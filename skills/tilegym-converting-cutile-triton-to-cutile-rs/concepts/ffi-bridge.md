# Python <-> Rust FFI Bridge

This is the current bridge pattern for generated cutile-rs kernels in this
skill. Agent D writes BOTH `ffi.rs` (the C-ABI host launcher) and the Python
`cffi` wrapper that calls it — the whole host boundary is D's (device/host
split). Agent B writes only the device `kernel.rs`.

Every op's kernel is compiled into ONE aggregated cdylib,
`libcutile_kernels.so`; there is no per-op `.so`. Tensors cross the boundary as
a single `TensorDesc` pointer per tensor (not raw ptr + loose dim/stride/dtype
args).

## Architecture

```text
Python torch Tensor        cffi const TensorDesc*        Rust cdylib ffi.rs
data_ptr(),shape,        cutile_{kernel_name}(...)   ->   borrow_tensor / DevicePointer
strides,dtype    ->      (one *const TensorDesc          kernel(...).generics()
current CUDA stream        per tensor)                    .grid(...).compile_options()
device index                                              .sync_on(&stream)
      |
 make_tensor_desc(ffi,t) packs each tensor into a TensorDesc
```

PyTorch owns the memory. Rust only borrows the raw device pointers (via the
descriptor) for the duration of the launch.

## The `TensorDesc` contract

Each tensor crosses the boundary as one `*const TensorDesc`. The struct lives in
`ops/cutile_rs/ffi_util.rs` and is shared by every op (`#[repr(C)]`, strides in
ELEMENTS):

```rust
#[repr(C)]
pub struct TensorDesc {
    pub ptr: u64,        // CUDA device pointer (data_ptr())
    pub ndim: i32,
    pub shape: [i64; 4],
    pub strides: [i64; 4],  // in elements, NOT bytes
    pub dtype: i32,      // f32=0, f16=1, bf16=2, i32=3
}
```

Helpers on the descriptor (also in `ffi_util.rs`):

- `desc.dim(i)` — the `i`-th logical dim as `i32`.
- `desc.strides[i]` — the `i`-th stride in elements.
- `dtype_str(desc.dtype) -> Option<&'static str>` — maps the dtype code to the
  generic name string (`"f32"`/`"f16"`/`"bf16"`/`"i32"`), `None` on an unknown
  code.
- `borrow_tensor::<E>(desc) -> ManuallyDrop<Tensor<E>>` — rebuilds a borrowed
  host `Tensor<E>` over the PyTorch device pointer (see below).

## Rust 2024 export shape

Use Rust 2024 export attributes:

```rust
use crate::ffi_util::{borrow_tensor, dtype_str, TensorDesc};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_{kernel_name}(
    out: *const TensorDesc,
    x: *const TensorDesc,
    /* tile sizes, latency, compile options, grid, ... */
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    if out.is_null() || x.is_null() {
        return -5;
    }
    let (out_d, x_d) = unsafe { (&*out, &*x) };
    ...
}
```

Return codes convention:

- `0`: success
- `-2`: unsupported dtype / invalid FFI enum
- `-3`: launch/JIT error
- `-4`: CUDA device/stream setup error
- `-5`: null `TensorDesc` pointer

Do not use legacy `#[no_mangle]`.

## Device + stream setup (multi-GPU correct)

The FFI takes a `device_id: i32` (the tensor's CUDA ordinal) so multi-GPU
launches target the right device. Never hardcode device 0.

```rust
use core::ffi::c_void;
use cuda_core::{Device, Stream};

let device = match Device::new(device_id.max(0) as usize) {
    Ok(d) => d,
    Err(e) => {
        eprintln!("cutile_{kernel_name}: Device::new failed: {e:?}");
        return -4;
    }
};
let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };
```

## Reading a descriptor

Read dtype, dims, and strides off the descriptor — do NOT reconstruct them from
loose FFI args (there are none anymore):

```rust
let dty: &'static str = match dtype_str(x_d.dtype) {
    Some(s) => s,
    None => return -2,
};
let m = x_d.dim(0);
let n = x_d.dim(1);
let s_m = x_d.strides[0] as i32;
let s_n = x_d.strides[1] as i32;
```

## Raw-pointer entries

Use raw pointers only when the kernel entry itself has raw `*mut E` /
`DevicePointer<E>` params, as with pointer scatter/gather IR (softmax, norms).
Wrap the descriptor's `ptr` field directly — do NOT `transmute` and do NOT call
`from_raw_parts`.

```rust
use cuda_async::device_buffer::DevicePointer;

let x_dp: DevicePointer<E> = unsafe { DevicePointer::from_cu_deviceptr(x_d.ptr) };
let y_dp: DevicePointer<E> = unsafe { DevicePointer::from_cu_deviceptr(y_d.ptr) };

let op = unsafe { kernel(y_dp, x_dp, m, n, s_m, s_n) }
    .generics(generics)
    .grid((grid_size as u32, 1, 1));
```

Do not pass `DevicePointer<T>` to an entry whose Rust params are `&Tensor`.
Worked example: `examples/softmax`.

## `&Tensor` entries

For view/TMA kernels, rebuild borrowed host `Tensor<E>` wrappers from the
descriptors with `borrow_tensor` and pass references into the host launcher.

```rust
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, TileKernel};

// ManuallyDrop<Tensor<E>> over PyTorch memory — dropped as a no-op, so the
// PyTorch buffer is NEVER freed. This is the ownership gate by construction.
let a_t = unsafe { borrow_tensor::<E>(a_d) };
let b_t = unsafe { borrow_tensor::<E>(b_d) };
let c_t = unsafe { borrow_tensor::<E>(c_d) };

let op = unsafe { kernel(&*a_t, &*b_t, &*c_t) }   // &* deref the ManuallyDrop
    .generics(generics)
    .grid((grid_size as u32, 1, 1))
    .compile_options(opts);
```

`borrow_tensor` returns a `ManuallyDrop<Tensor<E>>`, so there is NO
`core::mem::forget` — the drop is already a no-op and PyTorch memory is never
freed. Op-level code must NOT call `Tensor::from_raw_parts` directly and must
NOT `transmute` a pointer. `from_raw_parts` still exists but ONLY inside
`borrow_tensor` in `ffi_util.rs`, where it is the 5-arg form
`from_raw_parts(ptr: u64, nbytes: usize, offset: 0, shape: Vec<i32>, strides: Vec<i32>)`.
Worked example: `examples/bmm`.

## Outputs: NEVER `&mut Tensor` — pass like an input, launch any grid

A kernel output is NEVER `&mut Tensor` / `Partition<&mut Tensor>` /
`MappedLaunchPartition` (see `references/no-mut-tensor-output.md`). The kernel
declares the output as a **read-only `&Tensor<E,{[-1,...]}>`** (written in-body
via `partition_full_mut`) or a **raw `*mut E`** (via `make_tensor_view`). The
host passes the output the SAME way it passes an input — NO `.partition()` /
`.map()` on the output, and NO grid lock.

Read-only `&Tensor` output (worked example: `examples/bmm`):

```rust
let c_t = unsafe { borrow_tensor::<E>(c_d) };   // like the inputs
let op = unsafe { kernel(&*a_t, &*b_t, &*c_t) } // &*c_t — read-only, like the inputs
    .generics(generics)
    .grid((grid_size as u32, 1, 1));            // ANY explicit grid (persistent OK) — no lock
op.sync_on(&stream)?;                           // c_t is ManuallyDrop — never frees PyTorch memory
```

Raw-pointer output (worked example: `examples/softmax`):

```rust
let c_dp: DevicePointer<E> = unsafe { DevicePointer::from_cu_deviceptr(c_d.ptr) };
let op = unsafe { kernel(a_dp, b_dp, c_dp, /*dims, strides*/) }
    .generics(generics)
    .grid((grid_size as u32, 1, 1));            // ANY explicit grid — no lock
```

Why no `&mut Tensor`: a `&mut` output emits `KernelOutputStored::grid`, and
`validate_grids` then forces the explicit launch grid to EQUAL the output's
inferred partition grid `(cdiv(M,bm), cdiv(N,bn), 1)` in descriptor dim order.
Any other grid (persistent, or a transposed block-id order) is rejected with
`"Specified launch grid does not match inferred tensor partition grid"` (rc=-3).
Read-only `&Tensor` / raw pointers emit NO inferred grid, so the host launches
whatever the kernel needs. Do NOT reach for `.partition().map()` to reconcile a
mismatch — that is the `&mut` path we are avoiding.

## Compile options

Agent B exposes compile-option levers; Agent D chooses values from
`analysis.json` and wrapper autotune configs.

```rust
let mut opts = CompileOptions::default();
if occupancy > 0 {
    opts = opts.occupancy(occupancy);
}
if num_cta_in_cga > 0 {
    opts = opts.num_cta_in_cga(num_cta_in_cga);
}

let op = op.compile_options(opts);
```

A non-positive value means leave the compiler default. Do not call setters with
sentinel values.

## Dtype dispatch skeleton

```rust
let dty: &'static str = match dtype_str(x_d.dtype) {
    Some(s) => s,
    None => return -2,
};

macro_rules! dispatch {
    ($E:ty) => {{
        let generics = vec![dty.to_string(), bm.to_string(), bn.to_string()];
        /* borrow_tensor / DevicePointer, build op, sync — no forget */
    }};
}

match dty {
    "f32" => dispatch!(f32),
    "f16" => dispatch!(f16),
    "bf16" => dispatch!(bf16),
    _ => -2,
}
```

If the kernel has an accumulator dtype generic, pass `"f32"` for that slot unless
the reference says otherwise. If f32 GEMM needs tf32 operands, pass a const
generic such as `CAST_TF32=1` only in the f32 branch.

## Build & crate layout

All ops build into ONE aggregated cdylib crate, `ops/cutile_rs/cutile_kernels/`,
producing a single `libcutile_kernels.so`. There is no per-op throwaway crate,
no `include!` into a monolithic `cutile-ffi/src/lib.rs`, and no per-op `.so`.

- Each op is wired into the aggregate crate as a module that `include!`s its
  device + FFI sources:

  ```rust
  mod {kernel_name} {
      include!("../../{kernel_name}_kernel/kernel.rs");
      include!("../../{kernel_name}_kernel/ffi.rs");
  }
  ```

- The shared `TensorDesc` / helpers are pulled in once via:

  ```rust
  #[path = "../../ffi_util.rs"]
  mod ffi_util;
  ```

- Dependencies are crates.io PINNED versions — NO cutile-rs checkout and NO
  `CUTILE_RS_ROOT` / path deps:

  ```toml
  cutile          = "=0.2.0"
  cutile-compiler = "=0.2.0"
  cutile-macro    = "=0.2.0"
  cuda-core       = "=0.2.0"
  cuda-async      = "=0.2.0"
  ```

## Python wrapper side (cffi)

Agent D's Python wrapper uses `cffi` (not `ctypes`). It:

- declares the C signature in a `_FFI_CDEF` string whose tensor args are
  `const TensorDesc*` and keeps it in sync with `ffi.rs`;
- binds the shared library with
  `ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)` (the shared
  `TensorDesc` typedef is prepended for you);
- packs each tensor with `make_tensor_desc(ffi, t)` (this reads
  `data_ptr()`/shape/strides/dtype into the descriptor) and keeps the packed
  descriptors alive until after the call;
- passes the tensor's device ordinal as `device_id` and the current CUDA stream
  as `raw_stream`;
- calls `check_rc(rc, _FFI_NAME)` after every launch;
- detaches grad-enabled inputs because this skill is forward-only.

Imports come from `tilegym.backend.cutile_rs.utils`
(`bind_kernel_function_cffi`, `make_tensor_desc`, `check_rc`, `get_num_sm`) and
`tilegym.backend.cutile_rs.autotuner` (`autotune_launch`).

```python
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import make_tensor_desc

_FFI_CDEF = """
int32_t cutile_{kernel_name}(
    const TensorDesc* y, const TensorDesc* x,
    int32_t bm, int32_t bn,
    int32_t num_cta_in_cga, int32_t occupancy,
    int32_t grid_size, int32_t device_id, uint64_t raw_stream);
"""

ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
yd = make_tensor_desc(ffi, y)
xd = make_tensor_desc(ffi, x)
rc = lib.cutile_{kernel_name}(yd, xd, bm, bn, num_cta_in_cga, occupancy,
                              grid_size, device_id, raw_stream)
check_rc(rc, _FFI_NAME)
```

Correctness and perf live in tilegym pytest, not ad hoc Python scripts under the
kernel output directory.
