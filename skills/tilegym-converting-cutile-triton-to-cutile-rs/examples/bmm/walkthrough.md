# Worked example — BMM (read-only `&Tensor` output, NO `&mut Tensor`)

Batched GEMM `C[Q,M,N] = A[Q,M,K] @ B[Q,K,N]`. This is the **canonical
tiled-output example**: the output is declared as a **read-only
`&Tensor<E, {[-1,-1,-1]}>`** and written in-body via **`partition_full_mut`** —
NOT `&mut Tensor`.

> Why no `&mut Tensor`? See [`references/no-mut-tensor-output.md`](../../references/no-mut-tensor-output.md).
> A `&mut Tensor` output makes the launcher emit `KernelOutputStored::grid(&out)`,
> which LOCKS the launch grid to the output partition grid (descriptor dim order,
> exact-match, no `-1`). That is the source of both the `{[-1,..]}` reject trap
> and the block-id↔dim axis-transpose lock (output heads/rows unwritten). A read-only
> `&Tensor` (this example) emits NO grid → no lock → free grid / block-id /
> persistent schedule, and `-1` wildcards are allowed.

## The pattern (`kernel.rs`)

A, B, **and the output C** are all `&Tensor<E, {[-1,-1,-1]}>` (read-only param
type, `-1` wildcards). The output is written through a mutable partition VIEW
built inside the kernel body:

```rust
unsafe fn bmm_kernel_static_persistent<E: ElementType, const BM, const BN, const BK, ...>(
    a: &Tensor<E, { [-1, -1, -1] }>,
    b: &Tensor<E, { [-1, -1, -1] }>,
    c: &Tensor<E, { [-1, -1, -1] }>,           // OUTPUT — read-only PARAM type, -1 OK
    /* logical dims, transpose flags ... */
) {
    let total_tiles = num_pid_in_batch * rt_q;
    let bid_x = get_tile_block_id().0;          // free block-id — NO grid lock
    let grid_x = get_num_tile_blocks().0;
    let mut c_part = unsafe { c.partition_full_mut(const_shape![1, BM, BN]) };
    //                          ^^^^^^^^^^^^^^^^^^ mutable VIEW from a read-only &Tensor
    //                          (NOT partition_mut — that re-offsets by block id)
    for tile in (bid_x .. total_tiles).step_by(grid_x as usize) {   // persistent grid-stride
        // ... grouped (GROUP_SIZE_M) schedule → (q, mi, ni) ...
        // ... inner K loop, f32 acc, mmaf ...
        store_view_tko_mut(&mut c_part, tile_c, [q, mi, ni], None, ...);  // explicit index
    }
}
```

Key: **`c.partition_full_mut(tile)`** turns the read-only `&Tensor` output into a
writable partition view at body time. Use `partition_full_mut` (NOT
`partition_mut`, which auto-re-offsets by block id) because the schedule computes
explicit `(q, mi, ni)` tile indices.

## Host side (`ffi.rs`)

Tensors cross the C-ABI as `*const TensorDesc` (dtype/shape/strides ride inside the
descriptor — no raw ptr + loose dim/stride/dtype args). The entry takes
`&Tensor<E,{[-1,-1,-1]}>` (TMA view pattern), so the host rebuilds borrowed
`Tensor`s from the descriptors via `borrow_tensor::<E>` (defined in
`ops/cutile_rs/ffi_util.rs`), then launches. `borrow_tensor` returns a
`ManuallyDrop<Tensor<E>>` over the PyTorch memory — the drop is already a no-op, so
there is NO op-level `Tensor::from_raw_parts` and NO `mem::forget` (that is the FFI
ownership gate by construction, Rule 10):

```rust
use crate::ffi_util::{borrow_tensor, dtype_str, TensorDesc};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_bmm(
    c: *const TensorDesc, a: *const TensorDesc, b: *const TensorDesc,
    /* tile sizes, transpose flags, compile options, grid, */
    device_id: i32, raw_stream: u64,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() { return -5; }
    let (a_d, b_d, c_d) = unsafe { (&*a, &*b, &*c) };
    let dty = match dtype_str(a_d.dtype) { Some(s) => s, None => return -2 };
    // Build Device from the tensor's own ordinal — never hardcode Device::new(0).
    let device = Device::new(device_id.max(0) as usize).unwrap();
    let stream = unsafe { Stream::borrow_raw(raw_stream as *mut c_void, &device) };

    let a_t = unsafe { borrow_tensor::<E>(a_d) };   // ManuallyDrop<Tensor<E>> — never freed
    let b_t = unsafe { borrow_tensor::<E>(b_d) };
    let c_t = unsafe { borrow_tensor::<E>(c_d) };   // OUTPUT — borrowed like an input
    let op = unsafe { bmm_kernel_static_persistent(&*a_t, &*b_t, &*c_t) }  // &* deref ManuallyDrop
        .generics(generics)
        .grid((grid_size as u32, 1, 1))            // explicit persistent grid — accepted, no lock
        .compile_options(opts);
    match op.sync_on(&stream) { Ok(_) => 0, Err(_) => -3 }
    // a_t/b_t/c_t drop as no-ops here — PyTorch memory is never freed.
}
```

Because the output is `&Tensor` (read-only param), the launcher does NOT derive a
grid from it → the explicit `.grid((grid_size,1,1))` persistent grid is accepted
as-is (no `validate_grids`). Native transpose: physical shapes/strides ride in the
descriptors unchanged; `trans_a`/`trans_b` become const generics.

The op is compiled into the single aggregated `cutile_kernels` cdylib
(`libcutile_kernels.so`) by adding a `mod bmm { include!(...) }` to
`cutile_kernels/src/lib.rs` — there is no per-op `.so`. See the full landed
version in [`ffi.rs`](./ffi.rs) and the Python cffi wrapper in
[`wrapper.py`](./wrapper.py).

## Takeaway

**Mutable outputs are declared read-only `&Tensor<{[-1,...]}>` and written via
`partition_full_mut` (this example), or raw `*mut E` + `make_tensor_view`. NEVER
`&mut Tensor`.** The read-only `&Tensor` form keeps the ergonomic Tensor boundary
(via `borrow_tensor::<E>` → `ManuallyDrop<Tensor>`, so no op-level `from_raw_parts`
and no `mem::forget`) while leaving the kernel free to pick any grid / block-id
convention (persistent grid-stride here) without the launcher's grid-axis lock.
