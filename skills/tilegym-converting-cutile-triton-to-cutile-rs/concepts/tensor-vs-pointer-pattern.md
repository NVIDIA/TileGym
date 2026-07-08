# &Tensor vs Raw Pointer Pattern

cutile-rs supports two kernel parameter patterns. Choosing the wrong one causes severe performance loss.

Both patterns cross the FFI boundary IDENTICALLY — every tensor is one
`const TensorDesc*` (see `concepts/ffi-bridge.md`). The distinction is purely
how `ffi.rs` UNPACKS the descriptor before calling the entry:

- **&Tensor entries** → `borrow_tensor::<E>(desc)` (a `ManuallyDrop<Tensor<E>>`),
  passed as `&*t`.
- **Raw-pointer entries** → `DevicePointer::<E>::from_cu_deviceptr(desc.ptr)`,
  with dims/strides read via `desc.dim(i)` / `desc.strides[i]`.

There is no `transmute` of pointers and no op-level `Tensor::from_raw_parts`.

## &Tensor Pattern (partition-based)

Use for: **matmul, attention, any kernel with regular tile access**

```rust
#[cutile::entry()]
unsafe fn kernel(
    z: &Tensor<f16, { [-1, -1] }>,       // OUTPUT: read-only PARAM type, -1 wildcards. NEVER &mut Tensor.
    x: &Tensor<f16, { [-1, -1] }>,       // input
) {
    let part = x.partition(const_shape![BM, BK]);
    let tile = part.load([pid.0, i]);
    let mut zp = z.partition_full_mut(const_shape![BM, BN]);   // mutable VIEW from read-only &Tensor
    store_view_tko_mut(&mut zp, result, [pid.0, pid.1], ...);  // explicit index
}
```

**Why read-only `&Tensor` (not `&mut Tensor`)**: the launcher keys on `&` vs
`&mut`. A `&mut Tensor` output emits `KernelOutputStored::grid` → the launch grid
is LOCKED to the output partition grid (descriptor dim order, exact shape, no
`-1`) — the axis-transpose / `-1` reject trap. A read-only `&Tensor` emits no
grid → free grid + `-1` OK. `partition_full_mut` is a device-body op the host
launcher never sees. See `references/no-mut-tensor-output.md`. TMA `strides=[?,1]`
still applies.

**Host side**: pass `z` like an input — `borrow_tensor::<E>(z_d)` (a
`ManuallyDrop<Tensor>`, no `mem::forget`), then `&*z_t`; launch any explicit
grid. (Worked example: `examples/bmm`.)

**IR produced**: `make_tensor_view ... strides=[?,1]` → `make_partition_view` (emitted by `tview.partition(...)` / `tview.partition_mut(...)` method form) → `load_view_tko` / `store_view_tko_mut`

## Raw Pointer Pattern (scatter/gather)

Use for: **layer norm, softmax, any kernel with irregular/masked access**

```rust
#[cutile::entry()]
unsafe fn kernel(x_ptr: *mut f32, ...) {
    let base = pointer_to_tile(x_ptr);
    let ptrs = base.reshape(const_shape![1]).broadcast(const_shape![N]);
    let loaded_ptrs = ptrs.offset_tile(offsets);
    let (vals, tok) = load_ptr_tko(loaded_ptrs, ordering::Weak, None::<scope::TileBlock>, ...);
}
```

**Why**: Scatter/gather access pattern can't be expressed as partition tiles.

**Host side**: Pass `DevicePointer<T>` via `DevicePointer::from_cu_deviceptr(desc.ptr)`
(never `transmute`). Read dims/strides from the descriptor (`desc.dim(i)` /
`desc.strides[i]`). (Worked example: `examples/softmax`.)

**IR produced**: `pointer_to_tile` → `reshape` → `broadcast` → `offset` → `load_ptr_tko`

## Decision Table

| Access Pattern | Example | Use |
|---------------|---------|-----|
| Regular 2D tiles | GEMM, batched | &Tensor |
| Regular 4D tiles | Attention Q/K/V | &Tensor + 4D partition |
| Row-wise reduction | Layer norm, RMS norm | Raw pointer |
| Masked gather/scatter | Softmax, dropout | Raw pointer |
| Mixed | Attention output + softmax | Raw pointer for softmax, &Tensor for output |

## Persistent Kernels (grid-stride loop)

Persistent kernels use `grid < num_tiles` with a grid-stride loop:
```
for row in (pid.0..n_rows).step_by(grid.0 as usize) { ... }
```

**`&mut Tensor` is BANNED entirely** (not just for persistent) — it emits an
inferred grid that LOCKS the launch grid to the output partition grid, so any
`grid != inferred` (persistent, or a transposed block-id order) is rejected:
```
error: Specified launch grid does not match inferred tensor partition grid
```
Use a read-only `&Tensor<{[-1,...]}>` + `partition_full_mut`, or a raw pointer +
`make_tensor_view`. Both emit NO inferred grid → any explicit grid is accepted
(persistent `grid < num_tiles` included). For the strides=[?,1] fix on the
raw-pointer path, use `Array::<{[-1, 1]}> { dims: &[stride_val] }`.

| Kernel Type | Output Pattern | Grid |
|-------------|---------------|------|
| Non-persistent (matmul, attention) | read-only `&Tensor` + `partition_full_mut` (or raw ptr) | any explicit grid |
| Persistent (softmax, layer norm) | read-only `&Tensor` + `partition_full_mut`, or raw ptr + `make_tensor_view` | grid < num_tiles |
