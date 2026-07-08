# Rule: kernel outputs are NEVER `&mut Tensor` (raw `*mut E` or read-only `&Tensor` only)

This rule is a hard ban on
`&mut Tensor<T, {...}>` as a kernel entry parameter for outputs. All converted
kernels must write through ONE of:

1. **read-only `&Tensor<T, {[-1, ...]}>` + in-kernel `partition_full_mut`**
   (worked example: `examples/bmm`) — preferred for
   tiled / persistent / transposed outputs; keeps the ergonomic Tensor boundary
   (host borrows the descriptor with `borrow_tensor::<E>(desc)` →
   `ManuallyDrop<Tensor<E>>`, never freed) while emitting NO grid lock.
2. **raw `*mut E` device pointer + in-kernel `make_tensor_view`** (worked
   example: `examples/softmax`) — when no Tensor
   boundary is wanted (host wraps `desc.ptr` with
   `DevicePointer::<E>::from_cu_deviceptr`).

`&mut Tensor` (and `MappedPartitionMut` / `Partition<&mut Tensor>` host args) are
**banned**. Why follows.

## Root cause — the launcher macro branches on `&` vs `&mut`

`cutile-macro/src/kernel_launcher_generator.rs` generates DIFFERENT host launch
code per parameter, keyed on `ty.mutability.is_some()`:

**`&mut Tensor` (mutable) branch:**
```rust
launch_grid_expr_strs.push("KernelOutputStored::grid(&{var})?");   // contributes inferred grid
// shape validation: EXACT equality, -1 NOT substituted
kernel_launch_assert(valid_shape == &given_shape, "partition shape mismatch ...");
```

**`&Tensor` (read-only) branch:**
```rust
KernelInputStored::push_kernel_args(...)   // treated as INPUT; launch_grid_expr_strs stays EMPTY
// shape validation: -1 substituted with the actual dim
let valid_shape_mixed = zip(valid, given).map(|(e,g)| if e == -1 { g } else { e });
```

So writing `&mut` flips TWO switches at once:

1. **Grid lock.** The mutable output is pushed into `inferred_grids`, and
   `infer_launch_grid` → `validate_grids` forces the launch grid to EQUAL the
   output's partition grid **on every dim, in descriptor dim order**
   (`(u32,u32,u32)` full-tuple `!=` check, `tile_kernel.rs`). The kernel's
   `get_tile_block_id().k` must therefore map to descriptor dim `k` for all `k`.
   Any transpose (e.g. attention wanting `head = bid(0)` while the output is
   `[B, num_head, …]` so dim0 = batch) is rejected with
   `"Specified launch grid does not match inferred tensor partition grid"`
   (FFI rc=-3) — output heads/rows unwritten, launcher reject. Persistent grids (`num_blocks < num_tiles`) are also
   impossible on this path.

2. **`-1` reject trap.** The mutable branch shape-matches EXACTLY (no `-1`
   substitution), so `out: &mut Tensor<E, {[-1, ...]}>` is rejected for every
   config (`partition shape mismatch. Expected [-1,..], got [..]`). A mutable
   output is forced to carry concrete tile dims, which then re-introduces #1.

A read-only `&Tensor` (mutated in-body via `partition_full_mut`) and a raw
`*mut E` pointer both emit **no** `KernelOutputStored::grid`, so neither
contributes to `inferred_grids` → **no `validate_grids` lock** → the kernel
launches any grid it wants and uses the real (possibly transposed / persistent)
block id. `partition_full_mut` is a DEVICE-side op the host launcher never sees,
so it does not re-trigger the lock.

## Evidence (converted kernels)

| kernel | output param | grid lock? | result |
|---|---|---|---|
| tiled / transposed output (`examples/bmm`) | `&Tensor<{[-1,…]}>` + `partition_full_mut` | none | PASS |
| row-wise output (`examples/softmax`) | raw `*mut E` + `make_tensor_view` | none | PASS |
| 2D output whose grid happens to align | `&mut Tensor<{[BM,BN]}>` | locked (dims happen to align) | PASS by luck |
| **multi-head / transposed output** | `&mut Tensor<{[1,TILE_H,TILE_D]}>` | **locked → head/batch transpose** | **FAILED** (launcher reject) |

## Notes

- Persistent + CGA perf (`occupancy`, `num_cta_in_cga`) is reachable on the
  raw-pointer / read-only `&Tensor` path, so banning `&mut Tensor` costs no
  performance.
- `validate_agent_b.sh` currently FAILs the `-1`-shaped mutable-output sub-case;
  a blanket check on ANY `&mut Tensor` entry param would enforce this policy in full.
