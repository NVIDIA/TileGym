# Op Mapping: CUDA Tile IR -> cutile-rs

**Verified** against the provided `cutile-rs` checkout.

> **Critical**: many math ops take explicit `rounding::Mode` + `ftz::Mode`
> static-typed args via `static_params`. Old "single-arg" call shapes will not
> compile. Also always use `#[cutile::entry()]` with parentheses; a bare
> `#[cutile::entry]` (no parentheses) is NOT recognized as an entry — the macro
> only matches a parenthesized attribute, so no launcher is generated.

## Block ID / Grid intrinsics (read this FIRST)

**Canonical block-id**: `get_tile_block_id() -> (i32, i32, i32)`. It returns
`(bid.x, bid.y, bid.z)` regardless of grid dimensionality.

```rust
// 1D grid
let pid = get_tile_block_id().0;

// 2D grid
let (m_idx, n_idx, _) = get_tile_block_id();

// 3D grid
let pid: (i32, i32, i32) = get_tile_block_id();
```

Anti-patterns:

```rust
let bid = block_id::<0>();
let bid = cuda_block_id();
let bid = bid(0);
```

Companion: `get_num_tile_blocks() -> (i32, i32, i32)`.

## Entry Attributes And Host Launchers

Use `#[cutile::entry()]` when the entry has no options:

```rust
#[cutile::entry()]
pub unsafe fn kernel_name<E: ElementType, const BM: i32>(...) {
    ...
}
```

Use option form only when the reference requires options:

```rust
#[cutile::entry(print_ir = false, unchecked_accesses = true)]
pub unsafe fn kernel_name(...) {
    ...
}
```

Do not use bare `#[cutile::entry]` (no parentheses). The macro only matches a
parenthesized attribute, so the bare form is NOT recognized as an entry at all —
no host launcher is generated, so `.generics()` is unavailable on the result.
The compiler registry error text also expects `#[entry(...)]`. Always write
`#[cutile::entry()]` or `#[cutile::entry(...)]`.

## Pointer Scatter/Gather

| IR Op | cutile-rs Rust |
|-------|----------------|
| `pointer_to_tile %ptr` | `pointer_to_tile(ptr)` -> `PointerTile<*mut E, {[]}>` |
| `reshape %ptr : tile<ptr> -> tile<1xptr>` | `ptr_tile.reshape(const_shape![1])` |
| `broadcast %r : tile<1xptr> -> tile<Nxptr>` | `ptr_1d.broadcast(const_shape![N])` |
| `offset %ptrs, %offsets` | `ptrs.offset_tile(offsets)` |

Split pointer tile chains into typed bindings.

```rust
let p0: PointerTile<*mut E, { [] }> = pointer_to_tile(ptr);
let p1: PointerTile<*mut E, { [1] }> = p0.reshape(const_shape![1]);
let ps: PointerTile<*mut E, { [N] }> = p1.broadcast(const_shape![N]);
```

### `load_ptr_tko`

```rust
pub fn load_ptr_tko<
    E: ElementType,
    const S: [i32; N],
    O: ordering::LoadMode,
    Sc: scope::Mode,
    const CYCLES: u32,
>(
    source: PointerTile<*mut E, S>,
    memory_ordering: O,
    memory_scope: Option<Sc>,
    mask: Option<Tile<bool, S>>,
    padding_value: Option<E>,
    token: Option<Token>,
    latency: Latency<CYCLES>,
) -> (Tile<E, S>, Token)
```

Call:

```rust
let (tile, tok) = load_ptr_tko(
    ptrs, ordering::Weak, None::<scope::TileBlock>,
    Some(mask), Some(pad), None, Latency::<0>
);
```

### `store_ptr_tko`

```rust
pub fn store_ptr_tko<
    E: ElementType,
    const S: [i32; N],
    O: ordering::StoreMode,
    Sc: scope::Mode,
    const CYCLES: u32,
>(
    destination: PointerTile<*mut E, S>,
    value: Tile<E, S>,
    memory_ordering: O,
    memory_scope: Option<Sc>,
    mask: Option<Tile<bool, S>>,
    token: Option<Token>,
    latency: Latency<CYCLES>,
) -> Token
```

Call:

```rust
let tok = store_ptr_tko(
    ptrs, val, ordering::Weak, None::<scope::TileBlock>,
    Some(mask), Some(tok), Latency::<0>
);
```

## Tensor / Partition

### Token creation

IR `cuda_tile.make_token` maps to `new_token_unordered()`.

```rust
let token: Token = new_token_unordered();
```

Use this token for manual `make_tensor_view(...)` construction. Most TMA/view
kernels should prefer typed `&Tensor` parameters and let the entry macro build
the tensor view.

### `make_tensor_view`

```rust
pub unsafe fn make_tensor_view<E: ElementType, const D: [i32; N], const C: [i32; N]>(
    base: PointerTile<*mut E, {[]}>,
    shape: Shape<D>,
    strides: Array<C>,
    token: Token,
) -> Tensor<E, D>
```

Required `unsafe`.

### Reading dynamic tensor dimensions

For `&Tensor` entry parameters, prefer metadata shape accessors when only scalar
dimensions are needed:

```rust
let a_shape: Shape<{ [-1, -1] }> = a.shape();
let b_shape: Shape<{ [-1, -1] }> = b.shape();
let m: i32 = a_shape[0];
let n: i32 = b_shape[1];
```

`tensor.shape()` uses `get_tensor_shape_meta`; it does not emit a real
`cuda_tile.get_tensor_shape` op. Avoid `get_tensor_shape(a)` unless the reference
IR itself has that op, because it adds real tensor-view layout attributes.

### Partition view construction

Method form is preferred for real loads/stores:

```rust
let part: Partition<E, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
let part_p: Partition<E, { [BK, BM] }> =
    a.partition_permuted(const_shape![BK, BM], const_array![1, 0]);
let mut out: PartitionMut<E, { [BM, BN] }> =
    unsafe { c.partition_full_mut(const_shape![BM, BN]) };
```

Low-level `make_partition_view` is the intended exception when the reference has
a structurally distinct no-padding metadata partition, for example TMA GEMM
querying `get_index_space_shape` separately from padded loads:

```rust
let token = get_tensor_token(a);
let a_iter: Partition<E, { [BM, BK] }> = make_partition_view(
    a,
    const_shape![BM, BK],
    padding::None,
    dim_map::Identity,
    token,
);
let a_space: [i32; 2] = get_index_space_shape(&a_iter);

let a_part: Partition<E, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
let b_part: Partition<E, { [BK, BN] }> = b.partition(const_shape![BK, BN]);
```

### Mutable output: NEVER `&mut Tensor` — use raw `*mut E` or read-only `&Tensor`

**HARD RULE (see [`no-mut-tensor-output.md`](no-mut-tensor-output.md)):**
a kernel output is NEVER declared `&mut Tensor` (nor `MappedPartitionMut` /
`Partition<&mut Tensor>` host arg). Write outputs ONE of two ways:

**Preferred — read-only `&Tensor<{[-1,...]}>` + in-kernel `partition_full_mut`**
(worked example: `examples/bmm/kernel.rs` + `examples/bmm/ffi.rs`):
```rust
// kernel:  c: &Tensor<E, { [-1, -1, -1] }>        // OUTPUT — READ-only param type, -1 OK
//          let mut cp = c.partition_full_mut(const_shape![1, BM, BN]);  // mut VIEW in body
//          store_view_tko_mut(&mut cp, tile, [q, mi, ni], ...);          // explicit index
// host:    c crosses as *const TensorDesc; borrow_tensor::<E>(c_d) (ManuallyDrop, never freed);
//          launch ANY grid (persistent OK). (partition_full_mut, NOT partition_mut — the
//          latter re-offsets by block id.)
```

**Alternative — raw `*mut E` pointer + in-kernel `make_tensor_view`** (worked
example: `examples/softmax`):
```rust
// kernel:  c_ptr: *mut E, c_d0,c_d1,c_d2: i32, c_s0,c_s1: i32   // pointer + dims + strides
//          let cv = make_tensor_view(c_ptr, [batch,m,n], [c_s0,c_s1,1], token);
//          let mut cp = cv.partition(const_shape![1,BM,BN]); store_view_tko_mut(&mut cp, tile, [..], ...);
// host:    c crosses as *const TensorDesc; read dims/strides off it and wrap the pointer with
//          DevicePointer::<E>::from_cu_deviceptr(c_d.ptr); launch ANY grid (persistent OK).
```

**Why NOT `&mut Tensor`:** the launcher macro keys on `&` vs `&mut`. A `&mut`
param emits `KernelOutputStored::grid(&out)` → the launch grid is **LOCKED** to
the output's partition grid in descriptor dim order (full-tuple match), and the
shape is EXACT-matched so `{[-1,...]}` is rejected. That produces the two
recurring footguns — the `{[-1,...]}` reject trap AND the block-id↔dim
axis-transpose lock (output heads/rows unwritten / rc=-3). A raw
pointer or a read-only `&Tensor` emits NO grid → no lock → the kernel chooses any
grid / block-id convention (persistent, transposed, CGA) freely.
`-1` wildcards are valid on read-only `&Tensor` inputs (x, y, and `&Tensor`
outputs); they are forbidden ONLY because `&mut` would EXACT-match them — which
is why we drop `&mut` entirely. Persistent + CGA perf (`occupancy`,
`num_cta_in_cga`) is reachable on the raw-pointer path (see `examples/bmm`).

### `order=(...)` loads and `partition_permuted`

Represent a load order once:

| Python / IR pattern | cutile-rs pattern |
|---------------------|-------------------|
| Load from original tensor with `order=(...)` | `tensor.partition_permuted(tile, const_array![...])` on the original logical tensor |
| Host already provides a physically swapped tensor | plain `tensor.partition(tile)` |

Do not both pre-swap the host tensor and call `partition_permuted`.

### `load_view_tko`

```rust
pub fn load_view_tko<
    E: ElementType,
    const D: [i32; N],
    O: ordering::LoadMode,
    Sc: scope::Mode,
    T: tma::Mode,
>(
    view: &Partition<E, D>,
    index: [i32; N],
    memory_ordering: O,
    memory_scope: Sc,
    latency: Option<i32>,
    tma: T,
) -> Tile<E, D>
```

Call:

```rust
let tile = load_view_tko(
    &part, [bid_m, bid_n],
    ordering::Weak, scope::TileBlock, Some(3i32), tma::Enabled
);
```

### `store_view_tko_mut`

```rust
pub unsafe fn store_view_tko_mut<
    E: ElementType,
    const D: [i32; N],
    O: ordering::StoreMode,
    Sc: scope::Mode,
    T: tma::Mode,
>(
    view: &mut PartitionMut<E, D>,
    tile: Tile<E, D>,
    index: [i32; N],
    memory_ordering: O,
    memory_scope: Sc,
    latency: Option<i32>,
    tma: T,
) -> Token
```

Call:

```rust
unsafe {
    store_view_tko_mut(
        &mut part_mut, tile, [bid_m, bid_n],
        ordering::Weak, scope::TileBlock, Some(1i32), tma::Enabled
    )
};
```

### Missing APIs

| Missing | Workaround |
|---------|------------|
| `make_strided_view` | Use `tview.partition(...)` plus tile-level index |
| legacy `load_from_view` | Use `load_view_tko` |
| legacy `store_to_view_mut` | Use `store_view_tko_mut` |
| `make_partition_view_padded` | `tview.partition(...)` auto-emits padding for loads |

## Math + Arithmetic with mode args

### `fma`

```rust
let z = fma(a, b, c, rounding::NearestEven, ftz::Enabled);
```

### `addf` / `subf` / `mulf`

```rust
let z = addf(a, b, rounding::NearestEven, ftz::Enabled);
```

Rust operators use default modes. Use named functions when the reference records
non-default rounding or FTZ.

### `maxf` / `minf`

```rust
let z = maxf(a, b, nan::Disabled, ftz::Enabled);
```

### `exp2`, `rsqrt`, `log`, `log2`, `true_div`

```rust
let p = exp2(qk, ftz::Enabled);
let r = rsqrt(x, ftz::Enabled);
let l = log2(x);
let q = true_div(a, b);
```

For `divi signed rounding<positive_inf>`, use `(a + b - 1) / b` or a scalar
ceil-div helper.

## Type Conversion

| IR Op | cutile-rs |
|-------|-----------|
| `exti signed` | `exti(x)` with return type annotation |
| `trunci` | `trunci(x, overflow::Wrap)` or matching overflow mode |
| `ftof` / `itof` / `ftoi` | `convert_tile(x)`; same-type conversions are DCE-elided |
| `ftof f32 -> tf32` | `let t: Tile<tf32, S> = convert_tile(f32_tile);` |

`tf32` is re-exported from `cutile::core::*`.

## Tile Creation

| IR Op | cutile-rs |
|-------|-----------|
| `constant` | `constant(0.0f32, const_shape![N])` for compile-time literals |
| runtime scalar tile | `broadcast_scalar(value, const_shape![N])` |
| `iota` | `iota(const_shape![N])` |
| `reshape` | `x.reshape(const_shape![...])` |
| `broadcast` | `x.broadcast(const_shape![...])` |
| `permute [1,0]` | `permute(x, const_array![1, 0])` |

## `mmaf`

```rust
pub fn mmaf<EIn: ElementType, EOut: ElementType,
            const LHS: [i32; N], const RHS: [i32; N], const ACC: [i32; N]>(
    lhs: Tile<EIn, LHS>,
    rhs: Tile<EIn, RHS>,
    acc: Tile<EOut, ACC>,
) -> Tile<EOut, ACC>
```

Mixed f16/bf16 input with f32 accumulate is native:

```rust
let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
acc = mmaf(a_f16_or_bf16, b_f16_or_bf16, acc);
```

For f32 GEMM-family inputs, check the f32 reference IR or cuTile-Python source.
If it has two `ftof ... -> ...xtf32` casts and `mmaf` operands are tf32,
reproduce that:

```rust
let a_tc: Tile<tf32, { [BM, BK] }> = convert_tile(a_f32);
let b_tc: Tile<tf32, { [BK, BN] }> = convert_tile(b_f32);
acc = mmaf(a_tc, b_tc, acc);
```

Do not cast f16/bf16 tiles to tf32 unless the reference does.

## Reduce

| IR Op | cutile-rs |
|-------|-----------|
| `reduce dim=0 { addf }` | `reduce_sum(x, 0)` |
| `reduce dim=1 { maxf }` | `reduce_max(x, 1)` |
| `reduce dim=1 { addf }` | `reduce_sum(x, 1)` |

For a 2D tile, reducing `dim=1` returns rank-1:

```rust
let row_max: Tile<E, { [M] }> = reduce_max(qk, 1);
let row_max_col: Tile<E, { [M, 1] }> = row_max.reshape(const_shape![M, 1]);
let qk_shifted = qk - row_max_col.broadcast(const_shape![M, N]);
```

## Entry launch hints: do not pin occupancy / num_cta_in_cga

Never bake `occupancy=N` or `num_cta_in_cga=N` into
`#[cutile::entry(...)]` unless the reference IR genuinely carries that value.
Leave them empty so runtime/autotune can set resident-CTA policy through FFI
compile options.

```rust
// WRONG unless the reference explicitly carries these exact hints:
#[cutile::entry(optimization_hints(occupancy = 1, num_cta_in_cga = 1))]

// RIGHT when the reference leaves hints unset:
#[cutile::entry()]
```

Pinning `occupancy=1` can preserve correctness while creating a large perf cliff.
This is entry/host-side, not kernel math.

## Control Flow

| IR Op | cutile-rs |
|-------|-----------|
| `for %i in (0 to N, step S)` | `for i in (0i32..N).step_by(S as usize)` |
| `if/else with yield` | statement `if` with explicit mutable assignments |

Prefer `for j in 0i32..num_tiles` when the step is 1.

## Assume Hints

```rust
pub unsafe fn assume_div_by<T, const DIVISOR: i32>(x: T) -> T;
pub unsafe fn assume_bounds_lower<T, const LOWER: i32>(x: T) -> T;
```

| IR Op | cutile-rs |
|-------|-----------|
| `assume div_by<16>` | `unsafe { assume_div_by::<_, 16>(x) }` |
| `assume bounded<0, ?>` | `unsafe { assume_bounds_lower::<_, 0>(x) }` |

Apply at the top of the `#[cutile::entry()]` body before usage.

## Verification provenance

Cross-checked:

- `cutile/src/_core.rs`: `get_tile_block_id`, token helpers, tensor view ops,
  load/store view ops, math modes, conversion, `tf32`, `mmaf`, assume ops.
- `cutile/src/tensor.rs`: `PartitionMut`, `MappedLaunchPartition`,
  partition-grid inference, and `KernelOutput` impls for partition/mapped
  partition outputs.
- `cutile/src/tile_kernel.rs`: explicit grid validation against inferred grids.
- `cutile-macro/src/_module.rs`: host launcher generation uses `KernelInput` and
  `KernelOutput` bounds.
- `cutile-compiler/src/compiler/_function.rs`: registry expects `#[entry(...)]`.
