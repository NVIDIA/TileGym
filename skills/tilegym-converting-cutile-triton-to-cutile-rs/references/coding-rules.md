# cutile-rs Conversion Rulebook (coding rules + known gaps)

Single source for converting kernels to cutile-rs. **Part 1** = prescriptive rules
(Agent B reads before writing `kernel.rs`; treat as a pre-build checklist + source/API
pin). **Part 2** = troubleshooting index + capability gaps (match a failure SIGNATURE,
then apply the fix). Rule numbers (1–49) are stable IDs cited elsewhere as "Rule N" — do
not renumber. Verified against cutile-rs 0.2.0.

## Index

**Part 1 — Rules by category** (numbers are Rule IDs; full text in numeric order below)

| Category | Rules |
|---|---|
| DSL syntax & macro form | 1, 2, 3, 6, 12, 13, 14, 22, 24, 25, 26 |
| Types & inference | 4, 5, 7, 15, 16, 23, 46, 47 |
| Memory / FFI / ABI | 9, 10, 11, 17, 31, 32, 33, 34, 35, 43, 44 |
| Performance | 8, 18, 19, 20, 21, 27, 28, 29, 30, 36, 41, 42, 45, 48 |
| Pipeline / process | 37, 38, 39, 40, 49 |
| `&mut`/output & transpose ownership | 34, 43, 44 (+ `no-mut-tensor-output.md`) |

**Part 2 — Troubleshooting & gaps** (match the signature)

- Forward-only port convention
- First-Build Blockers
- Correctness / runtime failure signatures
- Compiler & JIT bugs (with cutile-rs 0.2.0 status)
- Missing APIs / design gaps
- Runtime / wrapper gaps
- Performance gaps
- Process pitfalls
- Minimal triage order

---

# Part 1 — Rules

## Quick Reference

| # | Rule | Failure prevented |
|---|------|-------------------|
| 1 | Bind every rank/type-changing call | E0283 inference failures |
| 2 | No scientific notation or qualified constants in JIT values | parser / dense-value failures |
| 3 | Token bindings are ordinary names | macro pattern failures |
| 4 | Bind reductions before later math | inference and shape drift |
| 5 | 2D reduce axis 1 returns rank 1 | RMS and softmax shape mismatch |
| 6 | Avoid const arithmetic in type/shape positions | stable Rust const-generic errors |
| 7 | Integer conversions need explicit intermediate tile types | `exti(scalar_to_tile(...))` E0283 |
| 8 | MMA accumulates in f32; fp32 GEMM uses tf32 when reference does | accuracy/perf loss |
| 9 | Dynamic `Shape` / `Array` literals carry only dynamic entries | bad tensor metadata |
| 10 | FFI borrows PyTorch memory via `borrow_tensor` (ManuallyDrop) / `DevicePointer::from_cu_deviceptr` | invalid frees / SIGABRT |
| 11 | Pipeline strides are full rank; kernel static stride dims are omitted | dump/layout mismatch |
| 12 | Match entry pattern and always use parenthesized `#[cutile::entry()]` | launcher not generated |
| 13 | Runtime `step_by` takes `usize` | loop compile error |
| 14 | No associated const padding such as `Some(E::ZERO)` in JIT calls | JIT expression-position failure |
| 15 | Avoid tuple destructuring before reductions unless return type is annotated | inference failure |
| 16 | Data dtype is generic `<E: ElementType>` | missing dtype dispatch |
| 17 | One source of truth: cdylib and pipeline include the same `kernel.rs` | generated IR differs from launched kernel |
| 18 | Runtime scalar tiles use `broadcast_scalar` | runtime treated as constant |
| 19 | Use `true_div` where reference uses true division | math/perf mismatch |
| 20 | Match latency, TMA, occupancy and `num_cta_in_cga` to analysis/reference | silent perf drift |
| 21 | Pointer `assume_div_by<16>` only when guaranteed; avoid false scalar assumes | corrupt boundary masks |
| 22 | Mutable tile state crossing `if` is assigned in both branches | dropped compiler state |
| 23 | `ct.Constant[int]` normally becomes a const generic | missed constant folding |
| 24 | Nested loop/if mutations use explicit carry behavior | lost updates |
| 25 | Do not write expression-style `if` returning a `Tile` | unsupported tile-valued control flow |
| 26 | Math APIs take explicit rounding/ftz/nan modes | wrong API shape |
| 27 | GEMM transpose has one owner | timed Python transpose or wrong values |
| 28 | Padded view loads keep TMA when reference uses TMA | ragged tile bugs |
| 29 | Persistent TMA latency is case-specific | persistent-kernel abort / reduction mismatch |
| 30 | Batched TMA descriptors require aligned outer strides | TMA descriptor failure |
| 31 | Accumulator dtype generic slots usually hardcode `"f32"` in FFI | MMA generic mismatch |
| 32 | Rust 2024 exports use `#[unsafe(no_mangle)]` | C ABI build failure |
| 33 | Standalone crates depend on package `cutile-compiler` | proc macro cannot resolve compiler crate |
| 34 | A transpose has one owner: host strides or `partition_permuted`, never both | axis-transpose bugs |
| 35 | Per-kernel ABI pieces must agree | import skips / dtype dispatch failures |
| 36 | Stable const-generic escape hatch is literal entry specialization | repeated nightly-feature loops |
| 37 | Python wrappers are autograd surfaces | grad-enabled tests fail |
| 38 | Run a fixed-config canary before autotune/CUPTI | hidden FFI aborts |
| 39 | In B-only mode B owns wrapper/canaries; in full pipeline D owns host wrapper | waiting for absent agents |
| 40 | Register `cutile-rs` only for dispatcher tests | coverage tests call direct backends |
| 41 | Do not clone/copy unconditionally in wrapper fast paths | CUPTI times copies |
| 42 | After correctness passes, perf cliffs usually start in wrapper/config/pattern drift | chasing math too early |
| 43 | Outputs are never `&mut Tensor`; use read-only tensor + `partition_full_mut` or raw pointer | launch-grid rc=-3 |
| 44 | `partition_full_mut` requires `PartitionMut` import | E0599 |
| 45 | Pointer tiled loops use ceil-div and tail masks when source uses `ct.cdiv` | non-pow2 N aborts/wrong output |
| 46 | Convert const generics to float via `scalar_to_tile` + `convert_tile`, not `N as f32` or `itof` | unsupported cast / signedness writer skew |
| 47 | Annotate `load_ptr_tko` result tuples when mask/padding does not constrain shape | JIT inference failure |
| 48 | Preserve source semantics beyond the one dumped shape | exact-cover IR hiding tail bugs |
| 49 | Final build log records every compile/JIT fix before validator | repeated rediscovery |

## Rule 1: Bind Rank/Type-Changing Calls

The macro and Rust inference are fragile when conversions are chained. Bind each step with an explicit type.

```rust
let p0: PointerTile<*mut E, { [] }> = pointer_to_tile(x_ptr);
let p1: PointerTile<*mut E, { [1] }> = p0.reshape(const_shape![1]);
let ps: PointerTile<*mut E, { [BLOCK] }> = p1.broadcast(const_shape![BLOCK]);

let bid_i32: Tile<i32, { [] }> = scalar_to_tile(bid_row);
let bid_i64: Tile<i64, { [] }> = exti(bid_i32);
```

Do not write:

```rust
let bid_i64: Tile<i64, { [] }> = exti(scalar_to_tile(bid_row));
```

That exact pattern produces E0283.

## Rule 2: No Scientific Notation Or Qualified Constants

Use ordinary decimal literals in JIT-visible calls:

```rust
let eps: Tile<f32, { [] }> = scalar_to_tile(0.000009999999747378752f32);
let neg = constant(-340282346638528859811704183484516925440.0f32, const_shape![BLOCK]);
```

Avoid `1e-5`, `std::f32::consts::*`, `core::f32::*`, and `f32::NEG_INFINITY` inside kernel DSL calls.

## Rule 3: Token Bindings Are Plain Identifiers

Use ordinary names and tolerate unused-variable warnings:

```rust
let (tile, tok): (Tile<E, { [BLOCK] }>, Token) = load_ptr_tko(...);
let _ = tok;
```

Avoid underscore patterns in macro-sensitive destructures.

## Rule 4: Bind Reductions Before Math

```rust
let sum: Tile<f32, { [] }> = reduce_sum(xf, 0i32);
let mean: Tile<f32, { [] }> = sum / denom;
let inv: Tile<f32, { [] }> = rsqrt(mean + eps, ftz::Enabled);
```

Apply this to `reduce_sum`, `reduce_max`, `reduce_min`, and reductions feeding `exp2`, `rsqrt`, or division.

## Rule 5: 2D Reduce Rank

For `Tile<T, {[M, N]}>`, reducing axis 1 returns `Tile<T, {[M]}>`, not `[M, 1]`.

```rust
let row_sum: Tile<f32, { [M] }> = reduce_sum(x, 1i32);
let row_sum_col: Tile<f32, { [M, 1] }> = row_sum.reshape(const_shape![M, 1]);
```

## Rule 6: Const Shapes And Stable Rust

Simple const shape generics are fine:

```rust
let cols: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
```

Stable Rust rejects const arithmetic in type/shape positions:

```rust
type Bad = Tile<f32, { [BLOCK + 1] }>;
```

Move arithmetic to value position, host code, or literal-specialized entries.

## Rule 7: Explicit Integer Conversion Types

`exti` and `trunci` need both input and output types visible.

```rust
let pid32: Tile<i32, { [] }> = scalar_to_tile(pid);
let pid64: Tile<i64, { [] }> = exti(pid32);
let pid32_again: Tile<i32, { [] }> = trunci(pid64, overflow::None);
```

Use i64 pointer offsets when Python/Triton casts offsets to int64 or the element count may exceed i32.

## Rule 8: MMA Accumulator And tf32

MMA accumulators are normally f32:

```rust
let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
acc = mmaf(a_tile, b_tile, acc);
let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
```

For fp32 GEMM-family kernels, inspect the reference. If it casts f32 A/B to `ct.tfloat32`, cutile-rs must do the same before `mmaf`:

```rust
let a_tf32: Tile<tf32, { [BM, BK] }> = convert_tile(a_f32);
let b_tf32: Tile<tf32, { [BK, BN] }> = convert_tile(b_f32);
acc = mmaf(a_tf32, b_tf32, acc);
```

## Rule 9: Dynamic Shape / Array Literals

Only dynamic entries appear in `dims`; static entries live in the type.

```rust
let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[m, n] };
let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[row_stride] };
```

## Rule 10: FFI Memory Ownership

Borrow PyTorch allocations; do not create owned buffers around them. Tensors
cross the boundary as `*const TensorDesc` (see Rule 35); the FFI unpacks them
without ever freeing PyTorch memory.

For `&Tensor` entries, borrow via `borrow_tensor::<E>(desc)`, which returns a
`ManuallyDrop<Tensor<E>>` — the ownership gate. Do NOT call `Tensor::from_raw_parts`
in op code and do NOT `core::mem::forget`; `from_raw_parts` now lives ONLY inside
`borrow_tensor`, and `ManuallyDrop` means the borrowed wrapper is never freed.

```rust
// desc: &TensorDesc (from unsafe { &*x } after a null check).
let x_t = unsafe { borrow_tensor::<E>(desc) };  // ManuallyDrop<Tensor<E>> — never freed

let rc = match op.sync_on(&stream) {
    Ok(_) => 0,
    Err(e) => {
        eprintln!("cutile_{kernel_name} error: {e:?}");
        -3
    }
};
rc  // x_t drops here as a no-op (ManuallyDrop) — no mem::forget needed
```

For raw pointer entries, use `DevicePointer::<E>::from_cu_deviceptr(desc.ptr)` in
FFI (no `borrow_tensor` / ownership gate needed — the kernel never holds a Tensor
wrapper). Do not use `transmute`.

## Rule 11: Static Stride Dims And Pipeline Strides

Inside kernels, omit static stride entries from `Array::dims`. At the pipeline boundary, pass full logical rank metadata:

```rust
const X_STRIDES: &[i32] = &[N, 1];
compiler.strides(&[("x", X_STRIDES)]);
```

## Rule 12: Entry Pattern And Attribute Form

Pick params from reference IR structure:

- `load_view_tko` / `store_view_tko`: use `&Tensor<E, {[-1, ...]}>` and partitions.
- `load_ptr_tko` / `store_ptr_tko`: use raw `*mut E` in the kernel and `DevicePointer<E>` in FFI.
- Mixed structural variants: write one entry per variant.

Always use the parenthesized function-like attribute:

```rust
#[cutile::entry()]
pub unsafe fn op_kernel<E: ElementType, const BLOCK: i32>(...) { ... }
```

A bare `#[cutile::entry]` (no parentheses) is simply NOT recognized as an entry —
the macro only matches a parenthesized attribute, so no launcher is generated at
all. Always write `#[cutile::entry()]` (no options) or `#[cutile::entry(...)]`
(with options).

## Rule 13: `step_by`

```rust
for tile_id in (bid..total_tiles).step_by(grid_x as usize) {
    ...
}
```

Use `for j in 0i32..num_tiles` when the step is 1.

## Rule 14: No Associated Const Padding In JIT Calls

This fails in pipeline JIT:

```rust
load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, mask, Some(E::ZERO), Some(token), Latency::<1>);
```

Use `None::<E>` padding plus an explicit mask/select pattern, or only use `None::<E>` without select when `analysis.json` proves exact cover for every correctness and perf shape.

```rust
let (raw, tok): (Tile<E, { [BLOCK] }>, Token) =
    load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(valid), None::<E>, Some(token), Latency::<1>);
let xf: Tile<f32, { [BLOCK] }> = convert_tile(raw);
let zero: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
let xf_safe: Tile<f32, { [BLOCK] }> = select(valid, xf, zero);
```

## Rule 15: Avoid Tuple Destructuring Before Reductions

If a loaded tile feeds a reduction, annotate the load return and then bind conversion/reduction separately. This keeps diagnostics local.

## Rule 16: Data Dtype Is Generic

Use `<E: ElementType>` for data dtype unless the reference is truly f32-only. Hardcode f32 for accumulators and secondary accumulator generic slots where the reference does.

## Rule 17: One Source Of Truth

`kernel.rs` is included by both the cdylib crate and pipeline test. Do not copy kernels into the test. Generated IR must come from the same entry functions that FFI launches.

## Rule 18: Runtime Scalar Broadcasts

Use `broadcast_scalar(value, const_shape![...])` for runtime scalar tiles. Do not route runtime values through `constant(...)`.

## Rule 19: f32 Sigmoid Reciprocal

For sigmoid-style f32 expressions, use `true_div` if the reference uses true division.

## Rule 20: Latency And TMA Hints

Mirror `Latency<N>`, TMA enablement, occupancy, and `num_cta_in_cga` from `analysis.json` and reference IR. Do not add hints because a different kernel used them.

## Rule 21: Assume Hints

Apply `assume_div_by::<_, 16>` to raw data pointers when alignment is guaranteed. Avoid scalar `assume_div_by` on runtime dims/strides unless the wrapper proves divisibility for every selected shape. False scalar assumptions corrupt masks and boundary handling.

## Rule 22: Mutable Tile State Across `if`

Both branches assign loop-carried tile state:

```rust
if cond {
    acc = new_acc;
} else {
    acc = acc;
}
```

## Rule 23: Python Constants Become Const Generics

`ct.Constant[int]` tile sizes and baked shape constants normally become const generics. Runtime args are for true runtime values, not autotune constants folded into the reference IR.

## Rule 24: Nested Mutation Carry Behavior

If source IR yields a value through a nested loop/if, keep it as a mutable binding and update it in every path.

## Rule 25: Avoid Tile-Valued Expression `if`

Use statement-style mutation for tile-valued control flow. Do not rely on Rust expression `if` returning a `Tile`.

## Rule 26: Math APIs With Modes

Many math ops take explicit static modes:

```rust
let z = addf(a, b, rounding::NearestEven, ftz::Enabled);
let m = maxf(a, b, nan::Disabled, ftz::Enabled);
let p = exp2(x, ftz::Enabled);
let q = fma(a, b, acc, rounding::NearestEven, ftz::Enabled);
```

Use named functions when the reference records non-default rounding, FTZ, or NaN behavior.

## Rule 27: GEMM Transpose Ownership

A GEMM-family transpose should be native: const generic transpose flags, physical load from original tensor, and `permute` after load when the reference does that. Do not implement timed Python `.t().contiguous()` unless the reference wrapper does it outside the measured region.

## Rule 28: Padded Loads Keep TMA

Use `tensor.partition(...)` for padded view loads and keep TMA enabled when the reference uses view/TMA ops. Do not fall back to scalar pointer loads for ragged edge tiles unless the reference is pointer-scatter.

## Rule 29: Persistent TMA Latency Is Case-Sensitive

Persistent kernels that autotune many configs often need `None` latency inside the persistent body unless `analysis.json` proves the reference requires a hint. Direct-launch (non-persistent) kernels may need recorded hints. Check the IR.

## Rule 30: Batched TMA Descriptors

Batched TMA descriptors require aligned outer strides. If an outer stride is not aligned, fix wrapper padding/fallback. Do not assert false scalar divisibility.

## Rule 31: Accumulator Generic Slots

If a kernel has a separate accumulator dtype generic such as `E2`, FFI usually passes `"f32"` for that slot unless the reference says otherwise.

## Rule 32: Rust 2024 Exports

```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_{kernel_name}(
    out: *const TensorDesc, /* ... more TensorDesc* + scalar args ... */
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    if out.is_null() /* || ... */ { return -5; }  // null descriptor
    // dtype: match dtype_str(desc.dtype) { Some(s) => s, None => return -2 }
    let device = Device::new(device_id.max(0) as usize)?;  // multi-GPU; was Device::new(0)
    // ...
}
```

Legacy `#[no_mangle]` fails under Rust 2024. Return `-5` on a null descriptor,
`-2` on an unknown dtype code, and derive the device from the FFI `device_id`
arg (`Device::new(device_id.max(0) as usize)`) rather than a hardcoded
`Device::new(0)`, so multi-GPU launches target the tensors' device.

## Rule 33: Aggregated Cargo Dependencies (crates.io, pinned 0.2.0)

There is ONE aggregated cdylib crate, `ops/cutile_rs/cutile_kernels/`, that
`include!`s every op's `kernel.rs` + `ffi.rs` (see `output-structure.md`). Its
`Cargo.toml` uses PINNED crates.io dependencies — no path deps, no
`CUTILE_RS_ROOT`:

```toml
cutile          = "=0.2.0"
cutile-compiler = "=0.2.0"
cutile-macro    = "=0.2.0"
cuda-core       = "=0.2.0"
cuda-async      = "=0.2.0"
```

The proc macro resolves `cutile-compiler` during expansion. Because everything
is a registry version pin, there is nothing to sed-replace and no per-op
throwaway crate. Build once with `cargo build --release` to produce
`libcutile_kernels.so`.

## Rule 34: Transpose Has One Owner

If FFI swaps logical shape/strides, the kernel uses identity partitioning. If the kernel uses `partition_permuted`, FFI passes original physical shape/strides. Do not do both.

## Rule 35: ABI Pieces Must Agree

The Python wrapper's cffi `_FFI_CDEF` (with `const TensorDesc*` params), the FFI
export signature, the Rust generic list, test skips, and reports must describe
the same ABI. All ops share ONE `libcutile_kernels.so`; each op's `_FFI_CDEF`
declares only its own `cutile_{kernel_name}` symbol (the shared `TensorDesc`
typedef is prepended by `bind_kernel_function_cffi`). Bind with
`bind_kernel_function_cffi`, pack tensors with `make_tensor_desc`, and check the
return code with `check_rc` — no ctypes `_FFI_ARGTYPES` list.

## Rule 36: Stable Const-Generic Escape Hatch

Do not add nightly flags. If const arithmetic in type positions keeps failing, collapse the expression to a literal entry specialization or pass a runtime scalar.

## Rule 37: Python Wrappers Are Autograd Surfaces

Perf tests may pass `requires_grad=True` tensors while checking forward only. Detach grad-enabled inputs before FFI. Backward tests skip `cutile-rs` until backward kernels exist.

## Rule 38: Fixed-Config Canary

Run a small fixed-config canary before autotune/CUPTI when inputs are easy to construct. It catches FFI ownership aborts and dtype dispatch bugs earlier than Agent D/E.

## Rule 39: B-Only Ownership

In B-only mode, Agent B owns compile, wrapper correctness, generated IR, and tiny canaries. In full-pipeline mode, Agent D owns wrapper correctness.

## Rule 40: Test Registration Scope

Register `"cutile-rs"` only in test classes that dispatch through `tilegym.ops.{op}`. Do not add it to coverage classes that call Triton/Cutile helpers directly or unsupported backward paths.

## Rule 41: Wrapper Fast Paths Avoid Copies

Do not clone/copy/contiguous large tensors unconditionally inside timed wrapper paths. CUPTI times that work. If layout conversion is required, gate it by contiguity/stride and report the cost.

## Rule 42: Perf Cliff Triage

After correctness passes, inspect in this order:

1. FFI launch count and wrapper allocations/copies.
2. `analysis.json` config fidelity: tile sizes, occupancy, `num_cta_in_cga`, latency, variant dispatch.
3. Generated IR op surface: TMA vs pointer ops, `mmaf` dtype, reductions.
4. Kernel math rewrites.

For GEMM fp32, check tf32 before `mmaf` first.

## Rule 43: Outputs Are Never `&mut Tensor`

A `&mut Tensor` output can lock explicit launch grid to inferred partition grid and reject persistent or swizzled grids with rc=-3. Declare outputs as read-only `&Tensor<E, {[-1, ...]}>` and write in-body via `partition_full_mut`, or use raw `*mut E` and pointer/tensor-view writes. Host passes the output as just another `*const TensorDesc` and borrows it with `borrow_tensor` (ManuallyDrop, never freed) or `DevicePointer::from_cu_deviceptr`.

See `references/no-mut-tensor-output.md`.

## Rule 44: `PartitionMut` Import

`partition_full_mut` comes from:

```rust
use cutile::prelude::PartitionMut;
```

Import it before rewriting working partition logic.

## Rule 45: Pointer Tiled Loops Use Ceil-Div And Tail Masks

If source analysis contains `ct.cdiv`, `ceil_div`, or `check_bounds = (num_tiles*TILE_SIZE != N)`, do not implement loop count as floor division.

Bad:

```rust
let num_tiles: i32 = N / TILE_SIZE;
for j in 0i32..num_tiles { ... }
```

Good for positive const generics:

```rust
let num_tiles: i32 = (N + TILE_SIZE - 1) / TILE_SIZE;
for j in 0i32..num_tiles {
    let tile_off_s: Tile<i32, { [] }> = scalar_to_tile(j * TILE_SIZE);
    let tile_off: Tile<i32, { [TILE_SIZE] }> =
        tile_off_s.reshape(const_shape![1]).broadcast(const_shape![TILE_SIZE]);
    let col: Tile<i32, { [TILE_SIZE] }> = tile_off + cols;

    let n_s: Tile<i32, { [] }> = scalar_to_tile(N);
    let n_b: Tile<i32, { [TILE_SIZE] }> =
        n_s.reshape(const_shape![1]).broadcast(const_shape![TILE_SIZE]);
    let valid: Tile<bool, { [TILE_SIZE] }> = cmpi(col, n_b, predicate::LessThan);

    let col64: Tile<i64, { [TILE_SIZE] }> = exti(col);
    let ptrs: PointerTile<*mut E, { [TILE_SIZE] }> = base.offset_tile(row_base + col64);

    let (raw, tok): (Tile<E, { [TILE_SIZE] }>, Token) =
        load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(valid), None::<E>, Some(token), Latency::<1>);
    let _ = tok;

    let xf: Tile<f32, { [TILE_SIZE] }> = convert_tile(raw);
    let z: Tile<f32, { [TILE_SIZE] }> = constant(0.0f32, const_shape![TILE_SIZE]);
    let xf_safe: Tile<f32, { [TILE_SIZE] }> = select(valid, xf, z);
    acc = fma(xf_safe, xf_safe, acc, rounding::NearestEven, ftz::Enabled);
}
```

Use the same `valid` mask for apply-loop loads and stores when columns may exceed the real `N`. A representative IR dump at an exact-cover shape does not prove floor division is safe for correctness shapes.

## Rule 46: Const Generic Numeric Conversion

Do not write `scalar_to_tile(N as f32)` inside a kernel. The macro may lower the cast as an unsupported runtime cast. Also avoid `itof` for int-to-float until the writer's `signedness` attribute path is proven for that op.

Use `convert_tile`:

```rust
let n_i32: Tile<i32, { [] }> = scalar_to_tile(N);
let n_f32: Tile<f32, { [] }> = convert_tile(n_i32);
```

The local API pins are:

- `scalar_to_tile<E: ElementType>(scalar: impl Scalar) -> Tile<E, { [] }>`
- `convert_tile<TO, FROM, S>(x: Tile<FROM, S>) -> Tile<TO, S>`
- `itof(x, rounding)` exists but can hit `missing attribute 'signedness'` in bytecode writing.

## Rule 47: Annotate `load_ptr_tko`

When mask is `None`, padding is `None`, or shape is generic, annotate the tuple:

```rust
let (xv, tok): (Tile<E, { [TILE_SIZE] }>, Token) = load_ptr_tko(
    ptrs,
    ordering::Weak,
    None::<scope::TileBlock>,
    Some(valid),
    None::<E>,
    Some(token),
    Latency::<1>,
);
```

The source signature is:

```rust
load_ptr_tko(source, ordering, scope, mask, padding_value, token, Latency::<CYCLES>)
    -> (Tile<E, S>, Token)
```

`store_ptr_tko` takes `(destination, value, ordering, scope, mask, token, latency)` and returns `Token`.

## Rule 48: Preserve Source Semantics Beyond The Dumped Shape

Agent A often dumps one representative shape per structural variant. If `analysis.json.ops_used` or variant text says `ct.cdiv`, `next_power_of_2`, `check_bounds`, `padding_mode=ZERO`, or dtype-specific lowering, preserve that source behavior even when the chosen IR shape folds to a simpler constant.

Example: a reference dumped at a shape where the tile count divides evenly (floor == ceil) hides the tail; correctness then fails on a shape where floor division drops the partial tile. Source semantics win.

## Rule 49: Build Log Is Part Of The Fix

Before returning `VERDICT: COMPILED`, write `reports/build_log.md` with:

- each cargo build attempt and top rustc error
- each pipeline/JIT compile iteration
- offending snippet
- fix applied
- whether the fix is now covered by this rulebook or a known gap

Then run `validate_agent_b.sh` and include its output in `<VALIDATOR_OUTPUT>`.


---

# Part 2 — Troubleshooting Index & Capability Gaps

This part was merged in from the former separate known-gaps troubleshooting index.
Match the exact compiler, JIT, Python wrapper,
pytest, or scorer signature before changing kernel math. In B-only eval mode, absent
Agent C/D/E/F reports are expected. Entries for APIs that cutile-rs 0.2.0 added
(`get_index_space_shape`, `divi rounding<positive_inf>`, unsigned `cmpi`) and the
never-existent `make_strided_view` were removed; the compiler-bug table carries a 0.2.0
status column.

## Forward-only port (skill convention)

cutile-rs is **forward-only** in this skill version. Convention, not compiler limit:

- `kernel.rs` / `ffi.rs` emit ONLY forward; never write a backward kernel.
- `wrapper.py` detaches grad-enabled inputs (`if x.requires_grad: x = x.detach()`) and returns a leaf tensor (no `grad_fn`). Do NOT define a `torch.autograd.Function`.
- `test_{kernel_name}.py` patches MUST skip backward parametrizations for cutile-rs:
  ```python
  if backend == "cutile-rs":
      pytest.skip("cutile-rs is forward-only (this skill version)")
  ```
  Apply to any `test_op_backward[...]` and any `test_perf[...]` that times backward / combined fwd+bwd. Forward `test_op[cutile_rs-...]` and forward-only `test_perf[cutile_rs-...]` remain enabled.

**Why**: a PyTorch-formula backward inside `autograd.Function` adds large wrapper-layer slowdown to combined fwd+bwd ratios, swamping Agent C's forward-kernel perf signal. This skill version ports forward only.

**`detach()` cost**: ~1-5us per call, no CUDA work, no allocation. Apply unconditionally when `requires_grad=True`.

**In-place reference ops** (those that mutate an input in place): after detach, do `x.detach().clone()` if the FFI mutates in-place — autograd-detached tensors share storage with the original.

## First-Build Blockers

| Failure signature | Bad pattern | Fix |
|-------------------|-------------|-----|
| `error[E0433]: cannot find module or crate cutile_compiler` under `#[cutile::module]` | `cutile_kernels` crate lacks the `cutile-compiler` package dependency | Add `cutile-compiler = "=0.2.0"` to `cutile_kernels/Cargo.toml` (Rule 33) |
| `error[E0432]` / `E0433` for `DevicePointer`, `CompileOptions`, or `DType` | Stale imports such as `cutile::DevicePointer` or `cutile::tensor::DType` | Use the import block in `agents/agent_b.md` |
| `built-in f16 is unstable`; `bf16 is not in scope` | Used Rust builtin or missing half imports | Import `cutile::core::{f16, bf16}` if concrete half names are needed |
| `call to unsafe function make_tensor_view is unsafe` | Unsafe helper called without a block | Wrap `unsafe { make_tensor_view(...) }` |
| `Pointers can only be used in unsafe kernel entry points` | Raw pointer entry declared safe | Use `#[cutile::entry()] unsafe fn` |
| `kernel_entry_generator.rs:70` / stride unwrap | `&Tensor` pipeline test omitted full-rank strides | Pass `.strides(...)` for every tensor rank, including trailing `1` |
| Validator OK but pytest all skipped | Backend availability or shared-lib loader missing | Wire `selector.py`, `ops/cutile_rs/__init__.py`, and `bind_kernel_function_cffi` |

## Failure Signatures

| Symptom | Smallest trigger | Fix |
|---------|------------------|-----|
| `error: generic parameters may not be used in const operations` at `#[cutile::module]`, followed by `help: add #![feature(generic_const_exprs)]`; adding the feature gives `E0554` on stable | Pointer-scatter variant with shape-driving const generic such as `TILE_SIZE` inside `Tile<..., {[TILE_SIZE]}>`, loops, masks, and extra derived const generics in the same generated module | Do not switch to nightly. Use literal entry specialization or a conservative literal block-size entry and keep `N` runtime. Dispatch in FFI. See coding-rules Rule 36. (Confirmed still present in 0.2.0: crate is stable Rust 1.89, no `generic_const_exprs`.) |
| pytest says `libcutile_kernels.so not found` after Agent B reported work | Cargo build of the aggregated `cutile_kernels` crate failed but wrapper/test registration was still written | Treat missing `.so` as build failure, not Python failure. Fix cargo first; do not spend time changing wrapper logic. |
| `NotImplementedError: cutile-rs {op}: backward not implemented` | Wrapper rejects `input.requires_grad`, while tests create `requires_grad=True` inputs or pass `gradient=...` | Do NOT reject `requires_grad`. Detach grad-enabled inputs (`if x.requires_grad: x = x.detach()`) and return a leaf tensor. Backward is out of scope; the backward test must skip cutile-rs. |
| Forward values close but gradient mismatch is huge | A backward-correctness test reached cutile-rs, which is forward-only | Skip the cutile-rs backend for backward parametrizations (`pytest.skip("cutile-rs is forward-only (this skill version)")`). Do not add a `torch.autograd.Function`. |
| Fatal Python error: Aborted in `tilegym/ops/cutile_rs/{kernel_name}.py:_run_ffi` during autotune | Bad persistent TMA config reaches CUPTI; common cause is `Some(LATENCY)` in a persistent grid-stride TMA loop | Run fixed-config canary before autotune. In persistent TMA loops, pass `None` latency for every view load/store unless analysis/reference IR prove a direct-launch exception. |
| Correctness: 0% matched, output near zero | FFI builds an input with swapped logical shape/strides AND the kernel also applies `partition_permuted` → double transpose | Pick one transpose owner. If the kernel permutes, pass the original physical shape/strides; if the FFI swaps the layout, the kernel uses identity `partition`. |
| Row-wise correctness mismatch begins at row `get_num_sm()*occupancy` and perf ratio is implausibly fast | Wrapper capped `grid_size = min(n_rows, num_sms * occupancy)` but kernel handles only `get_tile_block_id().0` once | Either launch `grid=n_rows` or add an explicit persistent loop `for row in (bid_x..n_rows).step_by(grid_x as usize)`. Occupancy is a compile/runtime hint, not a substitute for logical grid coverage. |
| Perf geomean is very fast, but correctness failed | Host scorer can still run perf after a wrong partial-output kernel | Ignore the speed until correctness passes. Look for unwritten rows/tiles, output sentinels, wrapper grid caps, and skipped boundary masks. |
| Fatal Python error: Aborted in `tilegym/ops/cutile_rs/{kernel_name}.py:_run_ffi` only on the largest resource (memory/batch) perf case after smaller cases pass | Resource-sensitive test case reached cutile-rs even though only unsupported dtype/transpose skips were mirrored | Mirror the reference backend's resource/OOM skips for cutile-rs when the same allocation and launch shape are used, or add a smaller supported cutile-rs perf surface. Do not let an intentionally unsupported giant case enter FFI. |
| Correctness command selects `test_perf` cases and aborts before coverage tests finish | `python -m pytest tests/ops/test_{kernel_name}.py -k cutile_rs -x` selects every cutile-rs parametrized test, not only `test_op` | Before adding `cutile-rs`, inspect all selected classes and add skips before dispatcher/FFI calls for unsupported resource, dtype, transpose, or backend-specific coverage cases. |
| Perf geomean present but correctness failed | Host scorer runs perf even if correctness failed | Fix correctness first. Do not optimize a failing wrapper/kernel unless the perf log identifies an independent abort. |
| `AttributeError` / undefined `cutile_{kernel_name}` symbol at load | `_KERNEL`, `_FFI_NAME`, or the exported Rust symbol disagree | Make all three consistently `cutile_{kernel_name}` / `{kernel_name}`. |
| FFI returns an error code but Python keeps running | Wrapper omits `check_rc(rc, _FFI_NAME)` after the call | Always `check_rc(rc, _FFI_NAME)` immediately after every FFI call. |
| Process aborts from a Rust panic across `extern "C"` | FFI panicked before returning `i32`; Python cannot catch it | Guard shapes/dtypes/resource cases in Python before FFI; return `-N` for expected errors instead of asserting/panicking. |

## Compiler And JIT Bugs Currently Open

Verified against cutile-rs **0.2.0** source (built + grepped from source).
Most error paths still exist in the 0.2.0 compiler (status column cites the file). Two
look RELAXED in 0.2.0 (re-test before relying on the workaround). `generic parameters may
not be used in const operations` is a rustc-stable error, not a cutile check — it persists
because the crate is stable Rust 1.89 with no `generic_const_exprs`.

| Failure signature | Smallest trigger | Workaround | 0.2.0 status |
|-------------------|------------------|------------|--------------|
| `binary Div requires operands of the same type` with `[M]` vs `[M,1]` | Annotating `reduce_sum(x, 1)` as `[M,1]` | Bind actual `[M]`, then reshape to `[M,1]` | STILL PRESENT (`compile_binary_op.rs`) |
| `return type is missing a compiled tile type` | Chained `pointer_to_tile(...).reshape(...).broadcast(...)` | Split into explicit typed bindings | STILL PRESENT (`compile_cuda_tile_op.rs`) |
| `Unexpected optimization hint key` | `simt_num_warps_in_cta` hint | Remove it; tileiras chooses warps | STILL PRESENT (`hints.rs`; whitelist = `num_cta_in_cga`/`occupancy`/`max_divisibility`) |
| JIT `Unexpected dense value` | Runtime or qualified value in `constant(...)` | Use `broadcast_scalar` for runtime values; use decimal literal for constants | STILL PRESENT (`compile_cuda_tile_op.rs`) |
| Unary expression not supported | `if !EVEN_N` or unary const bool in macro body | Invert branch: `if EVEN_N { x=x; } else { ... }` | STILL PRESENT (`compile_expression.rs`/`generics.rs`) |
| `qualified paths not supported` | `Some(f32::NEG_INFINITY)` or `constant(std::f32::consts::LOG2_E, ...)` | Use decimal literals and explicit pad/select tiles | LIKELY RELAXED — hard error string is gone; 0.2.0 added `dense_const_path_parts`/`dense_module_const_path_value` (`compile_cuda_tile_op.rs`). Re-test qualified consts before assuming failure. |
| Value assigned in `if` not visible after block | Mutable tile only assigned in one branch | Add `else { x = x; }` or split branches | MAYBE RELAXED — `compile_expression.rs` now has if-branch `carry_vars` handling. Re-test. |
| Type inference failure after tuple destruct in loop | `let (tile, tok) = load_ptr_tko(...)` directly feeding reduce in loop | Predefine typed lets or bind explicitly | UNVERIFIED (no single error string; needs a live repro) |
| Store partition view lacks `padding_value = zero` | Mutable partition stores | Accept; ensure load-side partition padding and store masks/bounds are correct | UNVERIFIED (`load_ptr_tko` takes `padding_value: Option<E>`; associated-const lowering not retested) |

## Missing APIs / Design Gaps

(0.2.0 added `get_index_space_shape`, `divi rounding<positive_inf>`, and unsigned `cmpi`; `make_strided_view` never existed and `make_tensor_view(base, shape, strides, token)` covers strided views — all removed from this table.)

| Missing or limited area | Workaround |
|-------------------------|------------|
| Explicit Triton `num_stages` knob | Use per-op latency only when reference IR has it and Rule 29 does not forbid it |

`&mut Tensor` kernel inputs/outputs: still grid-coupling in 0.2.0 (accepted only via a `Partition` whose shape pins the launch grid). The rule + workaround are Rule 43/44 (Part 1 above) and `references/no-mut-tensor-output.md`.

## Performance Gaps

(`assume_div_by` pointers-only and "occupancy is not a grid cap" rules live once as Rule 21 / Rule 20 in Part 1 above; the occupancy failure SIGNATURE stays in the Failure Signatures table above.)

| Gap | Impact | Workaround |
|-----|--------|------------|
| TMA disabled when reference uses TMA | Often 1.5x-10x slower | Keep `tma::Enabled`; fix layout/strides instead |
| Scalar `ct.Constant[int]` left runtime | Can be several times slower | Use const generics unless Rule 36 compile failure occurs |
| f16 MMA accumulator | Slower and less accurate | Accumulate in f32 |
| `/` for f32 sigmoid | Slow division | Use `true_div` |
| Extra `clone`, `zeros`, `ones`, or `contiguous` inside autotune lambda | CUPTI times extra kernels/copies | Allocate outputs with `torch.empty` only inside `kernel_fn` |
| Autotune tries invalid configs first | Abort or bad cache | Fixed-config canary and config filtering before `autotune_launch` |
| Large-shape resource cases inherited blindly from test_perf | Abort or timeout after many passing cases | Mirror reference skip rules when cutile-rs has the same allocation footprint, and document any intentionally unsupported perf cases in `reports/agent_b.md`. |

## Process Pitfalls

| Pitfall | Prevention |
|---------|------------|
| Chasing missing Agent C/D/E/F reports in B-only | Ignore; only Agent B runs |
| Reporting `COMPILED` after cargo failed | Check `.so`, symbol, pipeline compile, and tiny wrapper canary |
| Reading perf before correctness | Correctness first; perf only after wrapper/kernel semantics pass |
| Treating wrapper errors as compiler gaps | If traceback is in `tilegym/ops/cutile_rs/*.py`, fix wrapper |
| Treating near-zero or partially zero output as math first | Check grid coverage, host tensor shape/stride, and transpose ownership first |
| Changing kernel math to fix autograd | Autograd is Python wrapper behavior unless backward kernel is required |
| Adding cutile-rs to every backend list by rote | Register only dispatcher paths or skip backend-specific/resource-excluded cases before FFI |
| Running full pytest matrix after each edit | Use a tiny canary, then `python -m pytest tests/ops/test_{kernel_name}.py -k cutile_rs -x --timeout=60` |

## Minimal Triage Order

1. Cargo build and `.so` symbol.
2. Pipeline compile and generated MLIR.
3. Entry pattern validator.
4. Python import/selector/backend registration.
5. Fixed-config wrapper canary on a small shape.
6. Grid/resource canary: for row-wise kernels, use a shape with more rows than `get_num_sm()*occupancy`; for resource-heavy kernels, run the largest registered case once outside CUPTI or skip it before FFI if it is intentionally unsupported.
7. Correctness pytest with `-k cutile_rs -x --timeout=60`.
8. Perf pytest only after correctness.

Record exact commands, pass/fail result, intentional pytest skips, and any grid/resource decisions in `reports/build_log.md` and `reports/agent_b.md`.

## Verification Provenance

Checked against the local cutile-rs tree:

- `cutile/src/_core.rs`: `scalar_to_tile`, `convert_tile`, `itof`, `exti`, `select`, `fma`, `cmpi`, `andi`, `load_ptr_tko`, `store_ptr_tko`, `load_view_tko`, `Latency`.
- `cutile-compiler/src/compile_api.rs`: `KernelCompiler::generics`, `strides`, `spec_args`, `grid`, `options`.
- `cutile-compiler/src/specialization.rs`: `SpecializationBits`, `DivHint`.
