# Native Transpose Support for GEMM-Family Kernels

## Overview

GEMM-family kernels support transpose via const generic + load + permute.
Physical tensor layout is passed as-is; the kernel loads tiles in physical order then permutes in registers.

TMA requires innermost stride=1, so Python-level `.t().contiguous()` is forbidden — it copies memory.
Instead, transpose is a zero-copy operation handled entirely inside the kernel.

## Three-Layer Implementation

### 1. Kernel (kernel.rs)

**Signature**: Add `TRANSPOSE_A`, `TRANSPOSE_B` as const generics (i32: 0 or 1).
Pass physical shape (`a_d0`, `a_d1`) and one stride (`a_stride0`). Innermost stride is always 1.

```rust
unsafe fn kernel<E1: ElementType, E2: ElementType,
    const BM: i32, const BN: i32, const BK: i32,
    const TRANSPOSE_A: i32, const TRANSPOSE_B: i32>(
    a_ptr: *mut E1, a_d0: i32, a_d1: i32, a_stride0: i32,
    b_ptr: *mut E1, b_d0: i32, b_d1: i32, b_stride0: i32,
    c_ptr: *mut E1, c_m: i32, c_n: i32, c_stride0: i32,
    ...
    rt_m: i32, rt_n: i32, rt_k: i32,  // LOGICAL dimensions
    acc_zero: E2,
)
```

**Tensor view**: Always uses physical layout.

```rust
let a_shape = Shape::<{[-1, -1]}> { dims: &[a_d0_a, a_d1_a] };
let a_strides = Array::<{[-1, 1]}> { dims: &[a_str0_a] };  // innermost=1 always
let a_tview = make_tensor_view(pointer_to_tile(a_ptr_a), a_shape, a_strides, token);
```

**MAC loop**: Load in physical order, permute if transposed.

```rust
// Default: A is [M, K], load [BM, BK]
// `tview.partition(shape)` (method form) auto-emits padding_value = zero in IR.
let a_part_nn: Partition<E1, { [BM, BK] }> = a_tview.partition(const_shape![BM, BK]);
let mut a_tile: Tile<E1, {[BM, BK]}> = load_view_tko(
    &a_part_nn, [bid_m, ki],
    ordering::Weak, scope::TileBlock, Some(5), tma::Enabled,
);

if TRANSPOSE_A == 1i32 {
    // A is physically [K, M], load [BK, BM], permute to [BM, BK]
    let a_part_t: Partition<E1, { [BK, BM] }> = a_tview.partition(const_shape![BK, BM]);
    let a_tile_t: Tile<E1, {[BK, BM]}> = load_view_tko(
        &a_part_t, [ki, bid_m],
        ordering::Weak, scope::TileBlock, Some(5), tma::Enabled,
    );
    // CRITICAL: return type annotation is mandatory — tileiras inference fails without it
    let a_perm: Tile<E1, {[BM, BK]}> = permute(a_tile_t, const_array![1, 0]);
    a_tile = a_perm;
}

// B: same pattern — normal [BK, BN], transposed load [BN, BK] → permute → [BK, BN]
```

**3D variant (batched)**: For batched kernels with batch dim Q:

```rust
// Load [1, BK, BM] then permute dims 1,2 → [1, BM, BK], then reshape to 2D
let a_tile_3d_t: Tile<E1, {[1, BK, BM]}> = load_view_tko(
    &a_part_t, [bid_q, ki, bid_m],
    ordering::Weak, scope::TileBlock, None, tma::Enabled,
);
let a_tile_perm: Tile<E1, {[1, BM, BK]}> = permute(a_tile_3d_t, const_array![0, 2, 1]);
a_tile = a_tile_perm.reshape(const_shape![BM, BK]);
```

**Key rules**:
- Partition indices flip: normal `[bid_m, ki]` → transposed `[ki, bid_m]`
- `permute()` return type MUST be explicitly annotated (tileiras limitation)
- `TRANSPOSE_A/B` are const generics → JIT dead-code-eliminates the unused branch

### 2. FFI (ffi.rs)

Tensors cross as `const TensorDesc*` (physical shape/strides/dtype travel inside
the descriptor). `trans_a` / `trans_b` remain plain `i32` FFI args and become
const generics via `.generics()`. Logical dims are derived from the physical
descriptor shapes + the transpose flags.

```rust
use crate::ffi_util::{borrow_tensor, dtype_str, TensorDesc};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutile_xxx(
    c: *const TensorDesc,
    a: *const TensorDesc,          // physical layout carried inside the descriptor
    b: *const TensorDesc,
    bm: i32, bn: i32, bk: i32,
    ...,
    trans_a: i32,  // 0=no, 1=yes
    trans_b: i32,
    device_id: i32,
    raw_stream: u64,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return -5;
    }
    let (a_d, b_d, c_d) = unsafe { (&*a, &*b, &*c) };

    // Compute logical (post-transpose) dimensions from the PHYSICAL shapes.
    //   A physical: trans_a==0 -> [M,K]; trans_a==1 -> [K,M]
    let rt_m = if trans_a != 0 { a_d.dim(1) } else { a_d.dim(0) };
    let rt_n = if trans_b != 0 { b_d.dim(0) } else { b_d.dim(1) };
    let rt_k = if trans_a != 0 { a_d.dim(0) } else { a_d.dim(1) };
}
```

Use a dispatch macro to reduce boilerplate across dtype variants. Unpack each
descriptor with `borrow_tensor` (&Tensor entries) or
`DevicePointer::from_cu_deviceptr(desc.ptr)` (raw-pointer entries) — never
`transmute` a pointer.

```rust
let dty: &'static str = match dtype_str(a_d.dtype) {
    Some(s) => s,
    None => return -2,
};

macro_rules! dispatch_sk {
    ($E:ty) => {{
        let generics = vec![
            dty.to_string(),
            bm.to_string(), bn.to_string(), bk.to_string(),
            trans_a.to_string(), trans_b.to_string(),
        ];
        // &Tensor entries: borrow_tensor over the descriptors (ManuallyDrop, no forget)
        let a_t = unsafe { borrow_tensor::<$E>(a_d) };
        let b_t = unsafe { borrow_tensor::<$E>(b_d) };
        let c_t = unsafe { borrow_tensor::<$E>(c_d) };
        let op = unsafe { kernel(&*a_t, &*b_t, &*c_t, rt_m, rt_n, rt_k) }
            .generics(generics)
            .grid(...);
        // ... sync_on(&stream)
    }};
}
match dty {
    "f32"  => dispatch_sk!(f32),
    "f16"  => dispatch_sk!(f16),
    "bf16" => dispatch_sk!(bf16),
    _ => -2,
}
```

### 3. Python FFI ({kernel_name}.py)

**cffi cdef**: Add `trans_a`, `trans_b` as `int32_t`; tensors are
`const TensorDesc*`.

```python
_FFI_CDEF = """
int32_t cutile_xxx(
    const TensorDesc* c, const TensorDesc* a, const TensorDesc* b,
    int32_t bm, int32_t bn, int32_t bk,
    ...,
    int32_t trans_a, int32_t trans_b,
    int32_t device_id, uint64_t raw_stream);
"""
```

**Call site**: Pass the physical tensor (no Python-level transpose). Pack each
tensor with `make_tensor_desc`; the transpose flags stay plain ints.

```python
def _run_ffi(a, b, bm, bn, bk, trans_a_int, trans_b_int, out_m, out_n):
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = a.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream

    c = torch.empty((out_m, out_n), device=a.device, dtype=a.dtype)
    cd = make_tensor_desc(ffi, c)
    ad = make_tensor_desc(ffi, a)   # physical — no .t()/.contiguous() transpose
    bd = make_tensor_desc(ffi, b)
    rc = lib.cutile_xxx(
        cd, ad, bd,
        int(bm), int(bn), int(bk),
        ...,
        trans_a_int, trans_b_int,
        int(device_id), int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return c
```

**Public API**: Compute logical dims for output allocation and autotune key.

```python
def {kernel_name}(A, B, *, trans_a=False, trans_b=False, **kwargs):
    A = A.contiguous()
    B = B.contiguous()

    M = A.shape[1] if trans_a else A.shape[0]
    K_a = A.shape[0] if trans_a else A.shape[1]
    K_b = B.shape[1] if trans_b else B.shape[0]
    N = B.shape[0] if trans_b else B.shape[1]

    result = autotune_launch(
        kernel_fn=lambda cfg: _run_ffi(A, B, cfg.BM, cfg.BN, cfg.BK,
                                        int(trans_a), int(trans_b), M, N),
        configs=configs(M, N, K_a),
        key=(M, N, K_a, A.dtype, "{kernel_name}", trans_a, trans_b),
        kernel_name="{kernel_name}",
    )
```

## Test Integration

Remove the transpose skip in test_op:

```python
# BEFORE
if framework == "cutile-rs":
    if trans_a or trans_b:
        pytest.skip("cutile-rs {kernel_name} does not support transpose")

# AFTER — remove the skip entirely
```

Add transpose variants to test_perf:

```python
@pytest.mark.parametrize("trans_a", [False, True])  # was [False]
@pytest.mark.parametrize("trans_b", [False, True])  # was [False]
```

## Performance

- `TRANSPOSE=0`: Zero overhead — dead code eliminated by JIT
- `TRANSPOSE=1`: permute compiles to shared memory shuffle, no global memory roundtrip

All 4 combos (NN/NT/TN/TT) compile to a shared-memory shuffle.

## Existing Examples

- **Batched** (`examples/bmm/kernel.rs`): 3D transpose with `permute(const_array![0, 2, 1])`
- **stream_k** (`cutile-kernels/stream_k/kernel.rs`): 2D transpose with `permute(const_array![1, 0])`

## Common Pitfalls

1. **Missing return type on permute**: `permute(tile, const_array![1,0])` without explicit `Tile<E1, {[BM, BK]}>` annotation → "Failed to infer all generic parameters for permute"
2. **Passing stride1**: Don't pass a second stride. TMA requires innermost=1. Use `Array<{[-1, 1]}>` always.
3. **Python-level .t().contiguous()**: Forbidden — copies memory. The whole point is zero-copy transpose.
4. **Forgetting to update autotune key**: Include `trans_a, trans_b` in the key — different transpose combos may have different optimal tile configs.
