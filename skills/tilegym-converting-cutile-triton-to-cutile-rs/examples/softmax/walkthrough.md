# Worked example — softmax (your first kernel)

This directory is the **complete, copy-pasteable starting point** for adding a
brand-new kernel to a fresh tilegym checkout (the cutile-rs Rust now lives inside
tilegym under `src/tilegym/ops/cutile_rs/` — there is no separate cutile-rs
checkout).  It exists so that a user with zero prior conversions can produce a
working kernel without having to reverse-engineer the conventions out of a larger
kernel.

The example is a row-wise softmax along the last dimension — small enough that
every load-bearing line is visible, large enough to exercise:

| Stack layer            | Demonstrated by                                    |
|------------------------|----------------------------------------------------|
| dtype generic          | `<E: ElementType>` in `kernel.rs` (Rule 16)        |
| const generics         | `<const BM, const BN, const LATENCY>`              |
| Boundary-safe partition | `tview.partition(const_shape![BM, BN])` (Rule 28 v2) |
| TMA + latency hints    | `tma::Enabled` + `Some(LATENCY)` on every op       |
| Tile-form reductions   | `reduce_max` / `reduce_sum`                        |
| `flush_to_zero` etc.   | `mulf(..., ftz::Enabled)`, `true_div`              |
| FFI dtype dispatch     | `match dtype_str(x_d.dtype) { Some("f32") => ... }` |
| TensorDesc boundary    | `*const TensorDesc` in `ffi.rs`; `make_tensor_desc` in wrapper |
| cffi wrapper           | `wrapper.py::_FFI_CDEF` mirrors `ffi.rs`            |
| CUPTI autotune         | `autotune_launch(kernel_fn=lambda cfg: ...)`       |
| pipeline rule 11 NO_FALLBACK | `raise NotImplementedError(...)` for bad combos    |
| tilegym test surface   | `Test_Softmax::test_op` + `test_perf`              |

## File map

| File | Install to | Role |
|------|-----------|------|
| `kernel.rs` | `${CUTILE_KERNEL_OUT_ROOT}/softmax/kernel.rs` | The cutile-rs kernel (single source of truth) |
| `ffi.rs` | `${CUTILE_KERNEL_OUT_ROOT}/softmax/ffi.rs` | C-ABI export (`*const TensorDesc` dtype dispatch + launch) |
| `softmax_pipeline.rs` | `${CUTILE_KERNEL_OUT_ROOT}/softmax/softmax_pipeline.rs` | `cargo test` compile + IR dump |
| `wrapper.py` | `{TILEGYM_PATH}/src/tilegym/ops/cutile_rs/softmax.py` | tilegym cffi wrapper (autotune + dispatch) |

The kernel Rust is wired into the aggregated `cutile_kernels` crate under
`{TILEGYM_PATH}/src/tilegym/ops/cutile_rs/cutile_kernels/` (one
`libcutile_kernels.so` for every op). There is no separate cutile-rs checkout.

Plus three small **registration snippets** (paste-into existing files):
see *Wiring* below.

## Bootstrap (zero → first kernel)

### 0. Prerequisites

* A cloned `tilegym` checkout — it now holds the cutile-rs Rust
  under `src/tilegym/ops/cutile_rs/`. No separate cutile-rs checkout is needed;
  the `cutile_kernels` crate pulls its deps from crates.io (pinned `=0.2.0`).
* CUDA toolkit (`CUDA_TOOLKIT_PATH`, default `/usr/local/cuda`) + Rust toolchain
  (`cargo`). `tileiras` lowers IR → cubin at launch.
* Python with PyTorch installed.

### 1. The load-bearing Python bits ship in-repo

The cutile-rs backend runtime is already present in the repository — no copy step:

```
${TILEGYM_PATH}/src/tilegym/backend/cutile_rs/utils.py      # bind_kernel_function_cffi, make_tensor_desc, check_rc, get_num_sm
${TILEGYM_PATH}/src/tilegym/backend/cutile_rs/autotuner.py  # autotune_launch
${TILEGYM_PATH}/src/tilegym/backend/cutile_rs/__init__.py
${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/__init__.py
```

Wrappers `import from tilegym.backend.cutile_rs.*`. You only add the per-op
wrapper under `src/tilegym/ops/cutile_rs/` and register it.

### 2. Install the example kernel files

```bash
# The per-kernel working dir holds Agent A–D artifacts (kernel.rs, ffi.rs,
# reference/, generated/, reports/). The eval harness depends on this layout.
export CUTILE_KERNEL_OUT_ROOT=/abs/path/to/cutile_kernel_out

# Rust kernel + FFI + pipeline test (per-kernel working dir)
mkdir -p ${CUTILE_KERNEL_OUT_ROOT}/softmax/{generated,reports/agent_logs}
cp ${SKILL}/examples/softmax/kernel.rs            ${CUTILE_KERNEL_OUT_ROOT}/softmax/kernel.rs
cp ${SKILL}/examples/softmax/ffi.rs               ${CUTILE_KERNEL_OUT_ROOT}/softmax/ffi.rs
cp ${SKILL}/examples/softmax/softmax_pipeline.rs  ${CUTILE_KERNEL_OUT_ROOT}/softmax/softmax_pipeline.rs

# Mirror the kernel Rust into the aggregated crate's sibling {op}_kernel/ dir so
# the `mod softmax { include!("../../softmax_kernel/kernel.rs"); ... }` line in
# cutile_kernels/src/lib.rs resolves (see step 3a).
mkdir -p ${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/softmax_kernel
cp ${SKILL}/examples/softmax/kernel.rs ${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/softmax_kernel/kernel.rs
cp ${SKILL}/examples/softmax/ffi.rs    ${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/softmax_kernel/ffi.rs

# tilegym Python wrapper
cp ${SKILL}/examples/softmax/wrapper.py            ${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/softmax.py
```

### 3. Wiring (3 small edits in existing files)

#### a. `cutile_kernels/src/lib.rs` — register the op in the aggregated crate

There is ONE aggregated cdylib crate for every op:
`{TILEGYM_PATH}/src/tilegym/ops/cutile_rs/cutile_kernels/`. Register softmax by
adding ONE `mod` that `include!`s the op's device + FFI sources from the sibling
`softmax_kernel/` dir (the shared `TensorDesc` helpers are pulled in once via
`ffi_util`):

```rust
// cutile_kernels/src/lib.rs
#[path = "../../ffi_util.rs"]
mod ffi_util;   // shared TensorDesc / borrow_tensor / dtype_str (once)

mod softmax {
    include!("../../softmax_kernel/kernel.rs");
    include!("../../softmax_kernel/ffi.rs");
}
```

Deps are crates.io PINNED (`cutile = "=0.2.0"`, …; Rule 33) — no path deps and no
cutile-rs checkout. Then build the single shared library:

```bash
# The loader autobuilds this on first use (CUTILE_RS_AUTOBUILD, on by default);
# building by hand is optional.
cd ${TILEGYM_PATH}/src/tilegym/ops/cutile_rs/cutile_kernels
cargo build --release
nm -D target/release/libcutile_kernels.so | grep cutile_softmax
# must print: T cutile_softmax
```

The wrapper.py template loads this single `libcutile_kernels.so` via
``bind_kernel_function_cffi("softmax", _FFI_CDEF)`` (cffi). One aggregated library
serves every op; each op only cdefs its own symbol. There is no per-op `.so` and
no legacy monolithic ``libcutile_ffi.so``.

#### b. `{TILEGYM_PATH}/src/tilegym/ops/cutile_rs/__init__.py` — register import

Append:
```python
from . import softmax  # noqa: F401  (registers @register_impl)
```

#### b'. `{TILEGYM_PATH}/src/tilegym/backend/selector.py` — register backend

The tilegym test harness skips every ``cutile-rs`` test with
``pytest.skip("Backend cutile-rs is not available")`` unless
``selector.is_cutile_rs_available()`` exists AND ``"cutile-rs"`` is a key in
``_check_backends_availability()``'s availability dict. Add both:

```python
def is_cutile_rs_available() -> bool:
    """True if the aggregated cutile_kernels crate is on disk and cargo is callable.

    We deliberately use a *permissive* probe — the single libcutile_kernels.so is
    autobuilt on-demand by the loader (CUTILE_RS_AUTOBUILD), so requiring the .so
    here would skip legitimately-installed environments. The Rust now lives inside
    tilegym, so probe the in-tree cutile_kernels crate (CUTILE_RS_KERNELS_DIR
    overrides its location); there is no CUTILE_RS_DIR / cutile-rs checkout.
    """
    import os, shutil
    kernels_dir = os.environ.get("CUTILE_RS_KERNELS_DIR") or os.path.join(
        os.path.dirname(__file__), "..", "ops", "cutile_rs", "cutile_kernels"
    )
    if not os.path.isfile(os.path.join(kernels_dir, "Cargo.toml")):
        return False
    return shutil.which("cargo") is not None


def _check_backends_availability() -> Dict[str, bool]:
    availability = {
        # ...existing entries (cutile, triton, tilecpp, ...)
        "cutile-rs": is_cutile_rs_available(),     # NEW
    }
    return availability
```

Verify with::

    python -c "from tilegym.backend.selector import is_cutile_rs_available as f; print(f())"
    # → True (the in-tree cutile_kernels crate exists and cargo is callable)

If you skip this step, your wrapper's ``@register_impl`` registration is
still correct, but ``tilegym.set_backend("cutile-rs")`` returns
``"cutile-rs is not available"`` and pytest will silently SKIP every
parametrized case.

#### c. `{TILEGYM_PATH}/src/tilegym/ops/ops.py` — add @dispatch

If softmax is not already a registered tilegym op, add:
```python
@dispatch("softmax", fallback_backend="triton")
def softmax_interface(x: torch.Tensor, dim: int = -1):
    """Row-wise softmax (last-dim by default)."""
    pass
```
(If it already exists with a `cutile` backend, this step is a no-op — your
`@register_impl("softmax", backend="cutile-rs")` plugs into the same
dispatch table.)

### 4. Compile + IR pipeline + correctness sanity

```bash
# Stage 1 — Rust pipeline test dumps bytecode (.tileirbc)
cd ${CUTILE_KERNEL_OUT_ROOT}/softmax
cargo test --release --test softmax_pipeline -- --nocapture
# expect: ✅ softmax kernel compiled. Bytecode written to ...generated.tileirbc

# Stage 2 — bytecode → MLIR text
cuda-tile-translate --cudatilebc-to-mlir \
  ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated.tileirbc \
  -o ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated.mlir

# Stage 3 — canonicalize + CSE for IR diff
${CUDA_TILE_OPT_BIN} --canonicalize --cse \
  ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated.mlir \
  -o ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated_canon.mlir

# tilegym pytest — the loader autobuilds libcutile_kernels.so on first use
# (CUTILE_RS_AUTOBUILD is on by default; tileiras lowers IR → cubin at launch).
cd ${TILEGYM_PATH}
python -m pytest tests/ops/test_softmax.py -k cutile_rs -v
# expect: passing on f32/f16/bf16 across the 3 shapes
```

### 5. (Optional) Drive the full A→B→C→D→E pipeline on a different kernel

Once the softmax example is green, you have proof that the install + wiring
work.  Now you can invoke the skill on a real kernel that lives in tilegym:

```
"Use the tilegym-converting-cutile-triton-to-cutile-rs skill to add a cutile-rs
backend to tests/ops/test_{kernel_name}.py"
```

The orchestrator will spawn Agent A → ... → Agent E.  Each agent's prompt
references `examples/softmax/` for the canonical pattern (see Agent B prompt
section "Pattern reference").

## What to swap when reusing this template for a new kernel

```
Replace globally           "softmax"          → "{kernel_name}"
                           softmax_module     → {kernel_name}_module
                           softmax_kernel     → {kernel_name}_kernel
                           cutile_softmax     → cutile_{kernel_name}

Per-file specifics:
  kernel.rs                Body (lines under "Online softmax body").
                           Const-generics list (BM, BN, LATENCY → your set).
                           Entry signature (pointers + dims your op needs).
                           optimization_hints (occupancy / num_cta_in_cga
                             from Agent A's analysis.json).

  ffi.rs                   FFI extern signature (tensors stay `*const TensorDesc`;
                             mirror the scalar args your new kernel.rs needs).
                           borrow_tensor::<E> vs DevicePointer::from_cu_deviceptr
                             (pick per the entry's param type).
                           generics vec! (mirror the const-generic order).

  wrapper.py               _FFI_CDEF (mirror ffi.rs; tensor args = `const TensorDesc*`).
                           make_tensor_desc packs each tensor (no argtypes list).
                           _configs() (autotune space — copy from Agent A
                             analysis.json.autotune_configs).
                           softmax(...) signature (match @dispatch sig).
```

After your edits, run:
```bash
bash ${SKILL}/scripts/validate_kernel.sh {kernel_name}
```
to verify all expected files are present + correctly wired (dtype generic,
`*const TensorDesc` FFI symbol in the aggregated `libcutile_kernels.so`,
`register_impl`, cffi `_FFI_CDEF`, test backend list, etc.).
