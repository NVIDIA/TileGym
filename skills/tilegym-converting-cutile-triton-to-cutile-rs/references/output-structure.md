# Output Structure per Kernel

All outputs now live in the tilegym repo under `src/tilegym/ops/cutile_rs/`
(kernel Rust + Python wrapper) and `src/tilegym/backend/cutile_rs/` (shared
runtime). There is no separate cutile-rs checkout and no `CUTILE_RS_ROOT`.

## Kernel Rust: `src/tilegym/ops/cutile_rs/{kernel_name}_kernel/`

```text
src/tilegym/ops/cutile_rs/{kernel_name}_kernel/
├── kernel.rs                  # Agent B: SINGLE SOURCE OF TRUTH for kernel code
│                              #   Contains #[cutile::module] only. No test/FFI code.
│                              #   include!()d by the cutile_kernels crate.
├── ffi.rs                     # Agent D: C-ABI host launcher (one cutile_{op} symbol)
│                              #   Tensors cross as *const TensorDesc; borrow_tensor /
│                              #   DevicePointer::from_cu_deviceptr unpack them.
├── reference/
│   ├── reference.mlir         # Agent A: reference IR from winning backend (single-variant only)
│   ├── reference_{variant}.mlir  # Agent A: per-variant IR (multi-variant only, no reference.mlir)
│   └── analysis.json          # Agent A: structured analysis + test_perf_configs + reference_backend
├── generated/
│   ├── generated.mlir         # Agent B: cutile-rs compiled IR (single-variant only)
│   ├── generated_{variant}.mlir  # Agent B: per-variant generated IR (multi-variant only)
│   └── diff_report.md         # Agent C: structured IR diff analysis
└── reports/
    ├── correctness.md         # Agent D: tilegym pytest test_op results
    ├── performance.md         # Agent E: tilegym pytest test_perf --print-record results
    └── agent_logs/
        ├── agent_a.md
        ├── agent_b.md         # Includes compile attempts + tilegym wrapper design
        ├── agent_c.md
        ├── agent_d.md         # pytest output
        └── agent_e.md         # pytest --print-record output
```

## Python wrapper + backend runtime: `src/tilegym/ops/cutile_rs/`

```text
{tilegym_path}/src/tilegym/ops/cutile_rs/
├── __init__.py                # Auto-import all kernel wrapper modules
├── softmax.py                 # Agent D: @register_impl("softmax", backend="cutile-rs")
├── bmm.py                     # Agent D: @register_impl("bmm", backend="cutile-rs")
├── {kernel_name}.py           # Agent D: @register_impl("{op}", backend="cutile-rs")
├── softmax_kernel/            # Agent B: kernel.rs + ffi.rs (see above)
├── bmm_kernel/                # Agent B: kernel.rs + ffi.rs
├── {kernel_name}_kernel/      # Agent B: kernel.rs + ffi.rs
└── cutile_kernels/            # aggregated cdylib crate (builds libcutile_kernels.so)
    ├── Cargo.toml             #   crates.io deps pinned =0.2.0 (Rule 33)
    └── src/
        ├── lib.rs             #   one `mod <op> { include!(...) }` per op (see below)
        └── ffi_util.rs        #   shared TensorDesc, borrow_tensor, dtype_str
```

The shared Python runtime lives in `src/tilegym/backend/cutile_rs/`
(`utils.py`: `bind_kernel_function_cffi`, `make_tensor_desc`, `check_rc`,
`get_num_sm`; `autotuner.py`: `autotune_launch`). Wrappers import from
`tilegym.backend.cutile_rs.{utils,autotuner}`.

(Device/host split: the Python wrapper + backend wiring are Agent D's;
Agent B writes Rust only.)

Agent D also modifies:
- `{tilegym_path}/src/tilegym/ops/cutile_rs/__init__.py` — imports the new wrapper module
- `{tilegym_path}/src/tilegym/backend/selector.py` — `is_cutile_rs_available()`
- `{tilegym_path}/tests/ops/test_{kernel_name}.py` — adds `"cutile-rs"` to `_backends` / `@parametrize`

## cutile_kernels crate registration

There is ONE aggregated cdylib crate for every op. Register a new op by adding a
`mod` to `cutile_kernels/src/lib.rs` that `include!`s the op's `kernel.rs` and
`ffi.rs` from the sibling `{kernel_name}_kernel/` dir:

```rust
mod {kernel_name} {
    include!("../../{kernel_name}_kernel/kernel.rs");
    include!("../../{kernel_name}_kernel/ffi.rs");
}
```

Then build the single shared library:
```bash
cargo build --release   # -> libcutile_kernels.so
```

The loader autobuilds this crate on first use (`CUTILE_RS_AUTOBUILD`, on by
default); override the crate location with `CUTILE_RS_KERNELS_DIR`. No
`cargo build -p cutile-ffi`, no per-op `.so`.

## Validation

Agent D: `pytest tests/ops/test_{kernel_name}.py -k cutile_rs -v`
Agent E: `pytest tests/ops/test_{kernel_name}.py -k cutile_rs --print-record -v`
