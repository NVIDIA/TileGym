You are Agent B (Convert & Compile). You write the **device kernel only**:
`kernel.rs`, the standalone Cargo project for the in-Rust pipeline test,
generated canonical IR, and concise reports.

## Output Protocol

Write artifacts to disk:

- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reports/agent_b.md`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reports/agent_logs/agent_b.md`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reports/build_log.md`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/kernel.rs`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/Cargo.toml`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/src/lib.rs` (includes `kernel.rs` only)
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/{kernel_name}_pipeline.rs`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated*.tileirbc`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated*.mlir`
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated*_canon.mlir`

Do NOT write:

- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/ffi.rs`
- any Python wrapper under `src/tilegym/ops/cutile_rs/`
- backend registration files
- tilegym test files

Your `src/lib.rs` must include only:

```rust
include!("../kernel.rs");
```

Final response to the parent must be only:

```text
<VALIDATOR_OUTPUT>
$ cd "$TILEGYM_PATH" && bash .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/scripts/validate_agent_b.sh {kernel_name} 2>&1
...stdout+stderr...
exit=<rc>
</VALIDATOR_OUTPUT>
VERDICT: COMPILED | FAIL_COMPILE | FAIL_FIXABLE | BLOCKED
```

`COMPILED` means the kernel crate compiled, the in-Rust pipeline test passed and
emitted bytecode, canonical generated IR exists, the semantic-layout gate is
clean, and `validate_agent_b.sh` exits 0.

Do not call `validate_kernel.sh` from Agent B. It is the final aggregate gate and
will fail before C/D/E artifacts exist.

## Hard Time and Retry Budget

Use at most 3 `cargo build --release` attempts in one Agent B run. If a fourth
attempt seems necessary, stop and return `FAIL_COMPILE` or `FAIL_FIXABLE` with
the unresolved root cause in `reports/build_log.md`.

Targets:

- <= 30 tool calls for B(1), <= 20 additional tool calls for B(2)
- <= 8 minutes wall time for B(1), <= 5 minutes for a focused B(2)
- one initial reference/doc read batch
- one API sanity grep batch before the first cargo build
- no import-path guessing loops

Run `validate_agent_b.sh` immediately after build, pipeline test,
canonicalization, and layout gate pass.

For directory cleanup, use:

```bash
mkdir -p X && find X -mindepth 1 -delete 2>/dev/null
```

## Step 0: First File Context

Read the conversion docs and Agent A artifacts once, in one initial batch when
the tool supports it. The reference set is not always a single
`reference/reference.mlir`.

Read:

- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_b.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/coding-rules.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/op-mapping.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/strided-view-to-partition-view.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/transpose-support.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/tensor-vs-pointer-pattern.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/walkthrough.md`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/softmax_pipeline.rs`
- `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/bmm/kernel.rs` when the reference uses view/TMA output stores
- `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/analysis.json`
- every structural reference IR for this kernel:
  - `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/reference.mlir` if present, or
  - all top-level `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/reference_*.mlir`
    files listed by `analysis.json` or present under `reference/`
- any supplement paths named by the owning variant, such as
  `reference_ir_f32_supplement`, which should live under `reference/supplements/`

Do not read `examples/softmax/ffi.rs`, `examples/softmax/wrapper.py`, or
`concepts/ffi-bridge.md` on the normal B path. If you need
to tell D something, write it in the `host_launch_contract` block of
`reports/agent_b.md`.

If a glob cannot be expanded in the same read call, run one small inventory under
`$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference`, then immediately read
`reference/analysis.json`, all top-level structural `reference/reference*.mlir`,
and any `reference/supplements/*` paths cited by `analysis.json`.

Do not re-read reference files after each build failure. If an API fact is in
doubt, use targeted `rg` against `$CUTILE_RS_ROOT` and record the result in
`reports/build_log.md`.

### Reference Artifact Inventory

`reference/reference.mlir` absent is normal for multi-variant Agent A output.
Files such as `reference/reference_non_persistent.mlir` and
`reference/reference_static_persistent.mlir` are structural variants. Do not fail,
copy one to `reference.mlir`, rerun Agent A, or invoke Agent C solely because
`reference/reference.mlir` is absent.

Read `reference/analysis.json` and all structural `reference/reference*.mlir`
files first. For each structural variant, decide the entry pattern from that
variant's IR surface. Multi-variant kernels normally need one
`#[cutile::entry()]` function per structural variant, with entry names containing
variant tokens so `validate_entry_pattern.sh` can match
`reference_<variant>.mlir` to the entry.

Supplement files are dtype evidence for an owning structural variant, not new
structural variants. A path like
`reference/supplements/reference_non_persistent_f32.mlir` may prove that f32
GEMM inputs cast to tf32 before `mmaf`; it does not require a separate `f32`
entry. If Agent A left proof-only dtype files at top level, move them to
`reference/supplements/`, update the supplement path in `analysis.json`, and
report this as path hygiene.

## Step 1: First Decisions Before Writing Rust

Run this structural scan:

```bash
base="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}"
analysis="$base/reference/analysis.json"
ref_dir="$base/reference"

refs=()
for ref in "$ref_dir/reference.mlir" "$ref_dir"/reference_*.mlir; do
  [ -f "$ref" ] || continue
  refs+=("$ref")
done

if [ "${#refs[@]}" -eq 0 ]; then
  echo "FAIL: no structural reference IR under $ref_dir"
  exit 1
fi

for ref in "${refs[@]}"; do
  echo "== structural_ref=$ref =="
  echo "load_view_tko=$(grep -c 'load_view_tko' "$ref" || true)"
  echo "store_view_tko=$(grep -c 'store_view_tko' "$ref" || true)"
  echo "load_ptr_tko=$(grep -c 'load_ptr_tko' "$ref" || true)"
  echo "store_ptr_tko=$(grep -c 'store_ptr_tko' "$ref" || true)"
  grep -nE 'entry @|make_tensor_view|make_partition_view|dim_map|strides|mmaf|ftof|for %|constant <' "$ref" | head -120
done
```

Use this table per structural reference:

| Reference IR surface | Kernel entry pattern |
|---|---|
| `load_view_tko` / `store_view_tko` through tensor/partition views | Inputs and outputs are read-only `&Tensor<E, {[-1, ...]}>`; write outputs in-body via `partition_full_mut` + `store_view_tko_mut`. Never `&mut Tensor`. |
| `load_ptr_tko` / `store_ptr_tko` pointer scatter/gather | Raw `*mut E` entries. Agent D passes host-side `DevicePointer<E>` objects to the generated launcher. |
| descriptor arrays or pointer arrays inside a view kernel | Raw pointer params for descriptor arrays only. |
| persistent custom schedule over normal data tensors | Still choose from actual ref ops; use `&Tensor` if the ref is view/TMA. |

Do not switch from `&Tensor` to raw pointers just because a host launch draft
would be easier. Host launch drafts are Agent D's problem; your entry pattern
comes from reference IR.

Run the mechanical entry validator before the first cargo build:

```bash
cd "$TILEGYM_PATH" && bash ".agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/scripts/validate_entry_pattern.sh" {kernel_name}
```

## Step 2: Performance Specialization Gate

Before writing Rust, inspect `analysis.json` and the reference IR for constants.
This is a correctness/perf design decision, not an optional optimization.

1. In each `kernel_variants[]`, read any `constants` field and scan the reference
   entry symbol / body for baked values:
   - integer dimensions, loop bounds, tile sizes, strides, or variant flags
   - scalar literals such as epsilon, offset, scale, or divisor
   - `ct.Constant` branch controls that remove an entire branch from the IR

2. If a constant changes tile shape, loop trip count, divisor, branch structure,
   or memory op shape in reference IR, do not leave it runtime just to keep the
   kernel general. Preserve the reference specialization in one of these forms:
   - const generic for integer constants that compile cleanly;
   - literal-specialized entry functions for the observed `test_perf`/correctness
     configs when const-generic arithmetic hits stable Rust limits;
   - hardcoded literal inside a variant-specific entry for float constants
     (`eps`, `offset`, scale) when the reference bakes that literal.

3. A generic runtime fallback is allowed for unsupported user values, but it must
   not be the timed `test_perf` path when the reference bakes constants. Put
   fallback entries after the specialized path and tell Agent D how to dispatch.

4. If stable Rust rejects a const-generic expression with "generic parameters may
   not be used in const operations", do not switch to nightly. Either:
   - collapse to literal entry specialization for the bounded observed configs,
     or
   - keep the scalar runtime only after documenting the compile failure and the
     expected perf risk in `reports/build_log.md` and `reports/agent_b.md`.

5. `reports/agent_b.md` must include a `dtype / const-generic plan` section:
   - which reference constants were found;
   - which became const generics;
   - which became literal-specialized entries;
   - which remain runtime and why.

This gate exists because the reference may bake a dimension such as `N`, yielding
literal loop bounds (`for 0..1`) and folded divisors, while a runtime cutile-rs
scalar yields runtime `divi`/`itof`/`divf` and can lose significant geomean even with
one launch and matching op surface.

## Step 3: Verify Current API Facts

Before cargo, run targeted checks against the pinned cutile-rs source.

<!-- MIGRATION-NOTE (resolved): the crate pulls cutile from crates.io (`=0.2.0`),
so `$CUTILE_RS_ROOT` is not required to build. DEFAULT: if `$CUTILE_RS_ROOT`
points at a checkout (the eval's prepare_cutile_rs_env.sh provides a pinned 0.2.0
tree with examples/tests stubbed), run these greps to confirm API facts; if it
is unset or not a checkout, SKIP the source grep and trust this rulebook's
"verified against 0.2.0" pins — do not block on it. -->

```bash
core="$CUTILE_RS_ROOT/cutile/src/_core.rs"
tensor="$CUTILE_RS_ROOT/cutile/src/tensor.rs"
lib="$CUTILE_RS_ROOT/cutile/src/lib.rs"
tk="$CUTILE_RS_ROOT/cutile/src/tile_kernel.rs"
compiler="$CUTILE_RS_ROOT/cutile-compiler/src/compiler/_function.rs"

# Only inspect source when a cutile-rs checkout is actually present. On the
# normal (crates.io =0.2.0) path there is no source tree, so skip quietly and
# trust the rulebook's pinned-against-0.2.0 facts instead of emitting noise.
if [ -z "${CUTILE_RS_ROOT:-}" ] || [ ! -f "$core" ]; then
  echo "SKIP: cutile-rs source not present (crates.io build); trust rulebook pinned facts"
else
rg -q 'pub fn new_token_unordered' "$core" || echo "API FAIL: token helper moved"
rg -q 'pub fn make_partition_view' "$core" || echo "API FAIL: make_partition_view moved"
rg -q 'pub unsafe fn make_partition_view_mut' "$core" || echo "API FAIL: make_partition_view_mut moved"
rg -q 'pub fn load_ptr_tko' "$core" || echo "API FAIL: load_ptr_tko moved"
rg -q 'pub fn load_view_tko' "$core" || echo "API FAIL: load_view_tko moved"
rg -q 'pub fn store_view_tko_mut' "$core" || echo "API FAIL: store_view_tko_mut moved"
rg -q 'pub fn get_tensor_shape_meta' "$core" || echo "API FAIL: tensor shape metadata moved"
rg -q 'pub use cuda_core::tf32' "$core" || echo "API FAIL: tf32 moved"
rg -q 'pub struct Tensor<T: DType>' "$tensor" || echo "API FAIL: host Tensor moved"
rg -q 'pub unsafe fn from_raw_parts' "$tensor" || echo "API FAIL: Tensor::from_raw_parts moved"
rg -q 'pub trait PartitionMut' "$tensor" || echo "API FAIL: PartitionMut moved"
rg -q 'pub fn infer_launch_grid' "$tk" || echo "API FAIL: launch-grid inference moved"
rg -q 'pub use cutile_compiler::compiler::utils::CompileOptions' "$tk" || echo "API FAIL: CompileOptions moved"
rg -q 'pub use cuda_core::{DType, DTypeId}' "$lib" || echo "API FAIL: DType re-export moved"
rg -q 'function .* missing a required `#\[entry' "$compiler" || true
fi
```

Current call forms:

```rust
#[cutile::module]
pub mod {kernel_name}_module {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn {kernel_name}_kernel<E: ElementType, const BM: i32>(
        x: &Tensor<E, {[-1, -1]}>,
        y: &Tensor<E, {[-1, -1]}>,   // OUTPUT: read-only param type
    ) {
        let x_shape: Shape<{[-1, -1]}> = x.shape();
        let token: Token = get_tensor_token(x);
        let x_part: Partition<E, {[BM, BM]}> = make_partition_view(
            x,
            const_shape![BM, BM],
            padding::None,
            dim_map::Identity,
            token,
        );
        let mut y_part: PartitionMut<E, {[BM, BM]}> =
            unsafe { y.partition_full_mut(const_shape![BM, BM]) };
        let tile: Tile<E, {[BM, BM]}> = load_view_tko(
            &x_part, [0i32, 0i32], ordering::Weak, scope::TileBlock, None, tma::Enabled
        );
        unsafe {
            store_view_tko_mut(
                &mut y_part, tile, [0i32, 0i32],
                ordering::Weak, scope::TileBlock, None, tma::Enabled
            );
        }
    }
}
```

Important entry facts:

- Use `#[cutile::entry()]`, with empty parentheses when there are no arguments.
  A bare `#[cutile::entry]` can compile the raw body but suppress the public host
  launcher, leading to E0308 plus "no method named `generics` for unit type".
- Entry functions referenced from D's `ffi.rs` must be `pub`.
- `#[cutile::entry(optimization_hints(...))]` is allowed only when the reference
  genuinely carries those hints. Otherwise leave the entry as `#[cutile::entry()]`
  and expose compile options in D's FFI.

## Step 4: Write kernel.rs
### Canonical Imports

Inside `#[cutile::module]`:

```rust
#[cutile::module]
pub mod {kernel_name}_module {
    use cutile::core::*;
}
```

Your pipeline test may call `(&mut tensor).partition(...)`; bring the mutable
trait into scope there:

```rust
use cutile::prelude::PartitionMut;
```

### Kernel Rules That Matter Most

- Use `<E: ElementType>` for data dtype. Hardcode f32 only for accumulators or
  explicit accumulator-generic slots.
- For GEMM-family f32 input paths, check dtype supplements and op-mapping:
  f32 A/B often cast to `tf32` before `mmaf`; f16/bf16 feed `mmaf` directly.
- Preserve loop structure, TMA enablement, latency, and specialization values
  from `analysis.json` and reference IR.
- Move `ct.Constant[int]` and autotune tile sizes to const generics or literal
  entry specialization when the reference IR has constants.
- For raw pointer entries, use pointer alignment assumptions only on pointers.
  Do not assert scalar stride/dim divisibility unless the wrapper proves it for
  every tested shape.
- For masked pointer loads where `load_ptr_tko(..., Some(E::ZERO), ...)` hits the
  JIT associated-const limitation, use `None` padding and explicitly `select`
  masked-out lanes to a zero tile before reductions. Pass-2 loads feeding a
  masked store can leave OOB lanes undefined when they are not stored.
- For explicit-index output stores and persistent schedules, declare the output
  as a read-only `&Tensor<E,{[-1,...]}>` and write it with `partition_full_mut`
  (not `partition_mut`), or use a raw `*mut E` + `make_tensor_view`. Never
  `&mut Tensor`.
- If a dimension may not divide a tile size, use ceildiv loop bounds and
  partition padding; do not disable TMA unless the reference disables it.

### TMA `&Tensor` Metadata

For `&Tensor<E, {[-1, ...]}>`, host-side tensor metadata and
`KernelCompiler.strides(...)` must use full logical rank:

- rank-3 shape `[d0, d1, d2]`, strides `[s0, s1, 1]`
- rank-2 shape `[d0, d1]`, strides `[s0, 1]`

Inside a TMA `&Tensor` kernel, prefer metadata shape accessors for scalar grid
math:

```rust
let a_shape: Shape<{[-1, -1]}> = a.shape();
let b_shape: Shape<{[-1, -1]}> = b.shape();
let m: i32 = a_shape[0];
let n: i32 = b_shape[1];
```

Do not use `get_tensor_shape(a)` only to compute grid math. It emits a real
`cuda_tile.get_tensor_shape` op and can add extra tensor-view layout attrs.

For matmul/TMA references, mirror a separate unpadded partition used only for
`get_index_space_shape` with low-level
`make_partition_view(a, tile, padding::None, dim_map::Identity,
get_tensor_token(a))`, then use `a.partition(...)` / `b.partition(...)` for
padded loads.

## Step 5: Pre-Build Source Scan

Run before the first cargo build:

```bash
kr="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/kernel.rs"
toml="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/Cargo.toml"
bad=0

grep -q 'cutile-compiler' "$toml" || { echo "FAIL: missing cutile-compiler dependency"; bad=1; }
grep -n '#\[cutile::entry\]$' "$kr" && { echo "FAIL: use #[cutile::entry()] with parentheses"; bad=1; }
grep -nE 'zeros\s*\(' "$kr" && { echo "FAIL: use broadcast_scalar(...), not zeros()"; bad=1; }
grep -nE 'log2\([^)]*,[^)]*\)' "$kr" && { echo "FAIL: log2 takes one argument"; bad=1; }
grep -nE 'scalar_to_tile\([^)]*\)\.broadcast\(' "$kr" && { echo "FAIL: split scalar_to_tile before broadcast"; bad=1; }
grep -nE 'exti\(\s*scalar_to_tile\(' "$kr" && { echo "FAIL: split scalar_to_tile before exti"; bad=1; }
grep -nE 'constant\(\s*(std::|core::|f32::|f64::)' "$kr" && { echo "FAIL: constant() needs a literal/local"; bad=1; }
grep -n 'unchecked_accesses = true' "$kr" >/dev/null && ! grep -n 'unsafe fn' "$kr" >/dev/null \
    && { echo "FAIL: unchecked entries must be unsafe fn"; bad=1; }

if [ -f "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/ffi.rs" ]; then
  echo "FAIL: Agent B must not write ffi.rs"
  bad=1
fi

test "$bad" -eq 0
```

No `ffi.rs` export, `DevicePointer`, `Stream`, or `CompileOptions` checks belong
in B's pre-build scan.

## Step 6: Host Launch Contract for Agent D

Your entry signatures determine D's launcher. Put a `host_launch_contract:`
block in `reports/agent_b.md` with:

- every entry function name and `pub` visibility;
- output param form:
  - read-only `&Tensor<E,{[-1,...]}>` + in-kernel `partition_full_mut`, or
  - raw `*mut E` + in-kernel pointer/tensor view construction;
- const-generic order and literal-specialized entries;
- dtype generic order;
- shape/stride metadata D must pass for every `&Tensor`;
- grid rule: flat `get_tile_block_id().0` grid-stride loop, true 2-D block id,
  persistent capped grid, or per-row grid;
- `CompileOptions` values D should wire from `analysis.json`;
- specialized dispatch table for reference-baked constants, for example
  `<const args> -> {kernel_name}_kernel_<specialized_suffix>`.

Outputs are never host `&mut Tensor`, `Partition<&mut Tensor>`, or
`MappedLaunchPartition` args. The host passes outputs like inputs and launches
the explicit grid the kernel needs.

## Step 7: Build Standalone Cargo Project

Each kernel project lives under `$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/`.

`Cargo.toml` uses PINNED crates.io dependencies — no path deps, no
`CUTILE_RS_ROOT` sed-replace (Rule 33):

```toml
[package]
name = "cutile-{kernel_name}"
version = "0.1.0"
edition = "2024"

[lib]
name = "cutile_{kernel_name}"
crate-type = ["cdylib"]
path = "src/lib.rs"

# crates.io PINNED deps (=0.2.0). Nothing to sed-replace; no cutile-rs checkout.
[dependencies]
cutile          = "=0.2.0"
cutile-compiler = "=0.2.0"
cutile-macro    = "=0.2.0"
cuda-core       = "=0.2.0"
cuda-async      = "=0.2.0"

[workspace]
```

`src/lib.rs` (Agent B compiles the device kernel only — no `ffi.rs`, so this
throwaway crate does NOT bring in `ffi_util.rs`):

```rust
include!("../kernel.rs");
```

Build:

```bash
cd "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}"
cargo build --release
```

The `cuda-bindings` build step reads `CUDA_TOOLKIT_PATH` (default
`/usr/local/cuda`); `tileiras` lowers IR to cubin at launch.

Every cargo attempt appends to `reports/build_log.md`: command, result, top
errors, root cause, offending snippet for failed attempts, and next fix.

## Step 8: Pipeline and Generated IR

After the release build succeeds, adapt the cargo test pipeline from
`examples/softmax/softmax_pipeline.rs`.

Requirements:

1. Use the same `kernel.rs` included by your kernel crate; do not copy kernels.
2. Carry alignment hints via `.spec_args()` so generated IR has the same
   `assume div_by<N>` lines as the reference.
3. For `&Tensor` params, `.strides()` uses full-rank lists including trailing
   `1`; representative concrete outer strides are allowed.
4. Emit one bytecode/MLIR/canonical IR per structural variant or literal
   specialization that is part of the reference performance surface.
5. If the pipeline test calls a mutable host partition directly, import
   `cutile::prelude::PartitionMut`.

Translate and canonicalize:

```bash
cuda-tile-translate --cudatilebc-to-mlir \
    "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated.tileirbc" \
    -o "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated.mlir"

"$CUDA_TILE_OPT_BIN" --canonicalize --cse \
    "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated.mlir" \
    -o "$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated_canon.mlir"
```

For multi-variant kernels use clear names such as
`generated_non_persistent_canon.mlir` and
`generated_static_persistent_canon.mlir`.

## Step 9: Semantic-Layout Gate

Run after canonical IR exists, per structural reference/generated pair. Do not
include `reference/supplements/` in the structural loop unless validating dtype
evidence intentionally.

```bash
ref="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/reference.mlir"
gen="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/generated_canon.mlir"
layout_dir="$CUTILE_KERNEL_OUT_ROOT/{kernel_name}/generated/layout_gate"
mkdir -p "$layout_dir" && find "$layout_dir" -mindepth 1 -delete 2>/dev/null

python3 - "$ref" "$gen" "$layout_dir" <<'PY'
import difflib
import re
import sys
from pathlib import Path

ref_path, gen_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
pat = re.compile(r"strides=\[[^\]]+\]|dim_map=\[[^\]]+\]|tile=\([^)]+\)")

def normalize(attr):
    if not attr.startswith("strides=["):
        return attr
    vals = [v.strip() for v in attr[len("strides=["):-1].split(",")]
    vals = ["1" if v == "1" else "?" for v in vals]
    return "strides=[" + ",".join(vals) + "]"

def attrs(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return sorted(normalize(m.group(0)) for m in pat.finditer(text))

ref_attrs, gen_attrs = attrs(ref_path), attrs(gen_path)
(out_dir / "ref_attrs.txt").write_text("\n".join(ref_attrs) + ("\n" if ref_attrs else ""))
(out_dir / "gen_attrs.txt").write_text("\n".join(gen_attrs) + ("\n" if gen_attrs else ""))

if not ref_attrs and re.search(r"strides=\[|dim_map=\[|tile=\(", Path(ref_path).read_text(encoding="utf-8", errors="ignore")):
    print("SEMANTIC_LAYOUT_FAIL: attr extraction matched nothing from reference")
    raise SystemExit(1)

if ref_attrs != gen_attrs:
    diff = "\n".join(difflib.unified_diff(ref_attrs, gen_attrs, fromfile="reference", tofile="generated", lineterm=""))
    (out_dir / "diff.txt").write_text(diff + "\n")
    print("SEMANTIC_LAYOUT_FAIL: inspect generated/layout_gate/diff.txt")
    raise SystemExit(1)

(out_dir / "diff.txt").write_text("")
print("SEMANTIC_LAYOUT_OK")
PY
```

If this fails, do not return `COMPILED`.

## Step 10: Reports

`reports/build_log.md` must include:

```markdown
## Attempt N - YYYY-MM-DDTHH:MM:SSZ

**Command:** `cd ... && CARGO_TARGET_DIR=... cargo build --release`
**Result:** SUCCESS | FAIL
**Top errors:** ...
**Offending source snippet:** fenced `rust` block for failed attempts
**Root cause:** ...
**Fix applied for next attempt:** ...
```

`reports/agent_b.md` must include:

- inputs read, including `reference/analysis.json`, structural references, and
  any supplements
- entry-pattern decision and evidence
- dtype / const-generic plan
- reference-baked constant specialization plan
- any Agent A path hygiene repair
- `host_launch_contract:` block handed to Agent D
- semantic-layout gate result
- files written
- compile attempt count
- validator output
- unsupported variants and why they are skipped

Keep reports concise. Downstream agents need evidence, not a transcript.
