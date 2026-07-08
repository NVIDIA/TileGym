# Performance Checklist

If a cutile-rs kernel is slower than cutile-python or perf is missing, check correctness and wrapper behavior first. A fast wrong kernel still scores poorly.

## Mandatory Pre-Submit Checklist

Record pass/fail evidence in `<VALIDATOR_OUTPUT>` when a validator or scorer asks for it.

- [ ] **Cargo artifact exists**: `ops/cutile_rs/cutile_kernels/target/release/libcutile_kernels.so` exists and `nm -D` shows `T cutile_{kernel_name}`.
- [ ] **Pipeline compiled**: `cargo test --release -p cutile --test {kernel_name}_pipeline -- --nocapture` produced bytecode/MLIR.
- [ ] **Entry pattern matches reference IR**: view/TMA reference uses `&Tensor`; pointer reference uses `*mut E`.
- [ ] **Wrapper import works**: `tilegym.set_backend("cutile-rs")` and `from tilegym.ops.cutile_rs import {kernel_name}` succeed.
- [ ] **Fixed-config canary stays outside timed/autotune hot paths**: one `_run_ffi` call may be used before perf timing, but never inside `kernel_fn(cfg)` and never before every cached launch.
- [ ] **Autograd contract checked**: forward-only wrappers detach grad-enabled inputs and tests skip backward surfaces for `cutile-rs`.
- [ ] **No double transpose**: for each permuted tensor, either FFI swaps shape/strides or kernel uses `partition_permuted`, not both.
- [ ] **Persistent TMA latency**: persistent grid-stride loops pass `None` for every `load_view_tko` / `store_view_tko_mut` inside the loop.
- [ ] **Autotune lambda is clean**: `kernel_fn(cfg)` allocates outputs with `torch.empty` only. No clone/zeros/ones/extra contiguous copies inside timing.
- [ ] **TMA state matches reference**: do not disable TMA when reference has TMA descriptors.
- [ ] **CompileOptions parity (both directions)**: match analysis.json/best_config exactly. If the reference autotunes an EXPLICIT `num_ctas` (matmul ships {1,2,4}), the wrapper MUST forward that per-config value so the CGA cluster size takes effect (leaving it auto pins the cluster to 1 → ~1.6x GEMM regression — the #1 perf trap). Only when the reference truly uses `num_ctas=None`/auto do you leave `CompileOptions::default()` unmodified. Decide per-kernel, not by blanket default.

## Correctness Before Perf

Run:

```bash
python -m pytest tests/ops/test_{kernel_name}.py -k cutile_rs -x --timeout=60
```

If this fails, do not interpret geomean as a useful optimization signal.

| Symptom | Check |
|---------|-------|
| `.so not found` | Cargo failed or wrong target dir |
| `backward not implemented` | Wrapper rejected `requires_grad`; detach forward-only inputs and skip backward tests |
| Forward close, gradients wrong | Backward surface should be skipped unless this task implements backward |
| Output near zero / 0% matched | Tensor shape/stride or transpose ownership |
| Fatal abort in autotune | Run fixed config outside CUPTI; inspect FFI args, mem::forget, persistent latency/layout |

## CompileOptions Parity

`CompileOptions` is part of the JIT cache key and changes codegen. The default value is meaningful:

```rust
let opts = CompileOptions::default();
```

In cutile-rs, `CompileOptions::default()` stores `occupancy=None` and `num_cta_in_cga=None`, which means compiler auto-pick. Calling `.occupancy(N)` / `.num_cta_in_cga(N)` pins those targets and is part of the JIT cache key.

The rule is **match the reference, in BOTH directions**:
- Reference auto-picks (`num_ctas=None`) → keep `default()`, do not call the setters. Pinning `num_cta_in_cga(1)` "to be safe" caps resident CTAs and can regress decode shapes.
- Reference autotunes explicit `num_ctas` (matmul: {1,2,4}; many persistent/tensor-core GEMMs) → you MUST call `.num_cta_in_cga(value)` with the per-config value. A persistent kernel left at the auto default runs a 1-CTA cluster and loses ~1.6x vs a reference using 2/4-CTA clusters. The wrapper carries the value; ffi.rs applies it under the `if num_cta_in_cga > 0` guard.

Reference mapping:

| Reference evidence | cutile-rs FFI behavior |
|---|---|
| cuTile wrapper says `num_ctas=None`, `num_ctas = None`, or "let compiler choose" | keep `CompileOptions::default()`; do not call occupancy or num_cta setters |
| reference/autotune config has explicit occupancy or num_cta | pass that value and call the setter |
| Agent E/F finds CTA-count-sensitive slowdown with forced occupancy | re-test auto/default first before chasing tile math |
| reference has empty `optimization_hints=<sm_100={}>` only | treat as cosmetic unless it contains concrete keys |

Recommended FFI sentinel pattern:

```rust
let mut opts = CompileOptions::default();
if occupancy > 0 {
    opts = opts.occupancy(occupancy);
}
if num_cta_in_cga > 0 {
    opts = opts.num_cta_in_cga(num_cta_in_cga);
}

let op = kernel_call
    .generics(generics)
    .grid(grid)
    .compile_options(opts);
```

Recommended Python wrapper pattern:

```python
_AUTO_COMPILE_OPTION = -1
_COMPILE_OCCUPANCY = None
_COMPILE_NUM_CTA_IN_CGA = None

def _compile_option_value(value):
    return _AUTO_COMPILE_OPTION if value is None else int(value)

rc = lib.cutile_op(
    ...,
    _compile_option_value(getattr(cfg, "OCCUPANCY", _COMPILE_OCCUPANCY)),
    _compile_option_value(getattr(cfg, "NUM_CTA_IN_CGA", _COMPILE_NUM_CTA_IN_CGA)),
    ...,
)
```

Separate grid sizing from compile options. A persistent wrapper may need `_GRID_OCCUPANCY` to decide how many CTAs to launch, but that does not mean ffi.rs should call `.occupancy(_GRID_OCCUPANCY)`.

## Launch Count Before IR Theory

When Agent E reports `INVESTIGATE`, first classify whether the measured function launches extra GPU work.

Use the cheapest reliable evidence:

- `DUMP_CUPTI_EVENTS=1` from the tilegym perf harness, or
- `nsys profile` / `nsys stats`, or
- static wrapper proof when both wrappers directly launch exactly one kernel and allocate with `torch.empty`.

Common wrapper/host causes:

- fixed-config canary in the hot path before `autotune_launch`
- `torch.zeros`, `torch.ones`, `.clone()`, or `.contiguous()` inside `kernel_fn(cfg)`
- preallocating one output and allocating a second output inside `kernel_fn`
- launching both a canary and the cached best config on every call
- hidden PyTorch fallback or helper kernels

If launch count is 1:1 and wrappers allocate only `torch.empty`, continue to config/IR.

## TMA And Strides

Check generated MLIR:

```bash
grep -n "tma_descriptor\|load_view_tko\|store_view_tko\|strides=" generated/generated_canon.mlir
```

- View/TMA reference should show TMA-style view ops in generated IR too.
- Rank-R tensor stride metadata must have R entries and end in `1` for contiguous innermost layouts.
- If a batch/outer stride is not 16-byte aligned, fix host padding/fallback. Do not scalar-assert false divisibility.
- If kernel uses `partition_permuted`, the wrapper/FFI should pass original physical shapes/strides.

## Accumulator Type

For reductions and MMA:

```bash
grep -n "mmaf\|reduce" generated/generated_canon.mlir
```

- `mmaf` accumulators should be f32.
- RMS and layer-norm reductions should accumulate in f32 and cast back at the end.
- If a kernel has `E2`, FFI should pass `"f32"` for that generic slot.

## Persistent Kernels

Persistent pattern:

```rust
for tile_id in (bid_x..total_tiles).step_by(grid_x as usize) {
    ...
}
```

Rules:

- Use `None` latency for view loads/stores inside the persistent body when the known persistent/TMA exception applies.
- Keep `tma::Enabled` if reference uses TMA.
- Use a fixed-config canary before autotune only as a preflight, not in the measured or cached hot path.
- Grid capping must match the Rust body. If the kernel has no grid-stride loop, launch full coverage.

## Autotune Search Space

Start with reference `analysis.json` configs. Expand only after correctness passes.

Inside `kernel_fn(cfg)`:

```python
def kernel_fn(cfg):
    y = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    _run_ffi(..., y, cfg)
    return y
```

Avoid:

- `clone()`
- `zeros()` / `ones()`
- `contiguous()` on large tensors inside the timed lambda
- reference PyTorch work inside `kernel_fn`
- canary launches inside `kernel_fn`

Cache key should include every shape/dtype/semantic flag that can affect best config, including transpose flags and variant dispatch flags. If occupancy or num_cta is autotuned, include it through the config object and let the autotuner key include the same semantic shape/dtype fields as the reference.

## Wrapper Autograd Perf Trap

Perf tests often create inputs with `requires_grad=True` even if they only benchmark forward. A wrapper like this fails perf before timing:

```python
if input.requires_grad:
    raise NotImplementedError("backward not implemented")
```

Current skill policy is forward-only unless the task explicitly asks for backward kernels. Detach grad-enabled inputs before calling FFI and patch backward tests to skip `cutile-rs`.

## Transpose Ownership

Check both wrapper/FFI and kernel:

- If FFI creates `Tensor` with swapped logical shape/strides, kernel uses identity `partition`.
- If kernel uses `partition_permuted`, FFI passes original physical shape/strides.
- If generated MLIR has both swapped strides and `dim_map`, expect wrong values even though IR compiles.

## Math Hot Spots

| Pattern | Preferred form |
|---------|----------------|
| f32 sigmoid division | `true_div(exp_x, exp_x_plus_one)` |
| FTZ reference math | named op with `ftz::Enabled` |
| f16/bf16 final output | compute in f32 then `convert_tile` |
| pointer bandwidth | `assume_div_by<16>` on pointers only |
| empty `sm_100={}` hint | do not chase unless concrete hint keys are present |
| reference auto occupancy | leave `CompileOptions::default()` auto |

## Profiling Command

Use after correctness passes:

```python
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        run_kernel()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Confirm the launched kernel name and config before changing tile sizes.

## Perf Result Interpretation

| Result | Likely next step |
|--------|------------------|
| No geomean, correctness failed | Fix correctness/wrapper/build |
| Process abort | Fixed-config canary, FFI args, borrowed Tensor ownership |
| Near-uniform 2x slower | Extra timed launch or canary in hot path |
| 1.0x-1.1x slower with CTA-count outlier | Check CompileOptions occupancy/num_cta parity before IR hints |
| 1.1x-1.5x slower | Check TMA, clean autotune lambda, constants, accumulator, config parity |
| More than 2x slower | TMA disabled, runtime constants, extra timed copies, wrong variant |
| Faster but wrong | Ignore perf until correctness passes |

## Agent F Handoff

If Agent F writes `reports/perf_investigation_*.md`, reflection should treat it as the highest-priority perf signal. Agent C sees IR; Agent F checks wrapper launch count, CompileOptions, autotune cache behavior, and source-level config mismatches. If F says the fix is template-propagation, update `examples/softmax/wrapper.py` or the relevant wrapper pattern so next Agent B does not regenerate the same bug.

## B-Only Reminder

In B-only mode, no Agent E writes a report. The host runs perf pytest directly after Agent B. Agent B should still leave enough evidence in `reports/build_log.md` and `reports/agent_b.md` for the next reflection to identify the layer that failed.
