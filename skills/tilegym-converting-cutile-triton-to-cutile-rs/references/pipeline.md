# Pipeline architecture, gates, and orchestrator rules

This file holds the load-bearing details that the slim `SKILL.md` links to. Read it before spawning the first sub-agent so the run follows the bounded state machine and does not spend time on inline debugging or no-op respawns.

## Contents

- Architecture
- Agent responsibilities
- Orchestrator contract
- Mechanical validator acceptance gate
- Inter-agent communication
- Gate decisions
- Retry policy
- Compiler bug escalation
- The 19 CRITICAL rules

## Architecture

```text
Orchestrator
  |
  | preflight.sh
  v
Agent A: IR dump, analysis.json, reference/tolerance, baseline perf
  |
  | PASS
  v
Agent B(1): kernel.rs (DEVICE ONLY), kernel-only cargo build, in-Rust pipeline test, generated IR
  |
  | COMPILED
  v
Agent D(1): build Python wrapper + backend wiring, THEN tilegym correctness
  |
  | ALL_PASS              FAIL / BLOCKED (route by fail_class / block_class)
  v                       v
Agent E(1): CUPTI perf    fail_class/block_class: host → Agent D(2)  (D owns ffi.rs/.so/wrapper)
  |                       fail_class/block_class: kernel  → Agent C → B(2)
  | DONE
  v
PIPELINE_COMPLETE

Agent E(1) DONE is TERMINAL → PIPELINE_COMPLETE; never spawn C/D(2)/F after DONE
(geomean ≤ 1.05 is the authoritative gate; a (1.0,1.05] geomean or an info-only
slow per-config row is NOT grounds to diagnose).
Agent E(1) INVESTIGATE/BLOCKED routes to Agent C once. C tags owner:
  owner: host → Agent D(2) (ffi.rs launcher/output-partition ABI/launch grid/autotune value/wrapper)
  owner: kernel  → Agent B(2) (kernel.rs device math/IR)
Then Agent D(2) -> Agent E(2) -> PIPELINE_COMPLETE.
Agent C PASS/PASS_WITH_NOTES does not respawn; it ends the run with reports for reflection.
```

The happy path is A -> B -> D -> E. Agent B writes Rust only; Agent D owns the
Python wrapper + backend wiring + correctness, so wrapper-caused mismatches and
wrapper-caused perf gaps route back to D, not B. Agent C is diagnostic-only and
tags each fixable finding `owner: host|kernel`. Agent F is optional after the
fix loop is spent.

## Agent Responsibilities

| Agent | Does | Does not do |
|---|---|---|
| A | Dump reference IR, analysis.json, tolerance, baseline perf | Write Rust kernel code |
| B | Write the device kernel.rs, compile kernel-only, pass in-Rust pipeline test, canonicalize generated IR, hand D a host_launch_contract | Write ffi.rs, build the .so, build the Python wrapper, register backend, run correctness/perf |
| D | Write ffi.rs (C-ABI launcher) + build the cdylib .so + Python wrapper + backend wiring, run tilegym correctness | Edit kernel.rs or the device module |
| C | Diff IR and perf-relevant attributes, classify differences, write action items + `owner: host\|kernel` | Edit code or run tests |
| E | Run CUPTI perf comparison and report ratios | Fix code |
| F | Optional residual perf diagnosis after fix-loop budget | Replace B/D/E |

Each agent writes `reports/agent_{x}.md` and ends its return with a literal `<VALIDATOR_OUTPUT>` block followed immediately by one enumerated `VERDICT:` line.

## Orchestrator Contract

The orchestrator is a router, not a worker. It does exactly this:

1. Run `scripts/preflight.sh`.
2. Spawn the next agent using the minimal pointer prompt from `SKILL.md`.
3. Check the returned `<VALIDATOR_OUTPUT>` block.
4. Route by the final `VERDICT:` enum and the tables below.
5. Run `scripts/validate_kernel.sh {kernel_name}` only after the pipeline reaches `PIPELINE_COMPLETE`.

The orchestrator must not read IR files or wrappers for routing, edit source, run cargo, run pytest, run benchmarks, or diagnose root causes inline. If root cause is needed, spawn Agent C when the route allows it.

## Mechanical Validator Acceptance Gate

Expected final response shape from a sub-agent:

```text
<VALIDATOR_OUTPUT>
$ bash .../validate_agent_x.sh kernel 2>&1
...stdout+stderr...
exit=0
</VALIDATOR_OUTPUT>
VERDICT: PASS
```

Acceptance algorithm:

1. Extract the first enumerated `VERDICT:` line.
2. Search for a literal `<VALIDATOR_OUTPUT>` ... `</VALIDATOR_OUTPUT>` block.
3. If the block exists, parse every `exit=<integer>` or `exit_code=<integer>` inside it.
4. If at least one exit is parseable and all exits are zero, accept and route by verdict.
5. If the block is missing but the text has a clean success verdict marker (`PASS`, `COMPILED`, `ALL_PASS`, `DONE`) and none of `FAIL`, `ERROR`, `Traceback`, `panic`, `exit=1`, `exit_code=1`, `nonzero`, or `timeout`, accept with `SOFT_VALIDATOR_MISSING`.
6. Otherwise re-spawn only the same agent letter once for validator repair. The repair prompt must tell the agent to reuse existing artifacts and return the same semantic verdict plus a valid block.
7. If repair still fails, proceed with `SOFT_VALIDATOR_FAILED` unless the semantic verdict itself blocks.

Hard locks:

- Missing validator block plus clean success marker and no failure marker is accepted; do not respawn for formatting alone.
- Maximum one validator repair per agent letter per run.
- Never cascade validator repair across agents.
- A non-zero validator in an agent's own block repairs that same agent only. It does not authorize another agent respawn.
- The aggregate `validate_kernel.sh` gate is not part of Agent B's validator block.

Minimum validator evidence by agent:

| Agent | Required evidence in `<VALIDATOR_OUTPUT>` |
|---|---|
| A | `validate_ir.sh exit=0`, `validate_analysis.sh exit=0`, and perf-log validation when A writes perf logs |
| B | compile/canonicalization status and `validate_agent_b.sh exit=0` |
| C | `validate_diff_report.sh exit=0`; include `validate_ir.sh` and `diff_ir.sh` output in reports, not necessarily the final block |
| D | correctness pytest command and `validate_agent_d.sh exit=0`; failing pytest may be non-zero only with D `FAIL` |
| E | perf pytest command, `validate_perf_log.sh` for cutile-rs and baseline logs, and `validate_agent_e.sh exit=0` |
| F | residual investigation commands or artifact-validation exits |

## Inter-Agent Communication

Sub-agents communicate through files on disk. The orchestrator passes paths, not summaries.

| Producer | Main outputs | Consumer |
|---|---|---|
| A | `reference/analysis.json`, `reference/*.mlir`, baseline perf logs | B, C, E |
| B | `kernel.rs`, kernel-only `Cargo.toml`, generated IR, build log, host_launch_contract | D, C |
| C | `generated/diff_report.md` (+ `owner:` line), `reports/agent_c.md` | B(2) on `owner: kernel`, D(2) on `owner: host`; reflection otherwise |
| D | `ffi.rs` + cdylib `.so` + Python wrapper + backend wiring, `reports/correctness.md` (+ `fail_class:`), correctness log | E, C, or D(2) |
| E | `reports/performance.md`, `perf_cutile_rs.txt`, baseline perf logs | C or reflection |
| F | `reports/perf_investigation_*.md` | reflection |

Good next-agent prompt: `Kernel: softmax. Agent C verdict: FAIL_FIXABLE. Read action items from ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/diff_report.md`.

Bad next-agent prompt: pasting C's whole analysis into B's spawn prompt. That causes context drift and route errors.

## Gate Decisions

### Agent A

| Verdict | Action |
|---|---|
| `PASS` | Spawn Agent B |
| `FAIL_FIXABLE` / `BLOCKED` | STOP and record `SKILL_METHODOLOGY_BLOCKED.txt` |

### Agent B(1)

| Verdict | Action |
|---|---|
| `COMPILED` | Spawn Agent D |
| `FAIL_COMPILE` / `FAIL_FIXABLE` / `BLOCKED` | STOP |

Agent B validates B-stage artifacts with `validate_agent_b.sh`. Do not ask Agent B to run `validate_kernel.sh`; final aggregate validation requires D/E outputs and belongs to the orchestrator after `PIPELINE_COMPLETE`.

### Agent D(1)

| Verdict | Action |
|---|---|
| `ALL_PASS` | Spawn Agent E |
| `FAIL` / `BLOCKED` (`fail_class`/`block_class: host`) | Spawn Agent D(2) — D owns the ffi.rs/.so/wrapper/wiring/autotune fix; read `host_fix:` |
| `FAIL` / `BLOCKED` (`fail_class`/`block_class: kernel`) | Spawn Agent C if C has not run — kernel/FFI fault; read `agent_b_followup:` |
| `BLOCKED` (`block_class: env`) | STOP (external) |

D now owns the Python wrapper, so most non-kernel mismatches (0 tests collected, argtype/CUDA-700, dtype dispatch, layout) are D's own to fix via D(2). Only a provably kernel-side fault (missing `.so`/symbol, FFI signature gap, math wrong across correct wrapper variants, in-kernel panic) goes to C→B. If the class line is absent, fall back to C.

### Agent E(1)

| Verdict | Action |
|---|---|
| `DONE` | `PIPELINE_COMPLETE` — **TERMINAL. MUST NOT spawn C, D(2), or F.** |
| `INVESTIGATE` / `BLOCKED` | Spawn Agent C if C has not run and B has not been respawned; otherwise `PIPELINE_COMPLETE` |

**`DONE` is binding and terminal — no discretionary diagnosis.** Agent E's
`VERDICT: DONE` means the perf gate already passed (geomean_ratio ≤ 1.05 is the
authoritative completion criterion). The orchestrator MUST route `DONE` straight
to `PIPELINE_COMPLETE` and end the run. Specifically you may NOT:
- spawn Agent C "to diagnose perf root cause" / "to confirm it's just noise" — C
  is reachable ONLY from `INVESTIGATE`/`BLOCKED`, never from `DONE`;
- reinterpret the task goal (e.g. "≤ baseline") to override the verdict — a
  geomean in `(1.0, 1.05]` is PASS, not a gap to chase; `DONE` already accounts
  for that parity window;
- treat an info-only slow per-config row in `performance.md` as a reason to
  investigate — per-row ratios are diagnostic only and do NOT gate the verdict.

The routing table is not a menu of optional levers; for `DONE`, `PIPELINE_COMPLETE`
is the only legal action. Spawning any agent after `DONE` is a routing violation.

Agent E does not use `FAIL` as a final verdict. Slow but valid perf is `INVESTIGATE` with severity and per-row ratios in `reports/performance.md`.

### Agent C

| Verdict | Action |
|---|---|
| `FAIL_FIXABLE` (`owner: kernel`) | Spawn Agent B(2) if B has not been respawned; otherwise `PIPELINE_COMPLETE` |
| `FAIL_FIXABLE` (`owner: host`) | Spawn Agent D(2) if D has not been respawned; otherwise `PIPELINE_COMPLETE` |
| `PASS` | `PIPELINE_COMPLETE` |
| `PASS_WITH_NOTES` | `PIPELINE_COMPLETE` |
| `FAIL_COMPILER_BUG` | `PIPELINE_COMPLETE` |
| `BLOCKED` | `PIPELINE_COMPLETE` |

Agent C verdict semantics:

- `FAIL_FIXABLE` means C found a concrete fixable action and tagged `owner:`. `owner: kernel` → B(2) (kernel.rs device math/IR); `owner: host` → D(2) (ffi.rs launcher, output-partition ABI, launch grid, cutile_kernels crate build, num_cta_in_cga/occupancy, cffi `_FFI_CDEF` / `TensorDesc` packing, dtype dispatch, layout, backend registration). Action items must be explicit in `generated/diff_report.md`. If `owner` is absent, default to `kernel`/B(2).
- `PASS` means no fixable owner was found.
- `PASS_WITH_NOTES` means remaining differences are known gaps, INFO-only IR deltas, semantic equivalents, or residual perf notes. These are not a reason for a respawn.
- `FAIL_COMPILER_BUG` and `BLOCKED` stop the in-run semantic loop.

This distinction prevents no-op respawns and sends each fix to its true owner (Rust→B, Python/host→D).

### Agent B(2), D(2), E(2)

D(2) is reachable two ways: from a host-owned C `FAIL_FIXABLE`/D `fail_class: host` (D fixes its own ffi.rs/wrapper), or from B(2) after a kernel fix. Either way D's semantic-run cap is 2.

| Stage | Verdict | Action |
|---|---|---|
| B(2) | `COMPILED` | Spawn D(2) |
| B(2) | any failure/block | STOP |
| D(2) | `ALL_PASS` | Spawn E(2) |
| D(2) | `FAIL` / `BLOCKED` | `PIPELINE_COMPLETE` |
| E(2) | `DONE` / `INVESTIGATE` / `BLOCKED` | `PIPELINE_COMPLETE` |

After E(2), the fix-loop budget is spent. Do not spawn C again in the same run.

### Agent F

Agent F is optional and runs only after the one-shot fix loop has been spent. F may diagnose residual perf, but it does not open another in-run B/D/E loop.

| F conclusion | Action |
|---|---|
| `FIXABLE` | `PIPELINE_COMPLETE` (action items recorded for reflection) |
| `ALIGNED` | `PIPELINE_COMPLETE` |
| `BLOCKED` | `PIPELINE_COMPLETE` |

## Retry Policy

| Phase | Max attempts | On exhaust |
|---|---:|---|
| A semantic run | 1 | STOP |
| B compile loop inside one B run | 3 cargo builds | B returns `FAIL_COMPILE` |
| B semantic runs | 2 total | STOP after B(2) failure |
| C semantic run | 1 | STOP / reflection mines report |
| D semantic runs | 2 total | STOP after D(2) failure |
| E semantic runs | 2 total | STOP with residual perf |
| F semantic run | 1 | STOP |
| Validator repair | 1 per agent letter | Proceed with soft marker unless semantic verdict blocks |

Retry boundaries:

- Semantic retries follow the gate table.
- Validator repair is local to the same agent letter.
- Missing validator block alone is not enough to respawn if the compatibility fallback accepts.
- Non-zero validator output paired with a success verdict is a same-agent repair event, not a reason for the orchestrator to diagnose, inspect IR, or rerun tests inline.

## Compiler Bug Escalation

1. Check `references/coding-rules.md`.
2. If a workaround exists, Agent C should report `FAIL_FIXABLE` and list the workaround for B.
3. If no workaround exists, Agent C reports `FAIL_COMPILER_BUG` and documents the limitation.
4. The orchestrator does not create a minimal repro inline. Reflection or a future Agent F can do that outside the current bounded route.

Compiler-bug escalation still obeys the validator gate.

## The 19 CRITICAL Rules

1. **test_perf_configs only**: tests and benchmarks use tilegym test_perf shapes. Do not invent custom benchmark shapes.
2. **Identical workload**: cutile-rs and the reference backend run the exact same input and variant.
3. **Forward first**: implement forward kernels before backward variants.
4. **Logs mandatory**: every spawned agent writes `reports/agent_{x}.md` or `reports/agent_logs/agent_{x}.md`.
5. **IR diff is diagnostic, not automatically actionable**: Agent C drives B(2) only with `FAIL_FIXABLE`. `PASS` and `PASS_WITH_NOTES` end the current run with reports.
6. **Autotuning**: if any backend uses autotune for this kernel, cutile-rs uses the CUPTI-based autotuner (`from tilegym.backend.cutile_rs import autotune_launch`) with configs recorded by Agent A.
7. **dtype generic**: kernels use `<E: ElementType>` unless a documented side buffer requires concrete f32.
8. **Reference from tilegym**: correctness reference comes from the tilegym test class, not a hand-written formula.
9. **Tolerance from tilegym**: Agent A records the exact atol/rtol used by tilegym tests.
10. **Multi-variant ops**: Agent A records all variants and Agent B converts all variants; Agent E compares each variant against its matching baseline.
11. **No fallback**: the cutile-rs wrapper must call the cutile-rs FFI kernel or raise a clear unsupported error. It must not silently fall back to PyTorch or another backend.
12. **CUPTI only for perf**: performance conclusions use tilegym `--print-record` CUPTI timing, not `torch.cuda.Event`.
13. **ct.Constant -> const generic**: prefer const generics for Python `ct.Constant[int]` values unless lower-IR evidence shows constant unrolling blocks software pipelining.
14. **Happy path skips C**: A -> B -> D -> E is the normal route. C runs only after D/E non-DONE. B(2) runs only after C `FAIL_FIXABLE`.
15. **Dual-backend reference selection**: Agent A benchmarks cuTile-Python and Triton-TileIR when available and records the winning reference backend.
16. **Autotuner lambda uses torch.empty**: output allocation in perf lambdas must use `torch.empty`, not clone/zeros/ones, to avoid extra GPU kernels in CUPTI.
17. **Perf investigation uses the right layer**: C checks IR attributes first; residual post-loop gaps belong to reflection or optional Agent F.
18. **Const generic tradeoffs**: default to const generic, but switch to runtime only when IR and nsys evidence show const unrolling hurts the generated loop.
19. **Mechanical validator block before routing**: validate the literal `<VALIDATOR_OUTPUT>` block before semantic routing. Same-agent repair is capped once; never cascade repairs; final `validate_kernel.sh` runs only after the pipeline is complete.
