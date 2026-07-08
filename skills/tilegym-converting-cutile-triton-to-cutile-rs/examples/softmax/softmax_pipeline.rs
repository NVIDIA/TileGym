// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

//
// Compile-and-bytecode-dump test for the softmax kernel.
//
// Lives at: ${CUTILE_KERNEL_OUT_ROOT}/softmax/softmax_pipeline.rs
// (the per-kernel working dir; run with `cargo test --test softmax_pipeline`).
// The cutile-rs Rust lives inside tilegym — there is no separate cutile-rs
// checkout / CUTILE_RS_ROOT.
//
// Two-stage IR pipeline:
//
//   Stage 1 (this file):  Rust test compiles the kernel and writes a portable
//                         **bytecode** artifact (`.tileirbc`). No external
//                         binary calls, no MLIR text serialization in Rust —
//                         keeps the test minimal and reproducible.
//
//   Stage 2 (shell):      `cuda-tile-translate --cudatilebc-to-mlir` lifts
//                         the bytecode to MLIR text; `cuda-tile-opt
//                         --canonicalize --cse` produces the canonicalized
//                         IR Agent C diffs against the reference.
//
// The kernel.rs source is the single source of truth — it is the sibling
// file in this same per-kernel working dir, and both lib.rs (FFI) and this
// test pull from it via `include!`. Do NOT duplicate kernel code here.

use cutile::compile_api::KernelCompiler;
use cutile_compiler::specialization::{DivHint, SpecializationBits};

// ─── single source of truth — pull kernel.rs (the sibling in this same
// ─── per-kernel working dir) into this test ─────────────
include!("kernel.rs");

// `cutile-macros` injects this `__module_ast_self` symbol inside every
// `#[cutile::module]` block. Importing it gives KernelCompiler a handle to
// the module's AST without re-declaring anything.
use softmax_module::__module_ast_self;

// Pick ONE concrete generic specialization for the bytecode dump. Any
// reasonable tile size works for first-run development; the autotuner / FFI
// will JIT-compile other (BM, BN, LATENCY) specializations at runtime.
const BM: i32 = 128;
const BN: i32 = 128;
const LATENCY: i32 = 4;

// Static stride hints: outer strides per pointer arg in element units.
// For row-major (M, N), outer stride is N (innermost stride 1 implicit).
const REP_N: i32 = 1024;
const Y_STRIDES: &[i32] = &[REP_N];
const X_STRIDES: &[i32] = &[REP_N];

const TARGET: &str = "sm_100";

// Alignment hints so the DUMP carries the same `assume div_by<N>` lines the
// reference IR does. Read `reference/reference.mlir` (`grep "assume div_by"`)
// and reproduce the per-arg divisor EXACTLY — do not blanket-fill 16; if the
// reference asserts div_by<8> on a dim, use a value whose divisor is 8 there.
// shape_div/stride_div are in ELEMENTS, base_ptr_div in BYTES; DivHint::from_value
// computes the largest power-of-2 divisor (clamped to 16). A pipeline test with
// ONLY .strides() dumps 0 assumes vs the reference's many → noisy Agent C diff.
fn spec_bits_1d(rep_n: i32) -> SpecializationBits {
    // softmax view is row-major [M, N]; here N=rep_n is the inner contiguous dim.
    SpecializationBits {
        shape_div: vec![DivHint::from_value(rep_n)], // N divisible by 16 (1024 → 16)
        stride_div: vec![DivHint::from_value(1)],    // innermost stride = 1
        stride_one: vec![true],
        base_ptr_div: DivHint::from_ptr(0x1000), // 16-byte aligned base ptr
        elements_disjoint: true,
    }
}

fn compile() -> Vec<u8> {
    let artifacts = KernelCompiler::new(__module_ast_self, "softmax_module", "softmax_kernel")
        // Generics in the SAME ORDER as the entry signature:
        //   <E: ElementType, const BM, const BN, const LATENCY>
        .generics(vec![
            "f32".to_string(), // E (data dtype)
            BM.to_string(),
            BN.to_string(),
            LATENCY.to_string(),
        ])
        .strides(&[("y_ptr", Y_STRIDES), ("x_ptr", X_STRIDES)])
        // Reproduce the reference's `assume div_by<N>` (see spec_bits_1d above).
        // Names MUST match the entry's pointer args. Omit an arg here only if
        // the reference shows NO assume for it.
        .spec_args(&[
            ("y_ptr", spec_bits_1d(REP_N)),
            ("x_ptr", spec_bits_1d(REP_N)),
        ])
        .target(TARGET)
        .compile()
        .expect(
            "softmax_kernel compile failed — check kernel.rs against the \
                 compile-fix-loop table in agents/agent_b.md",
        );

    artifacts.bytecode().expect("bytecode serialization failed")
}

fn write_bytecode(bc: &[u8]) {
    // NOTE (path): this template's include! and the path math below still
    // reference the old `cutile-kernels/` include layout — the migrated layout
    // wires kernel.rs into the aggregated cutile_kernels crate via
    // `mod softmax { include!("../../softmax_kernel/kernel.rs"); ... }` and dumps
    // generated IR under ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/. Only the
    // path plumbing changed; the compile/spec-bits logic is unchanged. (Rust
    // path code left as-is per the migration scope — comments only.)
    let path = format!(
        "{}/cutile-kernels/softmax/generated/generated.tileirbc",
        env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/cutile")
            .unwrap_or(env!("CARGO_MANIFEST_DIR")),
    );
    std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).ok();
    std::fs::write(&path, bc).expect("failed to write bytecode file");
    println!("✅ softmax kernel compiled. Bytecode written to {path}");
    println!("   Next step (shell):");
    println!("     cuda-tile-translate --cudatilebc-to-mlir {path} \\");
    println!("       -o ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated.mlir");
    println!("     $CUDA_TILE_OPT_BIN --canonicalize --cse \\");
    println!("       ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated.mlir \\");
    println!("       -o ${CUTILE_KERNEL_OUT_ROOT}/softmax/generated/generated_canon.mlir");
}

#[test]
fn softmax_compile_default() {
    let bc = compile();
    assert!(
        !bc.is_empty(),
        "bytecode is empty — compile produced nothing"
    );
    write_bytecode(&bc);
}
