// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Single cdylib aggregating every cutile-rs kernel. Each op contributes a
// pure-.rs pair under ../<op>_kernel/ (kernel.rs = device module `<op>_module`,
// ffi.rs = `#[no_mangle] extern "C" fn cutile_<op>` using crate::ffi_util). Each
// op is wrapped in its own `mod <op>` so the per-op `use` statements don't
// collide at crate root; the `#[no_mangle]` symbol is still exported globally,
// producing one libcutile_kernels.so with every cutile_<op> symbol. To add an
// op: drop its <op>_kernel/{kernel.rs,ffi.rs} and add a `mod <op> { ... }` block.

// Shared FFI marshalling (TensorDesc + borrow_tensor + dtype helpers); every
// op's ffi.rs uses it via `crate::ffi_util::...`.
#[path = "../../ffi_util.rs"]
mod ffi_util;

mod matmul {
    include!("../../matmul_kernel/kernel.rs");
    include!("../../matmul_kernel/ffi.rs");
}

mod bmm {
    include!("../../bmm_kernel/kernel.rs");
    include!("../../bmm_kernel/ffi.rs");
}

mod silu_and_mul {
    include!("../../silu_and_mul_kernel/kernel.rs");
    include!("../../silu_and_mul_kernel/ffi.rs");
}

mod swiglu {
    include!("../../swiglu_kernel/kernel.rs");
    include!("../../swiglu_kernel/ffi.rs");
}

mod attention_sink {
    include!("../../attention_sink_kernel/kernel.rs");
    include!("../../attention_sink_kernel/ffi.rs");
}
