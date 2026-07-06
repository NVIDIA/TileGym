# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs ops package — wrappers self-register via @register_impl.

The import side-effect of each wrapper module runs
``@register_impl("<op>", backend="cutile-rs")``, making
``tilegym.set_backend("cutile-rs"); matmul_interface(a, b)`` route to the
cutile-rs FFI kernel.
"""

from tilegym.backend import is_backend_available

# Only import if cutile-rs backend is available
if is_backend_available("cutile-rs"):
    from . import attention_sink  # noqa: F401  (register dispatch entry)
    from . import bmm  # noqa: F401  (register dispatch entry)
    from . import matmul  # noqa: F401  (register dispatch entry)
    from . import silu_and_mul  # noqa: F401  (register dispatch entry)
    from . import swiglu  # noqa: F401  (register dispatch entry)
