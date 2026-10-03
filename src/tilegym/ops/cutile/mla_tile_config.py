# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Shape-aware head-tile selection for the cuTile MLA decode kernels.

``TILE_H=16`` makes every QK/PV ``ct.mma`` an N=16 tall-skinny UTCHMMA and
leaves the tensor pipe at ~21% of its per-SM rate. Widening fixes the MMA
shape, but the optimum is shape-dependent, so the choice is a measured table
rather than a formula.
"""

from .utils import next_power_of_2

_DEFAULT_TILE_H = 16
_WIDE_TILE_H = 64
_KV_WIDE_THRESHOLD = 8192
_WIDE_CC_MAJOR = 10


def _ceil_div(a: int, b: int) -> int:
    return -(a // -b)


def _shape_can_widen(num_heads: int, s_kv: int, cc_major: int) -> bool:
    """Gates that depend only on the shape, not on how the KV is split."""
    return (
        cc_major == _WIDE_CC_MAJOR
        and s_kv >= _KV_WIDE_THRESHOLD
        and num_heads >= _WIDE_TILE_H
        and num_heads % _WIDE_TILE_H == 0  # never pad the head dimension
    )


def _wide_grid_is_big_enough(num_heads: int, batch: int, num_kv_splits: int, sm_count: int) -> bool:
    """The measured winner ran 128 CTAs on 148 SMs, so "fill a wave" is too strict."""
    wide_ctas = _ceil_div(num_heads, _WIDE_TILE_H) * batch * num_kv_splits
    return wide_ctas * 2 >= sm_count  # at least half a wave


def select_tile_h(
    num_heads: int,
    s_kv: int,
    batch: int,
    num_kv_splits: int,
    sm_count: int,
    cc_major: int,
) -> int:
    """Pick the head tile when the split count is already fixed.

    Pure integer arithmetic, no device queries, so it is testable without a GPU.
    ``num_kv_splits`` must be >= 1; it is 1 for the plain (non-split) kernel.
    """
    assert num_kv_splits >= 1, num_kv_splits
    if not _shape_can_widen(num_heads, s_kv, cc_major):
        return _DEFAULT_TILE_H
    if not _wide_grid_is_big_enough(num_heads, batch, num_kv_splits, sm_count):
        return _DEFAULT_TILE_H
    return _WIDE_TILE_H


def _wave_split_len(num_heads: int, s_kv: int, batch: int, tile_n: int, sm_count: int, tile_h: int) -> int:
    """Split length that fills roughly one wave for a given head tile."""
    target_splits = max(1, sm_count // (_ceil_div(num_heads, tile_h) * batch))
    return max(next_power_of_2(max(1, s_kv // target_splits)), tile_n)


def _default_split_len(s_kv: int, batch: int, tile_n: int, sm_count: int) -> int:
    """Split length for the default tile, kept as-is so unwidened shapes do not move.

    Not ``_wave_split_len(tile_h=16)``: this formula omits the head-tile factor
    from the grid, so the two diverge by roughly ``ceil(H / 16)``.
    """
    return max(next_power_of_2(s_kv // max(1, sm_count // batch)), tile_n)


def plan_split_kv(
    num_heads: int,
    s_kv: int,
    batch: int,
    tile_n: int,
    sm_count: int,
    cc_major: int,
    force_tile_h: "int | None" = None,
) -> "tuple[int, int]":
    """Choose ``(TILE_H, kv_len_per_split)`` together for the auto-split path.

    The grid is ``ceil(H / TILE_H) * B * NUM_KV_SPLITS``, so sizing splits
    before knowing the head tile aims at the wrong wave. ``force_tile_h`` pins
    the tile (a user override) and still sizes the split for it, so each tile
    gets the split length it was measured with.
    """
    if force_tile_h is not None:
        if force_tile_h <= 0 or num_heads % force_tile_h != 0:
            raise ValueError(f"unsupported TILE_H override: {force_tile_h}")
        # Forcing the default tile reproduces the unwidened path exactly, split
        # included; any other tile is sized for the grid it actually launches.
        if force_tile_h == _DEFAULT_TILE_H:
            return force_tile_h, _default_split_len(s_kv, batch, tile_n, sm_count)
        return force_tile_h, _wave_split_len(num_heads, s_kv, batch, tile_n, sm_count, force_tile_h)

    if _shape_can_widen(num_heads, s_kv, cc_major):
        # The wave check needs the split count, which is itself sized for the
        # wide tile, so it can only run once the wide split length is known.
        wide_kv_len = _wave_split_len(num_heads, s_kv, batch, tile_n, sm_count, _WIDE_TILE_H)
        if _wide_grid_is_big_enough(num_heads, batch, _ceil_div(s_kv, wide_kv_len), sm_count):
            return _WIDE_TILE_H, wide_kv_len
    return _DEFAULT_TILE_H, _default_split_len(s_kv, batch, tile_n, sm_count)
