# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from tilegym.backend import register_impl


@ct.kernel
def _layer_norm_fwd_kernel(
    x,  # (N, C, W)
    y,  # (N, C, W)
    w,  # (C,)
    b,  # (C,)
    mean,  # (N, W)
    rstd,  # (N, W)
    C: ct.Constant[int],
    W: ct.Constant[int],
    EPS: ct.Constant[float],
    WEIGHT_SHIFT: ct.Constant[float],
    BLOCK_SIZE_C: ct.Constant[int],
    BLOCK_SIZE_W: ct.Constant[int],
):
    """
    Grid(N, W // BLOCK_SIZE_W)
    Each block normalizes the (C, BLOCK_SIZE_W) channel column for one (row, W-tile).

    ``BLOCK_SIZE_C`` is capped by the host launcher (see CHANNEL_BLOCK_CAP), so
    for large C this loop runs multiple iterations, each reducing a bounded
    (BLOCK_SIZE_C, BLOCK_SIZE_W) tile and accumulating into ``_mean`` / ``_var``.
    This is an exact strided partial reduction over the channel axis and keeps a
    single block from materialising the whole C run (register-spill cliff);
    the same cap protects the W==1 gather kernel below.

    The inputs are kept as their natural (N, C, W) / (N, W) / (C,) tensors and the
    per-block sub-tiles are read/written with block-indexed ``ct.load``/``ct.store``
    (TMA), so each block moves a contiguous sub-tile rather than addressing every
    element individually.

    Out-of-bounds handling is provided by the hardware instead of explicit masks:
      * ``padding_mode=ZERO`` on ``ct.load`` zeroes the [C, BLOCK_SIZE_C) rows and
        [W, BLOCK_SIZE_W) columns of the last tile, so they contribute 0 to the
        mean/variance reductions. Because the load indexes the genuine ``C`` axis of
        the 3-D tensor (not a flattened buffer), an OOB channel cannot alias the
        next row's data.
      * ``ct.store`` only writes the in-range extent, so OOB channel rows / W columns
        of the last tile are never written, with no offset redirection needed.
    """
    row = ct.bid(0)
    wtile = ct.bid(1)

    # compute mean over the channel axis C for each of the BLOCK_SIZE_W positions
    _mean = ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32)
    for col_start in range(0, C, BLOCK_SIZE_C):
        col_tile = col_start // BLOCK_SIZE_C
        x_tile = ct.load(
            x,
            index=(row, col_tile, wtile),
            shape=(1, BLOCK_SIZE_C, BLOCK_SIZE_W),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_tile = ct.astype(ct.reshape(x_tile, (BLOCK_SIZE_C, BLOCK_SIZE_W)), ct.float32)
        _mean = _mean + x_tile

    mean_val = ct.sum(_mean, axis=0) / C  # (BLOCK_SIZE_W,)
    ct.store(mean, index=(row, wtile), tile=ct.reshape(mean_val, (1, BLOCK_SIZE_W)), allow_tma=False)

    # compute variance
    mean_row = ct.reshape(mean_val, (1, BLOCK_SIZE_W))
    _var = ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32)
    for col_start in range(0, C, BLOCK_SIZE_C):
        col_tile = col_start // BLOCK_SIZE_C
        x_tile = ct.load(
            x,
            index=(row, col_tile, wtile),
            shape=(1, BLOCK_SIZE_C, BLOCK_SIZE_W),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_tile = ct.astype(ct.reshape(x_tile, (BLOCK_SIZE_C, BLOCK_SIZE_W)), ct.float32)
        # OOB rows/cols loaded as 0; subtracting the mean would make them nonzero,
        # so re-zero the OOB channel rows before the sum of squares. Only rows can
        # alias a real mean; guard the [C, BLOCK_SIZE_C) rows explicitly.
        col_offsets = col_start + ct.arange(BLOCK_SIZE_C, dtype=ct.int32)
        mask_C = ct.reshape(col_offsets < C, (BLOCK_SIZE_C, 1))
        x_centered = ct.where(mask_C, x_tile - mean_row, ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32))
        _var = _var + x_centered * x_centered

    var_val = ct.sum(_var, axis=0) / C  # (BLOCK_SIZE_W,)
    rstd_val = ct.rsqrt(var_val + EPS)
    ct.store(rstd, index=(row, wtile), tile=ct.reshape(rstd_val, (1, BLOCK_SIZE_W)), allow_tma=False)

    # normalization and affine transformation
    rstd_row = ct.reshape(rstd_val, (1, BLOCK_SIZE_W))
    for col_start in range(0, C, BLOCK_SIZE_C):
        col_tile = col_start // BLOCK_SIZE_C
        x_tile = ct.load(
            x,
            index=(row, col_tile, wtile),
            shape=(1, BLOCK_SIZE_C, BLOCK_SIZE_W),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_tile = ct.astype(ct.reshape(x_tile, (BLOCK_SIZE_C, BLOCK_SIZE_W)), ct.float32)

        w_tile = ct.load(w, index=(col_tile,), shape=(BLOCK_SIZE_C,), padding_mode=ct.PaddingMode.ZERO)
        w_tile = ct.reshape(ct.astype(w_tile, ct.float32), (BLOCK_SIZE_C, 1)) + WEIGHT_SHIFT
        b_tile = ct.load(b, index=(col_tile,), shape=(BLOCK_SIZE_C,), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.reshape(ct.astype(b_tile, ct.float32), (BLOCK_SIZE_C, 1))

        x_hat = (x_tile - mean_row) * rstd_row
        y_tile = x_hat * w_tile + b_tile
        y_tile = ct.astype(y_tile, x.dtype)
        ct.store(
            y,
            index=(row, col_tile, wtile),
            tile=ct.reshape(y_tile, (1, BLOCK_SIZE_C, BLOCK_SIZE_W)),
            allow_tma=False,
        )


@ct.kernel
def _layer_norm_fwd_kernel_gather(
    x,
    y,
    w,
    b,
    mean,
    rstd,
    STRIDE_N: ct.Constant[int],
    STRIDE_C: ct.Constant[int],
    STRIDE_W: ct.Constant[int],
    C: ct.Constant[int],
    W: ct.Constant[int],
    EPS: ct.Constant[float],
    WEIGHT_SHIFT: ct.Constant[float],
    BLOCK_SIZE_C: ct.Constant[int],
    BLOCK_SIZE_W: ct.Constant[int],
    N_ELEMENTS: ct.Constant[int],
    STAT_N_ELEMENTS: ct.Constant[int],
):
    """
    Grid(N, W // BLOCK_SIZE_W)

    Coalescing-friendly path, selected by the host when the innermost (W) extent
    is narrower than TMA_MIN_INNER_EXTENT. Flat gather/scatter addressing walks
    the contiguous C*W slab directly, so a BLOCK_SIZE_C channel run is fetched as
    wide accesses regardless of how small W is. The block-indexed ct.load tiles
    used by _layer_norm_fwd_kernel above have an innermost box extent of only W
    elements here, which serialises the channel reads.

    Each block gets (1, C, BLOCK_SIZE_W) input data, matching (C,) weights.
    Will use a for loop to access (1, BLOCK_SIZE_C, BLOCK_SIZE_W) data every iter.
    """
    row = ct.bid(0)
    tub_start = ct.bid(1) * BLOCK_SIZE_W

    # compute mean
    if BLOCK_SIZE_W == 1:
        _mean = ct.zeros((BLOCK_SIZE_C,), dtype=ct.float32)
    else:
        _mean = ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32)

    tub_offsets = tub_start + ct.arange(BLOCK_SIZE_W, dtype=ct.int32)
    mask_W = tub_offsets < W
    tub_offsets_strided = tub_offsets * STRIDE_W

    for col_start in range(0, C, BLOCK_SIZE_C):
        col_offsets = col_start + ct.arange(BLOCK_SIZE_C, dtype=ct.int32)
        mask_C = col_offsets < C

        if BLOCK_SIZE_W == 1:
            indices = row * STRIDE_N + col_offsets * STRIDE_C
            x_tile = ct.gather(x, indices, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            # Mask out elements beyond C (they point to next row's data!)
            x_tile = ct.where(mask_C, x_tile, ct.zeros((BLOCK_SIZE_C,), dtype=ct.float32))
            _mean = _mean + x_tile
        else:
            offsets = ct.reshape(col_offsets, (BLOCK_SIZE_C, 1)) * STRIDE_C + ct.reshape(
                tub_offsets_strided, (1, BLOCK_SIZE_W)
            )
            offsets = row * STRIDE_N + offsets
            mask = ct.bitwise_and(
                ct.reshape(mask_C, (BLOCK_SIZE_C, 1)),
                ct.reshape(mask_W, (1, BLOCK_SIZE_W)),
            )

            x_tile = ct.gather(x, offsets, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            # Mask out OOB C/W lanes: ct.gather returns the padding value only
            # when the computed offset is outside the buffer, but for OOB
            # col_offsets (>= C) the offset stride_c * col_offsets still lands
            # inside the buffer — in the next row's data — and gather returns
            # that real (wrong) value. Explicit mask mirrors what the
            # BLOCK_SIZE_W == 1 branch and the _var path already do.
            x_tile = ct.where(mask, x_tile, ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32))
            _mean = _mean + x_tile

    mean_val = ct.sum(_mean, axis=0) / C

    if BLOCK_SIZE_W == 1:
        mean_offsets = ct.full((1,), row * W, dtype=ct.int32)
        mean_val_reshaped = ct.reshape(mean_val, (1,))
        ct.scatter(mean, mean_offsets, mean_val_reshaped)
    else:
        mean_offsets = row * W + tub_offsets
        # Redirect OOB W lanes (tub_offsets >= W) to an out-of-bounds sentinel
        # so they don't corrupt the next row's mean slot in the flat (N*W,) buffer.
        safe_mean_offsets = ct.where(mask_W, mean_offsets, ct.full((BLOCK_SIZE_W,), STAT_N_ELEMENTS, dtype=ct.int32))
        ct.scatter(mean, safe_mean_offsets, mean_val)

    # compute std
    if BLOCK_SIZE_W == 1:
        _var = ct.zeros((BLOCK_SIZE_C,), dtype=ct.float32)
    else:
        _var = ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32)

    for col_start in range(0, C, BLOCK_SIZE_C):
        col_offsets = col_start + ct.arange(BLOCK_SIZE_C, dtype=ct.int32)
        mask_C = col_offsets < C

        if BLOCK_SIZE_W == 1:
            indices = row * STRIDE_N + col_offsets * STRIDE_C
            x_tile = ct.gather(x, indices, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            x_centered = ct.where(
                mask_C,
                x_tile - mean_val,
                ct.zeros((BLOCK_SIZE_C,), dtype=ct.float32),
            )
        else:
            offsets = ct.reshape(col_offsets, (BLOCK_SIZE_C, 1)) * STRIDE_C + ct.reshape(
                tub_offsets_strided, (1, BLOCK_SIZE_W)
            )
            offsets = row * STRIDE_N + offsets
            mask = ct.bitwise_and(
                ct.reshape(mask_C, (BLOCK_SIZE_C, 1)),
                ct.reshape(mask_W, (1, BLOCK_SIZE_W)),
            )

            x_tile = ct.gather(x, offsets, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            mean_val_reshaped = ct.reshape(mean_val, (1, BLOCK_SIZE_W))
            x_centered = ct.where(
                mask,
                x_tile - mean_val_reshaped,
                ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32),
            )

        _var = _var + x_centered * x_centered

    var_val = ct.sum(_var, axis=0) / C
    rstd_val = ct.rsqrt(var_val + EPS)

    if BLOCK_SIZE_W == 1:
        rstd_offsets = ct.full((1,), row * W, dtype=ct.int32)
        rstd_val_reshaped = ct.reshape(rstd_val, (1,))
        ct.scatter(rstd, rstd_offsets, rstd_val_reshaped)
    else:
        rstd_offsets = row * W + tub_offsets
        # Same OOB redirection as the mean scatter above.
        safe_rstd_offsets = ct.where(mask_W, rstd_offsets, ct.full((BLOCK_SIZE_W,), STAT_N_ELEMENTS, dtype=ct.int32))
        ct.scatter(rstd, safe_rstd_offsets, rstd_val)

    # normalization and affine transformation
    if BLOCK_SIZE_W != 1:
        mean_val = ct.reshape(mean_val, (1, BLOCK_SIZE_W))
        rstd_val = ct.reshape(rstd_val, (1, BLOCK_SIZE_W))

    for col_start in range(0, C, BLOCK_SIZE_C):
        col_offsets = col_start + ct.arange(BLOCK_SIZE_C, dtype=ct.int32)
        mask_C = col_offsets < C

        if BLOCK_SIZE_W == 1:
            indices = row * STRIDE_N + col_offsets * STRIDE_C
            x_tile = ct.gather(x, indices, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            w_tile = ct.gather(w, col_offsets, padding_value=0)
            w_tile = w_tile + WEIGHT_SHIFT
            b_tile = ct.gather(b, col_offsets, padding_value=0)
        else:
            offsets = ct.reshape(col_offsets, (BLOCK_SIZE_C, 1)) * STRIDE_C + ct.reshape(
                tub_offsets_strided, (1, BLOCK_SIZE_W)
            )
            offsets = row * STRIDE_N + offsets
            mask = ct.bitwise_and(
                ct.reshape(mask_C, (BLOCK_SIZE_C, 1)),
                ct.reshape(mask_W, (1, BLOCK_SIZE_W)),
            )

            x_tile = ct.gather(x, offsets, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            w_tile = ct.gather(w, col_offsets, padding_value=0)
            w_tile = ct.reshape(w_tile, (BLOCK_SIZE_C, 1))
            w_tile = w_tile + WEIGHT_SHIFT
            b_tile = ct.gather(b, col_offsets, padding_value=0)
            b_tile = ct.reshape(b_tile, (BLOCK_SIZE_C, 1))

        x_hat = (x_tile - mean_val) * rstd_val
        y_tile = x_hat * w_tile + b_tile
        y_tile = ct.astype(y_tile, x.dtype)

        if BLOCK_SIZE_W == 1:
            indices = row * STRIDE_N + col_offsets * STRIDE_C
            # Redirect invalid indices to out-of-bounds (n_elements) to prevent race conditions
            safe_indices = ct.where(mask_C, indices, ct.full((BLOCK_SIZE_C,), N_ELEMENTS, dtype=ct.int32))
            ct.scatter(y, safe_indices, y_tile)
        else:
            # Redirect invalid indices to out-of-bounds (n_elements)
            safe_offsets = ct.where(mask, offsets, ct.full((BLOCK_SIZE_C, BLOCK_SIZE_W), N_ELEMENTS, dtype=ct.int32))
            ct.scatter(y, safe_offsets, y_tile)


def _squash_axis(x, start_dim, end_dim):
    """
    Squashes x to shape (N, C, W) where C are axes from start_dim to end_dim.
    """
    shape = x.shape
    # correct negative indexing
    if start_dim < 0:
        start_dim += len(shape)
    if end_dim < 0:
        end_dim += len(shape)
    assert start_dim < end_dim

    # squash N
    N = 1
    for i in range(start_dim):
        N *= shape[i]
    # squash C
    C = 1
    for i in range(start_dim, end_dim):
        C *= shape[i]

    return x.view(N, C, -1)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, start_dim, end_dim, weight, bias, eps, weight_shift=0.0):
        # The kernel reads/writes per-block sub-tiles with block-indexed
        # ct.load/ct.store on the natural (N, C, W) tensors. Force contiguous
        # layouts so the (N, C, W) view is canonical (stride triple C*W, W, 1)
        # and matches the block indexing.
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        x_squashed = _squash_axis(x, start_dim, end_dim)
        N, C, W = x_squashed.shape
        # y shares x's memory layout; view it with the same squashed (N, C, W)
        # shape so ct.store block indices line up with the x load indices.
        y_squashed = _squash_axis(y, start_dim, end_dim)

        mean = torch.empty((N, W), dtype=torch.float32, device="cuda")
        rstd = torch.empty((N, W), dtype=torch.float32, device="cuda")

        def next_power_of_2(n):
            return 1 if n == 0 else 2 ** (n - 1).bit_length()

        BLOCK_SIZE_W = min(1024, next_power_of_2(W))
        MAX_FUSED_SIZE = 65536 // BLOCK_SIZE_W // x.element_size()

        # Which addressing wins is decided by the innermost (W) extent of the
        # per-block tile, not by any single W value. The block-indexed
        # ct.load/ct.store tiles of _layer_norm_fwd_kernel are (1, BLOCK_SIZE_C,
        # BLOCK_SIZE_W), but the underlying (N, C, W) tensor only has W
        # contiguous elements per channel, so the box can never fetch more than
        # W elements per row. Once W drops below a warp-quarter the channel
        # reads serialise and the whole kernel collapses.
        #
        # Measurement puts the turn at 16 *elements* in both dtypes, not at a
        # fixed byte count, so gate on the element extent. W == 1 is simply the
        # far end of this curve; W = 2/4/8 regress just as badly and must take
        # the same path.
        TMA_MIN_INNER_EXTENT = 16
        use_tiles = W >= TMA_MIN_INNER_EXTENT

        if use_tiles:
            # Per-block channel-tile cap for the tiled path. Without it,
            # BLOCK_SIZE_C = min(MAX_FUSED_SIZE, next_power_of_2(C)) lets a
            # single block materialise the whole C run in one tile and the f32
            # working set spills.
            # Each kernel's ``for col_start in range(0, C, BLOCK_SIZE_C)`` loop
            # then performs an EXACT strided partial reduction over the channel
            # axis (every channel is still summed, none dropped), so the cap is
            # numerics-preserving for arbitrary C. It only binds for large C;
            # the NCHW C=32 workload keeps BLOCK_SIZE_C == C and is unaffected.
            CHANNEL_BLOCK_CAP = 1024
            BLOCK_SIZE_C = min(MAX_FUSED_SIZE, next_power_of_2(C), CHANNEL_BLOCK_CAP)
        else:
            # The gather path is not register-bound the same way and the cap
            # is a net loss for it.
            BLOCK_SIZE_C = min(MAX_FUSED_SIZE, next_power_of_2(C))
        num_warps = min(max(BLOCK_SIZE_C // 256, 1), 8)

        if use_tiles:
            grid = (N, (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _layer_norm_fwd_kernel,
                (
                    x_squashed,
                    y_squashed,
                    weight,
                    bias,
                    mean,
                    rstd,
                    C,
                    W,
                    eps,
                    weight_shift,
                    BLOCK_SIZE_C,
                    BLOCK_SIZE_W,
                ),
            )
        else:
            stride_n, stride_c, stride_w = x_squashed.stride()
            x_flat = x_squashed.reshape(-1)
            y_flat = y_squashed.reshape(-1)
            mean_flat = mean.reshape(-1)
            rstd_flat = rstd.reshape(-1)
            # n_elements / stat_n_elements are the OOB sentinels the gather
            # kernel redirects masked-off scatter lanes to. y_flat holds N*C*W
            # elements, mean/rstd only N*W, so they need separate sentinels.
            n_elements = y_flat.numel()
            stat_n_elements = mean_flat.numel()
            grid = (N, (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _layer_norm_fwd_kernel_gather,
                (
                    x_flat,
                    y_flat,
                    weight,
                    bias,
                    mean_flat,
                    rstd_flat,
                    stride_n,
                    stride_c,
                    stride_w,
                    C,
                    W,
                    eps,
                    weight_shift,
                    BLOCK_SIZE_C,
                    BLOCK_SIZE_W,
                    n_elements,
                    stat_n_elements,
                ),
            )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.BLOCK_SIZE_C = BLOCK_SIZE_C
        ctx.BLOCK_SIZE_W = BLOCK_SIZE_W
        ctx.num_warps = num_warps
        ctx.weight_shift = weight_shift

        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("LayerNorm backward is not implemented for this backend")


@register_impl("layer_norm", backend="cutile")
def layer_norm(x, start_dim, end_dim, weight, bias, eps, weight_shift=0.0, **kwargs):
    r"""
    Returns the LayerNorm of input. Normalization is performed starting from ``start_dim``
    and ending with ``end_dim`` (non inclusive).

    Args:
        input: Tensor of shape (\*, C1, ..., Ck, \*)
            where C1 is at start_dim and Ck is at end_dim-1
        start_dim: integer value indicating start of normalized dimension
        end_dim: integer value indicating end of normalized dimension
        weight: Tensor of shape (C1, ..., Ck)
        bias: Tensor of shape (C1, ..., Ck)
        eps: small scaler to be added to
            variance calculation prior to division.
        weight_shift: float value to be added to the weight
        **kwargs: Additional arguments for backend-specific configurations
    """
    return _LayerNorm.apply(x, start_dim, end_dim, weight, bias, eps, weight_shift)
