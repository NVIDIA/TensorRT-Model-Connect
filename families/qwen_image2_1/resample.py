# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Channel layout of the parameter-free upsampling shortcut in the Qwen-Image 2.1 VAE decoder."""

from __future__ import annotations

import numpy as np


def shortcut_channels(in_dim: int, out_dim: int, factor_t: int, factor_s: int = 2) -> np.ndarray:
    """Source channel of every upsampled sub-position in the parameter-free shortcut.

    The reference repeats each input channel ``out_dim * factor / in_dim`` times, splits the result
    into ``[out_dim, factor_t, factor_s, factor_s]`` and moves the three factors to the frame, row
    and column axes. A single input frame keeps only the last of its ``factor_t`` output frames.
    Returns ``[out_dim * factor_s * factor_s]`` indices ordered by output channel, row offset and
    column offset.
    """
    factor = factor_t * factor_s * factor_s
    if (out_dim * factor) % in_dim:
        raise ValueError("shortcut channel counts are not compatible")
    repeats = out_dim * factor // in_dim
    spatial = factor_s * factor_s
    repeated = np.arange(out_dim * factor, dtype=np.int64).reshape(out_dim, factor_t, spatial)
    return (repeated[:, factor_t - 1, :] // repeats).reshape(-1).astype(np.int32)
