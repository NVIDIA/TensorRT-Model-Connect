# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision patch budget of a bundle built for a declared image size.

``config`` is a ``GlmOcrConfig`` (or any object with the same ``processor`` and
``max_image_patches`` attributes).
"""

from __future__ import annotations

import math


def smart_resize(height: int, width: int, config) -> tuple[int, int] | None:
    """The processor's resize rule as the runtime implements it; None for a refused aspect ratio."""
    p = config.processor
    factor = p.patch_size * p.merge_size
    frames = p.temporal_patch_size
    h, w = height, width
    if h < factor or w < factor:
        scale = max(factor / h, factor / w)
        h, w = int(h * scale), int(w * scale)
    if min(h, w) <= 0 or max(h, w) / min(h, w) > 200:
        return None
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if frames * h_bar * w_bar > p.max_pixels:
        beta = math.sqrt(frames * h * w / p.max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif frames * h_bar * w_bar < p.min_pixels:
        beta = math.sqrt(p.min_pixels / (frames * h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def patch_budget(height: int, width: int, config) -> int:
    """The most vision patches any image of at most ``height x width`` pixels resizes to.

    Without enlargement the patch count grows with the size, so ``height x width`` itself
    bounds it. The resize enlarges an image with a side below the merge factor, or with fewer
    pixels than ``min_pixels`` after rounding; such an image can need more patches than the
    declared size, so every size in that region is checked.
    """
    p = config.processor
    factor = p.patch_size * p.merge_size
    cell = p.temporal_patch_size * factor * factor  # pixels of one rounded factor x factor cell

    def patches(h: int, w: int) -> int:
        size = smart_resize(h, w, config)
        return 0 if size is None else (size[0] // p.patch_size) * (size[1] // p.patch_size)

    # Neither enlargement applies: rounding and the max_pixels reduction never exceed this.
    budget = math.ceil(height / factor) * math.ceil(width / factor) * p.merge_size * p.merge_size
    # A side below the factor: the image is first scaled up until both sides reach it.
    for h in range(1, min(height, factor - 1) + 1):
        for w in range(1, width + 1):
            budget = max(budget, patches(h, w))
    for w in range(1, min(width, factor - 1) + 1):
        for h in range(factor, height + 1):
            budget = max(budget, patches(h, w))
    # Fewer than min_pixels after rounding: the image is scaled up to at least min_pixels.
    for h in range(factor, height + 1):
        if round(h / factor) * cell >= p.min_pixels:
            break
        for w in range(factor, width + 1):
            if round(h / factor) * round(w / factor) * cell >= p.min_pixels:
                break
            budget = max(budget, patches(h, w))
    return min(budget, config.max_image_patches)
