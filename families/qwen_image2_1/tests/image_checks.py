# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image helpers of the family E2E that need no GPU, so that ``test_layout.py`` can pin them."""

from __future__ import annotations

import numpy as np


def to_png_bytes(image: np.ndarray) -> np.ndarray:
    """A float image in [0, 1] as the 8-bit values a PNG stores: clipped, rounded to nearest."""
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def png_levels(image: np.ndarray) -> np.ndarray:
    """A float image as it reads back from a PNG: every value a multiple of 1 / 255.

    The CLI's output is a PNG, the SDK and the reference pipeline return float pixels. Bringing
    all of them to these levels makes the comparisons measure the same quantity; comparing an
    image that went through the PNG with one that did not would add the rounding to one side.
    """
    return to_png_bytes(image).astype(np.float32) / np.float32(255.0)


def spatial_std(image: np.ndarray) -> float:
    """The largest standard deviation, over the pixels, of one channel of an ``[H, W, C]`` image.

    A flat image scores 0 whatever its colour. The standard deviation over all axes would also
    count the difference between channels, which a uniform colour other than grey has.
    """
    return float(image.astype(np.float64).std(axis=(0, 1)).max())
