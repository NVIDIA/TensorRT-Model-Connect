# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image helpers of the family E2E that need no GPU, so that ``test_layout.py`` can pin them."""

from __future__ import annotations

import numpy as np


def to_png_bytes(image: np.ndarray) -> np.ndarray:
    """A float image in [0, 1] as the 8-bit values a PNG stores: clipped, rounded to nearest.

    An image with a NaN or an infinite value has no such values and raises ``ValueError``:
    clipping keeps a NaN, and what the cast to 8 bits makes of one is not defined.
    """
    if not np.isfinite(image).all():
        raise ValueError("an image with NaN or infinite pixels has no PNG values")
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


def _block_means(image: np.ndarray, block: int) -> np.ndarray:
    """The mean of every ``block`` x ``block`` tile of an ``[H, W, C]`` image, per channel."""
    height = image.shape[0] // block * block
    width = image.shape[1] // block * block
    return (
        image[:height, :width]
        .reshape(height // block, block, width // block, block, image.shape[2])
        .mean(axis=(1, 3))
    )


def layout_correlation(actual: np.ndarray, expected: np.ndarray, block: int) -> float:
    """Correlation of the block means of two ``[H, W, C]`` images: is the layout the same?

    Every channel is taken around its own mean over the blocks, so only variation across the
    image counts. Around one mean for all channels, the difference between the channels'
    averages would count too: both images share it when they have the same overall colour, and
    it can outweigh a layout that differs. Images whose block means are flat score 0.
    """

    def centred(image: np.ndarray) -> np.ndarray:
        low = _block_means(image.astype(np.float64), block)
        return (low - low.mean(axis=(0, 1), keepdims=True)).reshape(-1)

    low_actual, low_expected = centred(actual), centred(expected)
    denominator = float(np.linalg.norm(low_actual) * np.linalg.norm(low_expected))
    return float(np.dot(low_actual, low_expected) / denominator) if denominator > 0 else 0.0
