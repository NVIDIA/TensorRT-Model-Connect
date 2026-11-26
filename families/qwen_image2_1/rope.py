# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary position tables of the Qwen-Image 2.1 transformer and of its Qwen3-VL text encoder.

Transformer
-----------

The joint sequence is the text tokens followed by the target image's latent tokens. Each attention
head rotates ``head_dim / 2`` complex pairs, split over three axes (frame, height, width):

* a text token at index ``i`` sits at position ``i`` on all three axes;
* every image token shares one frame position, the number of text tokens before the image, and
  lies on a height / width grid centred on zero.

Only the image's frame position depends on the prompt; the denoiser engine computes that part from
its ``text_length`` input and takes everything else from the constants built here. All arithmetic
is float32, as in the reference (``torch.outer`` of positions and float32 inverse frequencies).
"""

from __future__ import annotations

import numpy as np

ROPE_THETA = 10000.0
# The reference precomputes positions [0, 8192) and [-1024, 0).
MAX_POSITIVE_POSITION = 8192
MAX_NEGATIVE_POSITION = 1024


def axis_inverse_frequencies(axis_dim: int) -> np.ndarray:
    """``1 / theta^(2j / axis_dim)`` for ``j`` in ``[0, axis_dim / 2)``, float32."""
    if axis_dim <= 0 or axis_dim % 2:
        raise ValueError(f"rotary axis dimension must be positive and even, got {axis_dim}")
    exponent = np.arange(0, axis_dim, 2, dtype=np.float32) / np.float32(axis_dim)
    return (np.float32(1.0) / np.power(np.float32(ROPE_THETA), exponent)).astype(np.float32)


def _angles(positions: np.ndarray, axis_dim: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size and (
        positions.max() >= MAX_POSITIVE_POSITION or positions.min() < -MAX_NEGATIVE_POSITION
    ):
        raise ValueError("rotary position is outside the range the model defines")
    return np.outer(positions.astype(np.float32), axis_inverse_frequencies(axis_dim)).astype(
        np.float32
    )


def centred_positions(extent: int) -> np.ndarray:
    """Grid positions ``-(extent - extent // 2) .. extent // 2 - 1``."""
    return np.arange(-(extent - extent // 2), extent // 2, dtype=np.int64)


def text_angles(text_tokens: int, axes_dim: tuple[int, int, int]) -> np.ndarray:
    """Angles ``[text_tokens, head_dim / 2]`` for text slots ``0 .. text_tokens - 1``."""
    positions = np.arange(text_tokens, dtype=np.int64)
    return np.concatenate([_angles(positions, axis_dim) for axis_dim in axes_dim], axis=1)


def image_grid_angles(height: int, width: int, axes_dim: tuple[int, int, int]) -> np.ndarray:
    """Height and width angles ``[height * width, (axes_dim[1] + axes_dim[2]) / 2]``, row-major."""
    row_positions = np.repeat(centred_positions(height), width)
    column_positions = np.tile(centred_positions(width), height)
    return np.concatenate(
        [_angles(row_positions, axes_dim[1]), _angles(column_positions, axes_dim[2])], axis=1
    )


def rotate_half_tables(sequence: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Text encoder tables: ``cos`` / ``sin`` of ``position * theta^(-2j / head_dim)``.

    Each is ``[sequence, head_dim / 2]`` float32. For text-only input the three multimodal position
    ids of a token are equal, so Qwen3-VL's interleaved MRoPE is plain RoPE on the token index.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"rotary head dimension must be positive and even, got {head_dim}")
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    inverse = (np.float32(1.0) / np.power(np.float32(theta), exponent)).astype(np.float32)
    angles = np.outer(np.arange(sequence, dtype=np.float32), inverse).astype(np.float32)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
