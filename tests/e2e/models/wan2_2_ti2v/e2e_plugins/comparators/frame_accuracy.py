# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming all-frame pixel metrics for Wan2.2 nightly qualification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


def _cosine(dot_product: int, reference_square_sum: int, actual_square_sum: int) -> float:
    if reference_square_sum == 0 or actual_square_sum == 0:
        return 1.0 if reference_square_sum == actual_square_sum else 0.0
    return dot_product / math.sqrt(reference_square_sum * actual_square_sum)


def _validated_paths(paths: Sequence[str], *, label: str) -> list[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError(f"{label} frame list is empty")
    expected_names = [f"frame_{index:04d}.png" for index in range(len(resolved))]
    names = [path.name for path in resolved]
    if names != expected_names:
        raise ValueError(
            f"{label} frame list is not contiguous: "
            f"expected={expected_names[:5]}, actual={names[:5]}"
        )
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise ValueError(f"{label} frame files are missing: {missing[:5]}")
    return resolved


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compare_png_sequences(
    reference_paths: Sequence[str],
    actual_paths: Sequence[str],
) -> dict[str, float]:
    """Compare every corresponding RGB PNG without retaining the video in memory."""

    references = _validated_paths(reference_paths, label="reference")
    actuals = _validated_paths(actual_paths, label="TensorRT")
    if len(references) != len(actuals):
        raise ValueError(
            f"Wan2.2 frame count mismatch: reference={len(references)}, TensorRT={len(actuals)}"
        )

    total_values = 0
    squared_error_sum = 0
    reference_square_sum = 0
    actual_square_sum = 0
    dot_product = 0
    minimum_frame_cosine = 1.0
    maximum_frame_rmse = 0.0

    for index, (reference_path, actual_path) in enumerate(zip(references, actuals)):
        reference = _load_rgb(reference_path)
        actual = _load_rgb(actual_path)
        if reference.shape != actual.shape:
            raise ValueError(
                f"Wan2.2 frame {index} shape mismatch: "
                f"reference={reference.shape}, TensorRT={actual.shape}"
            )

        reference_i64 = reference.astype(np.int64)
        actual_i64 = actual.astype(np.int64)
        difference = actual_i64 - reference_i64
        frame_squared_error = int(np.square(difference).sum(dtype=np.int64))
        frame_reference_square = int(np.square(reference_i64).sum(dtype=np.int64))
        frame_actual_square = int(np.square(actual_i64).sum(dtype=np.int64))
        frame_dot_product = int(np.multiply(reference_i64, actual_i64).sum(dtype=np.int64))

        total_values += int(difference.size)
        squared_error_sum += frame_squared_error
        reference_square_sum += frame_reference_square
        actual_square_sum += frame_actual_square
        dot_product += frame_dot_product
        minimum_frame_cosine = min(
            minimum_frame_cosine,
            _cosine(frame_dot_product, frame_reference_square, frame_actual_square),
        )
        maximum_frame_rmse = max(
            maximum_frame_rmse,
            math.sqrt(frame_squared_error / difference.size),
        )

    return {
        "frame_count": float(len(references)),
        "cosine_uint8": _cosine(dot_product, reference_square_sum, actual_square_sum),
        "minimum_frame_cosine_uint8": minimum_frame_cosine,
        "rmse_uint8": math.sqrt(squared_error_sum / total_values),
        "maximum_frame_rmse_uint8": maximum_frame_rmse,
    }
