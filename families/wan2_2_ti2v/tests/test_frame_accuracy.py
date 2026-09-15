# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frame comparator accepts both producers without losing sequence checks."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from .frame_accuracy import compare_png_sequences


_REFERENCE_NAMES = [f"frame_{index:04d}.png" for index in range(3)]
_NATIVE_NAMES = [f"frame-{index:06d}.png" for index in range(3)]


def _write_frames(root: Path, names: list[str]) -> list[str]:
    root.mkdir()
    paths = []
    for index, name in enumerate(names):
        path = root / name
        pixels = np.full((2, 3, 3), 20 * (2**index), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("reference_names", [_REFERENCE_NAMES, _NATIVE_NAMES])
@pytest.mark.parametrize("actual_names", [_REFERENCE_NAMES, _NATIVE_NAMES])
def test_producer_filenames_preserve_identical_frame_metrics(
    tmp_path: Path, reference_names: list[str], actual_names: list[str]
) -> None:
    references = _write_frames(tmp_path / "reference", reference_names)
    actuals = _write_frames(tmp_path / "native", actual_names)

    assert compare_png_sequences(references, actuals) == pytest.approx(
        {
            "frame_count": 3.0,
            "cosine_uint8": 1.0,
            "minimum_frame_cosine_uint8": 1.0,
            "rmse_uint8": 0.0,
            "maximum_frame_rmse_uint8": 0.0,
            "reference_temporal_mae_uint8": 30.0,
            "trt_temporal_mae_uint8": 30.0,
            "reference_active_transition_fraction": 1.0,
            "trt_active_transition_fraction": 1.0,
            "temporal_motion_ratio": 1.0,
            "temporal_profile_correlation": 1.0,
        }
    )


def test_native_filenames_still_compare_the_last_frame(tmp_path: Path) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", _NATIVE_NAMES)
    Image.fromarray(np.full((2, 3, 3), 100, dtype=np.uint8)).save(actuals[-1])

    metrics = compare_png_sequences(references, actuals)

    assert metrics["frame_count"] == 3.0
    assert metrics["maximum_frame_rmse_uint8"] == 20.0
    assert metrics["rmse_uint8"] == pytest.approx(20 / np.sqrt(3))
    assert metrics["temporal_motion_ratio"] == pytest.approx(4 / 3)


@pytest.mark.parametrize("names", [_REFERENCE_NAMES, _NATIVE_NAMES])
@pytest.mark.parametrize(
    "indices",
    [[0, 2], [1, 0, 2], [0, 0, 1], [1, 2]],
    ids=["missing-index", "out-of-order", "duplicate-index", "nonzero-start"],
)
def test_both_filename_formats_reject_invalid_sequences(
    tmp_path: Path, names: list[str], indices: list[int]
) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", names)

    with pytest.raises(ValueError, match="TensorRT frame list is not contiguous"):
        compare_png_sequences(references, [actuals[index] for index in indices])


@pytest.mark.parametrize(
    "names",
    [
        ["frame_0000.png", "frame-000001.png", "frame-000002.png"],
        ["frame-000000.png", "frame_0001.png", "frame_0002.png"],
        ["frame-0000.png", "frame-0001.png", "frame-0002.png"],
        ["frame_000000.png", "frame_000001.png", "frame_000002.png"],
    ],
    ids=["reference-then-native", "native-then-reference", "short-native", "long-reference"],
)
def test_filename_formats_cannot_be_mixed_or_reformatted(
    tmp_path: Path, names: list[str]
) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", names)

    with pytest.raises(ValueError, match="TensorRT frame list is not contiguous"):
        compare_png_sequences(references, actuals)


@pytest.mark.parametrize("names", [_REFERENCE_NAMES, _NATIVE_NAMES])
def test_contiguous_sequences_must_have_equal_counts(tmp_path: Path, names: list[str]) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", names[:2])

    with pytest.raises(ValueError, match="frame count mismatch: reference=3, TensorRT=2"):
        compare_png_sequences(references, actuals)


@pytest.mark.parametrize("names", [_REFERENCE_NAMES, _NATIVE_NAMES])
def test_contiguous_sequences_must_have_all_files(tmp_path: Path, names: list[str]) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", names)
    Path(actuals[1]).unlink()

    with pytest.raises(ValueError, match="TensorRT frame files are missing"):
        compare_png_sequences(references, actuals)


@pytest.mark.parametrize("empty_reference", [True, False])
def test_empty_sequences_are_rejected(tmp_path: Path, empty_reference: bool) -> None:
    references = _write_frames(tmp_path / "reference", _REFERENCE_NAMES)
    actuals = _write_frames(tmp_path / "native", _NATIVE_NAMES)
    label = "reference" if empty_reference else "TensorRT"

    with pytest.raises(ValueError, match=f"{label} frame list is empty"):
        compare_png_sequences([] if empty_reference else references, actuals if empty_reference else [])
