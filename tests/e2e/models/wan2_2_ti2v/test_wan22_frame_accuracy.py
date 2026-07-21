# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed coverage for Wan2.2 all-frame Nightly accuracy metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.e2e.models.wan2_2_ti2v.e2e_plugins.comparators import frame_accuracy


def _frame_paths(root: Path, count: int) -> list[str]:
    root.mkdir()
    paths = []
    for index in range(count):
        path = root / f"frame_{index:04d}.png"
        path.touch()
        paths.append(str(path))
    return paths


def test_rejects_noncontiguous_frame_numbering(tmp_path: Path) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 2)
    actuals = _frame_paths(tmp_path / "frames", 2)
    renamed = Path(actuals[1]).with_name("frame_0002.png")
    Path(actuals[1]).rename(renamed)
    actuals[1] = str(renamed)

    with pytest.raises(ValueError, match="TensorRT frame list is not contiguous"):
        frame_accuracy.compare_png_sequences(references, actuals)


def test_rejects_unequal_frame_lists(tmp_path: Path) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 2)
    actuals = _frame_paths(tmp_path / "frames", 1)

    with pytest.raises(ValueError, match="reference=2, TensorRT=1"):
        frame_accuracy.compare_png_sequences(references, actuals)


def test_rejects_missing_trt_frame(tmp_path: Path) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 1)
    actuals = _frame_paths(tmp_path / "frames", 1)
    Path(actuals[0]).unlink()

    with pytest.raises(ValueError, match="TensorRT frame files are missing"):
        frame_accuracy.compare_png_sequences(references, actuals)


def test_rejects_late_frame_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    pixels = {path: np.zeros((1, 1, 3), dtype=np.uint8) for path in [*references, *actuals]}
    pixels[actuals[120]] = np.zeros((2, 1, 3), dtype=np.uint8)
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    with pytest.raises(ValueError, match="frame 120 shape mismatch"):
        frame_accuracy.compare_png_sequences(references, actuals)
