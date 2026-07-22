# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed coverage for Wan2.2 all-frame Nightly accuracy metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.e2e.models.wan2_2_ti2v.e2e_plugins.comparators import frame_accuracy
from tests.e2e.models.wan2_2_ti2v.e2e_plugins.comparators.diffusion import (
    DiffusionComparator,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


MODEL_DIR = Path(__file__).resolve().parent


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


def test_compares_all_121_frame_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    loaded: list[str] = []
    pixels = {path: np.full((1, 1, 3), 100, dtype=np.uint8) for path in references + actuals}

    def load_rgb(path: Path) -> np.ndarray:
        loaded.append(str(path))
        return pixels[str(path)]

    monkeypatch.setattr(frame_accuracy, "_load_rgb", load_rgb)
    case = load_manifest(MODEL_DIR / "manifests/wan22-ti2v-5b.json")
    output = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 121,
            "frame_paths": actuals,
            "frame_stats": {
                "width": 1280,
                "height": 704,
                "dimensions_consistent": True,
                "mean": 0.5,
                "std": 0.2,
            },
        },
    )
    reference = StageOutput(
        stage_name="end_to_end",
        data={"returncode": 0, "num_frames": 121, "frame_paths": references},
    )
    comparator = DiffusionComparator()
    threshold = ThresholdProfile(
        task_strategy="diffusion_media_generation", metrics=case.threshold_overrides
    )
    stage = StageSpec(name="end_to_end")

    result = comparator.compare(output, reference, threshold, stage)
    assert result.status == "passed"
    assert result.metrics["all_reference_frames_compared"].value == 121
    assert loaded == [path for pair in zip(references, actuals) for path in pair]

    pixels[actuals[-1]] = np.asarray([[[200, 100, 100]]], dtype=np.uint8)
    result = comparator.compare(output, reference, threshold, stage)
    assert result.status == "failed"
    assert not result.metrics["maximum_frame_rmse_uint8"].passed


def test_l0_comparator_enforces_shape_and_threshold_sidecar() -> None:
    case = load_manifest(MODEL_DIR / "manifests/wan22-ti2v-5b-l0.json")
    output = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 5,
            "frame_stats": {
                "width": 672,
                "height": 384,
                "dimensions_consistent": True,
                "mean": 0.5,
                "std": 0.2,
            },
        },
    )
    reference = StageOutput(stage_name="end_to_end", data={"_invariant_only": True})
    comparator = DiffusionComparator()
    stage = StageSpec(name="end_to_end")
    threshold = ThresholdProfile(
        task_strategy="diffusion_media_generation", metrics=case.threshold_overrides
    )

    assert comparator.compare(output, reference, threshold, stage).status == "passed"
    output.data["frame_stats"]["width"] = 671
    assert comparator.compare(output, reference, threshold, stage).status == "failed"
    missing = comparator.compare(
        output,
        reference,
        ThresholdProfile(task_strategy="diffusion_media_generation", metrics={}),
        stage,
    )
    assert missing.status == "failed"
    assert "threshold sidecar is incomplete" in missing.message


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
