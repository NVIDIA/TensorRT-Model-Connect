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
_MANIFEST_POLICY = object()


def _frame_paths(root: Path, count: int) -> list[str]:
    root.mkdir()
    paths = []
    for index in range(count):
        path = root / f"frame_{index:04d}.png"
        path.touch()
        paths.append(str(path))
    return paths


def _compare_full_profile(
    references: list[str],
    actuals: list[str],
    *,
    native_acceptance: object = _MANIFEST_POLICY,
    invariant_only: bool = False,
):
    case = load_manifest(MODEL_DIR / "manifests/wan22-ti2v-5b.json")
    reference_data = (
        {"_invariant_only": True}
        if invariant_only
        else {
            "returncode": 0,
            "num_frames": 121,
            "frame_paths": references,
        }
    )
    if native_acceptance is _MANIFEST_POLICY:
        reference_data["native_acceptance"] = case.metadata["native_acceptance"]
    elif native_acceptance is not None:
        reference_data["native_acceptance"] = native_acceptance
    return DiffusionComparator().compare(
        StageOutput(
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
        ),
        StageOutput(stage_name="end_to_end", data=reference_data),
        ThresholdProfile(
            task_strategy="diffusion_media_generation",
            metrics=case.threshold_overrides,
        ),
        StageSpec(name="end_to_end"),
    )


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


def test_rejects_reference_without_temporal_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 2)
    actuals = _frame_paths(tmp_path / "frames", 2)
    pixels = {
        path: np.full((1, 1, 3), 100, dtype=np.uint8)
        for path in references + actuals
    }
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    with pytest.raises(ValueError, match="reference video has no temporal activity"):
        frame_accuracy.compare_png_sequences(references, actuals)


def test_compares_all_121_frame_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    loaded: list[str] = []
    pixels = {}
    for index, (reference, actual) in enumerate(zip(references, actuals)):
        value = 50 + (index * index) % 150
        frame = np.full((1, 1, 3), value, dtype=np.uint8)
        pixels[reference] = frame
        pixels[actual] = frame

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


def test_native_visual_acceptance_keeps_all_raw_pixel_metrics_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    pixels = {}
    for index, (reference, actual) in enumerate(zip(references, actuals)):
        value = 50 + (index * index) % 150
        pixels[reference] = np.asarray([[[value, 0, 0]]], dtype=np.uint8)
        pixels[actual] = np.asarray([[[0, value, 0]]], dtype=np.uint8)
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])
    accepted = _compare_full_profile(references, actuals)

    assert accepted.status == "passed"
    assert not accepted.metrics["cosine_uint8"].passed
    assert not accepted.metrics["maximum_frame_rmse_uint8"].passed
    assert "diagnostic under native visual semantic" in accepted.metrics["cosine_uint8"].note
    assert accepted.metrics["all_reference_frames_compared"].passed

    strict = _compare_full_profile(
        references, actuals, native_acceptance=None)
    assert strict.status == "failed"

    invariant_only = _compare_full_profile(
        references, actuals, invariant_only=True)
    assert invariant_only.status == "failed"
    assert "cannot be used with an invariant-only reference" in invariant_only.message

    invalid_policy = {
        **load_manifest(
            MODEL_DIR / "manifests/wan22-ti2v-5b.json"
        ).metadata["native_acceptance"],
        "requires_nightly_vlm": False,
    }
    invalid = _compare_full_profile(
        references,
        actuals,
        native_acceptance=invalid_policy,
    )
    assert invalid.status == "failed"
    assert "native_acceptance policy is invalid" in invalid.message


def test_native_visual_acceptance_rejects_frozen_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    pixels = {}
    for index, reference in enumerate(references):
        value = 50 + (index * index) % 150
        pixels[reference] = np.asarray([[[value, 0, 0]]], dtype=np.uint8)
    for actual in actuals:
        pixels[actual] = np.asarray([[[0, 100, 0]]], dtype=np.uint8)
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    result = _compare_full_profile(references, actuals)

    assert result.status == "failed"
    assert not result.metrics["trt_temporal_mae_uint8"].passed
    assert not result.metrics["trt_active_transition_fraction"].passed
    assert not result.metrics["temporal_motion_ratio_min"].passed


def test_native_visual_acceptance_rejects_wrong_temporal_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    reference_values = [(index * 73) % 256 for index in range(121)]
    swapped_indices = [
        index ^ 1 if index < 120 else index
        for index in range(121)
    ]
    pixels = {}
    for path, value in zip(references, reference_values):
        pixels[path] = np.full((1, 1, 3), value, dtype=np.uint8)
    for path, index in zip(actuals, swapped_indices):
        value = reference_values[index]
        pixels[path] = np.full((1, 1, 3), value, dtype=np.uint8)
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    result = _compare_full_profile(references, actuals)

    assert result.status == "failed"
    assert result.metrics["temporal_motion_ratio_min"].passed
    assert result.metrics["temporal_motion_ratio_max"].passed
    assert result.metrics["reference_active_transition_fraction"].passed
    assert result.metrics["trt_active_transition_fraction"].passed
    assert not result.metrics["temporal_profile_correlation"].passed


def test_all_frame_metrics_detect_repeated_frame_stutter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _frame_paths(tmp_path / "hf_frames", 121)
    actuals = _frame_paths(tmp_path / "frames", 121)
    reference_values = [(index * 73) % 256 for index in range(121)]
    pixels = {}
    for path, value in zip(references, reference_values):
        pixels[path] = np.full((1, 1, 3), value, dtype=np.uint8)
    for index, path in enumerate(actuals):
        pixels[path] = np.full(
            (1, 1, 3),
            reference_values[index // 2],
            dtype=np.uint8,
        )
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    metrics = frame_accuracy.compare_png_sequences(references, actuals)

    assert metrics["reference_active_transition_fraction"] > 0.9
    assert metrics["trt_active_transition_fraction"] == pytest.approx(0.5)


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
