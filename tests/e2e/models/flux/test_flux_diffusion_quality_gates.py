from __future__ import annotations

from tests.e2e.models.flux.e2e_plugins.comparators.diffusion import DiffusionComparator
from tests.e2e_harness.contracts import StageOutput, ThresholdProfile


def _threshold(**overrides: float) -> ThresholdProfile:
    metrics = {
        "min_pixel_mean": 0.15,
        "max_pixel_mean": 0.85,
        "min_pixel_std": 0.05,
        "reference_min_pixel_std_for_ratio": 0.08,
        "min_reference_std_ratio": 0.35,
    }
    metrics.update(overrides)
    return ThresholdProfile(
        task_strategy="diffusion_media_generation", metrics=metrics)


def test_flux_comparator_fails_when_trt_contrast_collapses_vs_reference():
    comparator = DiffusionComparator()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "frames_dir": "/tmp/frames",
            "num_frames": 17,
            "frame_stats": {"mean": 0.5462, "std": 0.0620},
        },
    )
    ref = StageOutput(
        stage_name="end_to_end",
        data={
            "frames_dir": "/tmp/ref_frames",
            "num_frames": 17,
            "frame_stats": {"mean": 0.5245, "std": 0.2509},
        },
    )

    result = comparator._compare_frames(trt, ref, _threshold().metrics)

    assert not result.passed
    assert result.metrics["pixel_std_min"].passed
    ratio = result.metrics["reference_pixel_std_ratio"]
    assert ratio.value < ratio.threshold
    assert not ratio.passed


def test_flux_comparator_uses_default_min_pixel_std_for_flat_images():
    comparator = DiffusionComparator()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 1,
            "frame_stats": {"mean": 0.4879, "std": 0.0193},
        },
    )
    ref = StageOutput(
        stage_name="end_to_end",
        data={
            "image_path": "/tmp/ref.png",
            "frame_stats": {"mean": 0.4305, "std": 0.3392},
        },
    )

    result = comparator._compare_frames(trt, ref, _threshold().metrics)

    assert not result.passed
    assert result.metrics["pixel_std_min"].threshold == 0.05
    assert not result.metrics["pixel_std_min"].passed


def test_comparator_fails_contrast_ratio_even_when_absolute_std_passes():
    comparator = DiffusionComparator()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 17,
            "frame_stats": {"mean": 0.5462, "std": 0.0620},
        },
    )
    ref = StageOutput(
        stage_name="end_to_end",
        data={"frame_stats": {"mean": 0.5245, "std": 0.2509}},
    )

    result = comparator._compare_frames(trt, ref, _threshold().metrics)

    assert not result.passed
    assert result.metrics["pixel_std_min"].passed
    ratio = result.metrics["reference_pixel_std_ratio"]
    assert ratio.value < ratio.threshold
    assert not ratio.passed
