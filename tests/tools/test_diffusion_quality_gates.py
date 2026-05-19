from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.comparators.diffusion import DiffusionComparator
from tests.e2e_harness.plugins.diffusion import DiffusionPlugin


def _case() -> E2ECase:
    return E2ECase(
        name="wan21-t2v-1.3b",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        family="wan_t2v",
        runtime_strategy="diffusion_wan",
        reference_family="diffusers_video_gen",
        user_contract="diffusion_video",
    )


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


def test_contract_fails_when_trt_contrast_collapses_vs_reference():
    plugin = DiffusionPlugin()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
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

    result = plugin.verify(trt, ref, _case(), _threshold())

    assert not result.passed
    assert result.metrics["pixel_std"].passed
    ratio = result.metrics["reference_pixel_std_ratio"]
    assert ratio.value < ratio.threshold
    assert not ratio.passed


def test_contract_uses_default_min_pixel_std_for_flat_images():
    plugin = DiffusionPlugin()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
            "image_path": "/tmp/frame.png",
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

    result = plugin.verify(trt, ref, _case(), _threshold())

    assert not result.passed
    assert result.metrics["pixel_std"].threshold == 0.05
    assert not result.metrics["pixel_std"].passed


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


def test_comparator_fails_when_frame_count_differs_from_reference():
    comparator = DiffusionComparator()
    trt = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 320,
            "frame_stats": {"mean": 0.5462, "std": 0.2509},
        },
    )
    ref = StageOutput(
        stage_name="end_to_end",
        data={
            "num_frames": 321,
            "frame_stats": {"mean": 0.5245, "std": 0.2509},
        },
    )

    result = comparator._compare_frames(trt, ref, _threshold().metrics)

    assert not result.passed
    frame_count = result.metrics["frame_count_match"]
    assert frame_count.value == 1.0
    assert frame_count.threshold == 0.0
    assert not frame_count.passed
    assert frame_count.note == "trt=320, ref=321"
