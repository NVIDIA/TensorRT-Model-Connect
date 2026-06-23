"""Flux-owned diffusion CLIP gate tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.e2e.models.flux.e2e_plugins.comparators.diffusion import (
    DiffusionComparator,
)
from tests.e2e.models.flux.e2e_plugins.comparators.clip_metrics import ClipMetrics
from tests.e2e_harness.contracts import (
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

_MODULE = "tests.e2e.models.flux.e2e_plugins.comparators.clip_metrics"


def _write_dummy_png(path: Path) -> None:
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    w = h = 4
    raw = b"".join(b"\x00" + bytes([128, 128, 128] * w) for _ in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _frames_dir(tmp: Path, name: str, n: int = 1) -> Path:
    d = tmp / name
    d.mkdir()
    for i in range(n):
        _write_dummy_png(d / f"frame_{i:04d}.png")
    return d


def _trt_output(frames_dir: str, prompt: str | None, num_frames: int = 1):
    return StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": num_frames,
            "frame_stats": {"mean": 0.5, "std": 0.2},
            "frames_dir": frames_dir,
            "prompt": prompt,
        },
        text=None,
        timing_s=0.0,
        metadata={},
    )


def _ref_output(frames_dir: str):
    return StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 1,
            "frame_stats": {"mean": 0.5, "std": 0.2},
            "frames_dir": frames_dir,
        },
        text=None,
        timing_s=0.0,
        metadata={},
    )


def _threshold(overrides: dict | None = None):
    metrics = {
        "psnr": 5.0,
        "ssim": 0.1,
        "min_pixel_mean": 0.15,
        "max_pixel_mean": 0.85,
        "min_pixel_std": 0.05,
        "reference_min_pixel_std_for_ratio": 0.08,
        "min_reference_std_ratio": 0.35,
        "max_prompt_clipscore_drop": 3.0,
        "min_hf_prompt_clipscore": 20.0,
        "min_trt_hf_image_clip_cosine": 0.0,
    }
    if overrides:
        metrics.update(overrides)
    return ThresholdProfile(
        task_strategy="diffusion_media_generation",
        profile_name="default",
        metrics=metrics,
    )


def _stage():
    return StageSpec(name="end_to_end", required=True)


def _fake_clip(delta: float = 0.0, hf_score: float = 25.0, img_cos: float = 0.9):
    return ClipMetrics(
        trt_prompt_clipscore=hf_score + delta,
        hf_prompt_clipscore=hf_score,
        prompt_clipscore_delta=delta,
        trt_hf_image_clip_cosine=img_cos,
        prompt_truncated=False,
    )


def test_clip_metric_keys_present_in_result(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip()):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold(),
            _stage(),
        )

    assert "prompt_clipscore_delta" in result.metrics
    assert "trt_hf_image_clip_cosine" in result.metrics
    assert "hf_prompt_clipscore" in result.metrics
    assert "trt_prompt_clipscore" in result.metrics


def test_large_negative_delta_fails_gate(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip(delta=-10.0)):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold({"max_prompt_clipscore_drop": 3.0}),
            _stage(),
        )

    assert not result.metrics["prompt_clipscore_delta"].passed
    assert result.status == StageStatus.FAILED.value


def test_positive_delta_always_passes(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip(delta=5.0)):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold(),
            _stage(),
        )

    assert result.metrics["prompt_clipscore_delta"].passed


def test_hf_floor_fail_when_reference_broken(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip(hf_score=5.0)):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold({"min_hf_prompt_clipscore": 20.0}),
            _stage(),
        )

    assert not result.metrics["hf_prompt_clipscore"].passed
    assert result.status == StageStatus.FAILED.value


def test_img_cosine_report_only_when_threshold_zero(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip(img_cos=-0.5)):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold({"min_trt_hf_image_clip_cosine": 0.0}),
            _stage(),
        )

    assert result.metrics["trt_hf_image_clip_cosine"].passed


def test_img_cosine_active_when_threshold_positive(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip(img_cos=0.5)):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold({"min_trt_hf_image_clip_cosine": 0.8}),
            _stage(),
        )

    assert not result.metrics["trt_hf_image_clip_cosine"].passed
    assert result.status == StageStatus.FAILED.value


def test_clip_skipped_when_no_prompt(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    result = DiffusionComparator().compare(
        _trt_output(str(trt_dir), prompt=None),
        _ref_output(str(ref_dir)),
        _threshold(),
        _stage(),
    )

    assert "prompt_clipscore_delta" not in result.metrics
    assert result.status is not None


def test_trt_prompt_clipscore_is_diagnostic_only(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt")
    ref_dir = _frames_dir(tmp_path, "ref")

    with patch(f"{_MODULE}.compute_clip_metrics", return_value=_fake_clip()):
        result = DiffusionComparator().compare(
            _trt_output(str(trt_dir), "a cat"),
            _ref_output(str(ref_dir)),
            _threshold(),
            _stage(),
        )

    assert result.metrics["trt_prompt_clipscore"].threshold is None


def test_clip_skipped_for_video_models(tmp_path):
    trt_dir = _frames_dir(tmp_path, "trt", n=4)
    ref_dir = _frames_dir(tmp_path, "ref", n=4)

    result = DiffusionComparator().compare(
        _trt_output(str(trt_dir), "a cat", num_frames=4),
        _ref_output(str(ref_dir)),
        _threshold(),
        _stage(),
    )

    assert "prompt_clipscore_delta" not in result.metrics
    assert "trt_hf_image_clip_cosine" not in result.metrics
