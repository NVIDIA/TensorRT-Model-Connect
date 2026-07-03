# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from PIL import Image

from tests.e2e.models.flux.e2e_plugins.contract import FluxDiffusionImagePlugin
from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile


def _write_frame(path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color=color).save(path)


def _case() -> E2ECase:
    return E2ECase(
        name="flux-batch2",
        hf_id="example/flux",
        family="flux",
        runtime_strategy="diffusion_flux",
        inputs={"expected_batch_size": 2},
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="diffusion_media_generation",
        metrics={
            "min_pixel_mean": 0.0,
            "max_pixel_mean": 1.0,
            "min_pixel_std": 0.0,
            "batch_min_pairwise_pixel_mae": 0.01,
        },
    )


def _output(frames_dir) -> StageOutput:
    return StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 2,
            "frames_dir": str(frames_dir),
            "frame_stats": {"mean": 0.5, "std": 0.2},
        },
    )


def test_flux_batch_contract_accepts_two_distinct_outputs(tmp_path) -> None:
    trt_dir = tmp_path / "trt"
    ref_dir = tmp_path / "ref"
    trt_dir.mkdir()
    ref_dir.mkdir()
    _write_frame(trt_dir / "frame_0.png", (255, 0, 0))
    _write_frame(trt_dir / "frame_1.png", (0, 0, 255))
    _write_frame(ref_dir / "frame_0.png", (255, 0, 0))
    _write_frame(ref_dir / "frame_1.png", (0, 0, 255))

    result = FluxDiffusionImagePlugin().verify(
        _output(trt_dir), _output(ref_dir), _case(), _threshold())

    assert result.passed
    assert result.metrics["trt_num_frames"].value == 2
    assert result.metrics["batch_pairwise_pixel_mae"].passed


def test_flux_batch_contract_rejects_duplicate_outputs(tmp_path) -> None:
    trt_dir = tmp_path / "trt"
    ref_dir = tmp_path / "ref"
    trt_dir.mkdir()
    ref_dir.mkdir()
    for directory in (trt_dir, ref_dir):
        _write_frame(directory / "frame_0.png", (255, 0, 0))
        _write_frame(directory / "frame_1.png", (255, 0, 0))

    result = FluxDiffusionImagePlugin().verify(
        _output(trt_dir), _output(ref_dir), _case(), _threshold())

    assert not result.passed
    assert not result.metrics["batch_pairwise_pixel_mae"].passed
