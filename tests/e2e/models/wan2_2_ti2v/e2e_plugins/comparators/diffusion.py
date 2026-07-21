# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-profile functional comparator for Wan2.2 TI2V-5B video output."""

from __future__ import annotations

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, ThresholdProfile

_REQUIRED_BUNDLE_TOKENS = (
    "Runtime strategy:   diffusion_wan2_2_ti2v",
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
    "config.json",
)

_REQUIRED_FRAME_THRESHOLDS = (
    "exact_num_frames",
    "exact_video_width",
    "exact_video_height",
    "min_pixel_mean",
    "max_pixel_mean",
    "min_pixel_std",
)


def _metric(value: float, threshold: float, operator: str, passed: bool) -> MetricResult:
    return MetricResult(value=value, threshold=threshold, operator=operator, passed=passed)


class DiffusionComparator:
    @property
    def task_strategy(self) -> str:
        return "diffusion_media_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        del ref
        if stage.name == "bundle_contract":
            return self._compare_bundle_contract(trt, stage)

        missing = sorted(set(_REQUIRED_FRAME_THRESHOLDS) - set(threshold.metrics))
        if missing:
            return CompareResult(
                stage_name=stage.name,
                status="failed",
                composite_rule="all model-owned frame thresholds must be loaded",
                message=f"Wan2.2 TI2V threshold sidecar is incomplete: {missing}",
            )

        expected_frames = int(threshold.metrics["exact_num_frames"])
        expected_width = int(threshold.metrics["exact_video_width"])
        expected_height = int(threshold.metrics["exact_video_height"])
        min_mean = float(threshold.metrics["min_pixel_mean"])
        max_mean = float(threshold.metrics["max_pixel_mean"])
        min_std = float(threshold.metrics["min_pixel_std"])

        data = trt.data or {}
        stats = data.get("frame_stats") or {}
        returncode = int(data.get("returncode", -1))
        frame_count = int(data.get("num_frames", 0))
        width = int(stats.get("width", 0))
        height = int(stats.get("height", 0))
        dimensions_consistent = bool(stats.get("dimensions_consistent", False))
        pixel_mean = float(stats.get("mean", 0.0))
        pixel_std = float(stats.get("std", 0.0))

        metrics = {
            "returncode": _metric(float(returncode), 0.0, "==", returncode == 0),
            "exact_num_frames": _metric(
                float(frame_count), float(expected_frames), "==", frame_count == expected_frames
            ),
            "exact_video_width": _metric(
                float(width), float(expected_width), "==", width == expected_width
            ),
            "exact_video_height": _metric(
                float(height), float(expected_height), "==", height == expected_height
            ),
            "frame_dimensions_consistent": _metric(
                float(dimensions_consistent), 1.0, "==", dimensions_consistent
            ),
            "pixel_mean_min": _metric(pixel_mean, min_mean, ">=", pixel_mean >= min_mean),
            "pixel_mean_max": _metric(pixel_mean, max_mean, "<=", pixel_mean <= max_mean),
            "pixel_std_min": _metric(pixel_std, min_std, ">=", pixel_std >= min_std),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule=(
                f"native command succeeds AND output is exactly {expected_frames} "
                f"{expected_width}x{expected_height} frames AND pixels are non-degenerate"
            ),
            message=f"Wan2.2 TI2V fixed-profile invariant contract: {'PASS' if passed else 'FAIL'}",
        )

    def _compare_bundle_contract(
        self,
        trt: StageOutput,
        stage: StageSpec,
    ) -> CompareResult:
        data = trt.data or {}
        stdout = str(data.get("stdout", ""))
        returncode = int(data.get("returncode", -1))
        strict_probe = bool(data.get("strict_model_plugin_probe", False))
        metrics = {
            "returncode": _metric(float(returncode), 0.0, "==", returncode == 0),
            "strict_model_plugin_probe": _metric(float(strict_probe), 1.0, "==", strict_probe),
        }
        for token in _REQUIRED_BUNDLE_TOKENS:
            present = token in stdout
            metrics[f"stdout_contains_{token}"] = _metric(float(present), 1.0, "==", present)
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule=(
                "strict source-bound model DSO probe succeeds AND bundle declares the "
                "Wan2.2 runtime strategy and every required section"
            ),
            message=f"Wan2.2 TI2V L0 bundle/plugin contract: {'PASS' if passed else 'FAIL'}",
        )


plugin = DiffusionComparator()
