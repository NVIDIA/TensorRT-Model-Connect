# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video-owned diffusion video contract plugin."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.contracts import CompareResult, MetricResult


def _contract_config(case):
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


def _make_pass(stage_name: str, metrics, rule: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="LTX-Video diffusion video contract verified",
    )


def _make_fail(stage_name: str, metrics, rule: str, message: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message,
    )


def _returncode_metric(output) -> MetricResult:
    rc = int((output.data or {}).get("returncode", -1))
    return MetricResult(value=float(rc), threshold=0.0, operator="==", passed=rc == 0)


def _frame_dir_metric(name: str, output) -> MetricResult:
    frame_dir = (output.data or {}).get("frames_dir")
    present = bool(frame_dir) and Path(str(frame_dir)).is_dir()
    if present:
        present = bool(list(Path(str(frame_dir)).glob("frame_*.png")))
    return MetricResult(
        value=1.0 if present else 0.0,
        threshold=1.0,
        operator="==",
        passed=present,
        note=f"{name} frames_dir={frame_dir}",
    )


def _pixel_metrics(output, thresholds: dict[str, float]) -> dict[str, MetricResult]:
    stats = (output.data or {}).get("frame_stats") or {}
    if not isinstance(stats, dict) or not stats:
        return {}
    pixel_mean = float(stats.get("mean", 0.0))
    pixel_std = float(stats.get("std", 0.0))
    min_mean = thresholds.get("min_pixel_mean", 0.15)
    max_mean = thresholds.get("max_pixel_mean", 0.85)
    min_std = thresholds.get("min_pixel_std", 0.05)
    return {
        "pixel_mean_range": MetricResult(
            value=pixel_mean,
            threshold=None,
            operator="in_range",
            passed=min_mean <= pixel_mean <= max_mean,
            note=f"[{min_mean}, {max_mean}]",
        ),
        "pixel_std_min": MetricResult(
            value=pixel_std,
            threshold=min_std,
            operator=">=",
            passed=pixel_std >= min_std,
        ),
    }


class LtxVideoDiffusionVideoPlugin:
    reference_families = ["diffusers_video_gen"]
    user_contract = "diffusion_video"

    def configure_reference(self, case):
        config = _contract_config(case)
        config.setdefault("use_diffusers", True)
        config.setdefault("video_mode", True)
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name or "end_to_end"
        metrics = {
            "trt_returncode": _returncode_metric(trt_output),
            "reference_returncode": _returncode_metric(ref_output),
        }
        metrics.update(_pixel_metrics(trt_output, threshold.metrics))

        if stage in {"end_to_end", "end_to_end_video", "generate", "frame_quality"}:
            declared_frames = int(case.inputs.get("video_num_frames", 1) or 1)
            min_frames = int(threshold.metrics.get("contract_min_num_frames", declared_frames))
            trt_frames = int((trt_output.data or {}).get("num_frames", 0))
            ref_frames = int((ref_output.data or {}).get("num_frames", 0))
            metrics["trt_num_frames"] = MetricResult(
                value=float(trt_frames),
                threshold=float(min_frames),
                operator=">=",
                passed=trt_frames >= min_frames,
            )
            metrics["reference_num_frames"] = MetricResult(
                value=float(ref_frames),
                threshold=float(min_frames),
                operator=">=",
                passed=ref_frames >= min_frames,
            )
            metrics["trt_frames_dir_present"] = _frame_dir_metric("trt", trt_output)
            metrics["reference_frames_dir_present"] = _frame_dir_metric("reference", ref_output)

        passed = all(metric.passed for metric in metrics.values())
        rule = "returncodes are zero AND declared video frames exist AND pixel stats pass"
        if passed:
            return _make_pass(stage, metrics, rule)
        return _make_fail(stage, metrics, rule, "LTX-Video diffusion video contract failed")


plugin = LtxVideoDiffusionVideoPlugin()
