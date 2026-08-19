# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned diffusion image contract plugin."""

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
        message="Flux diffusion image contract verified",
    )


def _make_fail(stage_name: str, metrics, rule: str, message: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message,
    )


def _returncode_metric(name: str, output) -> MetricResult:
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


def _minimum_pairwise_pixel_mae(frames_dir: str) -> float | None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if len(frame_paths) < 2:
        return None
    frames = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        for path in frame_paths
    ]
    return min(
        float(np.mean(np.abs(frames[left] - frames[right])))
        for left in range(len(frames))
        for right in range(left + 1, len(frames))
    )


class FluxDiffusionImagePlugin:
    reference_families = ["diffusers_image_gen"]
    user_contract = "diffusion_image"

    def configure_reference(self, case):
        config = _contract_config(case)
        config.setdefault("use_diffusers", True)
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name or "end_to_end"
        metrics = {
            "trt_returncode": _returncode_metric("trt", trt_output),
            "reference_returncode": _returncode_metric("reference", ref_output),
        }
        metrics.update(_pixel_metrics(trt_output, threshold.metrics))

        final_stages = {"end_to_end", "end_to_end_video", "generate", "frame_quality"}
        if stage in final_stages:
            expected_batch_size = int(case.inputs.get("expected_batch_size", 1))
            min_frames = int(threshold.metrics.get(
                "contract_min_num_frames", expected_batch_size))
            trt_frames = int((trt_output.data or {}).get("num_frames", 0))
            ref_frames = int((ref_output.data or {}).get("num_frames", 0))
            exact_batch = expected_batch_size > 1
            metrics["trt_num_frames"] = MetricResult(
                value=float(trt_frames),
                threshold=float(expected_batch_size if exact_batch else min_frames),
                operator="==" if exact_batch else ">=",
                passed=(trt_frames == expected_batch_size if exact_batch
                        else trt_frames >= min_frames),
            )
            metrics["reference_num_frames"] = MetricResult(
                value=float(ref_frames),
                threshold=float(expected_batch_size if exact_batch else 1),
                operator="==" if exact_batch else ">=",
                passed=(ref_frames == expected_batch_size if exact_batch
                        else ref_frames >= 1),
            )
            metrics["trt_frames_dir_present"] = _frame_dir_metric("trt", trt_output)
            metrics["reference_frames_dir_present"] = _frame_dir_metric("reference", ref_output)
            if exact_batch:
                frames_dir = str((trt_output.data or {}).get("frames_dir", ""))
                pairwise_mae = _minimum_pairwise_pixel_mae(frames_dir)
                min_mae = float(threshold.metrics.get(
                    "batch_min_pairwise_pixel_mae", 0.001))
                metrics["batch_pairwise_pixel_mae"] = MetricResult(
                    value=float(pairwise_mae or 0.0),
                    threshold=min_mae,
                    operator=">=",
                    passed=pairwise_mae is not None and pairwise_mae >= min_mae,
                    note="distinct prompts use the same seed",
                )

        passed = all(metric.passed for metric in metrics.values())
        rule = (
            "returncodes are zero AND expected frame count exists AND "
            "batch outputs are distinct AND pixel stats pass"
        )
        if passed:
            return _make_pass(stage, metrics, rule)
        return _make_fail(stage, metrics, rule, "Flux diffusion image contract failed")


plugin = FluxDiffusionImagePlugin()
