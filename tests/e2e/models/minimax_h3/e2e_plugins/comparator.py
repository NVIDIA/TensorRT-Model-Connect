# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed decoded-frame comparator for MiniMax-H3."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _normalized(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / np.iinfo(array.dtype).max
    return array.astype(np.float32)


def _decoded_metrics(
    reference_path: Path, candidate_path: Path
) -> tuple[tuple[int, ...], float, float, float]:
    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {reference.shape} != {candidate.shape}")

    squared_sum = 0.0
    absolute_sum = 0.0
    maximum = 0.0
    count = 0
    for index in range(reference.shape[0]):
        reference_frame = _normalized(reference[index])
        candidate_frame = _normalized(candidate[index])
        if not np.isfinite(candidate_frame).all():
            raise ValueError(f"candidate video contains non-finite pixels in frame {index}")
        error = candidate_frame - reference_frame
        squared_sum += float(np.square(error).sum(dtype=np.float64))
        absolute_sum += float(np.abs(error).sum(dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(error))))
        count += int(error.size)
    if count == 0:
        raise ValueError("decoded video contains no pixels")
    return tuple(reference.shape), squared_sum / count, absolute_sum / count, maximum


class MiniMaxH3DecodedVideoComparator:
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
        if int(trt.data.get("returncode", -1)) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Native MiniMax-H3 failed (rc={trt.data.get('returncode')})",
            )
        if int(ref.data.get("returncode", -1)) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"HF MiniMax-H3 reference failed (rc={ref.data.get('returncode')})",
            )
        if trt.data.get("source_revision") != ref.data.get("source_revision"):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="TRT and HF receipts do not identify the same source revision",
            )

        reference_path = Path(str(ref.data.get("frames_path", "")))
        candidate_path = Path(str(trt.data.get("frames_path", "")))
        if not reference_path.is_file() or not candidate_path.is_file():
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="MiniMax-H3 decoded frame arrays are missing",
            )

        metrics_config = threshold.metrics
        required = {
            "exact_num_frames",
            "exact_video_height",
            "exact_video_width",
            "maximum_mean_absolute_error",
            "minimum_psnr_db",
        }
        missing = sorted(required - metrics_config.keys())
        if missing:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"MiniMax-H3 threshold sidecar is missing {missing}",
            )

        shape, mse, mae, max_error = _decoded_metrics(reference_path, candidate_path)
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        expected_frames = int(metrics_config["exact_num_frames"])
        expected_height = int(metrics_config["exact_video_height"])
        expected_width = int(metrics_config["exact_video_width"])
        minimum_psnr = float(metrics_config["minimum_psnr_db"])
        maximum_mae = float(metrics_config["maximum_mean_absolute_error"])
        decoded_shape = shape[:3] if len(shape) >= 3 else shape
        frames = decoded_shape[0] if len(decoded_shape) > 0 else 0
        height = decoded_shape[1] if len(decoded_shape) > 1 else 0
        width = decoded_shape[2] if len(decoded_shape) > 2 else 0
        metrics = {
            "num_frames": MetricResult(
                value=float(frames),
                threshold=float(expected_frames),
                operator="==",
                passed=frames == expected_frames,
            ),
            "video_height": MetricResult(
                value=float(height),
                threshold=float(expected_height),
                operator="==",
                passed=height == expected_height,
            ),
            "video_width": MetricResult(
                value=float(width),
                threshold=float(expected_width),
                operator="==",
                passed=width == expected_width,
            ),
            "psnr_db": MetricResult(
                value=psnr,
                threshold=minimum_psnr,
                operator=">=",
                passed=psnr >= minimum_psnr,
            ),
            "mean_absolute_error": MetricResult(
                value=mae,
                threshold=maximum_mae,
                operator="<=",
                passed=mae <= maximum_mae,
            ),
            "maximum_absolute_error": MetricResult(
                value=max_error,
                threshold=None,
                operator="diagnostic",
                passed=True,
            ),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "exact decoded shape AND PSNR >= minimum_psnr_db AND "
                "mean_absolute_error <= maximum_mean_absolute_error"
            ),
            message=(
                f"{'PASS' if passed else 'FAIL'}: PSNR={psnr:.4f} dB, "
                f"MAE={mae:.8f}, max_abs={max_error:.8f}"
            ),
        )


comparator = MiniMaxH3DecodedVideoComparator()
