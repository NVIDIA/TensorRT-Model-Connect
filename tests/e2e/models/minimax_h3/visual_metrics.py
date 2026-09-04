# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoded-video metrics for the MiniMax-H3 visual parity contract.

The acceptance contract compares low-frequency scene layout, chroma, and motion
instead of requiring pixel identity. Diffusion
implementations can differ in high-frequency texture while producing the same
coherent scene. Pixel-space PSNR and MAE remain diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np


REQUIRED_VISUAL_THRESHOLDS = frozenset(
    {
        "exact_num_frames",
        "exact_video_height",
        "exact_video_width",
        "low_frequency_block_size",
        "minimum_frame_low_frequency_correlation",
        "minimum_mean_low_frequency_correlation",
        "maximum_frame_brightness_absolute_error",
        "maximum_temporal_activity_absolute_error",
        "minimum_temporal_activity_ratio",
        "maximum_temporal_activity_ratio",
        "minimum_frame_std_ratio",
        "maximum_frame_std_ratio",
        "maximum_chroma_absolute_error_p95",
    }
)


@dataclass(frozen=True)
class DecodedVisualMetrics:
    """Metrics accumulated one decoded frame at a time."""

    shape: tuple[int, ...]
    mse: float
    mean_absolute_error: float
    maximum_absolute_error: float
    psnr_db: float
    frame_low_frequency_correlation_minimum: float
    frame_low_frequency_correlation_mean: float
    brightness_profile_correlation: float
    frame_brightness_absolute_error_maximum: float
    temporal_activity_correlation: float
    temporal_activity_absolute_error_maximum: float
    temporal_activity_ratio: float
    frame_std_ratio_minimum: float
    frame_std_ratio_maximum: float
    chroma_absolute_error_mean: float
    chroma_absolute_error_p95: float
    chroma_absolute_error_maximum: float


@dataclass(frozen=True)
class VisualGateResult:
    """One self-describing visual metric result."""

    value: float
    threshold: float | None
    operator: str
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "passed": self.passed,
            "note": self.note,
        }


def _normalized_frame(frame: np.ndarray) -> np.ndarray:
    if np.issubdtype(frame.dtype, np.integer):
        return frame.astype(np.float32) / np.iinfo(frame.dtype).max
    return frame.astype(np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a stable Pearson correlation, including constant profiles."""

    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.size != right_flat.size:
        raise ValueError("correlation inputs have different sizes")
    if left_flat.size == 0:
        return 1.0

    left_centered = left_flat - float(left_flat.mean())
    right_centered = right_flat - float(right_flat.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(left_flat, right_flat, atol=1.0e-8, rtol=1.0e-6) else 0.0
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _block_means(frame: np.ndarray, block_size: int) -> np.ndarray:
    height, width, channels = frame.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"decoded frame dimensions {height}x{width} are not divisible by "
            f"low-frequency block size {block_size}"
        )
    return frame.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
        channels,
    ).mean(axis=(1, 3), dtype=np.float64)


def visual_block_size(thresholds: Mapping[str, float]) -> int:
    """Validate and return the configured low-frequency block size."""

    missing = sorted(REQUIRED_VISUAL_THRESHOLDS - thresholds.keys())
    if missing:
        raise ValueError(f"MiniMax-H3 threshold sidecar is missing {missing}")
    raw_value = float(thresholds["low_frequency_block_size"])
    if not math.isfinite(raw_value) or raw_value <= 0 or not raw_value.is_integer():
        raise ValueError("low_frequency_block_size must be a positive integer")
    return int(raw_value)


def compute_decoded_visual_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    block_size: int,
) -> DecodedVisualMetrics:
    """Compare two frame arrays without materializing the complete videos."""

    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {reference.shape} != {candidate.shape}")
    if len(reference.shape) != 4 or reference.shape[-1] != 3:
        raise ValueError(f"decoded frames must have shape [T,H,W,3], got {reference.shape}")
    if any(dimension <= 0 for dimension in reference.shape):
        raise ValueError(f"decoded video has an empty dimension: {reference.shape}")
    if block_size <= 0:
        raise ValueError("low-frequency block size must be positive")

    squared_sum = 0.0
    absolute_sum = 0.0
    maximum_error = 0.0
    pixel_count = 0
    frame_correlations: list[float] = []
    reference_brightness: list[float] = []
    candidate_brightness: list[float] = []
    reference_activity: list[float] = []
    candidate_activity: list[float] = []
    std_ratios: list[float] = []
    chroma_errors: list[float] = []
    previous_reference_blocks: np.ndarray | None = None
    previous_candidate_blocks: np.ndarray | None = None

    for index in range(reference.shape[0]):
        reference_frame = _normalized_frame(reference[index])
        candidate_frame = _normalized_frame(candidate[index])
        if not np.isfinite(reference_frame).all():
            raise ValueError(f"reference video contains non-finite pixels in frame {index}")
        if not np.isfinite(candidate_frame).all():
            raise ValueError(f"candidate video contains non-finite pixels in frame {index}")
        if float(reference_frame.min()) < 0.0 or float(reference_frame.max()) > 1.0:
            raise ValueError(f"reference video contains pixels outside [0, 1] in frame {index}")
        if float(candidate_frame.min()) < 0.0 or float(candidate_frame.max()) > 1.0:
            raise ValueError(f"candidate video contains pixels outside [0, 1] in frame {index}")

        error = candidate_frame - reference_frame
        squared_sum += float(np.square(error).sum(dtype=np.float64))
        absolute_sum += float(np.abs(error).sum(dtype=np.float64))
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
        pixel_count += int(error.size)

        reference_blocks = _block_means(reference_frame, block_size)
        candidate_blocks = _block_means(candidate_frame, block_size)
        frame_correlations.append(_correlation(reference_blocks, candidate_blocks))
        reference_brightness.append(float(reference_blocks.mean()))
        candidate_brightness.append(float(candidate_blocks.mean()))

        luma_weights = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        reference_luma = np.sum(reference_blocks * luma_weights, axis=-1)
        candidate_luma = np.sum(candidate_blocks * luma_weights, axis=-1)
        reference_chroma = np.stack(
            (reference_blocks[..., 2] - reference_luma, reference_blocks[..., 0] - reference_luma),
            axis=-1,
        )
        candidate_chroma = np.stack(
            (candidate_blocks[..., 2] - candidate_luma, candidate_blocks[..., 0] - candidate_luma),
            axis=-1,
        )
        chroma_errors.append(float(np.mean(np.abs(candidate_chroma - reference_chroma))))

        reference_std = float(reference_frame.std(dtype=np.float64))
        candidate_std = float(candidate_frame.std(dtype=np.float64))
        if reference_std <= np.finfo(np.float64).eps:
            ratio = 1.0 if candidate_std <= np.finfo(np.float64).eps else math.inf
        else:
            ratio = candidate_std / reference_std
        std_ratios.append(ratio)

        if previous_reference_blocks is not None and previous_candidate_blocks is not None:
            reference_activity.append(
                float(np.mean(np.abs(reference_blocks - previous_reference_blocks)))
            )
            candidate_activity.append(
                float(np.mean(np.abs(candidate_blocks - previous_candidate_blocks)))
            )
        previous_reference_blocks = reference_blocks
        previous_candidate_blocks = candidate_blocks

    mse = squared_sum / pixel_count
    mean_absolute_error = absolute_sum / pixel_count
    psnr_db = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    reference_brightness_array = np.asarray(reference_brightness, dtype=np.float64)
    candidate_brightness_array = np.asarray(candidate_brightness, dtype=np.float64)
    reference_activity_array = np.asarray(reference_activity, dtype=np.float64)
    candidate_activity_array = np.asarray(candidate_activity, dtype=np.float64)
    activity_denominator = float(reference_activity_array.sum())
    activity_numerator = float(candidate_activity_array.sum())
    if activity_denominator <= np.finfo(np.float64).eps:
        activity_ratio = 1.0 if activity_numerator <= np.finfo(np.float64).eps else math.inf
    else:
        activity_ratio = activity_numerator / activity_denominator

    temporal_error = np.abs(candidate_activity_array - reference_activity_array)
    return DecodedVisualMetrics(
        shape=tuple(int(value) for value in reference.shape),
        mse=mse,
        mean_absolute_error=mean_absolute_error,
        maximum_absolute_error=maximum_error,
        psnr_db=psnr_db,
        frame_low_frequency_correlation_minimum=float(min(frame_correlations)),
        frame_low_frequency_correlation_mean=float(np.mean(frame_correlations)),
        brightness_profile_correlation=_correlation(
            reference_brightness_array, candidate_brightness_array
        ),
        frame_brightness_absolute_error_maximum=float(
            np.max(np.abs(candidate_brightness_array - reference_brightness_array))
        ),
        temporal_activity_correlation=_correlation(
            reference_activity_array, candidate_activity_array
        ),
        temporal_activity_absolute_error_maximum=(
            float(np.max(temporal_error)) if temporal_error.size else 0.0
        ),
        temporal_activity_ratio=activity_ratio,
        frame_std_ratio_minimum=float(min(std_ratios)),
        frame_std_ratio_maximum=float(max(std_ratios)),
        chroma_absolute_error_mean=float(np.mean(chroma_errors)),
        chroma_absolute_error_p95=float(np.quantile(chroma_errors, 0.95)),
        chroma_absolute_error_maximum=float(np.max(chroma_errors)),
    )


def evaluate_visual_quality(
    metrics: DecodedVisualMetrics,
    thresholds: Mapping[str, float],
) -> dict[str, VisualGateResult]:
    """Apply the human-visible MiniMax-H3 contract to computed metrics."""

    visual_block_size(thresholds)
    expected_frames = int(thresholds["exact_num_frames"])
    expected_height = int(thresholds["exact_video_height"])
    expected_width = int(thresholds["exact_video_width"])
    expected_shape = (expected_frames, expected_height, expected_width, 3)
    if any(value <= 0 for value in expected_shape):
        raise ValueError("MiniMax-H3 exact decoded shape must be positive")

    minimum_activity_ratio = float(thresholds["minimum_temporal_activity_ratio"])
    maximum_activity_ratio = float(thresholds["maximum_temporal_activity_ratio"])
    minimum_std_ratio = float(thresholds["minimum_frame_std_ratio"])
    maximum_std_ratio = float(thresholds["maximum_frame_std_ratio"])
    if not (0.0 < minimum_activity_ratio <= maximum_activity_ratio):
        raise ValueError("invalid MiniMax-H3 temporal activity ratio interval")
    if not (0.0 < minimum_std_ratio <= maximum_std_ratio):
        raise ValueError("invalid MiniMax-H3 frame standard-deviation ratio interval")
    maximum_chroma_error = float(thresholds["maximum_chroma_absolute_error_p95"])
    if not math.isfinite(maximum_chroma_error) or maximum_chroma_error <= 0.0:
        raise ValueError("maximum_chroma_absolute_error_p95 must be positive and finite")

    gates = {
        "num_frames": VisualGateResult(
            float(metrics.shape[0]),
            float(expected_frames),
            "==",
            metrics.shape[0] == expected_frames,
        ),
        "video_height": VisualGateResult(
            float(metrics.shape[1]),
            float(expected_height),
            "==",
            metrics.shape[1] == expected_height,
        ),
        "video_width": VisualGateResult(
            float(metrics.shape[2]), float(expected_width), "==", metrics.shape[2] == expected_width
        ),
        "video_channels": VisualGateResult(
            float(metrics.shape[3]), 3.0, "==", metrics.shape[3] == 3
        ),
        "finite_pixels": VisualGateResult(1.0, 1.0, "==", True),
        "chroma_absolute_error_p95": VisualGateResult(
            metrics.chroma_absolute_error_p95,
            float(thresholds["maximum_chroma_absolute_error_p95"]),
            "<=",
            metrics.chroma_absolute_error_p95
            <= float(thresholds["maximum_chroma_absolute_error_p95"]),
            "Mean absolute error in aligned B-Y and R-Y channels.",
        ),
        "frame_low_frequency_correlation_minimum": VisualGateResult(
            metrics.frame_low_frequency_correlation_minimum,
            float(thresholds["minimum_frame_low_frequency_correlation"]),
            ">=",
            metrics.frame_low_frequency_correlation_minimum
            >= float(thresholds["minimum_frame_low_frequency_correlation"]),
        ),
        "frame_low_frequency_correlation_mean": VisualGateResult(
            metrics.frame_low_frequency_correlation_mean,
            float(thresholds["minimum_mean_low_frequency_correlation"]),
            ">=",
            metrics.frame_low_frequency_correlation_mean
            >= float(thresholds["minimum_mean_low_frequency_correlation"]),
        ),
        "brightness_profile_correlation": VisualGateResult(
            metrics.brightness_profile_correlation,
            None,
            "diagnostic",
            True,
            "Pearson correlation is unstable for nearly constant brightness profiles.",
        ),
        "frame_brightness_absolute_error_maximum": VisualGateResult(
            metrics.frame_brightness_absolute_error_maximum,
            float(thresholds["maximum_frame_brightness_absolute_error"]),
            "<=",
            metrics.frame_brightness_absolute_error_maximum
            <= float(thresholds["maximum_frame_brightness_absolute_error"]),
        ),
        "temporal_activity_correlation": VisualGateResult(
            metrics.temporal_activity_correlation,
            None,
            "diagnostic",
            True,
            "Pearson correlation is unstable for low-amplitude activity profiles.",
        ),
        "temporal_activity_absolute_error_maximum": VisualGateResult(
            metrics.temporal_activity_absolute_error_maximum,
            float(thresholds["maximum_temporal_activity_absolute_error"]),
            "<=",
            metrics.temporal_activity_absolute_error_maximum
            <= float(thresholds["maximum_temporal_activity_absolute_error"]),
        ),
        "temporal_activity_ratio_minimum": VisualGateResult(
            metrics.temporal_activity_ratio,
            minimum_activity_ratio,
            ">=",
            metrics.temporal_activity_ratio >= minimum_activity_ratio,
        ),
        "temporal_activity_ratio_maximum": VisualGateResult(
            metrics.temporal_activity_ratio,
            maximum_activity_ratio,
            "<=",
            metrics.temporal_activity_ratio <= maximum_activity_ratio,
        ),
        "frame_std_ratio_minimum": VisualGateResult(
            metrics.frame_std_ratio_minimum,
            minimum_std_ratio,
            ">=",
            metrics.frame_std_ratio_minimum >= minimum_std_ratio,
        ),
        "frame_std_ratio_maximum": VisualGateResult(
            metrics.frame_std_ratio_maximum,
            maximum_std_ratio,
            "<=",
            metrics.frame_std_ratio_maximum <= maximum_std_ratio,
        ),
        "psnr_db": VisualGateResult(
            metrics.psnr_db,
            None,
            "diagnostic",
            True,
            "Pixel identity is not part of the visual-quality acceptance contract.",
        ),
        "mean_absolute_error": VisualGateResult(
            metrics.mean_absolute_error,
            None,
            "diagnostic",
            True,
            "High-frequency pixel drift is allowed.",
        ),
        "maximum_absolute_error": VisualGateResult(
            metrics.maximum_absolute_error, None, "diagnostic", True
        ),
        "chroma_absolute_error_mean": VisualGateResult(
            metrics.chroma_absolute_error_mean, None, "diagnostic", True
        ),
        "chroma_absolute_error_maximum": VisualGateResult(
            metrics.chroma_absolute_error_maximum, None, "diagnostic", True
        ),
    }
    return gates


def visual_quality_passed(gates: Mapping[str, VisualGateResult]) -> bool:
    return all(result.passed for result in gates.values())
