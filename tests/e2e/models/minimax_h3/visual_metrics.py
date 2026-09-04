# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoded-video metrics for the MiniMax-H3 visual parity contract.

The acceptance contract compares aligned multi-scale structure, low-frequency
scene layout, chroma, and motion instead of requiring pixel identity. Diffusion
implementations can differ in high-frequency texture while producing the same
coherent scene. Pixel-space PSNR and MAE remain diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

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
        "perceptual_frame_count",
        "perceptual_maximum_dimension",
        "ms_ssim_window_size",
        "maximum_ms_ssim_distance_p95",
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
    ms_ssim_distance_mean: float
    ms_ssim_distance_p95: float
    ms_ssim_distance_maximum: float
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


def perceptual_settings(thresholds: Mapping[str, float]) -> tuple[int, int, int]:
    """Validate and return sampled perceptual metric settings."""

    missing = sorted(REQUIRED_VISUAL_THRESHOLDS - thresholds.keys())
    if missing:
        raise ValueError(f"MiniMax-H3 threshold sidecar is missing {missing}")

    values: list[int] = []
    for key in (
        "perceptual_frame_count",
        "perceptual_maximum_dimension",
        "ms_ssim_window_size",
    ):
        raw_value = float(thresholds[key])
        if not math.isfinite(raw_value) or raw_value <= 0 or not raw_value.is_integer():
            raise ValueError(f"{key} must be a positive integer")
        values.append(int(raw_value))
    if values[2] % 2 == 0:
        raise ValueError("ms_ssim_window_size must be odd")
    return values[0], values[1], values[2]


def _stratified_frame_indices(num_frames: int, sample_count: int) -> list[int]:
    if sample_count >= num_frames:
        return list(range(num_frames))
    indices = np.rint(np.linspace(0, num_frames - 1, sample_count)).astype(np.int64)
    return [int(index) for index in np.unique(indices)]


def _perceptual_dimensions(height: int, width: int, maximum_dimension: int) -> tuple[int, int]:
    scale = min(1.0, maximum_dimension / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize_perceptual_frame(frame: np.ndarray, height: int, width: int) -> Any:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[-2:] != (height, width):
        tensor = functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return tensor.squeeze(0)


def _sampled_perceptual_metrics(
    reference_frames: list[Any],
    candidate_frames: list[Any],
    *,
    window_size: int,
) -> dict[str, float]:
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "MiniMax-H3 visual parity requires pytorch-msssim==1.0.0"
        ) from exc

    if len(reference_frames) != len(candidate_frames) or not reference_frames:
        raise ValueError("sampled perceptual frame lists must have the same non-zero length")
    minimum_dimension = min(reference_frames[0].shape[-2:])
    required_dimension = (window_size - 1) * 16
    if minimum_dimension <= required_dimension:
        raise ValueError(
            "MS-SSIM evaluation dimensions must be greater than "
            f"{required_dimension} for window size {window_size}"
        )

    ms_ssim_distances: list[float] = []
    chroma_errors: list[float] = []
    batch_size = 4
    luma_weights = torch.tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    with torch.inference_mode():
        for offset in range(0, len(reference_frames), batch_size):
            reference = torch.stack(reference_frames[offset : offset + batch_size])
            candidate = torch.stack(candidate_frames[offset : offset + batch_size])
            similarity = ms_ssim(
                reference,
                candidate,
                data_range=1.0,
                size_average=False,
                win_size=window_size,
            )
            ms_ssim_distances.extend(
                float(1.0 - value) for value in similarity.reshape(-1)
            )

            reference_luma = (reference * luma_weights).sum(dim=1, keepdim=True)
            candidate_luma = (candidate * luma_weights).sum(dim=1, keepdim=True)
            reference_chroma = torch.cat(
                (reference[:, 2:3] - reference_luma, reference[:, 0:1] - reference_luma),
                dim=1,
            )
            candidate_chroma = torch.cat(
                (candidate[:, 2:3] - candidate_luma, candidate[:, 0:1] - candidate_luma),
                dim=1,
            )
            per_frame_chroma_error = (candidate_chroma - reference_chroma).abs().mean(
                dim=(1, 2, 3)
            )
            chroma_errors.extend(float(value) for value in per_frame_chroma_error)

    return {
        "ms_ssim_distance_mean": float(np.mean(ms_ssim_distances)),
        "ms_ssim_distance_p95": float(np.quantile(ms_ssim_distances, 0.95)),
        "ms_ssim_distance_maximum": float(np.max(ms_ssim_distances)),
        "chroma_absolute_error_mean": float(np.mean(chroma_errors)),
        "chroma_absolute_error_p95": float(np.quantile(chroma_errors, 0.95)),
        "chroma_absolute_error_maximum": float(np.max(chroma_errors)),
    }


def compute_decoded_visual_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    block_size: int,
    perceptual_frame_count: int,
    perceptual_maximum_dimension: int,
    ms_ssim_window_size: int,
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
    if perceptual_frame_count <= 0 or perceptual_maximum_dimension <= 0:
        raise ValueError("perceptual frame count and maximum dimension must be positive")
    if ms_ssim_window_size <= 0 or ms_ssim_window_size % 2 == 0:
        raise ValueError("MS-SSIM window size must be a positive odd integer")

    perceptual_indices = _stratified_frame_indices(
        int(reference.shape[0]), perceptual_frame_count
    )
    perceptual_index_set = set(perceptual_indices)
    perceptual_height, perceptual_width = _perceptual_dimensions(
        int(reference.shape[1]), int(reference.shape[2]), perceptual_maximum_dimension
    )
    reference_perceptual_frames: list[Any] = []
    candidate_perceptual_frames: list[Any] = []

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

        if index in perceptual_index_set:
            reference_perceptual_frames.append(
                _resize_perceptual_frame(
                    reference_frame, perceptual_height, perceptual_width
                )
            )
            candidate_perceptual_frames.append(
                _resize_perceptual_frame(
                    candidate_frame, perceptual_height, perceptual_width
                )
            )

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
    perceptual = _sampled_perceptual_metrics(
        reference_perceptual_frames,
        candidate_perceptual_frames,
        window_size=ms_ssim_window_size,
    )
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
        **perceptual,
    )


def evaluate_visual_quality(
    metrics: DecodedVisualMetrics,
    thresholds: Mapping[str, float],
) -> dict[str, VisualGateResult]:
    """Apply the human-visible MiniMax-H3 contract to computed metrics."""

    visual_block_size(thresholds)
    perceptual_settings(thresholds)
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
    for key in (
        "maximum_ms_ssim_distance_p95",
        "maximum_chroma_absolute_error_p95",
    ):
        value = float(thresholds[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be a positive finite threshold")

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
        "ms_ssim_distance_p95": VisualGateResult(
            metrics.ms_ssim_distance_p95,
            float(thresholds["maximum_ms_ssim_distance_p95"]),
            "<=",
            metrics.ms_ssim_distance_p95
            <= float(thresholds["maximum_ms_ssim_distance_p95"]),
            "Stratified zero-lag aligned frames at the configured evaluation resolution.",
        ),
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
        "ms_ssim_distance_mean": VisualGateResult(
            metrics.ms_ssim_distance_mean, None, "diagnostic", True
        ),
        "ms_ssim_distance_maximum": VisualGateResult(
            metrics.ms_ssim_distance_maximum, None, "diagnostic", True
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
