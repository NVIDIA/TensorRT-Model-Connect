# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic numerical metrics over complete task output artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def generated_media_metrics(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    """Compare deterministic generated image/video artifacts without family semantics."""
    import numpy as np
    from PIL import Image

    actual, actual_count, actual_indices = _media_artifacts(candidate, "candidate")
    expected, expected_count, expected_indices = _media_artifacts(reference, "reference")
    if actual_count != expected_count:
        raise ValueError("candidate and reference media counts differ")
    common_indices = sorted(set(actual_indices) & set(expected_indices))
    required_indices = sorted({0, actual_count // 2, actual_count - 1})
    if not set(required_indices).issubset(common_indices):
        raise ValueError("candidate and reference media do not retain the required frames")
    actual_by_index = dict(zip(actual_indices, actual, strict=True))
    expected_by_index = dict(zip(expected_indices, expected, strict=True))
    psnr_values = []
    ssim_values = []
    for index in required_indices:
        left_path, right_path = actual_by_index[index], expected_by_index[index]
        left = np.asarray(Image.open(left_path).convert("RGB"), dtype=np.float64)
        right = np.asarray(Image.open(right_path).convert("RGB"), dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError("candidate and reference media dimensions differ")
        delta = left - right
        mse = float(np.mean(delta * delta))
        psnr_values.append(100.0 if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
        left_mean, right_mean = float(left.mean()), float(right.mean())
        left_var, right_var = float(left.var()), float(right.var())
        covariance = float(np.mean((left - left_mean) * (right - right_mean)))
        c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
        numerator = (2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)
        denominator = (left_mean**2 + right_mean**2 + c1) * (left_var + right_var + c2)
        ssim_values.append(numerator / denominator)
    return {
        "media_count": float(actual_count),
        "compared_frames": float(len(required_indices)),
        "min_psnr": min(psnr_values),
        "min_ssim": min(ssim_values),
    }


def generated_media_passes(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    minimum_psnr = _finite_threshold(thresholds, "min_psnr")
    minimum_ssim = _finite_threshold(thresholds, "min_ssim")
    return metrics["min_psnr"] >= minimum_psnr and metrics["min_ssim"] >= minimum_ssim


def generated_audio_metrics(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    """Compare complete generated waveforms using stable signal-level metrics."""
    import numpy as np

    actual, actual_rate = _read_wav(
        _artifact_path(candidate.get("audio_artifact"), "candidate audio"), np
    )
    expected, expected_rate = _read_wav(
        _artifact_path(reference.get("audio_artifact"), "reference audio"), np
    )
    if actual_rate <= 0 or expected_rate <= 0 or actual.size == 0 or expected.size == 0:
        raise ValueError("candidate and reference audio must be non-empty")
    actual = actual.astype(np.float64, copy=False)
    expected = expected.astype(np.float64, copy=False)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("candidate and reference audio must be finite")
    duration_ratio = (actual.size / actual_rate) / (expected.size / expected_rate)
    actual_rms = float(np.sqrt(np.mean(actual * actual)))
    expected_rms = float(np.sqrt(np.mean(expected * expected)))
    rms_ratio = actual_rms / max(expected_rms, 1.0e-12)
    length = min(actual.size, expected.size)
    # Compare a bounded aligned prefix; generation duration is gated separately.
    fft_size = min(length, 262144)
    window = np.hanning(fft_size)
    left = np.log1p(np.abs(np.fft.rfft(actual[:fft_size] * window)))
    right = np.log1p(np.abs(np.fft.rfft(expected[:fft_size] * window)))
    log_spectral_distance = float(np.sqrt(np.mean((left - right) ** 2)))
    return {
        "duration_ratio": float(duration_ratio),
        "rms_ratio": float(rms_ratio),
        "log_spectral_distance": log_spectral_distance,
    }


def generated_audio_passes(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    return (
        metrics["duration_ratio"] >= _finite_threshold(thresholds, "min_duration_ratio")
        and metrics["duration_ratio"] <= _finite_threshold(thresholds, "max_duration_ratio")
        and metrics["rms_ratio"] >= _finite_threshold(thresholds, "min_rms_ratio")
        and metrics["rms_ratio"] <= _finite_threshold(thresholds, "max_rms_ratio")
        and metrics["log_spectral_distance"]
        <= _finite_threshold(thresholds, "max_log_spectral_distance")
    )


def _media_artifacts(summary: Mapping[str, Any], label: str) -> tuple[list[Path], int, list[int]]:
    values = summary.get("frame_artifacts", summary.get("image_artifacts"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} media artifacts are missing")
    paths = [_artifact_path(value, f"{label} media artifact") for value in values]
    declared_value = summary.get(
        "media_count", summary.get("num_frames", summary.get("generated_frames"))
    )
    declared = _integer(declared_value, f"{label} media_count", minimum=1)
    configured_indices = summary.get("artifact_indices")
    indices = list(range(len(paths))) if configured_indices is None else configured_indices
    if (
        not isinstance(indices, list)
        or len(indices) != len(paths)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
        or sorted(set(indices)) != indices
        or indices[0] < 0
        or indices[-1] >= declared
    ):
        raise ValueError(f"{label} media artifact indices are invalid")
    return paths, declared, indices


def _read_wav(path: Path, np):
    import struct

    payload = path.read_bytes()
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("audio artifact must be a RIFF/WAVE file")
    chunks = {}
    offset = 12
    while offset + 8 <= len(payload):
        name = payload[offset : offset + 4]
        size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        start, end = offset + 8, offset + 8 + size
        if end > len(payload):
            raise ValueError("audio artifact contains a truncated WAV chunk")
        if name in {b"fmt ", b"data"}:
            chunks[name] = payload[start:end]
        offset = end + (size & 1)
    if len(chunks.get(b"fmt ", b"")) < 16 or b"data" not in chunks:
        raise ValueError("audio artifact has no complete format and data chunks")
    kind, channels, rate, _, alignment, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if channels < 1 or rate < 1 or alignment != channels * (bits // 8):
        raise ValueError("audio artifact has an invalid WAV format")
    dtype = {(3, 32): "<f4", (1, 16): "<i2", (1, 32): "<i4"}.get((kind, bits))
    if dtype is None:
        raise ValueError("audio artifact must use Float32, PCM16, or PCM32 samples")
    values = np.frombuffer(chunks[b"data"], dtype=dtype)
    if values.size == 0 or values.size % channels:
        raise ValueError("audio artifact contains no complete frames")
    values = values.reshape(-1, channels).astype(np.float64)
    if kind == 1:
        values /= float(1 << (bits - 1))
    return values.mean(axis=1), rate


def _finite_threshold(thresholds: Mapping[str, Any], name: str) -> float:
    value = thresholds.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"threshold {name} must be finite")
    return float(value)


def metric_geometry_metrics(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    """Compare two complete metric-geometry outputs on their common valid grid."""
    import numpy as np

    actual = _metric_geometry(candidate, "candidate", np)
    expected = _metric_geometry(reference, "reference", np)
    actual_points, actual_depth, actual_mask, actual_intrinsics = actual
    ref_points, ref_depth, ref_mask, ref_intrinsics = expected
    if actual_depth.shape != ref_depth.shape:
        raise ValueError("candidate and reference geometry grids differ")
    actual_valid = actual_mask.astype(bool, copy=False)
    ref_valid = ref_mask.astype(bool, copy=False)
    common = actual_valid & ref_valid
    union = actual_valid | ref_valid
    if not np.any(common):
        raise ValueError("candidate and reference geometry have no common valid pixels")

    actual_depth_valid = actual_depth[common].astype(np.float64)
    ref_depth_valid = ref_depth[common].astype(np.float64)
    depth_delta = actual_depth_valid - ref_depth_valid
    actual_points_valid = actual_points[common].astype(np.float64)
    ref_points_valid = ref_points[common].astype(np.float64)
    point_delta = actual_points_valid - ref_points_valid
    cosine_denominator = np.maximum(
        np.linalg.norm(actual_points_valid, axis=-1) * np.linalg.norm(ref_points_valid, axis=-1),
        1.0e-12,
    )
    ref_intrinsics64 = ref_intrinsics.astype(np.float64)
    intrinsics_delta = actual_intrinsics.astype(np.float64) - ref_intrinsics64
    nonzero = np.abs(ref_intrinsics64) > 1.0e-12
    metrics = {
        "mask_iou": float(common.sum() / union.sum()),
        "depth_absrel_mean": float(
            np.mean(np.abs(depth_delta) / np.maximum(np.abs(ref_depth_valid), 1.0e-12))
        ),
        "depth_rel_l2": float(
            np.linalg.norm(depth_delta) / max(float(np.linalg.norm(ref_depth_valid)), 1.0e-12)
        ),
        "points_rel_l2": float(
            np.linalg.norm(point_delta) / max(float(np.linalg.norm(ref_points_valid)), 1.0e-12)
        ),
        "points_cosine": float(
            np.mean(np.sum(actual_points_valid * ref_points_valid, axis=-1) / cosine_denominator)
        ),
        "intrinsics_max_relative_error": float(
            np.max(np.abs(intrinsics_delta[nonzero]) / np.abs(ref_intrinsics64[nonzero]))
        ),
        "point_depth_consistency": float(
            np.max(np.abs(actual_points[..., 2][actual_valid] - actual_depth[actual_valid]))
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("metric-geometry comparison produced non-finite metrics")
    return metrics


def metric_geometry_passes(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    """Apply the public metric-geometry threshold vocabulary."""
    required = {
        "mask_iou",
        "depth_absrel_mean",
        "depth_rel_l2",
        "points_rel_l2",
        "points_cosine",
        "intrinsics_max_relative_error",
        "point_depth_consistency",
    }
    if set(metrics) != required or not required.issubset(thresholds):
        raise ValueError("metric-geometry thresholds are incomplete")
    minimums = {"mask_iou", "points_cosine"}
    for name in required:
        threshold = thresholds[name]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"metric-geometry threshold {name} must be numeric")
        if not math.isfinite(float(threshold)):
            raise ValueError(f"metric-geometry threshold {name} must be finite")
        if name in minimums:
            if metrics[name] < float(threshold):
                return False
        elif metrics[name] > float(threshold):
            return False
    return True


def robot_action_metrics(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    """Compare every value in two robot action chunks."""
    actual = _robot_actions(candidate, "candidate")
    expected = _robot_actions(reference, "reference")
    if len(actual) != len(expected):
        raise ValueError("candidate and reference action chunks differ in size")
    deltas = [float(left) - float(right) for left, right in zip(actual, expected, strict=True)]
    metrics = {
        "action_max_abs_error": max(abs(value) for value in deltas),
        "action_mean_abs_error": math.fsum(abs(value) for value in deltas) / len(deltas),
        "action_rmse": math.sqrt(math.fsum(value * value for value in deltas) / len(deltas)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("robot action comparison produced non-finite metrics")
    return metrics


def robot_action_passes(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    """Apply the public robot-action threshold vocabulary."""
    required = {
        "action_max_abs_error",
        "action_mean_abs_error",
        "action_rmse",
    }
    if set(metrics) != required or not required.issubset(thresholds):
        raise ValueError("robot action thresholds are incomplete")
    for name in required:
        threshold = thresholds[name]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"robot action threshold {name} must be numeric")
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError(f"robot action threshold {name} must be finite and non-negative")
        if metrics[name] > float(threshold):
            return False
    return True


def _robot_actions(summary: Mapping[str, Any], label: str) -> list[float]:
    steps = _integer(summary.get("action_steps"), f"{label} action_steps", minimum=1)
    dimension = _integer(summary.get("action_dim"), f"{label} action_dim", minimum=1)
    count = _integer(summary.get("action_values"), f"{label} action_values", minimum=1)
    values = summary.get("actions")
    if count != steps * dimension:
        raise ValueError(f"{label} action shape does not match its value count")
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} action values are missing or incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{label} action values must be finite numbers")
    if label == "candidate" and summary.get("within_training_bounds") is not True:
        raise ValueError("candidate action values are outside the training bounds")
    if label == "reference" and summary.get("finite") is not True:
        raise ValueError("reference action values are not declared finite")
    return [float(value) for value in values]


def _metric_geometry(summary: Mapping[str, Any], label: str, np):
    height = _integer(summary.get("height"), f"{label} height", minimum=1)
    width = _integer(summary.get("width"), f"{label} width", minimum=1)
    pixels = height * width
    if _integer(summary.get("geometry_images"), f"{label} geometry_images", minimum=1) != 1:
        raise ValueError(f"{label} geometry must contain one image")
    if _integer(summary.get("geometry_pixels"), f"{label} geometry_pixels", minimum=1) != pixels:
        raise ValueError(f"{label} geometry pixel count differs from its grid")
    if summary.get("point_shape") != [height, width, 3]:
        raise ValueError(f"{label} point shape differs from [H,W,3]")
    if (
        summary.get("units") != "meters"
        or summary.get("camera_axes") != ["right", "down", "forward"]
        or summary.get("intrinsics_coordinates") != "normalized_uv"
    ):
        raise ValueError(f"{label} geometry coordinate contract is invalid")
    points = _float_artifact(summary, "points_artifact", (height, width, 3), label, np)
    depth = _float_artifact(summary, "depth_artifact", (height, width), label, np)
    mask = _mask_artifact(summary, pixels, (height, width), label, np)
    intrinsics = np.asarray(summary.get("normalized_intrinsics"), dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"{label} normalized intrinsics are invalid")
    valid = mask.astype(bool, copy=False)
    if not np.any(valid):
        raise ValueError(f"{label} geometry has no valid pixels")
    if not np.isfinite(points[valid]).all():
        raise ValueError(f"{label} valid points are not finite")
    if not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0.0):
        raise ValueError(f"{label} valid depth is not finite and positive")
    declared_valid = _integer(summary.get("valid_pixels"), f"{label} valid_pixels")
    if declared_valid != int(valid.sum()):
        raise ValueError(f"{label} valid pixel count differs from its mask")
    return points, depth, mask, intrinsics


def _float_artifact(summary, name, shape, label, np):
    path = _artifact_path(summary.get(name), f"{label} {name}")
    values = np.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{label} {name} has {values.size} values; expected {expected}")
    return values.reshape(shape)


def _mask_artifact(summary, pixels, shape, label, np):
    path = _artifact_path(summary.get("valid_mask_artifact"), f"{label} valid_mask_artifact")
    values = np.fromfile(path, dtype=np.uint8)
    if values.size != pixels or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{label} validity mask is invalid")
    return values.reshape(shape)


def _artifact_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{label} is unavailable")
    return path


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value
