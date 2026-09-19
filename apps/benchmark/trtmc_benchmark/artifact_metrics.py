# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic numerical metrics over complete task output artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


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
        np.linalg.norm(actual_points_valid, axis=-1)
        * np.linalg.norm(ref_points_valid, axis=-1),
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
            np.linalg.norm(depth_delta)
            / max(float(np.linalg.norm(ref_depth_valid)), 1.0e-12)
        ),
        "points_rel_l2": float(
            np.linalg.norm(point_delta)
            / max(float(np.linalg.norm(ref_points_valid)), 1.0e-12)
        ),
        "points_cosine": float(
            np.mean(
                np.sum(actual_points_valid * ref_points_valid, axis=-1)
                / cosine_denominator
            )
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


def metric_geometry_passes(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> bool:
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


def robot_action_passes(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> bool:
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
