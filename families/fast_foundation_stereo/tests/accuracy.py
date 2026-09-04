# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Middlebury task-accuracy math."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def scene_statistics(
    candidate: np.ndarray,
    reference: np.ndarray,
    ground_truth: np.ndarray,
    valid_nonocc: np.ndarray,
) -> dict[str, float]:
    """Return sufficient statistics for one prepared Middlebury scene."""
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    valid_nonocc = np.asarray(valid_nonocc, dtype=bool)
    if not (candidate.shape == reference.shape == ground_truth.shape == valid_nonocc.shape):
        raise ValueError("candidate, reference, ground truth, and mask shapes must match")
    if not valid_nonocc.any():
        raise ValueError("task-accuracy scene has no valid non-occluded pixels")
    if not all(
        np.isfinite(values[valid_nonocc]).all() for values in (candidate, reference, ground_truth)
    ):
        raise ValueError("task-accuracy inputs must be finite on valid pixels")

    candidate_error = np.abs(
        candidate[valid_nonocc].astype(np.float64) - ground_truth[valid_nonocc].astype(np.float64)
    )
    reference_error = np.abs(
        reference[valid_nonocc].astype(np.float64) - ground_truth[valid_nonocc].astype(np.float64)
    )
    valid_pixels = int(valid_nonocc.sum())
    return {
        "valid_nonocc_pixels": float(valid_pixels),
        "candidate_nonocc_abs_error_sum_px": float(candidate_error.sum()),
        "reference_nonocc_abs_error_sum_px": float(reference_error.sum()),
        "candidate_nonocc_bad2_pixel_count": float((candidate_error > 2.0).sum()),
        "reference_nonocc_bad2_pixel_count": float((reference_error > 2.0).sum()),
        "candidate_nonocc_epe_px": float(candidate_error.mean()),
        "reference_nonocc_epe_px": float(reference_error.mean()),
        "candidate_nonocc_bp2_fraction": float(np.mean(candidate_error > 2.0)),
        "reference_nonocc_bp2_fraction": float(np.mean(reference_error > 2.0)),
    }


def aggregate_task_accuracy(
    scenes: Iterable[Mapping[str, float]],
    *,
    epe_allowance_px: float,
    bp2_allowance_fraction: float,
) -> dict[str, float | int | bool]:
    """Apply the pixel-weighted, reference-relative task gates."""
    rows = list(scenes)
    if not rows:
        raise ValueError("task-accuracy aggregate requires at least one scene")
    required = (
        "valid_nonocc_pixels",
        "candidate_nonocc_abs_error_sum_px",
        "reference_nonocc_abs_error_sum_px",
        "candidate_nonocc_bad2_pixel_count",
        "reference_nonocc_bad2_pixel_count",
    )
    if any(name not in row for row in rows for name in required):
        raise ValueError("task-accuracy scene is missing sufficient statistics")
    totals = {name: sum(float(row[name]) for row in rows) for name in required}
    valid_pixels = totals["valid_nonocc_pixels"]
    if valid_pixels <= 0:
        raise ValueError("task-accuracy aggregate has no valid non-occluded pixels")

    candidate_epe = totals["candidate_nonocc_abs_error_sum_px"] / valid_pixels
    reference_epe = totals["reference_nonocc_abs_error_sum_px"] / valid_pixels
    candidate_bp2 = totals["candidate_nonocc_bad2_pixel_count"] / valid_pixels
    reference_bp2 = totals["reference_nonocc_bad2_pixel_count"] / valid_pixels
    epe_limit = reference_epe + float(epe_allowance_px)
    bp2_limit = reference_bp2 + float(bp2_allowance_fraction)
    return {
        "valid_nonocc_pixels": int(valid_pixels),
        "candidate_nonocc_epe_px": candidate_epe,
        "reference_nonocc_epe_px": reference_epe,
        "candidate_nonocc_bp2_fraction": candidate_bp2,
        "reference_nonocc_bp2_fraction": reference_bp2,
        "candidate_nonocc_epe_px_max": epe_limit,
        "candidate_nonocc_bp2_fraction_max": bp2_limit,
        "candidate_nonocc_epe_passed": candidate_epe <= epe_limit,
        "candidate_nonocc_bp2_passed": candidate_bp2 <= bp2_limit,
    }
