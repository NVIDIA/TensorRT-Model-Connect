# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 query-patch views, independent of the numerical passing criteria."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tools.e2e_evidence import evidence_enabled, record_evidence, record_reference_comparison


_VIRIDIS = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
_ERROR = np.asarray([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]])


def record_validated_comparison(actual: dict, expected: dict, thresholds: dict) -> None:
    """Describe a successful original validator using its unchanged formulas.

    Call only after ``validation._assert_parity`` returns. Recomputed values
    explain that validator's checks; reporting never supplies a passing gate.
    """
    if not evidence_enabled():
        return
    try:
        from families.dinov3.validation import _cosine

        candidate = np.asarray(actual["last_hidden_state"], dtype=np.float64).reshape(
            actual["last_hidden_state_shape"]
        )
        reference = np.asarray(expected["last_hidden_state"], dtype=np.float64)
        candidate_pooler = np.asarray(actual["pooler_output"], dtype=np.float64).reshape(
            actual["pooler_output_shape"]
        )
        reference_pooler = np.asarray(expected["pooler_output"], dtype=np.float64)
        values = {
            "full_cosine": _cosine(candidate, reference),
            "cls_cosine": _cosine(candidate[:, 0, :], reference[:, 0, :]),
            "pooler_cosine": _cosine(candidate_pooler, reference_pooler),
            "relative_frobenius": float(
                np.linalg.norm(candidate - reference)
                / max(float(np.linalg.norm(reference)), 1e-12)
            ),
        }
        register_count = int(expected["num_register_tokens"])
        patch_start = 1 + register_count
        if register_count:
            values["register_cosine"] = _cosine(
                candidate[:, 1:patch_start, :], reference[:, 1:patch_start, :]
            )
        candidate_patches = candidate[:, patch_start:, :].reshape(-1, candidate.shape[-1])
        reference_patches = reference[:, patch_start:, :].reshape(-1, reference.shape[-1])
        denominators = np.linalg.norm(candidate_patches, axis=1) * np.linalg.norm(
            reference_patches, axis=1
        )
        patch_cosines = np.divide(
            np.sum(candidate_patches * reference_patches, axis=1),
            denominators,
            out=np.zeros_like(denominators),
            where=denominators != 0.0,
        )
        values["mean_patch_cosine"] = float(np.mean(patch_cosines))
        values["p01_patch_cosine"] = float(np.percentile(patch_cosines, 1.0))
        checks = [
            {
                "name": name,
                "scope": "independent_reference",
                "actual": value,
                "operator": "<=" if name == "relative_frobenius" else ">=",
                "expected": float(thresholds[name]),
                "passed": value <= float(thresholds[name])
                if name == "relative_frobenius"
                else value >= float(thresholds[name]),
            }
            for name, value in values.items()
        ]
        record_evidence(
            "comparison_provenance",
            {
                "validator": "families.dinov3.validation._assert_parity",
                "validator_completed": True,
                "display_values": "Recomputed with the original validator's formulas after it returned.",
            },
        )
        record_reference_comparison("DINOv3 full-feature validator", actual, expected, checks)
    except Exception as error:
        record_evidence("comparison_unavailable", f"{type(error).__name__}: {error}")


def query_patch_maps(actual: dict, expected: dict) -> list[dict]:
    """Use each reference query for both maps and exactly the same color scale."""
    candidate = np.array(actual["last_hidden_state"], dtype=np.float64, copy=True).reshape(actual["last_hidden_state_shape"])
    reference = np.array(expected["last_hidden_state"], dtype=np.float64, copy=True)
    if candidate.shape != reference.shape or reference.ndim != 3 or reference.shape[0] != 1:
        raise ValueError("query maps require matching [1, tokens, channels] feature tensors")
    start = 1 + int(expected["num_register_tokens"])
    candidate, reference = candidate[0, start:], reference[0, start:]
    grid = math.isqrt(len(reference))
    if grid < 1 or grid * grid != len(reference) or len(reference) > 1024 or reference.size > 4 * 1024 * 1024:
        raise ValueError("query maps require a bounded square patch grid")
    for features in (candidate, reference):
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.isfinite(features).all() or np.any(norm == 0):
            raise ValueError("query maps require finite, nonzero patch features")
        features /= norm
    anchors = sorted({min(grid - 1, grid * quarter // 4) for quarter in (1, 2, 3)})
    queries = [row * grid + column for row in anchors for column in anchors
               if grid < 3 or (row, column) != (anchors[1], anchors[1])][:8]
    result = []
    for query in queries:
        reference_map = np.clip(reference @ reference[query], -1, 1).reshape(grid, grid)
        native_map = np.clip(candidate @ reference[query], -1, 1).reshape(grid, grid)
        low = float(reference_map.min())
        if 1.0 - low < 0.05:
            low = -1.0
        result.append({"query": query, "native": native_map, "reference": reference_map,
                       "absolute_error": np.abs(native_map - reference_map), "scale": (low, 1.0)})
    return result


def _png(values: np.ndarray, path: Path, low: float, high: float, colors: np.ndarray) -> None:
    from PIL import Image

    normalized = np.clip((values - low) / (high - low), 0, 1)
    stops = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(normalized, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    Image.fromarray(np.rint(rgb).astype(np.uint8)).resize((280, 280), Image.Resampling.NEAREST).save(path)


def record_report_views(actual: dict, expected: dict, directory: Path) -> None:
    from tools.e2e_evidence import evidence_enabled

    if not evidence_enabled():
        return
    try:
        maps = query_patch_maps(actual, expected)
        directory.mkdir(parents=True, exist_ok=True)
        views = []
        for item in maps:
            for role in ("reference", "native", "absolute_error"):
                path = directory / f"query-{item['query']}-{role}.png"
                low, high = (0.0, 0.01) if role == "absolute_error" else item["scale"]
                _png(item[role], path, low, high, _ERROR if role == "absolute_error" else _VIRIDIS)
                views.append({"title": f"Query {item['query']} / {role}", "image": path,
                              "caption": f"Shared scale [{low:.5g}, {high:.5g}]; colors clip outside this display range. This is a visualization, not a passing threshold."})
        record_evidence("views", views)
    except (KeyError, TypeError, ValueError, OSError) as error:
        record_evidence("views", {"unavailable": str(error)})
