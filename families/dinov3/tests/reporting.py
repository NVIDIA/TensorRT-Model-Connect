# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 query-patch views, independent of the numerical passing criteria."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tools.e2e_evidence import record_evidence


_VIRIDIS = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
_ERROR = np.asarray([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]])


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
