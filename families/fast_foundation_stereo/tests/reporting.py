# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stereo disparity views with shared reference/native scales."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.e2e_evidence import record_evidence


_VIRIDIS = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
_ERROR = np.asarray([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]])


def disparity_views(actual: dict, expected: dict) -> dict:
    candidate = np.asarray(actual["disparity"], dtype=np.float64)
    reference = np.asarray(expected["disparity"], dtype=np.float64)
    if candidate.size != 700 * 700 or reference.size != candidate.size:
        raise ValueError("stereo views require two 700 x 700 disparity maps")
    candidate, reference = candidate.reshape(700, 700), reference.reshape(700, 700)
    finite = reference[np.isfinite(reference) & (reference >= 0)]
    if not finite.size:
        raise ValueError("reference disparity has no finite nonnegative values")
    return {"native": candidate, "reference": reference,
            "absolute_error": np.abs(candidate - reference),
            "scale": (0.0, max(float(np.percentile(finite, 99)), 1.0))}


def _png(values: np.ndarray, path: Path, high: float, colors: np.ndarray) -> None:
    from PIL import Image

    invalid = ~np.isfinite(values) | (values < 0)
    normalized = np.clip(np.nan_to_num(values, nan=0.0, posinf=high, neginf=0.0) / high, 0, 1)
    stops = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(normalized, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    rgb[invalid] = [255, 0, 255]
    Image.fromarray(np.rint(rgb).astype(np.uint8)).save(path)


def record_report_views(actual: dict, expected: dict, directory: Path) -> None:
    from tools.e2e_evidence import evidence_enabled

    if not evidence_enabled():
        return
    try:
        maps = disparity_views(actual, expected)
        directory.mkdir(parents=True, exist_ok=True)
        views = []
        for role in ("reference", "native", "absolute_error"):
            path = directory / f"{role}.png"
            high = 2.0 if role == "absolute_error" else maps["scale"][1]
            _png(maps[role], path, high, _ERROR if role == "absolute_error" else _VIRIDIS)
            views.append({"title": f"Stereo / {role}", "image": path,
                          "caption": f"Scale [0, {high:.5g}] pixels; magenta marks invalid values. Native/reference share a scale. Display clipping does not change the numerical gates."})
        record_evidence("views", views)
    except (KeyError, TypeError, ValueError, OSError) as error:
        record_evidence("views", {"unavailable": str(error)})
