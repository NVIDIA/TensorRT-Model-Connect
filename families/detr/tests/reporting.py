# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display this family's existing detections in original-image coordinates."""

import math
from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


_MAX_BOXES = 32
_COLORS = {"Native": (14, 145, 150), "Reference": (224, 113, 25)}


def _rows(records):
    rows = []
    for item in records[:_MAX_BOXES]:
        box = [float(value) for value in item["box"]]
        score = float(item["score"])
        if len(box) != 4 or not all(math.isfinite(value) for value in (*box, score)):
            raise ValueError("detection coordinates and scores must be finite")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("detection corners are not ordered xyxy")
        rows.append({"box": box, "label": str(item["label"]), "score": score})
    return rows, len(records)


def _native_rows(actual):
    labels = actual["classes"]
    scores, boxes = actual["scores"], actual["boxes"]
    if len(boxes) != 4 * len(scores) or len(labels) != len(scores):
        raise ValueError("native detection arrays have inconsistent lengths")
    rows, _ = _rows(
        [
            {
                "box": boxes[index * 4 : index * 4 + 4],
                "label": labels[index],
                "score": scores[index],
            }
            for index in range(min(len(scores), _MAX_BOXES))
        ]
    )
    return rows, len(scores)


def _view(role, rows, total, source: Path, directory: Path, sample_index: int, operation: str):
    from PIL import Image, ImageDraw

    with Image.open(source) as original:
        picture = original.convert("RGB")
    source_size = picture.size
    picture.thumbnail((768, 768))
    scale_x, scale_y = picture.width / source_size[0], picture.height / source_size[1]
    draw = ImageDraw.Draw(picture)
    for row in rows:
        x0, y0, x1, y1 = row["box"]
        rectangle = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        draw.rectangle(rectangle, outline=_COLORS[role], width=2)
        text = f"{row['label'][:80]} {row['score']:.3f}"
        position = (max(0, rectangle[0]), max(0, rectangle[1] - 12))
        draw.text(position, text, fill=_COLORS[role])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{role.lower()}-detections.png"
    picture.save(path)
    labels = "; ".join(f"{row['label'][:80]} ({row['score']:.3f})" for row in rows)
    caption = (
        f"{operation} / {source.name}: showing first {len(rows)} of {total} detections. "
        "Boxes use original-image xyxy pixels; the display is resized only. "
        + (labels or "No detections.")
    )
    return {
        "title": f"Sample {sample_index} / {role}",
        "image": path,
        "caption": caption,
    }, {
        "operation": operation,
        "source_size": list(source_size),
        "displayed": len(rows),
        "total": total,
        "detections": rows,
    }


def record_detection_views(actual, expected, source: Path, directory: Path, *, sample_index=0, operation="CLI"):
    """Record independent overlays without changing model data or test outcomes."""
    if not evidence_enabled():
        return
    views, summary = [], {}
    for role in ("Native", "Reference"):
        try:
            rows, total = _native_rows(actual) if role == "Native" else _rows(expected)
            view, details = _view(role, rows, total, source, directory, sample_index, operation)
            views.append(view)
            summary[role.lower()] = details
        except Exception as error:
            message = f"Detection preview unavailable: {type(error).__name__}: {error}"
            views.append({"title": f"Sample {sample_index} / {role}", "caption": message})
            summary[role.lower()] = {"unavailable": message}
    record_evidence("detection_view_summary", summary)
    record_evidence("views", views)
