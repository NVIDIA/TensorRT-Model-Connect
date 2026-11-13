# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display YOLOv10 detections already expressed in original-image pixels."""

import math
from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


_MAX_BOXES = 32
_COLORS = {"Native": (14, 145, 150), "Reference": (224, 113, 25)}


def begin_detection_operation(source: Path, operation: str) -> None:
    """Keep earlier operation results in history, not the current preview."""
    record_evidence("inputs", {"image": source, "operation": operation})
    record_evidence("native", None)
    record_evidence("reference", None)
    record_evidence("views", [])
    record_evidence("detection_view_summary", None)


def _rows(payload):
    boxes, scores, classes = payload["boxes"], payload["scores"], payload["classes"]
    if len(boxes) != 4 * len(scores) or len(classes) != len(scores):
        raise ValueError("detection arrays have inconsistent lengths")
    rows = []
    for index in range(min(len(scores), _MAX_BOXES)):
        box = [float(value) for value in boxes[index * 4 : index * 4 + 4]]
        score = float(scores[index])
        class_id = int(classes[index])
        if not all(math.isfinite(value) for value in (*box, score)):
            raise ValueError("detection coordinates and scores must be finite")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("detection corners are not ordered XYXY")
        rows.append({"box": box, "class_id": class_id, "score": score})
    return rows, len(scores)


def _view(role, rows, total, source, directory, sample_index, operation):
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
        draw.text(
            (max(0, rectangle[0]), max(0, rectangle[1] - 12)),
            f"ID {row['class_id']} {row['score']:.3f}",
            fill=_COLORS[role],
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{role.lower()}-detections.png"
    picture.save(path)
    detections = "; ".join(
        f"class ID {row['class_id']}, score {row['score']:.6g}, "
        f"XYXY [{', '.join(f'{value:.6g}' for value in row['box'])}]"
        for row in rows
    )
    caption = (
        f"{operation} / {source.name}: showing first {len(rows)} of {total} detections. "
        "XYXY uses original-image pixels; display only is resized to at most 768 x 768. "
        "Coordinates and scores below are rounded for display. " + (detections or "No detections.")
    )
    return {
        "title": f"Sample {sample_index} / {role}",
        "image": path,
        "caption": caption,
    }, {
        "operation": operation,
        "source_size": list(source_size),
        "display_size": list(picture.size),
        "displayed": len(rows),
        "total": total,
        "detections": rows,
    }


def record_detection_views(
    actual, reference, source: Path, directory: Path, *, sample_index=0, operation="CLI"
) -> None:
    """Create bounded display copies; leave complete outputs and test gates intact."""
    if not evidence_enabled():
        return
    views, summary = [], {}
    for role, payload in (("Native", actual), ("Reference", reference)):
        if payload is None:
            continue
        try:
            rows, total = _rows(payload)
            view, details = _view(role, rows, total, source, directory, sample_index, operation)
            views.append(view)
            summary[role.lower()] = details
        except Exception as error:
            message = f"Detection preview unavailable: {type(error).__name__}: {error}"
            views.append({"title": f"Sample {sample_index} / {role}", "caption": message})
            summary[role.lower()] = {"unavailable": message}
    record_evidence("detection_view_summary", summary)
    record_evidence("views", views)
