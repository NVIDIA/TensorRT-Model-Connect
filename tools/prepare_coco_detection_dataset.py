#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a deterministic COCO val2017 object-detection subset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root",
        required=True,
        type=Path,
        help="Directory containing annotations/instances_val2017.json and val2017/.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", default=100, type=_positive_int)
    return parser


def _objects(raw: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not all(isinstance(value, Mapping) for value in raw):
        raise ValueError(f"COCO {label} must be a list of objects")
    return [dict(value) for value in raw]


def _valid_annotation(annotation: Mapping[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    return (
        not bool(annotation.get("iscrowd", 0))
        and isinstance(bbox, list)
        and len(bbox) == 4
        and float(bbox[2]) > 0.0
        and float(bbox[3]) > 0.0
        and float(annotation.get("area", float(bbox[2]) * float(bbox[3]))) > 0.0
    )


def _select_images(
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
    limit: int,
) -> list[int]:
    image_by_id = {int(image["id"]): image for image in images}
    by_category: dict[int, list[int]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    for annotation in sorted(annotations, key=lambda value: int(value["id"])):
        if not _valid_annotation(annotation):
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        image = image_by_id.get(image_id)
        if image is None or category_id not in category_ids:
            continue
        bbox = annotation["bbox"]
        image_area = float(image["width"]) * float(image["height"])
        box_area = float(annotation.get("area", float(bbox[2]) * float(bbox[3])))
        pair = (category_id, image_id)
        if box_area >= image_area * 0.01 and pair not in seen_pairs:
            by_category[category_id].append(image_id)
            seen_pairs.add(pair)

    selected: list[int] = []
    selected_set: set[int] = set()
    offset = 0
    while len(selected) < limit:
        made_progress = False
        for category_id in category_ids:
            rows = by_category.get(category_id, [])
            if offset >= len(rows):
                continue
            made_progress = True
            image_id = rows[offset]
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            if len(selected) == limit:
                return selected
        if not made_progress:
            break
        offset += 1
    raise ValueError(f"COCO annotations provide only {len(selected)} eligible unique images")


def prepare(coco_root: Path, output_root: Path, limit: int) -> Path:
    annotation_path = coco_root / "annotations" / "instances_val2017.json"
    image_root = coco_root / "val2017"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotation file must contain an object")
    images = _objects(payload.get("images"), "images")
    annotations = _objects(payload.get("annotations"), "annotations")
    categories = sorted(
        _objects(payload.get("categories"), "categories"), key=lambda value: int(value["id"])
    )
    category_ids = [int(category["id"]) for category in categories]
    category_indices = {category_id: index for index, category_id in enumerate(category_ids)}
    selected_ids = _select_images(images, annotations, category_ids, limit)
    selected_set = set(selected_ids)
    image_by_id = {int(image["id"]): image for image in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in selected_set or not _valid_annotation(annotation):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        category_id = int(annotation["category_id"])
        annotations_by_image[image_id].append(
            {
                "id": int(annotation["id"]),
                "category_id": category_id,
                "category_index": category_indices[category_id],
                "bbox_xyxy": [x, y, x + width, y + height],
                "area": float(annotation.get("area", width * height)),
            }
        )

    output_dir = output_root / "COCO2017_object_detection"
    requests = []
    for image_id in selected_ids:
        image = image_by_id[image_id]
        filename = str(image["file_name"])
        relative = Path("images") / filename
        source = image_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"COCO image is unavailable: {source}")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        requests.append(
            {
                "id": f"coco2017_{image_id:012d}",
                "image": relative.as_posix(),
                "image_id": image_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": sorted(
                    annotations_by_image[image_id], key=lambda value: int(value["id"])
                ),
            }
        )

    manifest = {
        "dataset": "COCO 2017 validation object detection",
        "sources": [
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ],
        "sampling": (
            "deterministic category-balanced round-robin over unique images; "
            "each selected category has a non-crowd instance covering at least 1% of the image"
        ),
        "label_spaces": {
            "coco-category-id": "category_id",
            "coco-contiguous-80": "category_index",
        },
        "categories": [
            {
                "id": int(category["id"]),
                "index": category_indices[int(category["id"])],
                "name": str(category["name"]),
            }
            for category in categories
        ],
        "requests": requests,
    }
    output = output_dir / "coco2017_object_detection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    arguments = _parser().parse_args()
    output = prepare(
        arguments.coco_root.resolve(), arguments.output_root.resolve(), arguments.limit
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
