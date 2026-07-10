#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare model-aligned classification and segmentation task-eval datasets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGENET_CLASS_IDS = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_imagenette(source_root: Path, output_root: Path) -> Path:
    validation_root = source_root / "val"
    output_dir = output_root / "Imagenette"
    requests: list[dict[str, Any]] = []
    for synset, (label, label_name) in IMAGENET_CLASS_IDS.items():
        class_dir = validation_root / synset
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Imagenette class directory not found: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            relative = Path("images") / "val" / synset / image_path.name
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            requests.append(
                {
                    "id": f"imagenette_{synset}_{image_path.stem}",
                    "image": relative.as_posix(),
                    "label": label,
                    "label_name": label_name,
                    "synset": synset,
                    "subset": "validation",
                }
            )
    output = output_dir / "imagenette2-320_task_eval.json"
    _write_json(
        output,
        {
            "dataset": "Imagenette 320 validation",
            "source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "label_space": "ImageNet-1k original class IDs",
            "requests": requests,
        },
    )
    return output


def prepare_ade20k(source_root: Path, output_root: Path) -> Path:
    image_root = source_root / "images" / "validation"
    annotation_root = source_root / "annotations" / "validation"
    output_dir = output_root / "ADE20K"
    requests: list[dict[str, Any]] = []
    for image_path in sorted(image_root.glob("*.jpg")):
        annotation_path = annotation_root / f"{image_path.stem}.png"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"ADE20K annotation not found: {annotation_path}")
        image_relative = Path("images") / "validation" / image_path.name
        mask_relative = Path("masks") / "validation" / annotation_path.name
        image_destination = output_dir / image_relative
        mask_destination = output_dir / mask_relative
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_destination)
        raw_mask = np.asarray(Image.open(annotation_path).convert("L"), dtype=np.uint8)
        normalized = np.where(raw_mask == 0, 255, raw_mask - 1).astype(np.uint8)
        Image.fromarray(normalized, mode="L").save(mask_destination)
        requests.append(
            {
                "id": f"ade20k_{image_path.stem}",
                "image": image_relative.as_posix(),
                "mask": mask_relative.as_posix(),
                "subset": "validation",
            }
        )
    output = output_dir / "ade20k_validation_task_eval.json"
    _write_json(
        output,
        {
            "dataset": "ADE20K validation",
            "source": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            "label_normalization": "source 1..150 -> 0..149; source 0 -> ignore 255",
            "num_classes": 150,
            "ignore_index": 255,
            "requests": requests,
        },
    )
    return output


def _interior_point(mask: np.ndarray) -> tuple[float, float]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot create a point prompt from an empty mask")
    center_x = int(round(float(xs.mean())))
    center_y = int(round(float(ys.mean())))
    if not (0 <= center_x < width and 0 <= center_y < height and mask[center_y, center_x]):
        middle = xs.size // 2
        center_x = int(xs[middle])
        center_y = int(ys[middle])
    return ((center_x + 0.5) / width, (center_y + 0.5) / height)


def _balanced_coco_groups(coco: Any, limit: int) -> list[tuple[int, int, list[dict[str, Any]]]]:
    grouped: dict[int, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for image_id in sorted(coco.imgs):
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            by_category[int(annotation["category_id"])].append(annotation)
        image = coco.imgs[image_id]
        image_area = float(image["height"] * image["width"])
        for category_id, category_annotations in by_category.items():
            instances = [ann for ann in category_annotations if not ann.get("iscrowd", 0)]
            if not instances:
                continue
            largest = max(float(ann.get("area", 0.0)) for ann in instances)
            if largest < image_area * 0.01:
                continue
            grouped[category_id].append((image_id, category_annotations))

    selected: list[tuple[int, int, list[dict[str, Any]]]] = []
    per_category = max(1, math.ceil(limit / max(1, len(grouped))))
    for offset in range(per_category):
        for category_id in sorted(grouped):
            rows = grouped[category_id]
            if offset >= len(rows):
                continue
            image_id, annotations = rows[offset]
            selected.append((image_id, category_id, annotations))
            if len(selected) >= limit:
                return selected
    return selected


def prepare_coco(source_root: Path, output_root: Path, limit: int) -> Path:
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise RuntimeError("COCO preparation requires pycocotools") from exc

    annotation_file = source_root / "annotations" / "instances_val2017.json"
    image_root = source_root / "val2017"
    coco = COCO(str(annotation_file))
    output_dir = output_root / "COCO2017_prompted_segmentation"
    requests: list[dict[str, Any]] = []
    copied_images: set[int] = set()
    for image_id, category_id, category_annotations in _balanced_coco_groups(coco, limit):
        image = coco.imgs[image_id]
        category = coco.cats[category_id]
        non_crowd = [
            ann for ann in category_annotations if not ann.get("iscrowd", 0)
        ]
        instance = max(non_crowd, key=lambda ann: float(ann.get("area", 0.0)))
        instance_mask = coco.annToMask(instance).astype(bool)
        category_mask = np.zeros_like(instance_mask)
        for annotation in category_annotations:
            category_mask |= coco.annToMask(annotation).astype(bool)
        point_x, point_y = _interior_point(instance_mask)

        image_relative = Path("images") / image["file_name"]
        if image_id not in copied_images:
            image_destination = output_dir / image_relative
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_root / image["file_name"], image_destination)
            copied_images.add(image_id)
        stem = f"{image_id:012d}_{category_id:03d}"
        instance_relative = Path("masks") / "instances" / f"{stem}.png"
        category_relative = Path("masks") / "categories" / f"{stem}.png"
        for relative, mask in (
            (instance_relative, instance_mask),
            (category_relative, category_mask),
        ):
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(destination)
        requests.append(
            {
                "id": f"coco2017_{stem}",
                "image": image_relative.as_posix(),
                "instance_mask": instance_relative.as_posix(),
                "category_mask": category_relative.as_posix(),
                "point_x": point_x,
                "point_y": point_y,
                "text_prompt": str(category["name"]),
                "category": str(category["name"]),
                "category_id": category_id,
                "image_id": image_id,
                "annotation_id": int(instance["id"]),
                "subset": "val2017",
            }
        )
    output = output_dir / "coco2017_prompted_segmentation.json"
    _write_json(
        output,
        {
            "dataset": "COCO 2017 validation prompted segmentation",
            "sources": [
                "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ],
            "sampling": "category-balanced round-robin; largest non-crowd instance >=1% image",
            "requests": requests,
        },
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenette-root", type=Path, required=True)
    parser.add_argument("--ade20k-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coco-limit", type=int, default=500)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = [
        prepare_imagenette(args.imagenette_root, args.output_root),
        prepare_ade20k(args.ade20k_root, args.output_root),
        prepare_coco(args.coco_root, args.output_root, args.coco_limit),
    ]
    for output in outputs:
        data = json.loads(output.read_text(encoding="utf-8"))
        print(f"{output}: {len(data['requests'])} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
