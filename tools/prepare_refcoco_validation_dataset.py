#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a pinned lscpku/RefCOCO_rec snapshot to TRTMC validation JSON."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

SOURCE_DATASET = "lscpku/RefCOCO_rec"
SOURCE_REVISION = "566810e1ad62821ed3c6ab569ea33d80f5bdb874"
SOURCE_LICENSE_STATUS = "not-declared-by-source-card"
DEFAULT_SOURCE = Path("/mnt/data/RefCOCO_rec/raw/lscpku/RefCOCO_rec")
DEFAULT_OUTPUT = Path("/mnt/data/RefCOCO_rec/unified")


def _ground_single_prompt(phrase: str) -> str:
    # Kept byte-identical to the public family helper so dataset preparation
    # does not require importing TensorRT-backed family modules.
    return f"Locate a single instance that matches the following description: {phrase}."


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return stem or "sample"


def _bbox_1000(bbox: list[Any]) -> list[float]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError(f"RefCOCO bbox must have 4 values, got {len(values)}")
    if max(abs(value) for value in values) <= 1.0:
        values = [value * 1000.0 for value in values]
    x1, y1, x2, y2 = values
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    normalized = [
        round(max(0.0, min(1000.0, value)), 3)
        for value in (left, top, right, bottom)
    ]
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        raise ValueError(f"RefCOCO bbox has no area after normalization: {bbox}")
    return normalized


def _format_box(box: list[float]) -> str:
    return "[" + ", ".join(
        f"{value:.3f}".rstrip("0").rstrip(".") for value in box
    ) + "]"


def _split_files(source_root: Path, split: str) -> list[Path]:
    files = sorted((source_root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for split {split!r} under {source_root / 'data'}"
        )
    return files


def _iter_rows(source_root: Path, split: str) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("RefCOCO conversion requires pyarrow") from exc
    for parquet_path in _split_files(source_root, split):
        parquet_file = parquet.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _write_image(
    row: dict[str, Any],
    *,
    image_dir: Path,
    split: str,
    sample_id: str,
) -> str:
    image = row.get("image")
    file_name = str(row.get("file_name") or sample_id)
    suffix = Path(file_name).suffix.lower() or ".jpg"
    image_path = image_dir / split / f"{_safe_stem(Path(file_name).stem)}{suffix}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.is_file():
        return str(image_path.relative_to(image_dir.parent))
    if isinstance(image, dict) and isinstance(image.get("bytes"), (bytes, bytearray)):
        image_path.write_bytes(image["bytes"])
    elif isinstance(image, dict) and image.get("path"):
        source = Path(str(image["path"]))
        if not source.is_file():
            raise ValueError(f"Sample {sample_id} image path does not exist: {source}")
        shutil.copyfile(source, image_path)
    else:
        raise ValueError(f"Sample {sample_id} has no image bytes or readable image path")
    return str(image_path.relative_to(image_dir.parent))


def convert_refcoco_rec(
    *,
    source_root: Path,
    output_dir: Path,
    splits: list[str],
    source_revision: str = SOURCE_REVISION,
    limit: int = 0,
    sample_seed: int | None = None,
) -> Path:
    rows = [
        (split, split_index, row)
        for split in splits
        for split_index, row in enumerate(_iter_rows(source_root, split))
    ]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError("RefCOCO conversion selected no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    requests: list[dict[str, Any]] = []
    for global_index, (split, split_index, row) in enumerate(rows):
        question_id = str(row.get("question_id") or f"{split}_{split_index:06d}")
        sample_id = _safe_stem(
            f"refcoco_{split}_{question_id}_{global_index:06d}"
        )
        expression = str(row.get("answer") or "").strip()
        if not expression:
            raise ValueError(f"Sample {sample_id} is missing the referring expression")
        box = _bbox_1000(row.get("bbox") or [])
        image_ref = _write_image(
            row, image_dir=image_dir, split=split, sample_id=sample_id
        )
        requests.append(
            {
                "id": sample_id,
                "subject": split,
                "answer": _format_box(box),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_ref},
                            {"type": "text", "text": _ground_single_prompt(expression)},
                        ],
                    }
                ],
                "metadata": {
                    "source_dataset": SOURCE_DATASET,
                    "source_revision": source_revision,
                    "split": split,
                    "split_index": split_index,
                    "question_id": question_id,
                    "expression": expression,
                    "question": row.get("question", ""),
                    "file_name": row.get("file_name", ""),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "bbox": [float(value) for value in (row.get("bbox") or [])],
                    "bbox_1000": box,
                    "segmentation": row.get("segmentation") or [],
                    "iscrowd": row.get("iscrowd", 0),
                },
            }
        )
    dataset = {
        "name": "RefCOCO_rec",
        "version": 1,
        "task": "visual_grounding",
        "source": SOURCE_DATASET,
        "source_url": f"https://huggingface.co/datasets/{SOURCE_DATASET}",
        "source_revision": source_revision,
        "source_license_status": SOURCE_LICENSE_STATUS,
        "coordinate_format": "normalized_0_1000_xyxy",
        "splits": splits,
        "requests": requests,
    }
    output_path = output_dir / "dataset.json"
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--split", action="append", dest="splits", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = convert_refcoco_rec(
        source_root=arguments.source_root,
        output_dir=arguments.output_dir,
        splits=arguments.splits or ["testA"],
        source_revision=arguments.source_revision,
        limit=arguments.limit,
        sample_seed=arguments.sample_seed,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
