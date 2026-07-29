#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the official ELLA DPG-Bench CSV into validation JSON.

The first ten records are a fixed, human-reviewable slice spanning counting,
attribute binding, spatial relationships, text rendering, and realistic scenes.
All remaining official prompts follow in their original order, so a full run
still evaluates exactly the complete benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


SOURCE_URL = "https://github.com/TencentQQGYLab/ELLA"
SOURCE_COMMIT = "3c228f1dc6c4d3cad0a47493816151a419f14db3"
LICENSE = "Apache-2.0"
REVIEW_FIRST_ITEM_IDS = (
    "partiprompts231",  # count, vertical order, color binding
    "partiprompts63",   # 2x2 layout, color, expression, attribute binding
    "partiprompts68",   # count, shape, color, relative placement
    "partiprompts207",  # multiple entities, action, scene and background
    "partiprompts178",  # exact rendered text plus character attributes
    "146",              # surreal scale and object interaction
    "46",               # count, material, color and spatial distribution
    "vrd3",             # person, apparel and equipment attributes
    "localized18",      # natural scene with localized objects
    "drawtext33",       # rendered text, material and background
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source_csv: Path) -> dict[str, Any]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "item_id", "text", "proposition_id", "dependency",
        "category_broad", "category_detailed", "tuple",
        "question_natural_language",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{source_csv}: missing columns {sorted(missing)}")

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["item_id"], []).append(row)
    original_ids = list(grouped)
    absent = set(REVIEW_FIRST_ITEM_IDS).difference(grouped)
    if absent:
        raise ValueError(f"review-first DPG items are absent: {sorted(absent)}")
    ordered_ids = list(REVIEW_FIRST_ITEM_IDS) + [
        item_id for item_id in original_ids if item_id not in REVIEW_FIRST_ITEM_IDS
    ]

    requests: list[dict[str, Any]] = []
    for eval_index, item_id in enumerate(ordered_ids):
        question_rows = grouped[item_id]
        source_index = original_ids.index(item_id)
        prompt = question_rows[0]["text"].strip()
        if not prompt:
            raise ValueError(f"{source_csv}: empty prompt for item {item_id}")
        questions = []
        categories = set()
        for row in question_rows:
            dependencies = [
                int(value.strip())
                for value in row["dependency"].split(",")
                if value.strip()
            ]
            question = {
                "proposition_id": int(row["proposition_id"]),
                "dependency": dependencies,
                "category_broad": row["category_broad"].strip(),
                "category_detailed": row["category_detailed"].strip(),
                "tuple": row["tuple"].strip(),
                "question": row["question_natural_language"].strip(),
            }
            categories.add(question["category_broad"])
            questions.append(question)
        requests.append({
            "sample_id": f"dpg_bench_{source_index:06d}",
            "dataset_index": source_index,
            "eval_index": eval_index,
            "item_id": item_id,
            "prompt": prompt,
            "category": ",".join(sorted(categories)),
            "challenge": "dense_prompt_following",
            "questions": questions,
            "question_count": len(questions),
            "review_first": eval_index < len(REVIEW_FIRST_ITEM_IDS),
        })

    return {
        "dataset": "DPG-Bench",
        "version": "ELLA-3c228f1",
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "license": LICENSE,
        "source_csv_sha256": _sha256(source_csv),
        "ordering": "ten curated review-first cases, then official source order",
        "request_count": len(requests),
        "question_count": sum(request["question_count"] for request in requests),
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    payload = convert(args.source_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {payload['request_count']} prompts / "
        f"{payload['question_count']} questions to {args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
