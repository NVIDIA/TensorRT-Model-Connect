#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare original pinned COCO images, with annotations and an input-selection audit."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")

    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    import pyarrow.parquet as pq
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from trtmc_aiperf_qual.gold_metrics import COCO_IDS

    # Reuse the qualification application's existing immutable dataset source.
    source = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             "config/suites/coco-val2017-500.yaml").read_text())["source"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = {"dataset": source["dataset"], "revision": source["revision"], "split": source["split"],
             "selection": "first eligible complete images in pinned shard/row order",
             "requested": args.count, "selected": [], "skipped": [], "failed": [], "source_files": []}
    files = sorted(path for path in HfApi().list_repo_files(source["dataset"], repo_type="dataset",
                   revision=source["revision"]) if path.startswith("data/val-") and path.endswith(".parquet"))
    try:
        for filename in files:
            shard = hf_hub_download(source["dataset"], filename, repo_type="dataset", revision=source["revision"])
            audit["source_files"].append(filename)
            with pq.ParquetFile(shard) as table:
                for batch in table.iter_batches(batch_size=16, use_threads=False):
                    for row in batch.to_pylist():
                        sample_id = str(row["image_id"])
                        try:
                            raw = row["image"]["bytes"]
                            with Image.open(io.BytesIO(raw)) as image:
                                if image.mode != "RGB" or image.format not in ("PNG", "JPEG"):
                                    audit["skipped"].append({"id": sample_id, "reason": "outside single RGB PNG/JPEG input contract"})
                                    continue
                                width, height, fmt = image.width, image.height, image.format
                                image.verify()
                            path = args.output_dir.resolve() / f"image-{len(audit['selected']):03d}{'.jpg' if fmt == 'JPEG' else '.png'}"
                            path.write_bytes(raw)
                            objects = row["objects"]
                            annotations = [{"bbox_xyxy": box, "category_index": int(category),
                                            "category_id": COCO_IDS[int(category)], "area": area}
                                           for box, category, area in zip(objects["bbox"], objects["category"], objects["area"], strict=True)]
                            audit["selected"].append({"id": sample_id, "image_path": str(path), "width": width,
                                "height": height, "image_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                "annotations": annotations})
                        except Exception as error:  # noqa: BLE001 - preserve failure provenance
                            audit["failed"].append({"id": sample_id, "reason": str(error)})
                            raise
                        if len(audit["selected"]) == args.count:
                            break
                    if len(audit["selected"]) == args.count:
                        break
            if len(audit["selected"]) == args.count:
                break
        if len(audit["selected"]) != args.count:
            raise ValueError("dataset ended before the requested selection was complete")
        (args.output_dir / "inputs.jsonl").write_text("".join(json.dumps({
            "session_id": item["id"], "image": ("data:image/jpeg;base64," if item["image_path"].endswith(".jpg") else "data:image/png;base64,") +
            base64.b64encode(Path(item["image_path"]).read_bytes()).decode("ascii")}) + "\n" for item in audit["selected"]))
        (args.output_dir / "annotations.json").write_text(json.dumps({"requests": [{
            "id": item["id"], "image": Path(item["image_path"]).name, "annotations": item["annotations"]}
            for item in audit["selected"]], "sampling": audit["selection"]}, indent=2) + "\n")
    finally:
        (args.output_dir / "selection.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(audit['selected'])} original COCO images in {args.output_dir}")


if __name__ == "__main__":
    main()
