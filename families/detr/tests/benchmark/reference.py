#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Transformers DETR reference for Accuracy qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _dtype(torch: Any, precision: str) -> Any:
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    samples = request.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("reference request must contain samples")

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_id = str(request["model"])
    revision = request.get("revision")
    options = {"revision": revision} if revision else {}
    processor = AutoImageProcessor.from_pretrained(model_id, **options)
    model = (
        AutoModelForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=_dtype(torch, str(request.get("precision", "fp32"))),
            **options,
        )
        .eval()
        .to("cuda")
    )
    score_threshold = float(request.get("request", {}).get("score_threshold", 0.5))
    result = []
    for sample in samples:
        image = Image.open(str(sample["image_path"])).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            name: value.to(
                device=model.device,
                dtype=next(model.parameters()).dtype if value.is_floating_point() else value.dtype,
            )
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        detections = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=torch.tensor([[image.height, image.width]], device=model.device),
        )[0]
        boxes = detections["boxes"].float().cpu().tolist()
        result.append(
            {
                "sample_id": str(sample["sample_id"]),
                "image_height": image.height,
                "image_width": image.width,
                "boxes": boxes,
                "scores": detections["scores"].float().cpu().tolist(),
                "class_ids": detections["labels"].cpu().tolist(),
            }
        )
    arguments.output.write_text(json.dumps({"samples": result}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
