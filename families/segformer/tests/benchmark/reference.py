#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Transformers SegFormer reference for Accuracy qualification."""

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
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    model_id = str(request["model"])
    revision = request.get("revision")
    options = {"revision": revision} if revision else {}
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **options)
    model = (
        AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            torch_dtype=_dtype(torch, str(request.get("precision", "fp32"))),
            **options,
        )
        .eval()
        .to("cuda")
    )
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
        mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0].cpu()
        result.append(
            {
                "sample_id": str(sample["sample_id"]),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "mask": mask.reshape(-1).tolist(),
            }
        )
    arguments.output.write_text(json.dumps({"samples": result}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
