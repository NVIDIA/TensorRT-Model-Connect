#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce model-agnostic TIMM image-classification reference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--request", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise ValueError("reference request must contain an object")

    import timm
    import torch
    from PIL import Image
    from timm.data import create_transform, resolve_model_data_config

    model_id = _string(request.get("model"), "model")
    revision = request.get("revision")
    if revision is not None:
        model_id += "@" + _string(revision, "revision")
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[_string(request.get("precision"), "precision")]
    batch_size = int(request.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    samples = request.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    device = torch.device("cuda")
    model = timm.create_model(f"hf-hub:{model_id}", pretrained=True)
    model = model.eval().to(device=device, dtype=dtype)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = []
        for index, sample in enumerate(batch, start=start):
            if not isinstance(sample, Mapping):
                raise ValueError(f"sample {index} must be an object")
            path = Path(_string(sample.get("image_path"), f"sample {index} image_path"))
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        pixels = torch.stack(images).to(device=device, dtype=dtype)
        with torch.inference_mode():
            classes = model(pixels).argmax(dim=-1).cpu().tolist()
        outputs.extend(
            {
                "sample_id": _string(sample.get("sample_id"), "sample_id"),
                "top_class": int(top_class),
            }
            for sample, top_class in zip(batch, classes, strict=True)
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
