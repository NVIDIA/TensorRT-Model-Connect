#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Transformers SAM3 reference for qualification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence


def _accuracy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _performance_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter-options-json", required=True)
    parser.add_argument("--timing-contract-json", required=True)
    parser.add_argument("--padding", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-task", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _offline_requested() -> bool:
    return (
        os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
    )


def _model_directory(model: str, revision: str | None, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


@dataclass(frozen=True)
class Reference:
    processor: Any
    model: Any
    dtype: Any


def _load_reference(model_dir: Path, precision: str) -> Reference:
    import torch
    from transformers import Sam3Model, Sam3Processor

    try:
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error
    processor = Sam3Processor.from_pretrained(model_dir)
    model = Sam3Model.from_pretrained(model_dir, torch_dtype=dtype).eval().to("cuda")
    return Reference(processor, model, dtype)


def _prepare(reference: Reference, image_path: str, prompt: str) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    encoded = reference.processor(images=image, text=prompt, return_tensors="pt")
    return {
        name: value.to("cuda") if hasattr(value, "to") else value for name, value in encoded.items()
    }


def _infer(reference: Reference, inputs: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        outputs = reference.model(**inputs)
    result = reference.processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].cpu().tolist(),
    )[0]
    masks = result["masks"].detach().float().cpu()
    scores = result["scores"].detach().float().cpu().reshape(-1)
    boxes = result["boxes"].detach().float().cpu().reshape(-1, 4)
    if masks.ndim != 3 or masks.shape[0] != scores.numel() or boxes.shape[0] != scores.numel():
        raise RuntimeError("SAM3 reference returned inconsistent masks, scores or boxes")
    height, width = (int(value) for value in masks.shape[-2:])
    count = int(masks.shape[0])
    return {
        "segmented_images": 1,
        "generated_masks": count,
        "num_masks": count,
        "mask_pixels": int(masks.numel()),
        "height": height,
        "width": width,
        "mask_kind": "binary",
        "masks": masks.reshape(-1).tolist(),
        "iou_scores": scores.tolist(),
        "boxes": boxes.tolist(),
        "box_coordinates": "original_image_pixels_xyxy",
    }


def _accuracy(arguments: argparse.Namespace) -> int:
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    samples = request.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("reference request must contain samples")
    model_dir = _model_directory(
        str(request["model"]),
        request.get("revision"),
        _offline_requested(),
    )
    reference = _load_reference(model_dir, str(request.get("precision", "fp32")))
    controls = request.get("request", {})
    if not isinstance(controls, Mapping):
        raise ValueError("reference request controls must be an object")
    results = []
    for sample in samples:
        prompt = str(sample.get("label_name", "")).strip()
        if not prompt:
            raise ValueError("SAM3 Accuracy sample has no label_name prompt")
        inputs = _prepare(reference, str(sample["image_path"]), prompt)
        results.append({"sample_id": str(sample["sample_id"]), **_infer(reference, inputs)})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _performance(arguments: argparse.Namespace) -> int:
    import torch

    if arguments.family != "sam3" or arguments.operation != "segment_prompted":
        raise ValueError("SAM3 reference only supports sam3 segment_prompted")
    if arguments.mode != "pytorch-eager":
        raise ValueError("SAM3 reference only supports pytorch-eager")
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    request = _json_object(arguments.request_json, "--request-json")
    timing = _json_object(arguments.timing_contract_json, "--timing-contract-json")
    expected_timing = {
        "timing_scope": "task-model-call-wall",
        "input_preparation_included": False,
        "asset_loading_included": False,
    }
    if timing != expected_timing:
        raise ValueError(f"unsupported timing contract: {timing}")
    model_dir = _model_directory(
        arguments.model,
        arguments.revision,
        arguments.local_files_only,
    )
    reference = _load_reference(model_dir, arguments.precision)
    inputs = _prepare(reference, str(request["image_path"]), str(request["prompt"]))
    for _ in range(arguments.warmup):
        _infer(reference, inputs)
    samples = []
    output_summary: dict[str, Any] = {}
    for _ in range(arguments.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_summary = _infer(reference, inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(samples)
    result = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "backend": "transformers",
        "framework": "pytorch",
        "model": arguments.model,
        "family": arguments.family,
        "operation": arguments.operation,
        "case_name": arguments.case_name,
        "selected_task": arguments.selected_task,
        "precision": arguments.precision,
        "mode": arguments.mode,
        **expected_timing,
        "measurement": {
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
        },
        "measurement_policy": {
            **expected_timing,
            "model_load_excluded": True,
            "compile_excluded": True,
            "warmup_excluded": True,
            "output_materialization_included": True,
        },
        "samples_ms": samples,
        "metrics": {
            "sample_count": len(samples),
            "latency_ms": {
                "p50": median,
                "min": min(samples),
                "max": max(samples),
                "mean": statistics.fmean(samples),
            },
        },
        "output_summary": output_summary,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--request" in values:
        return _accuracy(_accuracy_parser().parse_args(values))
    return _performance(_performance_parser().parse_args(values))


if __name__ == "__main__":
    raise SystemExit(main())
