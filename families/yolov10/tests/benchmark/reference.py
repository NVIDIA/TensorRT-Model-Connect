#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Ultralytics YOLOv10 archive for qualification."""

from __future__ import annotations

import argparse
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


def _load_model(model_dir: Path, precision: str) -> Any:
    import torch
    from ultralytics import YOLO

    from families.yolov10.checkpoint import Checkpoint

    try:
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architecture = config.get("model") if isinstance(config, dict) else None
    supported = {"yolov10n.yaml", "yolov10s.yaml", "yolov10x.yaml"}
    if not isinstance(architecture, str) or architecture not in supported:
        raise ValueError(f"unsupported YOLOv10 reference architecture: {architecture!r}")
    model = YOLO(architecture, task="detect").model
    checkpoint = Checkpoint.open(model_dir, framework="pt")
    raw = {name: reader.get_tensor(name) for name, reader in checkpoint.tensor_map.items()}
    state = {
        name[len("model.") :]: value for name, value in raw.items() if name.startswith("model.")
    }
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"YOLOv10 reference is missing checkpoint tensors: {missing[:5]}")
    return model.eval().to(device="cuda", dtype=dtype)


def _prepare_image(image_path: str, dtype: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image
    import torch

    size = 640
    source = Image.open(image_path).convert("RGB")
    scale = min(size / source.height, size / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - resized.width) // 2
    pad_y = (size - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None].copy() / 255.0
    tensor = torch.from_numpy(pixels).to(device="cuda", dtype=dtype)
    geometry = {
        "height": source.height,
        "width": source.width,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return tensor, geometry


def _detect(
    model: Any, pixels: Any, geometry: Mapping[str, Any], threshold: float
) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        raw = model(pixels)
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    rows = raw[0]
    kept = rows[rows[:, 4] >= threshold].detach().float().cpu().tolist()
    scale = float(geometry["scale"])
    pad_x = int(geometry["pad_x"])
    pad_y = int(geometry["pad_y"])
    boxes = [
        [
            (float(row[0]) - pad_x) / scale,
            (float(row[1]) - pad_y) / scale,
            (float(row[2]) - pad_x) / scale,
            (float(row[3]) - pad_y) / scale,
        ]
        for row in kept
    ]
    return {
        "image_height": int(geometry["height"]),
        "image_width": int(geometry["width"]),
        "detections": len(kept),
        "boxes": boxes,
        "scores": [float(row[4]) for row in kept],
        "class_ids": [int(row[5]) for row in kept],
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
    model = _load_model(model_dir, str(request.get("precision", "fp32")))
    threshold = float(request.get("request", {}).get("score_threshold", 0.25))
    results = []
    for sample in samples:
        pixels, geometry = _prepare_image(str(sample["image_path"]), next(model.parameters()).dtype)
        results.append(
            {"sample_id": str(sample["sample_id"]), **_detect(model, pixels, geometry, threshold)}
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _performance(arguments: argparse.Namespace) -> int:
    import torch

    if arguments.family != "yolov10" or arguments.operation != "detect":
        raise ValueError("YOLOv10 reference only supports yolov10 detect")
    if arguments.mode != "pytorch-eager":
        raise ValueError("YOLOv10 reference only supports pytorch-eager")
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
    model = _load_model(model_dir, arguments.precision)
    pixels, geometry = _prepare_image(str(request["image_path"]), next(model.parameters()).dtype)
    threshold = float(request.get("score_threshold", 0.25))
    for _ in range(arguments.warmup):
        _detect(model, pixels, geometry, threshold)
    samples = []
    output_summary: dict[str, Any] = {}
    for _ in range(arguments.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_summary = _detect(model, pixels, geometry, threshold)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(samples)
    result = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "backend": "ultralytics",
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
            "torch": torch.__version__,
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
