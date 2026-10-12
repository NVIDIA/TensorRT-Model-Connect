#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Ultralytics YOLOv5 archive for qualification."""

from __future__ import annotations

import argparse
import copy
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
    stages: Any
    anchors: Any
    strides: Any
    head: tuple[Any, ...]
    biases: tuple[Any, ...]
    per_anchor: int
    dtype: Any


def _load_model(model_dir: Path, precision: str) -> Reference:
    import torch
    from ultralytics.nn.tasks import parse_model

    from families.yolov5.checkpoint import _collect, _install_placeholders

    try:
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error
    archive_path = model_dir / "yolov5n.pt"
    if not archive_path.is_file():
        raise FileNotFoundError(f"YOLOv5 model directory has no yolov5n.pt: {model_dir}")
    _install_placeholders()
    blob = torch.load(str(archive_path), map_location="cpu", weights_only=False)
    archive = blob.get("model") if isinstance(blob, dict) else None
    if archive is None:
        raise ValueError(f"YOLOv5 archive has no model entry: {archive_path}")
    spec = copy.deepcopy(archive.__dict__["yaml"])
    spec["head"].pop()
    stages, _ = parse_model(spec, ch=3, verbose=False)
    weights: dict[str, Any] = {}
    _collect(archive, "", weights)
    body = {
        name[len("model.") :]: value.float()
        for name, value in weights.items()
        if name.startswith("model.")
    }
    head_index = str(len(stages))
    missing, unexpected = stages.load_state_dict(
        {name: value for name, value in body.items() if not name.startswith(f"{head_index}.")},
        strict=False,
    )
    if missing:
        raise ValueError(f"YOLOv5 reference is missing checkpoint tensors: {missing[:5]}")
    if any(not name.endswith("num_batches_tracked") for name in unexpected):
        raise ValueError(f"YOLOv5 reference has unexpected checkpoint tensors: {unexpected[:5]}")
    for module in stages.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
    classes = len(archive.__dict__["names"])
    strides = archive.__dict__["stride"].float().to(device="cuda", dtype=dtype)
    anchors = weights[f"model.{head_index}.anchors"].float().to(device="cuda", dtype=dtype)
    head = tuple(
        body[f"{head_index}.m.{level}.weight"].to(device="cuda", dtype=dtype)
        for level in range(len(strides))
    )
    biases = tuple(
        body[f"{head_index}.m.{level}.bias"].to(device="cuda", dtype=dtype)
        for level in range(len(strides))
    )
    return Reference(
        stages=stages.eval().to(device="cuda", dtype=dtype),
        anchors=anchors,
        strides=strides,
        head=head,
        biases=biases,
        per_anchor=classes + 5,
        dtype=dtype,
    )


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
    model: Reference, pixels: Any, geometry: Mapping[str, Any], threshold: float
) -> dict[str, Any]:
    import torch
    import torchvision

    with torch.inference_mode():
        outputs: list[Any] = []
        tensor = pixels
        for stage in model.stages:
            if stage.f != -1:
                taken = [tensor if index == -1 else outputs[index] for index in stage.f]
                tensor = stage(taken)
            else:
                tensor = stage(tensor)
            outputs.append(tensor)

        levels = []
        for level, source_index in enumerate((17, 20, 23)):
            raw = torch.nn.functional.conv2d(
                outputs[source_index],
                model.head[level],
                model.biases[level],
            )
            rows, columns = raw.shape[-2], raw.shape[-1]
            values = raw.view(1, len(model.anchors[level]), model.per_anchor, rows, columns)
            values = values.permute(0, 1, 3, 4, 2).sigmoid()
            grid_y, grid_x = torch.meshgrid(
                torch.arange(rows, device=raw.device),
                torch.arange(columns, device=raw.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).to(dtype=model.dtype)
            centre = (values[..., 0:2] * 2 - 0.5 + grid) * model.strides[level]
            extent = (
                (values[..., 2:4] * 2) ** 2
                * model.anchors[level].view(1, -1, 1, 1, 2)
                * model.strides[level]
            )
            levels.append(
                torch.cat((centre, extent, values[..., 4:]), dim=-1).reshape(
                    1, -1, model.per_anchor
                )
            )
        predictions = torch.cat(levels, dim=1)[0]
        confidence = predictions[:, 4:5] * predictions[:, 5:]
        best, labels = confidence.max(dim=1)
        survivors = best > threshold
        centre = predictions[survivors, 0:2]
        extent = predictions[survivors, 2:4]
        corners = torch.cat((centre - extent / 2, centre + extent / 2), dim=1)
        order = torchvision.ops.batched_nms(
            corners,
            best[survivors],
            labels[survivors],
            0.7,
        )[:300]
        kept = torch.cat(
            (
                corners[order],
                best[survivors][order, None],
                labels[survivors][order, None].to(dtype=model.dtype),
            ),
            dim=1,
        )
    values = kept.detach().float().cpu().tolist()
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
        for row in values
    ]
    return {
        "image_height": int(geometry["height"]),
        "image_width": int(geometry["width"]),
        "detections": len(values),
        "boxes": boxes,
        "scores": [float(row[4]) for row in values],
        "class_ids": [int(row[5]) for row in values],
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
        pixels, geometry = _prepare_image(str(sample["image_path"]), model.dtype)
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

    if arguments.family != "yolov5" or arguments.operation != "detect":
        raise ValueError("YOLOv5 reference only supports yolov5 detect")
    if arguments.mode != "pytorch-eager":
        raise ValueError("YOLOv5 reference only supports pytorch-eager")
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
    pixels, geometry = _prepare_image(str(request["image_path"]), model.dtype)
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
