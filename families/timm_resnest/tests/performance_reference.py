#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time ResNeSt inference and host logits, not classification/reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time


TIMING = {
    "timing_scope": "task-model-call-wall",
    "input_preparation_included": False,
    "asset_loading_included": False,
}


def _load_reference(arguments, request):
    import timm
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_model_data_config

    checkpoint = Path(arguments.model)
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(
            repo_id=arguments.model, revision=arguments.revision,
            allow_patterns=("config.json", "model.safetensors"),
            local_files_only=arguments.local_files_only,
        ))
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    model = timm.create_model(config["architecture"], pretrained=False,
                              num_classes=int(config["num_classes"]))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    model.pretrained_cfg = config["pretrained_cfg"]
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[arguments.precision]
    model = model.eval().to(device="cuda", dtype=dtype)
    image_path = Path(request["image_path"])
    if not image_path.is_absolute():
        image_path = arguments.manifest.resolve().parent.parent / image_path
    with Image.open(image_path) as image:
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        inputs = transform(image.convert("RGB")).unsqueeze(0).to(device="cuda", dtype=dtype)

    def invoke():
        with torch.inference_mode():
            # The SDK returns complete, host-owned float32 scores synchronously.
            return model(inputs).to(device="cpu", dtype=torch.float32).numpy()

    return invoke, torch.cuda.synchronize, f"timm-{timm.__version__}"


def _measure(invoke, synchronize, warmup, iterations):
    for _ in range(warmup):
        invoke()
        synchronize()
    samples = []
    output = None
    for _ in range(iterations):
        output = None  # As in the native worker, exclude prior-result destruction.
        synchronize()
        started = time.perf_counter()
        output = invoke()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, output


def _summary(scores):
    import numpy as np

    if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] == 0:
        raise ValueError("ResNeSt must return one nonempty class-score vector")
    if not np.isfinite(scores).all():
        raise ValueError("ResNeSt returned nonfinite class scores")
    return {
        "top_class": int(scores[0].argmax()), "shape": list(scores.shape),
        "element_count": int(scores.size), "finite": True,
        "scores": scores[0].tolist(),
    }


def run(arguments):
    """Execute the existing family-reference protocol with reporting untimed."""
    if (arguments.family != "timm_resnest" or arguments.operation != "classify"
            or arguments.selected_task != "image_to_class_scores"):
        raise ValueError("reference requires timm_resnest image_to_class_scores/classify")
    if arguments.warmup < 0 or arguments.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    timing = json.loads(arguments.timing_contract_json)
    if timing != TIMING or any(type(timing[key]) is not type(value) for key, value in TIMING.items()):
        raise ValueError("reference requires its declared model-call timing policy")
    if json.loads(arguments.adapter_options_json) or arguments.trust_remote_code:
        raise ValueError("reference does not accept adapter options or remote code")
    request = json.loads(arguments.request_json)
    if (not isinstance(request, dict) or set(request) - {"image_path", "batch_size"}
            or not isinstance(request.get("image_path"), str) or not request["image_path"]
            or type(request.get("batch_size", 1)) is not int or request.get("batch_size", 1) != 1):
        raise ValueError("reference requires one image_path and no runtime Config")
    invoke, synchronize, framework = _load_reference(arguments, request)
    samples, scores = _measure(invoke, synchronize, arguments.warmup, arguments.iterations)
    return {
        "schema_version": "trtmc.perf-baseline/v1", "status": "completed",
        "model": arguments.model, "family": arguments.family, "operation": arguments.operation,
        "case_name": arguments.case_name, "selected_task": arguments.selected_task,
        "precision": arguments.precision, "mode": arguments.mode, "framework": framework,
        "measurement": {"warmup": arguments.warmup, "iterations": arguments.iterations},
        "measurement_policy": timing, **timing, "samples_ms": samples,
        "metrics": {"latency_ms": {"p50": statistics.median(samples)}},
        "output_summary": _summary(scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "family", "operation", "selected-task", "request-json",
                 "adapter-options-json", "timing-contract-json", "case-name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), required=True)
    parser.add_argument("--mode", choices=("hf-eager",), required=True)
    parser.add_argument("--padding", choices=("longest",), default="longest")
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
