#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the Transformers DINOv2 encoder and host features, not reporting."""

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
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from PIL import Image

    checkpoint = Path(arguments.model)
    if not checkpoint.is_dir():
        checkpoint = Path(
            snapshot_download(
                repo_id=arguments.model,
                revision=arguments.revision,
                allow_patterns=("config.json", "preprocessor_config.json", "model.safetensors"),
                local_files_only=arguments.local_files_only,
            )
        )
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    model_class = {
        "dinov2": transformers.Dinov2Model,
        "dinov2_with_registers": transformers.Dinov2WithRegistersModel,
    }[config["model_type"]]
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[
        arguments.precision
    ]
    model = model_class.from_pretrained(checkpoint, torch_dtype=dtype).eval().to("cuda")
    image_path = Path(request["image_path"])
    if not image_path.is_absolute():
        image_path = arguments.manifest.resolve().parent.parent / image_path
    # The family reproduces the Pillow processor; pin it rather than AutoImageProcessor.
    processor = transformers.BitImageProcessor.from_pretrained(checkpoint)
    with Image.open(image_path) as image:
        pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device="cuda", dtype=dtype)

    def invoke():
        with torch.inference_mode():
            outputs = model(pixel_values=pixels)
            # The SDK returns complete, host-owned float32 token and pooled features.
            return (
                outputs.last_hidden_state.to(device="cpu", dtype=torch.float32).numpy(),
                outputs.pooler_output.to(device="cpu", dtype=torch.float32).numpy(),
            )

    return invoke, torch.cuda.synchronize, f"transformers-{transformers.__version__}"


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


def _summary(features):
    import numpy as np

    hidden, pooled = features
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] == 0 or hidden.shape[2] == 0:
        raise ValueError("DINOv2 must return one nonempty token-feature matrix")
    if pooled.shape != (1, hidden.shape[2]):
        raise ValueError("DINOv2 must return one pooled feature per hidden feature")
    if not (np.isfinite(hidden).all() and np.isfinite(pooled).all()):
        raise ValueError("DINOv2 returned nonfinite features")
    return {
        "last_hidden_state_shape": list(hidden.shape),
        "pooler_output_shape": list(pooled.shape),
        "element_count": int(hidden.size + pooled.size),
        "finite": True,
    }


def run(arguments):
    """Execute the existing family-reference protocol with reporting untimed."""
    if (
        arguments.family != "dinov2"
        or arguments.operation != "extract_features"
        or arguments.selected_task != "image_to_token_and_pooled_features"
    ):
        raise ValueError(
            "reference requires dinov2 image_to_token_and_pooled_features/extract_features"
        )
    if arguments.warmup < 0 or arguments.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    timing = json.loads(arguments.timing_contract_json)
    if timing != TIMING or any(
        type(timing[key]) is not type(value) for key, value in TIMING.items()
    ):
        raise ValueError("reference requires its declared model-call timing policy")
    if json.loads(arguments.adapter_options_json) or arguments.trust_remote_code:
        raise ValueError("reference does not accept adapter options or remote code")
    request = json.loads(arguments.request_json)
    if (
        not isinstance(request, dict)
        or set(request) - {"image_path", "batch_size"}
        or not isinstance(request.get("image_path"), str)
        or not request["image_path"]
        or type(request.get("batch_size", 1)) is not int
        or request.get("batch_size", 1) != 1
    ):
        raise ValueError("reference requires one image_path and no runtime Config")
    invoke, synchronize, framework = _load_reference(arguments, request)
    samples, features = _measure(invoke, synchronize, arguments.warmup, arguments.iterations)
    return {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "model": arguments.model,
        "family": arguments.family,
        "operation": arguments.operation,
        "case_name": arguments.case_name,
        "selected_task": arguments.selected_task,
        "precision": arguments.precision,
        "mode": arguments.mode,
        "framework": framework,
        "measurement": {"warmup": arguments.warmup, "iterations": arguments.iterations},
        "measurement_policy": timing,
        **timing,
        "samples_ms": samples,
        "metrics": {"latency_ms": {"p50": statistics.median(samples)}},
        "output_summary": _summary(features),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "family",
        "operation",
        "selected-task",
        "request-json",
        "adapter-options-json",
        "timing-contract-json",
        "case-name",
    ):
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
