# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official Accuracy and Performance reference for Fast Foundation Stereo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping


def _model_root() -> Path:
    root = Path(sys.prefix) / "trtmc-reference/Fast-FoundationStereo"
    required = (
        root / "core/foundation_stereo.py",
        root / "core/submodule.py",
        root / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Fast Foundation Stereo reference is incomplete: {root}")
    return root


def _load_model(max_disparity: int, valid_iters: int):
    import torch

    root = _model_root()
    previous = Path.cwd()
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        model = torch.load(
            root / "weights/23-36-37/model_best_bp2_serialize.pth",
            map_location="cpu",
            weights_only=False,
        )
    finally:
        os.chdir(previous)
    model.args.max_disp = max_disparity
    model.args.valid_iters = valid_iters
    model.args.normalize = True
    return model.cuda().eval(), InputPadder, AMP_DTYPE


def _image(path: str, *, height: int, width: int):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.shape != (height, width, 3):
        raise ValueError(f"stereo image must be {height}x{width} RGB, got {pixels.shape}")
    return pixels


def _inference(
    model: Any,
    input_padder: Any,
    amp_dtype: Any,
    left: Any,
    right: Any,
    *,
    height: int,
    width: int,
    valid_iters: int,
):
    import numpy as np
    import torch

    left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = input_padder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
        disparity = model.forward(
            left_tensor,
            right_tensor,
            iters=valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    values = padder.unpad(disparity.float()).cpu().numpy().reshape(height, width)
    return np.clip(values, 0, None).astype("<f4", copy=False)


def _options(request: Mapping[str, Any]) -> tuple[int, int, int, int]:
    height = int(request.get("height", 700))
    width = int(request.get("width", 700))
    max_disparity = int(request.get("max_disp", 192))
    valid_iters = int(request.get("valid_iters", 8))
    if (height, width, max_disparity, valid_iters) != (700, 700, 192, 8):
        raise ValueError(
            "Fast Foundation Stereo reference requires 700x700, max_disp=192, valid_iters=8"
        )
    return height, width, max_disparity, valid_iters


def _summary(values: Any, artifact: Path) -> dict[str, Any]:
    import numpy as np

    values.tofile(artifact)
    return {
        "height": int(values.shape[0]),
        "width": int(values.shape[1]),
        "element_count": int(values.size),
        "finite_fraction": float(np.isfinite(values).mean()),
        "nonnegative_fraction": float((values >= 0).mean()),
        "disparity_artifact": str(artifact.resolve()),
    }


def _run_accuracy(arguments: argparse.Namespace) -> int:
    payload = json.loads(arguments.request.read_text(encoding="utf-8"))
    request = payload.get("request", {})
    samples = payload.get("samples")
    if not isinstance(request, Mapping) or not isinstance(samples, list) or not samples:
        raise ValueError("Accuracy reference requires request and samples")
    height, width, max_disparity, valid_iters = _options(request)
    model, input_padder, amp_dtype = _load_model(max_disparity, valid_iters)
    output_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Accuracy sample {index} must be an object")
        left = _image(str(sample["left_image_path"]), height=height, width=width)
        right = _image(str(sample["right_image_path"]), height=height, width=width)
        values = _inference(
            model,
            input_padder,
            amp_dtype,
            left,
            right,
            height=height,
            width=width,
            valid_iters=valid_iters,
        )
        artifact = arguments.output.with_name(
            f"{arguments.output.stem}.{index}.disparity.f32"
        )
        output_samples.append(
            {"sample_id": str(sample["sample_id"]), **_summary(values, artifact)}
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": output_samples}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _run_performance(arguments: argparse.Namespace) -> int:
    import torch

    request = json.loads(arguments.request_json)
    timing = json.loads(arguments.timing_contract_json)
    height, width, max_disparity, valid_iters = _options(request)
    model, input_padder, amp_dtype = _load_model(max_disparity, valid_iters)
    left = _image(str(request["left_image_path"]), height=height, width=width)
    right = _image(str(request["right_image_path"]), height=height, width=width)

    def invoke():
        return _inference(
            model,
            input_padder,
            amp_dtype,
            left,
            right,
            height=height,
            width=width,
            valid_iters=valid_iters,
        )

    result = None
    for _ in range(arguments.warmup):
        result = invoke()
    samples_ms = []
    for _ in range(arguments.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        result = invoke()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    if result is None:
        raise RuntimeError("reference produced no disparity output")
    artifact = arguments.output.with_suffix(".disparity.f32")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(result, artifact)
    value = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "model": arguments.model,
        "family": arguments.family,
        "operation": arguments.operation,
        "case_name": arguments.case_name,
        "selected_task": arguments.selected_task,
        "precision": arguments.precision,
        "mode": arguments.mode,
        "framework": f"torch-{torch.__version__}",
        "measurement": {
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
        },
        "measurement_policy": timing,
        **timing,
        "samples_ms": samples_ms,
        "metrics": {"latency_ms": {"p50": statistics.median(samples_ms)}},
        "output_summary": summary,
    }
    arguments.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return 0


def _accuracy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _performance_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in (
        "model",
        "family",
        "manifest",
        "operation",
        "selected-task",
        "request-json",
        "adapter-options-json",
        "timing-contract-json",
        "precision",
        "mode",
        "padding",
        "case-name",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    if "--request" in sys.argv[1:]:
        return _run_accuracy(_accuracy_parser().parse_args())
    return _run_performance(_performance_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
