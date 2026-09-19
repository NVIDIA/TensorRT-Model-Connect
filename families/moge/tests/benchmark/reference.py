# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official Accuracy and Performance reference for MoGe-2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np

from families.moge.tests.reference_support import OfficialReference, read_image


def _source_root() -> Path:
    root = Path(sys.prefix) / "trtmc-reference/MoGe"
    if not (root / "moge/model/v2.py").is_file():
        raise RuntimeError(f"MoGe reference is incomplete: {root}")
    return root


def _checkpoint(model: str, revision: str | None, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model,
            revision=revision,
            filename="model.pt",
            local_files_only=local_files_only,
        )
    )


def _options(request: Mapping[str, Any]) -> tuple[int, float | None]:
    unknown = sorted(set(request) - {"image_path", "num_tokens", "fov_x"})
    if unknown:
        raise ValueError("unsupported MoGe request fields: " + ", ".join(unknown))
    num_tokens = request.get("num_tokens")
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens != 1800:
        raise ValueError("MoGe reference requires num_tokens=1800")
    fov_x = request.get("fov_x")
    if fov_x is not None:
        fov_x = float(fov_x)
        if not 0.0 < fov_x < 180.0:
            raise ValueError("MoGe fov_x must be between zero and 180 degrees")
    return num_tokens, fov_x


def _write_summary(arrays: Mapping[str, Any], prefix: Path) -> dict[str, Any]:
    points = np.asarray(arrays["points"], dtype="<f4")
    depth = np.asarray(arrays["depth"], dtype="<f4")
    mask = np.asarray(arrays["mask"], dtype=np.uint8)
    intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float32)
    if depth.ndim != 2 or points.shape != (*depth.shape, 3) or mask.shape != depth.shape:
        raise ValueError("MoGe reference returned inconsistent geometry shapes")
    height, width = depth.shape
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "points_artifact": Path(str(prefix) + ".points.f32"),
        "depth_artifact": Path(str(prefix) + ".depth.f32"),
        "valid_mask_artifact": Path(str(prefix) + ".mask.u8"),
    }
    points.tofile(paths["points_artifact"])
    depth.tofile(paths["depth_artifact"])
    mask.tofile(paths["valid_mask_artifact"])
    return {
        "geometry_images": 1,
        "geometry_pixels": height * width,
        "height": height,
        "width": width,
        "point_shape": [height, width, 3],
        "valid_pixels": int(mask.astype(bool).sum()),
        "normalized_intrinsics": intrinsics.tolist(),
        "units": "meters",
        "camera_axes": ["right", "down", "forward"],
        "intrinsics_coordinates": "normalized_uv",
        **{name: str(path.resolve()) for name, path in paths.items()},
    }


def _run_accuracy(arguments: argparse.Namespace) -> int:
    payload = json.loads(arguments.request.read_text(encoding="utf-8"))
    request = payload.get("request", {})
    samples = payload.get("samples")
    if not isinstance(request, Mapping) or not isinstance(samples, list) or not samples:
        raise ValueError("Accuracy reference requires request and samples")
    num_tokens, fov_x = _options(request)
    model = OfficialReference(
        _source_root(),
        _checkpoint(
            str(payload["model"]),
            payload.get("revision"),
            os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
        ),
    )
    output_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Accuracy sample {index} must be an object")
        arrays = model.infer(
            read_image(Path(str(sample["image_path"]))),
            num_tokens=num_tokens,
            fov_x=fov_x,
        )
        summary = _write_summary(
            arrays, arguments.output.with_name(f"{arguments.output.stem}.{index}.geometry")
        )
        output_samples.append({"sample_id": str(sample["sample_id"]), **summary})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": output_samples}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _run_performance(arguments: argparse.Namespace) -> int:
    import torch

    request = json.loads(arguments.request_json)
    timing = json.loads(arguments.timing_contract_json)
    num_tokens, fov_x = _options(request)
    reference = OfficialReference(
        _source_root(),
        _checkpoint(arguments.model, arguments.revision, arguments.local_files_only),
    )
    pixels = read_image(Path(str(request["image_path"])))

    def invoke():
        return reference.infer(pixels, num_tokens=num_tokens, fov_x=fov_x)

    arrays = None
    for _ in range(arguments.warmup):
        arrays = invoke()
    samples_ms = []
    for _ in range(arguments.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        arrays = invoke()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    if arrays is None:
        raise RuntimeError("MoGe reference produced no geometry")
    summary = _write_summary(arrays, arguments.output.with_suffix(".geometry"))
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
