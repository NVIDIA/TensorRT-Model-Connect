#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a native Wan2.2 SDPA raw buffer with the official BF16 capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_as_float64(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.uint32) << np.uint32(16)
    return words.view(np.float32).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--engine-config", required=True)
    args = parser.parse_args()

    capture = json.loads(args.capture_manifest.read_text())
    reference_path = Path(capture["files"]["o"]["path"])
    reference = np.fromfile(reference_path, dtype=np.uint16)
    actual = np.fromfile(args.actual, dtype=np.uint16)
    if actual.shape != reference.shape:
        raise ValueError(
            f"element count mismatch: actual={actual.size}, reference={reference.size}"
        )

    exact_mask = actual == reference
    reference_fp64 = _bf16_as_float64(reference)
    actual_fp64 = _bf16_as_float64(actual)
    delta = actual_fp64 - reference_fp64
    reference_norm = np.linalg.norm(reference_fp64)
    actual_norm = np.linalg.norm(actual_fp64)
    cosine = float(np.dot(reference_fp64, actual_fp64) / (reference_norm * actual_norm))
    report = {
        "kind": "wan2_2_ti2v_native_cudnn_sdpa_qualification",
        "implementation": args.implementation,
        "engine_config": args.engine_config,
        "capture_manifest": str(args.capture_manifest),
        "reference": {
            "path": str(reference_path),
            "bytes": reference_path.stat().st_size,
            "sha256": _sha256(reference_path),
        },
        "actual": {
            "path": str(args.actual),
            "bytes": args.actual.stat().st_size,
            "sha256": _sha256(args.actual),
        },
        "logical_shape_bhsd": capture["files"]["o"]["logical_shape_bhsd"],
        "logical_stride_bhsd": capture["files"]["o"]["logical_stride_bhsd"],
        "physical_shape_bshd": capture["files"]["o"]["physical_shape_bshd"],
        "metrics": {
            "bitwise_equal": bool(np.all(exact_mask)),
            "exact_elements": int(np.count_nonzero(exact_mask)),
            "elements": int(reference.size),
            "exact_fraction": float(np.mean(exact_mask)),
            "max_abs_error": float(np.max(np.abs(delta))),
            "mean_abs_error": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "cosine_similarity": cosine,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["metrics"]["bitwise_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
