# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native Wan2.2 scheduler outputs with an official latent trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


LATENT_SHAPE = (48, 31, 44, 80)
LATENT_COUNT = int(np.prod(LATENT_SHAPE))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(values))).hexdigest()


def _compare(official: np.ndarray, native: np.ndarray) -> dict:
    official_bits = official.view(np.uint32).reshape(-1)
    native_bits = native.view(np.uint32).reshape(-1)
    mismatches = official_bits != native_bits
    mismatch_count = int(np.count_nonzero(mismatches))
    first_mismatch = None
    if mismatch_count:
        index = int(np.flatnonzero(mismatches)[0])
        first_mismatch = {
            "index": index,
            "official_bits": f"{int(official_bits[index]):08x}",
            "native_bits": f"{int(native_bits[index]):08x}",
        }
        delta = native.astype(np.float64) - official.astype(np.float64)
        official64 = official.astype(np.float64).reshape(-1)
        native64 = native.astype(np.float64).reshape(-1)
        denominator = np.linalg.norm(official64) * np.linalg.norm(native64)
        cosine = float(np.dot(official64, native64) / denominator)
        max_abs_error = float(np.max(np.abs(delta)))
        mean_abs_error = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
    else:
        cosine = 1.0
        max_abs_error = 0.0
        mean_abs_error = 0.0
        rmse = 0.0
    return {
        "official_sha256": _sha256(official),
        "native_sha256": _sha256(native),
        "bitwise_mismatch_count": mismatch_count,
        "bitwise_match_fraction": (LATENT_COUNT - mismatch_count) / LATENT_COUNT,
        "first_mismatch": first_mismatch,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = _parse_args()
    trace_dir = args.trace_dir.resolve()
    reference_path = args.reference.resolve()
    trajectory = torch.load(
        reference_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    expected_shape = (50, *LATENT_SHAPE)
    if tuple(trajectory.shape) != expected_shape or trajectory.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected official trajectory: shape={tuple(trajectory.shape)}, "
            f"dtype={trajectory.dtype}"
        )

    steps = []
    for step in range(expected_shape[0]):
        native_path = trace_dir / f"step_{step}_output_latents.f32"
        if native_path.stat().st_size != LATENT_COUNT * np.dtype(np.float32).itemsize:
            raise RuntimeError(f"Native trace has the wrong size: {native_path}")
        official = trajectory[step].numpy()
        native = np.memmap(native_path, mode="r", dtype=np.float32, shape=LATENT_SHAPE)
        steps.append({"step": step + 1, **_compare(official, native)})

    first_divergence = next(
        (record["step"] for record in steps if record["bitwise_mismatch_count"]),
        None,
    )
    payload = {
        "schema_version": 1,
        "kind": "wan2_2_ti2v_native_scheduler_trace_comparison",
        "trace_dir": str(trace_dir),
        "reference": str(reference_path),
        "shape": list(expected_shape),
        "first_divergence_step": first_divergence,
        "all_steps_bitwise_exact": first_divergence is None,
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "first_divergence_step": first_divergence,
                "all_steps_bitwise_exact": first_divergence is None,
                "final_step": steps[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
