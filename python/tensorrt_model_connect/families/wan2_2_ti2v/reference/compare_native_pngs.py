# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native Wan2.2 PNG frames with the official FP32 video tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_MINIMUM_COSINE = 0.998


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(4 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-cosine",
        type=_cosine_threshold,
        default=DEFAULT_MINIMUM_COSINE,
        help="Minimum whole-video uint8 cosine (default: %(default)s)",
    )
    parser.add_argument(
        "--min-frame-cosine",
        type=_cosine_threshold,
        default=DEFAULT_MINIMUM_COSINE,
        help="Minimum worst-frame uint8 cosine (default: %(default)s)",
    )
    return parser.parse_args()


def _cosine_threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("cosine thresholds must be in [0, 1]")
    return threshold


def _accuracy_failures(
    *,
    cosine: float,
    minimum_frame_cosine: float,
    min_cosine: float,
    min_frame_cosine: float,
) -> list[str]:
    failures = []
    if cosine < min_cosine:
        failures.append(f"cosine_uint8={cosine:.12f} < {min_cosine:.12f}")
    if minimum_frame_cosine < min_frame_cosine:
        failures.append(
            "minimum_frame_cosine_uint8="
            f"{minimum_frame_cosine:.12f} < {min_frame_cosine:.12f}"
        )
    return failures


def main() -> None:
    args = _parse_args()
    reference_path = args.reference.resolve()
    frames_dir = args.frames_dir.resolve()
    output_path = args.output.resolve()

    video = torch.load(reference_path, map_location="cpu", weights_only=True)
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError("Wan2.2 reference must be a rank-4 tensor [C,T,H,W]")
    channels, frame_count, height, width = (int(value) for value in video.shape)
    if channels != 3:
        raise ValueError(f"Wan2.2 reference must have three channels, got {channels}")

    expected_paths = [frames_dir / f"frame_{index:04d}.png" for index in range(frame_count)]
    actual_paths = sorted(frames_dir.glob("frame_*.png"))
    if actual_paths != expected_paths:
        missing = [str(path) for path in expected_paths if not path.is_file()]
        extras = [str(path) for path in actual_paths if path not in set(expected_paths)]
        raise ValueError(
            f"Native frame set is not contiguous: missing={missing[:5]}, extras={extras[:5]}"
        )

    total_values = frame_count * height * width * channels
    differing_values = 0
    absolute_error_sum = 0
    squared_error_sum = 0
    maximum_absolute_error = 0
    exact_frames = 0
    minimum_frame_cosine = 1.0
    worst_frame_index = 0
    expected_square_sum = 0
    actual_square_sum = 0
    dot_product = 0

    for index, path in enumerate(expected_paths):
        expected = (
            ((video[:, index].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        with Image.open(path) as image:
            actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if actual.shape != (height, width, channels):
            raise ValueError(
                f"{path} has shape {actual.shape}, expected {(height, width, channels)}"
            )

        expected_i64 = expected.astype(np.int64)
        actual_i64 = actual.astype(np.int64)
        difference = actual_i64 - expected_i64
        absolute = np.abs(difference)
        frame_differing = int(np.count_nonzero(difference))
        differing_values += frame_differing
        if frame_differing == 0:
            exact_frames += 1
        absolute_error_sum += int(absolute.sum(dtype=np.int64))
        squared_error_sum += int(np.square(difference).sum(dtype=np.int64))
        maximum_absolute_error = max(maximum_absolute_error, int(absolute.max(initial=0)))

        frame_expected_square = int(np.square(expected_i64).sum(dtype=np.int64))
        frame_actual_square = int(np.square(actual_i64).sum(dtype=np.int64))
        frame_dot = int(np.multiply(expected_i64, actual_i64).sum(dtype=np.int64))
        expected_square_sum += frame_expected_square
        actual_square_sum += frame_actual_square
        dot_product += frame_dot
        denominator = math.sqrt(frame_expected_square * frame_actual_square)
        frame_cosine = frame_dot / denominator if denominator else 1.0
        if frame_cosine < minimum_frame_cosine:
            minimum_frame_cosine = frame_cosine
            worst_frame_index = index

    denominator = math.sqrt(expected_square_sum * actual_square_sum)
    cosine = dot_product / denominator if denominator else 1.0
    mean_absolute_error = absolute_error_sum / total_values
    rmse = math.sqrt(squared_error_sum / total_values)
    accuracy_failures = _accuracy_failures(
        cosine=cosine,
        minimum_frame_cosine=minimum_frame_cosine,
        min_cosine=args.min_cosine,
        min_frame_cosine=args.min_frame_cosine,
    )
    report = {
        "schema": "trtmc.wan2_2_ti2v.native-png-reference.v1",
        "reference": str(reference_path),
        "reference_sha256": _sha256_file(reference_path),
        "frames_dir": str(frames_dir),
        "frames_sha256": _directory_sha256(expected_paths),
        "shape": [frame_count, height, width, channels],
        "frame_count": frame_count,
        "exact_frames": exact_frames,
        "total_values": total_values,
        "differing_values": differing_values,
        "maximum_absolute_error_uint8": maximum_absolute_error,
        "mean_absolute_error_uint8": mean_absolute_error,
        "rmse_uint8": rmse,
        "cosine_uint8": cosine,
        "minimum_frame_cosine_uint8": minimum_frame_cosine,
        "worst_frame_index": worst_frame_index,
        "exact": differing_values == 0,
        "qualification": {
            "min_cosine_uint8": args.min_cosine,
            "min_frame_cosine_uint8": args.min_frame_cosine,
            "passed": not accuracy_failures,
            "failures": accuracy_failures,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if accuracy_failures:
        raise SystemExit("Wan2.2 accuracy qualification failed: " + "; ".join(accuracy_failures))


if __name__ == "__main__":
    main()
