#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two SANA-WM frame directories by decoded RGB component values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _frames(path: Path) -> list[Path]:
    return sorted(path.glob("frame_*.png"))


def compare(reference_dir: Path, actual_dir: Path) -> dict:
    reference = _frames(reference_dir)
    actual = _frames(actual_dir)
    result: dict = {
        "reference_count": len(reference),
        "actual_count": len(actual),
        "count_match": len(reference) == len(actual),
        "names_match": [path.name for path in reference]
        == [path.name for path in actual],
        "exact_pixel_frames": 0,
        "different_components": 0,
        "total_components": 0,
        "max_abs": 0,
        "mean_abs": 0.0,
        "first_mismatch": None,
    }
    absolute_sum = 0
    for reference_path, actual_path in zip(reference, actual):
        expected = np.asarray(Image.open(reference_path).convert("RGB"))
        observed = np.asarray(Image.open(actual_path).convert("RGB"))
        if expected.shape != observed.shape:
            result["first_mismatch"] = {
                "frame": reference_path.name,
                "reference_shape": list(expected.shape),
                "actual_shape": list(observed.shape),
            }
            result["different_components"] += 1
            result["total_components"] += 1
            result["max_abs"] = 255
            absolute_sum += 255
            continue

        delta = np.abs(expected.astype(np.int16) - observed.astype(np.int16))
        different = int(np.count_nonzero(delta))
        result["different_components"] += different
        result["total_components"] += int(delta.size)
        result["max_abs"] = max(result["max_abs"], int(delta.max(initial=0)))
        absolute_sum += int(delta.sum(dtype=np.int64))
        if different == 0:
            result["exact_pixel_frames"] += 1
        elif result["first_mismatch"] is None:
            y, x, _ = np.argwhere(delta != 0)[0]
            result["first_mismatch"] = {
                "frame": reference_path.name,
                "x": int(x),
                "y": int(y),
                "reference_rgb": expected[y, x].tolist(),
                "actual_rgb": observed[y, x].tolist(),
            }

    if result["total_components"]:
        result["mean_abs"] = absolute_sum / result["total_components"]
    result["exact"] = bool(
        result["count_match"]
        and result["names_match"]
        and result["different_components"] == 0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(args.reference, args.actual)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
