# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed visual parity comparator for decoded MiniMax-H3 frame arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference_raw = np.load(args.reference)
    candidate_raw = np.load(args.candidate)

    def normalized(array):
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32) / np.iinfo(array.dtype).max
        return array.astype(np.float32)

    reference = normalized(reference_raw)
    candidate = normalized(candidate_raw)
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate video contains non-finite pixels")
    thresholds = json.loads(Path(args.thresholds).read_text())["accuracy"]
    error = candidate - reference
    mse = float(np.mean(np.square(error), dtype=np.float64))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    receipt = {
        "shape": list(reference.shape),
        "mse": mse,
        "mean_absolute_error": mae,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "psnr_db": psnr,
        "minimum_psnr_db": float(thresholds["minimum_psnr_db"]),
        "maximum_mean_absolute_error": float(thresholds["maximum_mean_absolute_error"]),
    }
    receipt["passed"] = (
        psnr >= receipt["minimum_psnr_db"] and mae <= receipt["maximum_mean_absolute_error"]
    )
    Path(args.output).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
