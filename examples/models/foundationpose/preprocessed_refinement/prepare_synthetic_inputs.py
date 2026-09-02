# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic preprocessed RGB+XYZ crop inputs for the example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    count = 3
    rendered = np.zeros((count, 160, 160, 6), dtype=np.float32)
    observed = np.zeros_like(rendered)
    y, x = np.mgrid[:160, :160].astype(np.float32)
    xn, yn = (x - 79.5) / 80.0, (y - 79.5) / 80.0
    mask = xn * xn + yn * yn <= 0.72
    for index in range(count):
        rendered[index, ..., 0][mask] = (0.08 * (index + 1) + 0.4 * (x / 159.0))[mask]
        rendered[index, ..., 1][mask] = (0.5 * (y / 159.0))[mask]
        rendered[index, ..., 2][mask] = 0.35
        rendered[index, ..., 3][mask] = (0.2 * xn)[mask]
        rendered[index, ..., 4][mask] = (0.2 * yn)[mask]
        rendered[index, ..., 5][mask] = 0.10 + 0.01 * index
        observed[index, ..., 0][mask] = (0.25 + 0.4 * (x / 159.0))[mask]
        observed[index, ..., 1][mask] = (0.10 + 0.5 * (y / 159.0))[mask]
        observed[index, ..., 2][mask] = 0.40
        observed[index, ..., 3][mask] = (0.2 * xn + 0.02 * index)[mask]
        observed[index, ..., 4][mask] = (0.2 * yn)[mask]
        observed[index, ..., 5][mask] = 0.11
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    poses[:, 0, 3] = (-0.02, 0.01, 0.02)
    poses.tofile(args.output / "candidate_poses.f32")
    rendered.tofile(args.output / "rendered_features.f32")
    observed.tofile(args.output / "observed_features.f32")


if __name__ == "__main__":
    main()
