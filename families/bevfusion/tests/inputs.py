# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare deterministic BEV feature fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(output, seed=0, feature_maps=None):
    output = Path(output) / "bevfusion" / "tests"
    (output / "data").mkdir(parents=True, exist_ok=True)
    manifests = sorted((Path(__file__).parent / "manifests").glob("*.json"))
    if feature_maps is None:
        feature_maps = json.loads(manifests[0].read_text())["testcases"][0]["feature_maps"]
    rng = np.random.default_rng(seed)
    for feature in feature_maps:
        shape = (1, *(feature[key] for key in ("channels", "height", "width")))
        rng.standard_normal(shape, dtype=np.float32).tofile(output / feature["values_path"])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    prepare(arguments.output, arguments.seed)
