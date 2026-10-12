# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stereo conversion-parity inputs: the 15 Middlebury v3 trainingQ scenes (pinned archives) cropped / padded to
the 700x700 profile by the family's preparation; ``python inputs.py --count N --output DIR`` prints the
first N as JSON records (left and right image paths; the ground truth rides along)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from families.fast_foundation_stereo.tests.prepare_middlebury_q import (DATA_URL, GROUND_TRUTH_URL, _download,
                                                                         prepare_archives)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = arguments.output / "dataset.json"
    if not manifest.is_file():
        arguments.output.mkdir(parents=True, exist_ok=True)
        data, ground_truth = arguments.output / "MiddEval3-data-Q.zip", arguments.output / "MiddEval3-GT0-Q.zip"
        _download(DATA_URL, data)
        _download(GROUND_TRUTH_URL, ground_truth)
        prepare_archives(data, ground_truth, arguments.output)
    requests = json.loads(manifest.read_text())["requests"][: arguments.count]
    print(json.dumps([{"id": request["sample_id"],
                       "request": {"left_image_path": request["inputs"]["left_image"],
                                   "right_image_path": request["inputs"]["right_image"]},
                       "label": {"disparity": request["inputs"]["ground_truth_disparity"],
                                 "valid": request["inputs"]["valid_nonocc_mask"]}} for request in requests]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
