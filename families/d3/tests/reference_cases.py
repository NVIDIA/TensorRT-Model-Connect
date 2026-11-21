# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate independent B1 oracle requests while loading one checkpoint once."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from reference import evaluate, load_model


def main():
    checkpoint, inputs, output = map(Path, sys.argv[1:4])
    model = load_model(checkpoint, os.environ.get("TRTMC_D3_REFERENCE_MODEL", "d3-edge"))
    cases = json.loads(inputs.read_text())
    items = cases["inputs"] + cases["same_length"]["inputs"]
    results = []
    for item in items:
        images = []
        if item["image"]:
            pixels = np.zeros((256, 256, 3), np.uint8)
            pixels[..., item["image_channel"]] = 255
            images = [Image.fromarray(pixels)]
        results.append(evaluate(model, json.loads(item["document"]), images))
    output.write_text(json.dumps(results, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
