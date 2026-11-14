#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write one image/question pair for AIPerf without recompressing the image."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.image.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB" or image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
            parser.error("image must be one RGB PNG or JPEG frame")
        mime = "jpeg" if image.format == "JPEG" else "png"
        image.verify()
    if not args.prompt.strip():
        parser.error("prompt must not be empty")
    row = {"session_id": "image-question", "texts": [args.prompt],
           "image": f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")}
    with args.output.open("x") as output:
        output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
