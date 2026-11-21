# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless transport of a generated PNG, outside the timed Task call."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any


def png_response(path: Path, *, max_pixels: int = 16_000_000) -> dict[str, Any]:
    from PIL import Image

    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or not 0 < image.width * image.height <= max_pixels:
            raise ValueError("generated output must be a PNG within the pixel limit")
        image.load()
        width, height = image.size
        channels = len(image.getbands())
    return {"b64_json": base64.b64encode(data).decode("ascii"),
            "sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height,
            "channels": channels}


def generated_image(workdir: Path) -> dict[str, Any]:
    """Read the existing worker artifact or encode the reference's HWC tensor."""
    paths = list(workdir.glob("output*.png"))
    if not paths and (workdir / "output.npy").is_file():
        import numpy as np
        from PIL import Image

        from .digests import _media_frames, _to_uint8

        frames = _media_frames(np.load(workdir / "output.npy"))
        if len(frames) != 1:
            raise ValueError("image endpoint requires exactly one output image")
        path = workdir / "output.png"
        Image.fromarray(_to_uint8(frames[0])).save(path)
        paths = [path]
    if len(paths) != 1:
        raise ValueError("image endpoint requires exactly one output PNG")
    return png_response(paths[0])
