# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged upstream E2E oracle outside the MC build process."""

from pathlib import Path
import sys

import numpy as np


def run(model_dir: Path, image: Path, output: Path) -> None:

    import torch
    from PIL import Image
    from ultralytics.utils.ops import non_max_suppression

    # The reference must run the archive under test, and must see the same
    # letterboxed pixels the runtime seam produced.
    blob = torch.load(str(model_dir / "yolov8n.pt"), map_location="cpu", weights_only=False)
    reference = blob["model"].float().eval()

    size = 640
    source = Image.open(image).convert("RGB")
    scale = min(size / source.height, size / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - resized.width) // 2
    pad_y = (size - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    with torch.no_grad():
        raw = reference(torch.from_numpy(pixels.astype(np.float32)))
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    kept = non_max_suppression(raw, conf_thres=0.25, iou_thres=0.7, max_det=300)[0].numpy()

    np.savez(output, kept=kept, scale=scale, pad_x=pad_x, pad_y=pad_y)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
