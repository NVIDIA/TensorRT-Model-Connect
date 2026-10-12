# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YOLO11's native pipeline for the trtmc-perf-serve reference backend (``detect``): the official Ultralytics
archive (yolo11n.pt), letterboxed to 640, Ultralytics NMS (IoU 0.7, at most 300 boxes) at the request's
``score_threshold``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


SIZE = 640  # the letterboxed input side
DTYPES = ("fp16", "fp32")


def _model_directory(spec: Any) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=spec.model, revision=spec.revision,
                                  local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"))


def _letterbox(image: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    """The image scaled into a 640x640 canvas (grey 114 padding, centered), as 0-1 NCHW pixels."""
    import numpy as np
    import torch
    from PIL import Image

    source = image.convert("RGB")
    scale = min(SIZE / source.height, SIZE / source.width)
    resized = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    pad_x, pad_y = (SIZE - resized.width) // 2, (SIZE - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None].copy() / 255.0
    return (torch.from_numpy(pixels).to(device="cuda", dtype=dtype),
            {"height": source.height, "width": source.width, "scale": scale, "pad_x": pad_x, "pad_y": pad_y})


def _observation(rows: list[list[float]], geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Detections (x1, y1, x2, y2, score, class) in letterbox pixels, as source-image boxes."""
    scale, pad_x, pad_y = float(geometry["scale"]), int(geometry["pad_x"]), int(geometry["pad_y"])
    boxes = [[(row[0] - pad_x) / scale, (row[1] - pad_y) / scale, (row[2] - pad_x) / scale, (row[3] - pad_y) / scale]
             for row in rows]
    return {"image_height": int(geometry["height"]), "image_width": int(geometry["width"]), "detections": len(rows),
            "boxes": boxes, "scores": [float(row[4]) for row in rows], "class_ids": [int(row[5]) for row in rows],
            "coordinates": "xyxy", "units": "pixels"}

ARCHIVE = "yolo11n.pt"


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import torch

        if spec.precision not in DTYPES:
            raise self.host.Error(f"YOLO11 reference runs at fp16 or fp32, not {spec.precision}")
        self.spec = spec
        blob = torch.load(str(_model_directory(spec) / ARCHIVE), map_location="cpu", weights_only=False)
        model = blob.get("model") if isinstance(blob, dict) else None
        if model is None:
            raise self.host.Error(f"{ARCHIVE} has no model entry")
        self.model = model.eval().to(device="cuda", dtype=spec.dtype)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        from ultralytics.utils.ops import non_max_suppression

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        pixels, geometry = _letterbox(image, self.spec.dtype)
        threshold = float(request.get("score_threshold", 0.25))

        def run() -> Any:
            raw = self.model(pixels)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            return non_max_suppression(raw, conf_thres=threshold, iou_thres=0.7, max_det=300)[0]

        kept, model_ms = self.host.timed(run)
        return self.host.invocation(_observation(kept.detach().float().cpu().tolist(), geometry), model_ms)
