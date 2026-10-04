# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YOLOv10's native pipeline for the trtmc-perf-serve reference backend (``detect``): the Ultralytics YOLOv10
architecture with the checkpoint's weights (the family's checkpoint reader), letterboxed to 640, NMS-free
(the one-to-one head's detections at the request's ``score_threshold``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from trtmc_perf_serving.backends.base import BackendError, Invocation
from trtmc_perf_serving.backends.reference.common import ReferenceSpec, invocation, load_image, required, timed

SIZE = 640  # the letterboxed input side
DTYPES = ("fp16", "fp32")


def _model_directory(spec: ReferenceSpec) -> Path:
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


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        import json

        from ultralytics import YOLO

        from families.yolov10.checkpoint import Checkpoint

        if spec.precision not in DTYPES:
            raise BackendError(f"YOLOv10 reference runs at fp16 or fp32, not {spec.precision}")
        self.spec = spec
        directory = _model_directory(spec)
        architecture = json.loads((directory / "config.json").read_text()).get("model")
        if architecture not in ("yolov10n.yaml", "yolov10s.yaml", "yolov10x.yaml"):
            raise BackendError(f"unsupported YOLOv10 architecture {architecture!r}")
        model = YOLO(architecture, task="detect").model
        checkpoint = Checkpoint.open(directory, framework="pt")
        state = {name[len("model."):]: reader.get_tensor(name) for name, reader in checkpoint.tensor_map.items()
                 if name.startswith("model.")}
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            raise BackendError(f"YOLOv10 checkpoint lacks tensors: {missing[:5]}")
        self.model = model.eval().to(device="cuda", dtype=spec.dtype)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        pixels, geometry = _letterbox(load_image(str(required(request, "image_path"))), self.spec.dtype)
        threshold = float(request.get("score_threshold", 0.25))

        def run() -> Any:
            raw = self.model(pixels)
            rows = (raw[0] if isinstance(raw, (list, tuple)) else raw)[0]
            return rows[rows[:, 4] >= threshold]

        kept, model_ms = timed(run)
        return invocation(_observation(kept.detach().float().cpu().tolist(), geometry), model_ms)
