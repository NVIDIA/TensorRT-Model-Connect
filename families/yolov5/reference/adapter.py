# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YOLOv5's native pipeline for the trtmc-perf-serve reference backend (``detect``): the official archive's
backbone and neck (Ultralytics' model parser, BatchNorm eps 1e-3) with the anchor head decoded as YOLOv5
does, letterboxed to 640, class-aware NMS (IoU 0.7, at most 300 boxes) at the request's ``score_threshold``.
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

ARCHIVE = "yolov5n.pt"
HEAD_SOURCES = (17, 20, 23)  # the stages feeding the three detection levels


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import copy

        import torch
        from ultralytics.nn.tasks import parse_model

        from families.yolov5.checkpoint import _collect, _install_placeholders

        if spec.precision not in DTYPES:
            raise self.host.Error(f"YOLOv5 reference runs at fp16 or fp32, not {spec.precision}")
        self.spec, dtype = spec, spec.dtype
        _install_placeholders()
        blob = torch.load(str(_model_directory(spec) / ARCHIVE), map_location="cpu", weights_only=False)
        archive = blob.get("model") if isinstance(blob, dict) else None
        if archive is None:
            raise self.host.Error(f"{ARCHIVE} has no model entry")
        layout = copy.deepcopy(archive.__dict__["yaml"])
        layout["head"].pop()  # the Detect layer: decoded below
        stages, _ = parse_model(layout, ch=3, verbose=False)
        weights: dict[str, Any] = {}
        _collect(archive, "", weights)
        body = {name[len("model."):]: value.float() for name, value in weights.items() if name.startswith("model.")}
        head = str(len(stages))
        missing, _ = stages.load_state_dict({name: value for name, value in body.items()
                                             if not name.startswith(f"{head}.")}, strict=False)
        if missing:
            raise self.host.Error(f"YOLOv5 checkpoint lacks tensors: {missing[:5]}")
        for module in stages.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = 1e-3
        self.stages = stages.eval().to(device="cuda", dtype=dtype)
        self.strides = archive.__dict__["stride"].float().to(device="cuda", dtype=dtype)
        self.anchors = weights[f"model.{head}.anchors"].float().to(device="cuda", dtype=dtype)
        levels = range(len(self.strides))
        self.weights = [body[f"{head}.m.{level}.weight"].to(device="cuda", dtype=dtype) for level in levels]
        self.biases = [body[f"{head}.m.{level}.bias"].to(device="cuda", dtype=dtype) for level in levels]
        self.per_anchor = len(archive.__dict__["names"]) + 5

    def _decode(self, pixels: Any, threshold: float) -> Any:
        import torch
        import torchvision

        outputs: list[Any] = []
        tensor = pixels
        for stage in self.stages:
            inputs = [tensor if index == -1 else outputs[index] for index in stage.f] if stage.f != -1 else tensor
            tensor = stage(inputs)
            outputs.append(tensor)
        levels = []
        for level, source in enumerate(HEAD_SOURCES):
            raw = torch.nn.functional.conv2d(outputs[source], self.weights[level], self.biases[level])
            rows, columns = raw.shape[-2], raw.shape[-1]
            values = raw.view(1, len(self.anchors[level]), self.per_anchor, rows, columns).permute(0, 1, 3, 4, 2).sigmoid()
            grid_y, grid_x = torch.meshgrid(torch.arange(rows, device=raw.device), torch.arange(columns, device=raw.device),
                                            indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1).to(dtype=self.spec.dtype)
            centre = (values[..., 0:2] * 2 - 0.5 + grid) * self.strides[level]
            extent = (values[..., 2:4] * 2) ** 2 * self.anchors[level].view(1, -1, 1, 1, 2) * self.strides[level]
            levels.append(torch.cat((centre, extent, values[..., 4:]), dim=-1).reshape(1, -1, self.per_anchor))
        predictions = torch.cat(levels, dim=1)[0]
        best, labels = (predictions[:, 4:5] * predictions[:, 5:]).max(dim=1)
        survivors = best > threshold
        centre, extent = predictions[survivors, 0:2], predictions[survivors, 2:4]
        corners = torch.cat((centre - extent / 2, centre + extent / 2), dim=1)
        order = torchvision.ops.batched_nms(corners, best[survivors], labels[survivors], 0.7)[:300]
        return torch.cat((corners[order], best[survivors][order, None],
                          labels[survivors][order, None].to(dtype=self.spec.dtype)), dim=1)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        image = self.host.load_image(str(self.host.required(request, "image_path")))
        pixels, geometry = _letterbox(image, self.spec.dtype)
        kept, model_ms = self.host.timed(lambda: self._decode(pixels, float(request.get("score_threshold", 0.25))))
        return self.host.invocation(_observation(kept.detach().float().cpu().tolist(), geometry), model_ms)
