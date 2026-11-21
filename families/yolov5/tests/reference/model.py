# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned eager PyTorch reference, independent of the TensorRT builder.

The trusted checkpoint supplies Torch layers, parameters and graph connections.
Archive-specific class names are mapped to local implementations; no upstream
Python package is imported. Unknown model operations fail closed.
"""

from pathlib import Path
import pickle
from types import SimpleNamespace

import torch
from torch import nn
from torchvision.ops import batched_nms


class Network(nn.Module):
    def forward(self, pixels):
        outputs = []
        current = pixels
        for layer in self.model:
            source = layer.f
            if isinstance(source, int):
                inputs = current if source == -1 else outputs[source]
            else:
                inputs = [current if index == -1 else outputs[index] for index in source]
            current = layer(inputs)
            outputs.append(current)
        return current


class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def forward(self, x):
        hidden = self.cv2(self.cv1(x))
        return x + hidden if self.add else hidden


class Concat(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, dim=self.d)


class SPPF(nn.Module):
    def forward(self, x):
        levels = [self.cv1(x)]
        for _ in range(3):
            levels.append(self.m(levels[-1]))
        return self.cv2(torch.cat(levels, dim=1))


class C3(nn.Module):
    def forward(self, x):
        processed = self.m(self.cv1(x))
        return self.cv3(torch.cat((processed, self.cv2(x)), dim=1))


class Detect(nn.Module):
    def forward(self, features):
        predictions = []
        for level, feature in enumerate(features):
            logits = self.m[level](feature)
            batch, _, height, width = logits.shape
            values = (
                logits.reshape(batch, self.na, self.no, height, width)
                .permute(0, 1, 3, 4, 2)
                .sigmoid()
            )
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=feature.dtype),
                torch.arange(width, device=feature.device, dtype=feature.dtype),
                indexing="ij",
            )
            grid = torch.stack((x, y), dim=-1)
            stride = float(self.stride[level])
            center = (2 * values[..., :2] - 0.5 + grid) * stride
            extent = (
                (2 * values[..., 2:4]).square()
                * self.anchors[level].reshape(1, -1, 1, 1, 2)
                * stride
            )
            predictions.append(
                torch.cat((center, extent, values[..., 4:]), dim=-1).reshape(batch, -1, self.no)
            )
        return torch.cat(predictions, dim=1)


_OPERATIONS = {
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "Concat": Concat,
    "SPPF": SPPF,
    "Detect": Detect,
    "Model": Network,
    "DetectionModel": Network,
    "C3": C3,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"models.common", "models.yolo"}:
            if name not in _OPERATIONS:
                raise ValueError(f"unsupported reference operation: {module}.{name}")
            return _OPERATIONS[name]
        return super().find_class(module, name)


def load_model(directory: Path) -> nn.Module:
    """Load only trusted .pt files; replacing classes does not sandbox pickle."""
    reader = SimpleNamespace(__name__=__name__, Unpickler=_Unpickler, load=pickle.load)
    archive = torch.load(
        directory / "yolov5n.pt", map_location="cpu", weights_only=False, pickle_module=reader
    )
    model = archive.get("model") if isinstance(archive, dict) else None
    if not isinstance(model, Network):
        raise ValueError("checkpoint has no supported detection model")
    return model.float().eval()


def detections(raw, threshold=0.25):
    """Single-image, class-aware suppression; coordinates remain letterboxed."""
    rows = raw[0]
    scores, labels = (rows[:, 4:5] * rows[:, 5:]).max(dim=1)
    selected = scores > threshold
    center, extent = rows[selected, :2], rows[selected, 2:4]
    corners = torch.cat((center - extent / 2, center + extent / 2), dim=1)
    scores, labels = scores[selected], labels[selected]
    keep = batched_nms(corners.float(), scores.float(), labels, 0.7)[:300]
    return torch.cat((corners[keep], scores[keep, None], labels[keep, None].to(raw.dtype)), dim=1)
