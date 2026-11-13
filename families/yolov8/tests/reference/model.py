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


class C2f(nn.Module):
    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class DFL(nn.Module):
    def forward(self, logits):
        batch, _, cells = logits.shape
        probabilities = logits.reshape(batch, 4, self.c1, cells).transpose(1, 2).softmax(dim=1)
        return self.conv(probabilities).reshape(batch, 4, cells)


class Detect(nn.Module):
    def forward(self, features):
        predictions, anchors, scales = [], [], []
        for level, feature in enumerate(features):
            batch, _, height, width = feature.shape
            outputs = torch.cat((self.cv2[level](feature), self.cv3[level](feature)), dim=1)
            predictions.append(outputs.reshape(batch, self.reg_max * 4 + self.nc, -1))
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5,
                torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5,
                indexing="ij",
            )
            anchors.append(torch.stack((x, y)).reshape(2, -1))
            scales.append(feature.new_full((1, height * width), float(self.stride[level])))
        predictions = torch.cat(predictions, dim=2)
        distances = self.dfl(predictions[:, : 4 * self.reg_max])
        points, stride = torch.cat(anchors, dim=1), torch.cat(scales, dim=1)
        minimum, maximum = points - distances[:, :2], points + distances[:, 2:]
        boxes = torch.cat(((minimum + maximum) / 2, maximum - minimum), dim=1) * stride
        return torch.cat((boxes, predictions[:, 4 * self.reg_max :].sigmoid()), dim=1)


_OPERATIONS = {
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "Concat": Concat,
    "SPPF": SPPF,
    "Detect": Detect,
    "DetectionModel": Network,
    "C2f": C2f,
    "DFL": DFL,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ultralytics."):
            if name not in _OPERATIONS:
                raise ValueError(f"unsupported reference operation: {module}.{name}")
            return _OPERATIONS[name]
        return super().find_class(module, name)


def load_model(directory: Path) -> nn.Module:
    """Load only trusted .pt files; replacing classes does not sandbox pickle."""
    reader = SimpleNamespace(__name__=__name__, Unpickler=_Unpickler, load=pickle.load)
    archive = torch.load(
        directory / "yolov8n.pt", map_location="cpu", weights_only=False, pickle_module=reader
    )
    model = archive.get("model") if isinstance(archive, dict) else None
    if not isinstance(model, Network):
        raise ValueError("checkpoint has no supported detection model")
    return model.float().eval()


def detections(raw, threshold=0.25):
    """Single-image, class-aware suppression; coordinates remain letterboxed."""
    rows = raw[0].transpose(0, 1)
    scores, labels = rows[:, 4:].max(dim=1)
    selected = scores > threshold
    center, extent = rows[selected, :2], rows[selected, 2:4]
    corners = torch.cat((center - extent / 2, center + extent / 2), dim=1)
    scores, labels = scores[selected], labels[selected]
    keep = batched_nms(corners.float(), scores.float(), labels, 0.7)[:300]
    return torch.cat((corners[keep], scores[keep, None], labels[keep, None].to(raw.dtype)), dim=1)
