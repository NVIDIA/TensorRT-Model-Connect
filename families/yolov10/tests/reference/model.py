# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager PyTorch YOLOv10 reference, independent of the TensorRT builder.

Build ordinary Torch modules from the checkpoint's tensor shapes. Keep batch
normalization and the two large-kernel branches explicit in the reference.
Only the one-to-one inference head participates in predictions.
"""

from pathlib import Path
import json

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F


class Weights:
    def __init__(self, tensors):
        self.tensors = tensors
        self.used = set()

    def take(self, key):
        self.used.add(key)
        return self.tensors[key].float()

    def conv(self, name, stride=1, activate=True):
        weight = self.take(name + ".conv.weight")
        outputs, inputs, height, width = weight.shape
        groups = outputs if inputs == 1 else 1
        layer = nn.Conv2d(
            inputs * groups,
            outputs,
            (height, width),
            stride,
            (height // 2, width // 2),
            groups=groups,
            bias=False,
        )
        layer.weight = nn.Parameter(weight, requires_grad=False)
        norm = nn.BatchNorm2d(outputs, eps=1e-3)
        norm.load_state_dict(
            {
                field: self.take(name + ".bn." + field)
                for field in ("weight", "bias", "running_mean", "running_var")
            },
            strict=False,
        )
        return nn.Sequential(layer, norm, nn.SiLU() if activate else nn.Identity())

    def linear_conv(self, name):
        weight, bias = self.take(name + ".weight"), self.take(name + ".bias")
        layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.bias = nn.Parameter(bias, requires_grad=False)
        return layer


class RepDepthwise(nn.Module):
    def __init__(self, weights, name):
        super().__init__()
        self.wide = weights.conv(name + ".conv", activate=False)
        self.narrow = weights.conv(name + ".conv1", activate=False)

    def forward(self, x):
        return F.silu(self.wide(x) + self.narrow(x))


class Residual(nn.Module):
    def __init__(self, sequence, enabled):
        super().__init__()
        self.sequence, self.enabled = sequence, enabled

    def forward(self, x):
        result = self.sequence(x)
        return result + x if self.enabled else result


class SplitBlocks(nn.Module):
    def __init__(self, weights, name, residual):
        super().__init__()
        self.entry = weights.conv(name + ".cv1")
        self.exit = weights.conv(name + ".cv2")
        indices = sorted(
            {
                int(key[len(name) + 3 :].split(".")[0])
                for key in weights.tensors
                if key.startswith(name + ".m.")
            }
        )
        if indices != list(range(len(indices))) or not indices:
            raise ValueError(f"invalid residual block sequence: {name}")
        blocks = []
        for index in indices:
            prefix = f"{name}.m.{index}"
            cib = prefix + ".cv1.0.conv.weight" in weights.tensors
            if cib:
                chain = []
                for step in range(5):
                    child = f"{prefix}.cv1.{step}"
                    pair = child + ".conv1.conv.weight" in weights.tensors
                    chain.append(RepDepthwise(weights, child) if pair else weights.conv(child))
            else:
                chain = [weights.conv(prefix + ".cv1"), weights.conv(prefix + ".cv2")]
            blocks.append(Residual(nn.Sequential(*chain), cib or residual))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        parts = list(self.entry(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.exit(torch.cat(parts, dim=1))


class PoolPyramid(nn.Module):
    def __init__(self, weights, name):
        super().__init__()
        self.entry, self.exit = weights.conv(name + ".cv1"), weights.conv(name + ".cv2")

    def forward(self, x):
        parts = [self.entry(x)]
        for _ in range(3):
            parts.append(F.max_pool2d(parts[-1], 5, stride=1, padding=2))
        return self.exit(torch.cat(parts, dim=1))


class SpatialAttention(nn.Module):
    def __init__(self, weights, name):
        super().__init__()
        self.qkv = weights.conv(name + ".qkv", activate=False)
        self.position = weights.conv(name + ".pe", activate=False)
        self.projection = weights.conv(name + ".proj", activate=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        heads, tokens = channels // 64, height * width
        query, key, value = (
            self.qkv(x).reshape(batch, heads, 128, tokens).split((32, 32, 64), dim=2)
        )
        probabilities = (query.transpose(-1, -2) @ key * (32**-0.5)).softmax(dim=-1)
        context = (value @ probabilities.transpose(-1, -2)).reshape_as(x)
        return self.projection(context + self.position(value.reshape_as(x)))


class PartialAttention(nn.Module):
    def __init__(self, weights, name):
        super().__init__()
        self.entry, self.exit = weights.conv(name + ".cv1"), weights.conv(name + ".cv2")
        self.attention = SpatialAttention(weights, name + ".attn")
        self.feed = nn.Sequential(
            weights.conv(name + ".ffn.0"), weights.conv(name + ".ffn.1", activate=False)
        )

    def forward(self, x):
        bypass, attended = self.entry(x).chunk(2, dim=1)
        attended = attended + self.attention(attended)
        attended = attended + self.feed(attended)
        return self.exit(torch.cat((bypass, attended), dim=1))


class DetectionHead(nn.Module):
    def __init__(self, weights, classes):
        super().__init__()
        prefix = "23."
        self.classes = classes
        self.register_buffer("bins", weights.take(prefix + "dfl.conv.weight").reshape(1, 1, -1, 1))
        self.boxes, self.scores = nn.ModuleList(), nn.ModuleList()
        for level in range(3):
            box = f"{prefix}one2one_cv2.{level}"
            self.boxes.append(
                nn.Sequential(
                    weights.conv(box + ".0"),
                    weights.conv(box + ".1"),
                    weights.linear_conv(box + ".2"),
                )
            )
            score = f"{prefix}one2one_cv3.{level}"
            self.scores.append(
                nn.Sequential(
                    weights.conv(score + ".0.0"),
                    weights.conv(score + ".0.1"),
                    weights.conv(score + ".1.0"),
                    weights.conv(score + ".1.1"),
                    weights.linear_conv(score + ".2"),
                )
            )

    def predictions(self, features):
        """Decode each image cell before score-dependent top-k ordering."""
        logits, anchors, scales = [], [], []
        for level, feature in enumerate(features):
            batch, _, height, width = feature.shape
            logits.append(
                torch.cat((self.boxes[level](feature), self.scores[level](feature)), dim=1).flatten(
                    2
                )
            )
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5,
                torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5,
                indexing="ij",
            )
            anchors.append(torch.stack((x, y)).reshape(1, 2, -1))
            scales.append(feature.new_full((1, 1, height * width), 8 * 2**level))
        raw = torch.cat(logits, dim=2)
        count = self.bins.numel()
        distances = (
            raw[:, : 4 * count].reshape(batch, 4, count, -1).softmax(dim=2) * self.bins
        ).sum(dim=2)
        points, strides = torch.cat(anchors, dim=2), torch.cat(scales, dim=2)
        corners = torch.cat((points - distances[:, :2], points + distances[:, 2:]), dim=1) * strides
        scores = raw[:, 4 * count :].sigmoid()
        return corners.transpose(1, 2), scores.transpose(1, 2)

    def forward(self, features):
        corners, scores = self.predictions(features)
        count = min(300, scores.shape[1])
        cells = scores.amax(dim=-1).topk(count, dim=-1).indices
        corners = corners.gather(1, cells[..., None].expand(-1, -1, 4))
        scores = scores.gather(1, cells[..., None].expand(-1, -1, self.classes))
        confidence, indices = scores.flatten(1).topk(count, dim=-1)
        boxes = corners.gather(1, (indices // self.classes)[..., None].expand(-1, -1, 4))
        return torch.cat(
            (boxes, confidence[..., None], (indices % self.classes)[..., None].to(scores.dtype)),
            dim=-1,
        )


class Network(nn.Module):
    def __init__(self, weights, classes):
        super().__init__()
        self.stages = nn.ModuleDict()
        for index in (0, 1, 3, 17):
            self.stages[str(index)] = weights.conv(str(index), stride=2)
        for index in (2, 4, 6, 8, 13, 16, 19, 22):
            self.stages[str(index)] = SplitBlocks(
                weights, str(index), residual=index not in (13, 16, 19)
            )
        for index in (5, 7, 20):
            self.stages[str(index)] = nn.Sequential(
                weights.conv(f"{index}.cv1"), weights.conv(f"{index}.cv2", stride=2, activate=False)
            )
        self.stages["9"] = PoolPyramid(weights, "9")
        self.stages["10"] = PartialAttention(weights, "10")
        self.head = DetectionHead(weights, classes)

    def _features(self, pixels):
        x, outputs = pixels, {}
        for index in range(11):
            x = self.stages[str(index)](x)
            outputs[index] = x
        neck = self.stages["13"](torch.cat((F.interpolate(x, scale_factor=2), outputs[6]), dim=1))
        small = self.stages["16"](
            torch.cat((F.interpolate(neck, scale_factor=2), outputs[4]), dim=1)
        )
        medium = self.stages["19"](torch.cat((self.stages["17"](small), neck), dim=1))
        large = self.stages["22"](torch.cat((self.stages["20"](medium), outputs[10]), dim=1))
        return small, medium, large

    def forward(self, pixels):
        return self.head(self._features(pixels))

    def predictions(self, pixels):
        """Stable cell/channel order for comparison with the recorded oracle."""
        boxes, scores = self.head.predictions(self._features(pixels))
        return torch.cat((boxes, scores), dim=-1).transpose(1, 2)


def load_model(directory: Path) -> nn.Module:
    config = json.loads((directory / "config.json").read_text())
    if config.get("model") not in {"yolov10n.yaml", "yolov10s.yaml", "yolov10x.yaml"}:
        raise ValueError(f"unsupported YOLOv10 reference architecture: {config.get('model')!r}")
    tensors = load_tensors(directory)
    if any(not key.startswith("model.model.") for key in tensors):
        raise ValueError("reference tensors must use model.model names")
    weights = Weights({key.removeprefix("model.model."): tensor for key, tensor in tensors.items()})
    model = Network(weights, len(config["names"]))
    extra = {
        key
        for key in weights.tensors.keys() - weights.used
        if not key.endswith("num_batches_tracked") and not key.startswith(("23.cv2.", "23.cv3."))
    }
    if extra:
        raise ValueError(f"unexpected reference tensors: {sorted(extra)[:5]}")
    return model.eval()


def detections(raw, threshold=0.25):
    rows = raw[0]
    return rows[rows[:, 4] >= threshold]


def load_tensors(directory: Path):
    """Read tensor data directly, independently of the production checkpoint reader."""
    single = directory / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as reader:
            return {key: reader.get_tensor(key) for key in reader.keys()}
    payload = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("reference checkpoint index has no weight_map")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(shard, str)
        or not shard
        or Path(shard).name != shard
        or shard in {".", ".."}
        for key, shard in mapping.items()
    ):
        raise ValueError("invalid reference checkpoint tensor mapping")
    tensors = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(directory / shard, framework="pt", device="cpu") as reader:
            for key, owner in mapping.items():
                if owner == shard:
                    tensors[key] = reader.get_tensor(key)
    return tensors
