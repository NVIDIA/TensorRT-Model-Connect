# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal PyTorch reference for the Uni3D-B point encoder.

Derived from baaivision/Uni3D (MIT License):
  https://github.com/baaivision/Uni3D
  revision 64e03c3c (models/point_encoder.py, models/uni3d.py)
Only the inference-time ``encode_pc`` path is reproduced; the ViT blocks are
timm ``eva02_base_patch14_448`` exactly as upstream builds them. Training,
dataset and CLIP code are not vendored.

The only custom CUDA op upstream is pointnet2_ops farthest point sampling. It
is replaced by ``farthest_point_indices``, a plain PyTorch transcription of the
kernel's rules, so the exported graph uses standard ONNX ops only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


NUM_GROUP = 512
GROUP_SIZE = 64
INPUT_DIM = 6
NUM_POINTS = 10000
ENCODER_DIM = 512
TRANS_DIM = 768
EMBED_DIM = 1024
# pointnet2_ops never samples a point with x^2 + y^2 + z^2 <= 1e-3.
FPS_MIN_SQUARED_NORM = 1e-3


def farthest_point_indices(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Farthest point sampling with the pointnet2_ops kernel rules.

    Starts at index 0, never selects a point whose squared norm is <= 1e-3
    (those points also keep their initial distance), and takes the first index
    on a tie. ``xyz`` is [B, N, 3] float32; returns [B, count] int64.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    valid = (x * x + y * y + z * z) > FPS_MIN_SQUARED_NORM
    distance = torch.full_like(x, 1e10)
    last = torch.zeros(xyz.shape[0], dtype=torch.long, device=xyz.device)
    picked = [last]
    for _ in range(count - 1):
        point = torch.gather(xyz, 1, last.view(-1, 1, 1).expand(-1, 1, 3))
        dx, dy, dz = x - point[..., 0], y - point[..., 1], z - point[..., 2]
        distance = torch.where(valid, torch.minimum(distance, dx * dx + dy * dy + dz * dz), distance)
        last = torch.argmax(torch.where(valid, distance, torch.full_like(distance, -1.0)), dim=1)
        picked.append(last)
    return torch.stack(picked, dim=1)


def farthest_point_indices_kernel(xyz: np.ndarray, count: int,
                                  min_squared_norm: float = FPS_MIN_SQUARED_NORM) -> np.ndarray:
    """Sequential float32 transcription of pointnet2_ops furthest_point_sampling_kernel."""
    xyz = np.asarray(xyz, dtype=np.float32)
    distance = np.full(xyz.shape[0], 1e10, dtype=np.float32)
    picked, last = [0], 0
    for _ in range(1, count):
        x1, y1, z1 = xyz[last]
        best_index, best = 0, np.float32(-1)
        for k, (x2, y2, z2) in enumerate(xyz):
            if x2 * x2 + y2 * y2 + z2 * z2 <= np.float32(min_squared_norm):
                continue
            d = min((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1), distance[k])
            distance[k] = d
            if d > best:
                best_index, best = k, d
        last = best_index
        picked.append(last)
    return np.array(picked)


class _MiniPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(), nn.Conv1d(512, ENCODER_DIM, 1))

    def forward(self, groups: torch.Tensor) -> torch.Tensor:  # [B, G, M, 6]
        batch, group, size, _ = groups.shape
        feature = self.first_conv(groups.reshape(batch * group, size, 6).transpose(2, 1))
        feature = torch.cat([feature.amax(dim=2, keepdim=True).expand(-1, -1, size), feature], dim=1)
        return self.second_conv(feature).amax(dim=2).reshape(batch, group, ENCODER_DIM)


class Uni3DPointEncoder(nn.Module):
    """Uni3D-B ``encode_pc``: points [B, N, 6] (xyz, rgb) -> embedding [B, 1024] float32.

    Grouping runs in float32. ``to_compute_dtype(torch.float16)`` moves the
    tokenizer and the ViT to half precision, the fp16 export of the family.
    """

    def __init__(self):
        super().__init__()
        import timm

        self.encoder = _MiniPointNet()
        self.encoder2trans = nn.Linear(ENCODER_DIM, TRANS_DIM)
        self.trans2embed = nn.Linear(TRANS_DIM, EMBED_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, TRANS_DIM))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, TRANS_DIM))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, TRANS_DIM))
        self.visual = timm.create_model("eva02_base_patch14_448", pretrained=False)
        self.compute_dtype = torch.float32

    def to_compute_dtype(self, dtype: torch.dtype) -> "Uni3DPointEncoder":
        self.compute_dtype = dtype
        return self.to(dtype)

    def group(self, xyz: torch.Tensor, color: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = xyz.shape[0]
        centers = torch.gather(
            xyz, 1, farthest_point_indices(xyz, NUM_GROUP).unsqueeze(-1).expand(-1, -1, 3))
        # Same squared-distance formula as upstream square_distance.
        squared = (-2 * centers @ xyz.transpose(1, 2) + (centers ** 2).sum(-1, keepdim=True)
                   + (xyz ** 2).sum(-1).unsqueeze(1))
        neighbors = torch.topk(squared, GROUP_SIZE, dim=-1, largest=False, sorted=False).indices
        flat = neighbors.reshape(batch, -1, 1)
        local = torch.gather(xyz, 1, flat.expand(-1, -1, 3)).view(batch, NUM_GROUP, GROUP_SIZE, 3)
        rgb = torch.gather(color, 1, flat.expand(-1, -1, 3)).view(batch, NUM_GROUP, GROUP_SIZE, 3)
        return centers, torch.cat([local - centers.unsqueeze(2), rgb], dim=-1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = points.float()
        centers, features = self.group(points[..., :3], points[..., 3:])
        dtype = self.compute_dtype
        tokens = self.encoder2trans(self.encoder(features.to(dtype)))
        batch = tokens.shape[0]
        x = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        x = x + torch.cat([self.cls_pos.expand(batch, -1, -1), self.pos_embed(centers.to(dtype))], dim=1)
        x = self.visual.pos_drop(x)
        for block in self.visual.blocks:
            x = block(x)
        x = self.visual.fc_norm(self.visual.norm(x[:, 0, :]))
        return self.trans2embed(x).float()


def load(model_dir: Path, device: str = "cpu") -> Uni3DPointEncoder:
    """Load ``modelzoo/uni3d-b/model.pt`` (or ``model.pt``) from ``model_dir``."""
    candidates = [model_dir / "modelzoo/uni3d-b/model.pt", model_dir / "model.pt"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no Uni3D-B model.pt under {model_dir}")
    state = torch.load(path, map_location="cpu", weights_only=True)["module"]
    prefix = "point_encoder."
    model = Uni3DPointEncoder()
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return model.to(device).eval()


def write_model_dir(model: Uni3DPointEncoder, destination: Path, precision: str) -> None:
    """Export ``model`` to the builder contract: ``uni3d.onnx`` plus ``config.json``."""
    destination.mkdir(parents=True, exist_ok=True)
    if precision == "fp16":
        model.to_compute_dtype(torch.float16)
    elif precision != "fp32":
        raise ValueError(f"unsupported precision {precision}")
    device = next(model.parameters()).device
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, NUM_POINTS, INPUT_DIM, device=device),),
                          str(destination / "uni3d.onnx"), input_names=["points"],
                          output_names=["embedding"], dynamic_axes={"points": {1: "num_points"}},
                          opset_version=17, dynamo=False)
    (destination / "config.json").write_text(json.dumps({
        "model_type": "uni3d",
        "architectures": ["Uni3D"],
        "num_points": NUM_POINTS,
        "num_group": NUM_GROUP,
        "group_size": GROUP_SIZE,
        "input_dim": INPUT_DIM,
        "embed_dim": EMBED_DIM,
        "precision": precision,
    }, indent=2) + "\n", encoding="utf-8")


def export_onnx(model_dir: Path, destination: Path, precision: str) -> None:
    """Export the upstream Uni3D-B checkpoint to ``destination``."""
    write_model_dir(load(model_dir, "cuda"), destination, precision)


def run(model_dir: Path, points: np.ndarray) -> np.ndarray:
    """Upstream checkpoint embedding in float32 for ``points`` [N, 6]."""
    model = load(model_dir, "cuda")
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(points, np.float32))[None].cuda())[0].cpu().numpy()
