# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal PyTorch reference for the PointNet S3DIS semantic-segmentation checkpoint.

Derived from yanx27/Pointnet_Pointnet2_pytorch (MIT License):
  https://github.com/yanx27/Pointnet_Pointnet2_pytorch
  revision eb64fe0b4c24055559cea26299cb485dcb43d8dd
Only the inference-time model definition is reproduced. Training, dataset,
augmentation, and utility code are intentionally not vendored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _STN3d(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9).repeat(batch, 1)
        return (x + identity).view(-1, 3, 3)


class _STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).reshape(1, self.k * self.k)
        return (x + identity.repeat(batch, 1)).view(-1, self.k, self.k)


class _PointNetEncoder(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.stn = _STN3d(channel)
        self.fstn = _STNkd(k=64)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        batch, dims, num_points = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if dims > 3:
            features = x[:, :, 3:]
            xyz = x[:, :, :3]
        else:
            features = None
            xyz = x
        xyz = torch.bmm(xyz, trans)
        x = torch.cat([xyz, features], dim=2) if features is not None else xyz
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        return torch.cat([x, point_feat], 1), trans, trans_feat


class S3DISSemSeg(nn.Module):
    def __init__(self, num_class: int = 13):
        super().__init__()
        self.feat = _PointNetEncoder(channel=9)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_classes, num_points = x.size(0), 13, x.size(2)
        x, _trans, _trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, num_classes), dim=-1)
        return x.view(batch, num_points, num_classes)


def _load(model_dir: Path, device: str) -> S3DISSemSeg:
    # Upstream stores class_avg_iou as a numpy float64; allow only that scalar
    # (under its numpy<2 pickled name) instead of unrestricted unpickling.
    scalar = np.float64(0).__reduce__()[0]
    numpy_scalar_globals = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
                            type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numpy_scalar_globals):
        checkpoint = torch.load(model_dir / "best_model.pth", map_location=device,
                                weights_only=True)
    model = S3DISSemSeg(num_class=13).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def export_onnx(model_dir: Path, destination: Path) -> None:
    """Export the upstream checkpoint to the builder's pointnet.onnx contract."""
    import onnx

    model = _load(model_dir, "cpu")
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, 9, 4096), destination,
                          input_names=["point"], output_names=["pred"],
                          dynamic_axes={"point": {2: "num_points"}, "pred": {1: "num_points"}},
                          opset_version=17, dynamo=False)
    # The builder's FP16 rewrite converts initializers only, so float Constant
    # nodes (the STN identity matrices) must become initializers too.
    exported = onnx.load(str(destination))
    for node in [node for node in exported.graph.node if node.op_type == "Constant"]:
        tensor = node.attribute[0].t
        if node.attribute[0].name == "value" and tensor.data_type == onnx.TensorProto.FLOAT:
            tensor.name = node.output[0]
            exported.graph.initializer.append(tensor)
            exported.graph.node.remove(node)
    onnx.checker.check_model(exported)
    onnx.save(exported, str(destination))


def run(model_dir: Path, points_path: Path, num_points: int) -> dict:
    """Run the upstream PyTorch checkpoint and return per-point logits and labels."""
    model = _load(model_dir, "cuda")
    points = np.fromfile(points_path, dtype=np.float32).reshape(num_points, 9)
    input_tensor = torch.from_numpy(points.T[None, :, :].copy()).cuda()  # [1, 9, N]
    with torch.no_grad():
        logits = model(input_tensor)[0].cpu().numpy()  # [N, 13] log-softmax
    return {"logits": logits, "labels": logits.argmax(axis=1)}
