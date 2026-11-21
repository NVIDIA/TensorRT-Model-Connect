# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelConfig — parse the PointNet config.json contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """PointNet segmentation configuration parsed from config.json."""

    model_type: str = "pointnet"
    architectures: list[str] = field(default_factory=lambda: ["PointNet"])
    num_points: int = 1024
    num_classes: int = 16
    input_dim: int = 3
    segmentation: bool = True
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        data = json.loads(text)
        architectures = data.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            architectures = ["PointNet"]
        return ModelConfig(
            model_type=data.get("model_type", "pointnet"),
            architectures=[str(value) for value in architectures],
            num_points=int(data.get("num_points", 1024)),
            num_classes=int(data.get("num_classes", 16)),
            input_dim=int(data.get("input_dim", 3)),
            segmentation=bool(data.get("segmentation", True)),
            raw=data,
        )

    @staticmethod
    def load(model_dir: Path) -> "ModelConfig":
        path = model_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError("PointNet model directory is missing config.json")
        return ModelConfig.from_json(path.read_text(encoding="utf-8"))
