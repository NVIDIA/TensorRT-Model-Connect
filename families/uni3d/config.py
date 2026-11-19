# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelConfig: parse the Uni3D config.json written next to uni3d.onnx."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Uni3D point-encoder configuration.

    ``precision`` is the precision the ONNX graph was exported with: grouping
    (farthest point sampling and kNN) is always fp32, and ``fp16`` exports run
    the mini-PointNet and the ViT in half precision.
    """

    model_type: str = "uni3d"
    num_points: int = 10000
    num_group: int = 512
    group_size: int = 64
    input_dim: int = 6
    embed_dim: int = 1024
    precision: str = "fp16"
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        data = json.loads(text)
        config = ModelConfig(
            model_type=str(data.get("model_type", "uni3d")),
            num_points=int(data["num_points"]),
            num_group=int(data["num_group"]),
            group_size=int(data["group_size"]),
            input_dim=int(data["input_dim"]),
            embed_dim=int(data["embed_dim"]),
            precision=str(data["precision"]),
            raw=data,
        )
        if config.precision not in {"fp16", "fp32"}:
            raise ValueError(f"Uni3D config.json precision must be fp16 or fp32, got {config.precision}")
        if not 0 < config.group_size <= config.num_group <= config.num_points:
            raise ValueError("Uni3D config.json needs 0 < group_size <= num_group <= num_points")
        return config

    @staticmethod
    def load(model_dir: Path) -> "ModelConfig":
        path = model_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError("Uni3D model directory is missing config.json")
        return ModelConfig.from_json(path.read_text(encoding="utf-8"))
