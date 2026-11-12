# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Table Transformer fields used by this family."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .support import validate_structure_config


@dataclass
class ModelConfig:
    model_type: str
    architecture: str
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        raw = json.loads(text)
        validate_structure_config(raw)
        return ModelConfig(
            model_type=raw["model_type"], architecture=raw["architectures"][0], raw=raw
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        return cls.from_json(config_path.read_text(encoding="utf-8"))
