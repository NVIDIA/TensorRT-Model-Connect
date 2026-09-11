# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the DETR fields used by this family."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str
    architecture: str
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        raw = json.loads(text)
        model_type = raw.get("model_type")
        architectures = raw.get("architectures")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("DETR config requires model_type")
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])
        else:
            architecture = raw.get("architecture", model_type)
        if not isinstance(architecture, str) or not architecture:
            architecture = model_type
        return ModelConfig(model_type=model_type, architecture=architecture, raw=raw)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        return cls.from_json(config_path.read_text(encoding="utf-8"))
