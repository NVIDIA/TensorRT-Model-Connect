# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the timm VGG fields used by this family."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str
    raw: dict

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        raw = json.loads(text)
        architecture = raw.get("architecture")
        model_type = raw.get("model_type") or architecture
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("timm VGG config requires architecture")
        return ModelConfig(model_type=model_type, raw=raw)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        return cls.from_json(config_path.read_text(encoding="utf-8"))
