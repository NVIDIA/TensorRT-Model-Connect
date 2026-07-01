# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Top-level identity and raw nested configuration consumed by SAM."""

    model_type: str = "sam"
    architectures: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        data = json.loads(text)
        architecture = data.get("architecture", "")
        architectures = data.get("architectures", [])
        if not architectures and architecture:
            architectures = [architecture]
        return ModelConfig(
            model_type=data.get("model_type", "sam") or "sam",
            architectures=architectures,
            raw=data,
        )

    @classmethod
    def create_tiny(cls, model_type: str = "sam", **overrides) -> ModelConfig:
        data = {"model_type": model_type, **overrides}
        return cls.from_json(json.dumps(data))

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text())
