# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer family model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Fields consumed by the PatchTSMixer builder and build metadata."""

    model_type: str = "patchtsmixer"
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 0
    hidden_act: str = "gelu"
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim or self.hidden_size

    @property
    def attention_size(self) -> int:
        return self.hidden_size

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        data = json.loads(text)
        hidden_size = int(data.get("d_model", data.get("hidden_size", 0)))
        architecture = data.get("architecture", "")
        architectures = data.get("architectures", [])
        if not architectures and architecture:
            architectures = [architecture]

        return ModelConfig(
            model_type=data.get("model_type", "patchtsmixer") or "patchtsmixer",
            architectures=architectures,
            vocab_size=int(data.get("vocab_size", 0)),
            hidden_size=hidden_size,
            intermediate_size=int(data.get("expansion_factor", 2) * hidden_size),
            num_hidden_layers=int(data.get("num_layers", 0)),
            rms_norm_eps=float(data.get("norm_eps", 1e-5)),
            max_position_embeddings=int(data.get("context_length", 0)),
            hidden_act=data.get("activation", "gelu") or "gelu",
            _head_dim=hidden_size,
            raw=data,
        )

    @classmethod
    def create_tiny(cls, model_type: str = "patchtsmixer", **overrides) -> ModelConfig:
        data = {
            "model_type": model_type,
            "d_model": 16,
            "num_layers": 2,
            "context_length": 32,
            "patch_length": 4,
            "patch_stride": 4,
        }
        data.update(overrides)
        return cls.from_json(json.dumps(data))

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text())
