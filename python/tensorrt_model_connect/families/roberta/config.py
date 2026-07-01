# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoBERTa family model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Fields consumed by the RoBERTa encoder and tensor-parallel builders."""

    model_type: str = "roberta"
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        if self._head_dim > 0:
            return self._head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        data = json.loads(text)
        num_heads = int(data.get("num_attention_heads", 1))
        architecture = data.get("architecture", "")
        architectures = data.get("architectures", [])
        if not architectures and architecture:
            architectures = [architecture]

        return ModelConfig(
            model_type=data.get("model_type", "roberta") or "roberta",
            architectures=architectures,
            vocab_size=int(data.get("vocab_size", 0)),
            hidden_size=int(data.get("hidden_size", 0)),
            intermediate_size=int(data.get("intermediate_size", 0)),
            num_hidden_layers=int(data.get("num_hidden_layers", 0)),
            num_attention_heads=num_heads,
            num_key_value_heads=int(data.get("num_key_value_heads", num_heads)),
            rms_norm_eps=float(data.get("layer_norm_eps", data.get("layer_norm_epsilon", 1e-5))),
            max_position_embeddings=int(data.get("max_position_embeddings", 512)),
            hidden_act=data.get("hidden_act", "gelu") or "gelu",
            _head_dim=int(data.get("head_dim", 0)),
            raw=data,
        )

    @classmethod
    def create_tiny(cls, model_type: str = "roberta", **overrides) -> ModelConfig:
        data = {
            "model_type": model_type,
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 128,
        }
        data.update(overrides)
        return cls.from_json(json.dumps(data))

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text())
