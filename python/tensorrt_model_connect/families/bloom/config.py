# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLOOM model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """BLOOM fields consumed by the family-owned weight and graph builders."""

    model_type: str = "bloom"
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim or self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        raw = json.loads(text)
        hidden_size = raw.get("hidden_size") or raw.get("d_model") or 0
        num_heads = (
            raw.get("num_attention_heads")
            or raw.get("attention_heads")
            or raw.get("n_head")
            or 1
        )
        num_layers = (
            raw.get("num_hidden_layers")
            or raw.get("num_layers")
            or raw.get("n_layer")
            or 0
        )
        return cls(
            model_type=raw.get("model_type") or "bloom",
            vocab_size=raw.get("vocab_size", 0),
            hidden_size=hidden_size,
            intermediate_size=(
                raw.get("intermediate_size")
                or raw.get("ffn_dim")
                or hidden_size * 4
            ),
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=raw.get("num_key_value_heads", num_heads),
            rms_norm_eps=(
                raw.get("rms_norm_eps")
                or raw.get("layer_norm_epsilon")
                or raw.get("layer_norm_eps")
                or 1e-5
            ),
            _head_dim=raw.get("head_dim", 0),
            raw=raw,
        )

    @classmethod
    def create_tiny(cls, model_type: str = "bloom", **overrides) -> ModelConfig:
        values = {
            "model_type": model_type,
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
        }
        values.update(overrides)
        return cls.from_json(json.dumps(values))

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> ModelConfig:
        return cls.from_json((Path(model_dir) / "config.json").read_text())
