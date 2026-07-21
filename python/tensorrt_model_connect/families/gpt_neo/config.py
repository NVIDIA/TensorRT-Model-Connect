# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-Neo family model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """GPT-Neo fields consumed by the family builder and bundle metadata."""

    model_type: str = "gpt_neo"
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 2048
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.hidden_size

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        data = json.loads(text)
        hidden_size = int(data.get("hidden_size", 0))
        num_heads = int(data.get("num_attention_heads", data.get("num_heads", 1)))
        architecture = data.get("architecture", "")
        architectures = data.get("architectures", [])
        if not architectures and architecture:
            architectures = [architecture]

        return ModelConfig(
            model_type=data.get("model_type", "gpt_neo") or "gpt_neo",
            architectures=architectures,
            vocab_size=int(data.get("vocab_size", 0)),
            hidden_size=hidden_size,
            intermediate_size=int(data.get("intermediate_size") or hidden_size * 4),
            num_hidden_layers=int(data.get("num_hidden_layers", data.get("num_layers", 0))),
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=float(
                data.get("layer_norm_epsilon", data.get("layer_norm_eps", 1e-5))
            ),
            bos_token_id=int(data.get("bos_token_id", -1) or -1),
            eos_token_id=int(data.get("eos_token_id", -1) or -1),
            pad_token_id=int(data.get("pad_token_id", -1) or -1),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            raw=data,
        )

    @classmethod
    def create_tiny(cls, model_type: str = "gpt_neo", **overrides) -> ModelConfig:
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
