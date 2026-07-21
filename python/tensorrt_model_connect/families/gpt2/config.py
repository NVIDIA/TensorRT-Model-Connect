# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 configuration fields consumed by the family implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 0

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        raw = json.loads(text)
        hidden = raw.get("hidden_size") or raw.get("n_embd") or 0
        heads = raw.get("num_attention_heads") or raw.get("n_head") or 1
        return cls(
            model_type=raw.get("model_type", ""),
            vocab_size=raw.get("vocab_size", 0),
            hidden_size=hidden,
            intermediate_size=(raw.get("intermediate_size") or raw.get("n_inner") or hidden * 4),
            num_hidden_layers=(raw.get("num_hidden_layers") or raw.get("n_layer") or 0),
            num_attention_heads=heads,
            num_key_value_heads=heads,
            rms_norm_eps=(raw.get("layer_norm_epsilon") or raw.get("layer_norm_eps") or 1e-5),
            max_position_embeddings=raw.get(
                "max_position_embeddings", raw.get("n_positions", 8192)
            ),
            raw=raw,
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> ModelConfig:
        return cls.from_json((Path(model_dir) / "config.json").read_text())
