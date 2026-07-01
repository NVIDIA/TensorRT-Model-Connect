# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral architecture fields used by the family implementation."""

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
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
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
        hidden_size = int(raw.get("hidden_size") or 0)
        num_heads = int(raw.get("num_attention_heads") or 1)
        return cls(
            model_type=str(raw.get("model_type") or ""),
            vocab_size=int(raw.get("vocab_size") or 0),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("intermediate_size") or hidden_size * 4),
            num_hidden_layers=int(raw.get("num_hidden_layers") or 0),
            num_attention_heads=num_heads,
            num_key_value_heads=int(raw.get("num_key_value_heads") or num_heads),
            rms_norm_eps=float(raw.get("rms_norm_eps") or 1e-5),
            rope_theta=float(raw.get("rope_theta") or 10000.0),
            max_position_embeddings=int(raw.get("max_position_embeddings") or 8192),
            _head_dim=int(raw.get("head_dim") or 0),
            raw=raw,
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> ModelConfig:
        return cls.from_json((Path(model_dir) / "config.json").read_text())
