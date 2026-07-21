# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration fields used by the Eagle VLM implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Llama-Nemotron-VL text-backbone configuration."""

    model_type: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        if self._head_dim > 0:
            return self._head_dim
        return self.hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 0

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        raw = json.loads(text)
        decoder = raw.get("llm_config", raw)
        if not isinstance(decoder, dict):
            raise ValueError("Eagle VLM llm_config must be an object")

        num_heads = decoder.get("num_attention_heads", 1)
        return ModelConfig(
            model_type=raw.get("model_type", ""),
            vocab_size=decoder.get("vocab_size", raw.get("vocab_size", 0)),
            hidden_size=decoder.get("hidden_size", 0),
            intermediate_size=decoder.get("intermediate_size", 0),
            num_hidden_layers=decoder.get("num_hidden_layers", 0),
            num_attention_heads=num_heads,
            num_key_value_heads=decoder.get("num_key_value_heads", num_heads),
            rms_norm_eps=decoder.get("rms_norm_eps", 1e-5),
            rope_theta=float(decoder.get("rope_theta", 10000.0)),
            _head_dim=decoder.get("head_dim", 0),
            raw=raw,
        )

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text())
