# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-MoE decoder dimensions, independent of optional Edge installation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str
    architectures: list[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    hidden_act: str
    _head_dim: int
    raw: dict = field(repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(value: str) -> ModelConfig:
        raw = json.loads(value)
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        return ModelConfig(
            model_type=raw["model_type"], architectures=raw.get("architectures", []),
            vocab_size=text["vocab_size"], hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10000000))),
            bos_token_id=text.get("bos_token_id", -1),
            eos_token_id=text.get("eos_token_id", -1),
            pad_token_id=text.get("pad_token_id", -1),
            tie_word_embeddings=text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            max_position_embeddings=text["max_position_embeddings"],
            hidden_act=text.get("hidden_act", "silu"),
            _head_dim=text["head_dim"], raw=raw,
        )

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
