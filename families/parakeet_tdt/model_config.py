# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-private build configuration for Parakeet TDT."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Minimal engine-builder view of the Parakeet TDT configuration."""

    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 8192
    hidden_act: str = ""
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
    def from_json(text: str) -> "ModelConfig":
        raw = json.loads(text)
        encoder = raw.get("encoder_config") or {}
        hidden_size = int(raw.get("hidden_size") or raw.get("decoder_hidden_size") or 0)
        num_heads = int(raw.get("num_attention_heads") or 1)
        return ModelConfig(
            model_type=str(raw.get("model_type", "")),
            architectures=[str(value) for value in raw.get("architectures", ())],
            vocab_size=int(raw.get("vocab_size", 0)),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("intermediate_size") or hidden_size * 4),
            num_hidden_layers=int(raw.get("num_hidden_layers") or raw.get("num_decoder_layers") or 0),
            num_attention_heads=num_heads,
            num_key_value_heads=int(raw.get("num_key_value_heads") or num_heads),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            bos_token_id=int(raw.get("bos_token_id", -1) if raw.get("bos_token_id") is not None else -1),
            eos_token_id=int(raw.get("eos_token_id", -1) if raw.get("eos_token_id") is not None else -1),
            pad_token_id=int(raw.get("pad_token_id", -1) if raw.get("pad_token_id") is not None else -1),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            max_position_embeddings=int(raw.get("max_position_embeddings") or encoder.get("max_position_embeddings") or 8192),
            hidden_act=str(raw.get("hidden_act", "")),
            _head_dim=int(raw.get("head_dim", 0)),
            raw=raw,
        )

    @classmethod
    def create_tiny(cls, model_type: str, **overrides) -> "ModelConfig":
        defaults = {
            "model_type": model_type,
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
        }
        defaults.update(overrides)
        return cls.from_json(json.dumps(defaults))

    @staticmethod
    def from_dir(model_dir: str | Path) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Parakeet TDT model directory is missing config.json: {config_path}"
            )
        return ModelConfig.from_json(config_path.read_text(encoding="utf-8"))
