# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM configuration parsed from Hugging Face ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str = "xglm"
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
    pad_token_id: int = 1
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 2048
    hidden_act: str = "gelu"
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.hidden_size

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        data = json.loads(text)
        heads = int(data.get("num_attention_heads", data.get("attention_heads", 1)))

        def token_id(name: str, default: int) -> int:
            value = data.get(name, default)
            return default if value is None else int(value)

        return cls(
            model_type=str(data.get("model_type", "xglm")),
            architectures=list(data.get("architectures", [])),
            vocab_size=int(data.get("vocab_size", 0)),
            hidden_size=int(data.get("hidden_size", data.get("d_model", 0))),
            intermediate_size=int(data.get("intermediate_size", data.get("ffn_dim", 0))),
            num_hidden_layers=int(data.get("num_hidden_layers", data.get("num_layers", 0))),
            num_attention_heads=heads,
            num_key_value_heads=heads,
            rms_norm_eps=float(data.get("layer_norm_eps", data.get("rms_norm_eps", 1e-5))),
            bos_token_id=token_id("bos_token_id", -1),
            eos_token_id=token_id("eos_token_id", -1),
            pad_token_id=token_id("pad_token_id", 1),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            hidden_act=str(data.get("activation_function", data.get("hidden_act", "gelu"))),
            raw=data,
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        return cls.from_json((Path(model_dir) / "config.json").read_text())

    @classmethod
    def create_tiny(cls, model_type: str = "xglm", **overrides) -> "ModelConfig":
        values = {
            "model_type": model_type,
            "vocab_size": 32,
            "d_model": 16,
            "ffn_dim": 32,
            "num_layers": 2,
            "attention_heads": 4,
            "max_position_embeddings": 128,
        }
        values.update(overrides)
        return cls.from_json(json.dumps(values))
