# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS Hugging Face configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture fields consumed by the GPT-OSS builder."""

    model_type: str = "gpt_oss"
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim or self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        raw = json.loads(text)
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        return cls(
            model_type=raw.get("model_type", "gpt_oss"),
            vocab_size=int(raw.get("vocab_size", 0)),
            hidden_size=int(raw.get("hidden_size", 0)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 0)),
            num_attention_heads=int(raw.get("num_attention_heads", 1)),
            num_key_value_heads=int(
                raw.get("num_key_value_heads", raw.get("num_attention_heads", 1))),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=float(raw.get("rope_theta", rope.get("rope_theta", 10000.0))),
            _head_dim=int(raw.get("head_dim", 0)),
            raw=raw,
        )

    @classmethod
    def create_tiny(cls, model_type: str, **overrides) -> "ModelConfig":
        """Create the small GPT-OSS architecture used by family tests."""
        values = {
            "model_type": model_type,
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        values.update(overrides)
        return cls.from_json(json.dumps(values))

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        return cls.from_json((Path(model_dir) / "config.json").read_text())
