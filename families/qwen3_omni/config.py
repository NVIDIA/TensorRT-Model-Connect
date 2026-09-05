# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen3-Omni checkpoint configuration used by this family."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _required_object(mapping: dict, name: str, owner: str) -> dict:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{owner} requires object {name}")
    return value


def _required_int(mapping: dict, name: str, owner: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner} requires integer {name}")
    return int(value)


def _required_positive_int(mapping: dict, name: str, owner: str) -> int:
    value = _required_int(mapping, name, owner)
    if value <= 0:
        raise ValueError(f"{owner}.{name} must be positive")
    return value


def _required_positive_float(mapping: dict, name: str, owner: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner} requires numeric {name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner}.{name} must be finite and positive")
    return result


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
    max_position_embeddings: int
    _head_dim: int
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self._head_dim

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Qwen3-Omni config.json must contain one object")
        if payload.get("model_type") != "qwen3_omni_moe":
            raise ValueError("Qwen3-Omni config model_type must be qwen3_omni_moe")
        architectures = payload.get("architectures")
        if architectures != ["Qwen3OmniMoeForConditionalGeneration"]:
            raise ValueError(
                "Qwen3-Omni config architecture must be Qwen3OmniMoeForConditionalGeneration"
            )
        thinker = _required_object(payload, "thinker_config", "Qwen3-Omni config")
        decoder = _required_object(thinker, "text_config", "Qwen3-Omni thinker_config")
        hidden = _required_positive_int(decoder, "hidden_size", "Qwen3-Omni Thinker")
        heads = _required_positive_int(decoder, "num_attention_heads", "Qwen3-Omni Thinker")
        kv_heads = _required_positive_int(decoder, "num_key_value_heads", "Qwen3-Omni Thinker")
        head_dim = _required_positive_int(decoder, "head_dim", "Qwen3-Omni Thinker")
        if heads % kv_heads != 0 or head_dim % 2 != 0:
            raise ValueError("Qwen3-Omni Thinker attention geometry is invalid")
        if decoder.get("hidden_act") != "silu":
            raise ValueError("Qwen3-Omni Thinker hidden_act must be silu")
        return ModelConfig(
            model_type="qwen3_omni_moe",
            architectures=list(architectures),
            vocab_size=_required_positive_int(decoder, "vocab_size", "Qwen3-Omni Thinker"),
            hidden_size=hidden,
            intermediate_size=_required_positive_int(
                decoder, "intermediate_size", "Qwen3-Omni Thinker"
            ),
            num_hidden_layers=_required_positive_int(
                decoder, "num_hidden_layers", "Qwen3-Omni Thinker"
            ),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            rms_norm_eps=_required_positive_float(decoder, "rms_norm_eps", "Qwen3-Omni Thinker"),
            rope_theta=_required_positive_float(decoder, "rope_theta", "Qwen3-Omni Thinker"),
            max_position_embeddings=_required_positive_int(
                decoder, "max_position_embeddings", "Qwen3-Omni Thinker"
            ),
            _head_dim=head_dim,
            raw=payload,
        )

    @staticmethod
    def from_dir(model_dir: str | Path) -> "ModelConfig":
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Qwen3-Omni requires config.json: {path}")
        return ModelConfig.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def create_tiny(cls, **overrides) -> "ModelConfig":
        values = {
            "model_type": "qwen3_omni_moe",
            "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "_head_dim": 4,
            "raw": {},
        }
        values.update(overrides)
        return cls(**values)
