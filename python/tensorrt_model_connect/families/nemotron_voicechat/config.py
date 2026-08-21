# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the pinned VoiceChat thinker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Fields consumed by the native VoiceChat thinker builder."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    _head_dim: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self._head_dim or self.hidden_size // self.num_attention_heads

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        raw = json.loads(text)
        return ModelConfig(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            _head_dim=int(raw.get("head_dim", 0)),
            raw=raw,
        )
