# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact nested Omni architecture; no unrelated checkpoint-schema heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


def positive(value, name):
    """Reject booleans and lossy conversions for topology dimensions."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DecoderConfig:
    raw: dict
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int

    @property
    def attention_size(self):
        return self.num_attention_heads * self.head_dim

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "nemotron_h":
            raise ValueError("Omni requires its nested Nemotron-H language configuration")
        fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        values = {key: positive(raw.get(key), key) for key in fields}
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("Omni attention heads must divide into complete KV groups")
        pattern = raw.get("hybrid_override_pattern")
        if (
            not isinstance(pattern, str)
            or len(pattern) != values["num_hidden_layers"]
            or any(ch not in "ME*-" for ch in pattern)
        ):
            raise ValueError("Omni hybrid pattern must specify every decoder layer")
        eps = raw.get("layer_norm_epsilon", raw.get("norm_eps", 1e-5))
        if type(eps) not in (float, int) or not 0 < eps < 1:
            raise ValueError("Omni layer normalization epsilon must be finite and positive")
        return cls(raw=dict(raw), rms_norm_eps=float(eps), **values)


@dataclass(frozen=True)
class OmniConfig:
    raw: dict
    decoder: DecoderConfig
    vision: dict
    audio: dict

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "NemotronH_Nano_Omni_Reasoning_V3":
            raise ValueError("Expected NemotronH_Nano_Omni_Reasoning_V3 checkpoint")
        decoder = DecoderConfig.from_dict(raw.get("llm_config"))
        vision, audio = raw.get("vision_config"), raw.get("sound_config")
        if not isinstance(vision, dict) or not isinstance(audio, dict):
            raise ValueError("Omni requires both visual and audio encoder configurations")
        for key in ("force_image_size", "patch_size", "vit_hidden_size", "projector_hidden_size"):
            positive(raw.get(key), key)
        if raw.get("downsample_ratio") != 0.5 or raw.get("ps_version") != "v2":
            raise ValueError("Omni requires its v2 2x2 pixel shuffle")
        if raw["force_image_size"] % (raw["patch_size"] * 2):
            raise ValueError("Omni image shape must contain complete 2x2 patch groups")
        return cls(dict(raw), decoder, dict(vision), dict(audio))

    @classmethod
    def from_dir(cls, model_dir):
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))
