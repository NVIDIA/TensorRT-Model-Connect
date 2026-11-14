# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Edge reasoning metadata, distinct from the Cosmos3 policy graph."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _positive(document: dict, name: str) -> int:
    value = document.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Cosmos3-Edge {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, int, int]

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelConfig":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        if not isinstance(raw, dict) or raw.get("model_type") != "cosmos3_edge":
            raise ValueError("expected the cosmos3_edge reasoning checkpoint")
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "cosmos3_edge_text":
            raise ValueError("Cosmos3-Edge requires its text_config")
        if text.get("hidden_act") != "relu2":
            raise ValueError("Cosmos3-Edge requires the non-gated squared-ReLU MLP")
        if any(text.get(k, False) for k in ("attention_bias", "mlp_bias", "qk_norm_for_text")):
            raise ValueError("Cosmos3-Edge text attention/MLP is bias-free and has no Q/K norm")
        names = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings")
        values = {name: _positive(text, name) for name in names}
        if values["hidden_size"] != values["num_attention_heads"] * values["head_dim"]:
            raise ValueError("Cosmos3-Edge attention width must equal hidden_size")
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("Cosmos3-Edge attention heads must be divisible by KV heads")
        rope = text.get("rope_parameters", {})
        if not isinstance(rope, dict) or rope.get("rope_type") != "default":
            raise ValueError("Cosmos3-Edge requires default interleaved MRoPE")
        sections = rope.get("mrope_section")
        if (not isinstance(sections, list) or len(sections) != 3
                or any(type(n) is not int or n <= 0 for n in sections)
                or sum(sections) * 2 != values["head_dim"]):
            raise ValueError("Cosmos3-Edge mrope_section must cover half of head_dim")
        eps, theta = float(text["rms_norm_eps"]), float(rope["rope_theta"])
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(theta) or theta <= 0:
            raise ValueError("Cosmos3-Edge norm epsilon and RoPE theta must be finite and positive")
        vision, projector = raw.get("vision_config"), raw.get("projector_config")
        if not isinstance(vision, dict) or not isinstance(projector, dict):
            raise ValueError("Cosmos3-Edge requires visual and projector configuration")
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                     "num_channels", "num_patches", "patch_size", "spatial_merge_size"):
            _positive(vision, name)
        if vision["hidden_size"] % vision["num_attention_heads"]:
            raise ValueError("Cosmos3-Edge visual heads must divide visual hidden_size")
        if math.isqrt(vision["num_patches"]) ** 2 != vision["num_patches"]:
            raise ValueError("Cosmos3-Edge visual position table must be square")
        if vision.get("hidden_act") != "gelu_pytorch_tanh":
            raise ValueError("Cosmos3-Edge visual MLP requires tanh GELU")
        if (projector.get("input_hidden_size") != vision["hidden_size"]
                or projector.get("out_hidden_size") != values["hidden_size"]
                or projector.get("spatial_merge_size") != vision["spatial_merge_size"]
                or projector.get("use_postshuffle_norm") is not False):
            raise ValueError("Cosmos3-Edge projector requires matching widths and pre-shuffle norm")
        _positive(projector, "merger_intermediate_size")
        return cls(raw=raw, **values, rms_norm_eps=eps, rope_theta=theta,
                   mrope_section=tuple(sections))

    @property
    def vision(self) -> dict:
        return self.raw["vision_config"]

    @property
    def projector(self) -> dict:
        return self.raw["projector_config"]
