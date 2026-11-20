# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-owned text topology, independent of the selected execution backend."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    layer_rope_theta: tuple[float, ...]
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    bos_token_id: int
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: dict, generation: dict | None = None) -> TextConfig:
        if raw.get("model_type") not in {"muse_glimmer", "muse_glimmer_text"}:
            raise ValueError("Expected a Muse-Glimmer checkpoint")
        c = raw.get("text_config", raw)
        integer_keys = (
            "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "sliding_window",
        )
        values = {}
        for key in integer_keys:
            value = c.get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f"Muse {key} must be a positive integer")
            values[key] = value
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("Muse query heads must be divisible by KV heads")
        if values["head_dim"] % 2:
            raise ValueError("Muse rotary head width must be even")
        if c.get("attention_bias", False) or c.get("hidden_activation", "silu") != "silu":
            raise ValueError("Muse native decoder requires bias-free attention and SwiGLU")
        if c.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)):
            raise ValueError("Muse expects its separately exported output projection")
        kinds = tuple(c.get("layer_types", ()))
        theta = tuple(c.get("layer_rope_theta", ()))
        if len(kinds) != values["num_hidden_layers"] or len(theta) != len(kinds):
            raise ValueError("Muse requires explicit attention and rotary policy for every layer")
        for kind, base in zip(kinds, theta):
            if kind not in {"sliding_attention", "full_attention"}:
                raise ValueError(f"Unknown Muse attention type: {kind}")
            if not isinstance(base, (int, float)) or not math.isfinite(base):
                raise ValueError("Muse rotary bases must be finite")
            if (kind == "sliding_attention" and base <= 0) or (kind == "full_attention" and base != 0):
                raise ValueError("Muse full attention is NoPE; sliding attention uses positive RoPE")
        for key in ("rms_norm_eps", "post_norm_eps", "qk_scale_factor", "output_multiplier",
                    "final_logit_softcapping"):
            value = c.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Muse {key} must be finite and positive")
            values[key] = float(value)
        if generation is None:
            generation = {}
        if not isinstance(generation, dict):
            raise ValueError("Muse generation_config.json must contain an object")
        # Provider generation defaults override model-topology token defaults.
        # In particular, ATEM end-of-turn is distinct from end-of-text.
        eos = generation.get("eos_token_id")
        if eos is None:
            eos = c.get("eos_token_id")
        eos = eos if isinstance(eos, list) else [eos]
        bos = generation.get("bos_token_id")
        if bos is None:
            bos = c.get("bos_token_id")
        if not eos or any(type(v) is not int or not 0 <= v < values["vocab_size"] for v in [bos, *eos]):
            raise ValueError("Muse BOS/EOS IDs must belong to the checkpoint vocabulary")
        return cls(**values, layer_types=kinds, layer_rope_theta=tuple(map(float, theta)),
                   bos_token_id=bos, eos_token_ids=tuple(eos))

    @classmethod
    def from_dir(cls, path: Path) -> TextConfig:
        generation_path = path / "generation_config.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else None
        return cls.from_dict(json.loads((path / "config.json").read_text(encoding="utf-8")), generation)
