# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    epsilon: float
    rotary_base: float

    @classmethod
    def from_dir(cls, path: Path):
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Nomic config must be an object")
        expected = {
            "model_type": "nomic_bert",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "activation_function": "swiglu",
            "prenorm": False,
            "parallel_block": False,
            "causal": False,
            "rotary_emb_fraction": 1.0,
            "rotary_emb_interleaved": False,
            "qkv_proj_bias": False,
            "mlp_fc1_bias": False,
            "mlp_fc2_bias": False,
            "type_vocab_size": 2,
            "pad_token_id": 0,
        }
        for key, value in expected.items():
            actual = raw.get(key)
            if actual != value or (type(value) in (int, bool) and type(actual) is not type(value)):
                raise ValueError(f"unsupported Nomic config {key}={actual!r}")
        for key in ("rotary_emb_scale_base", "rotary_scaling_factor"):
            if raw.get(key) is not None:
                raise ValueError(f"Nomic does not implement {key}")
        if (
            raw.get("num_experts", 0) != 0
            or raw.get("rotary_head_dim", False)
            or raw.get("norm_mlp", False)
            or raw.get("num_heads_kv") not in (None, 12)
        ):
            raise ValueError(
                "Nomic does not implement expert, MLP norm, GQA or head-axis rotary variants"
            )
        vocab = raw.get("vocab_size")
        if type(vocab) is not int or not 103 <= vocab <= 1000000:
            raise ValueError("invalid Nomic vocabulary size")
        epsilon = raw.get("layer_norm_epsilon")
        base = raw.get("rotary_emb_base")
        for key, value in (("layer_norm_epsilon", epsilon), ("rotary_emb_base", base)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid Nomic {key}")
        rope = raw.get("rope_parameters")
        if rope is not None and (
            not isinstance(rope, dict)
            or rope.get("rope_type") != "default"
            or rope.get("rope_theta") != base
        ):
            raise ValueError("Nomic supports only matching default rope_parameters")
        return cls(vocab, float(epsilon), float(base))
