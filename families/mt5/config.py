# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact mT5 encoder/decoder dimensions, including independent attention width."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    raw: dict
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    @classmethod
    def from_dir(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        if raw.get("model_type") != "mt5" or raw.get("feed_forward_proj") != "gated-gelu":
            raise ValueError("mt5 requires the gated-gelu mT5 architecture")
        if raw.get("tie_word_embeddings", False):
            raise NotImplementedError("mt5 requires an independent language-model head")
        for key in ("d_model", "d_ff", "d_kv", "num_heads", "num_layers", "vocab_size"):
            if not isinstance(raw.get(key), int) or raw[key] <= 0:
                raise ValueError(f"mt5 config requires positive {key}")
        return cls(raw, "mt5", raw["d_model"], raw["d_ff"], raw["num_layers"],
                   raw["num_heads"], raw["vocab_size"], -1, raw.get("eos_token_id", 1),
                   raw.get("pad_token_id", 0))
