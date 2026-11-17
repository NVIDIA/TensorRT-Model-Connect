# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART checkpoint loading and the generation settings written to runtime.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.bart import model
from families.bart.config import ModelConfig

D, FF, VOCAB, POSITIONS = 16, 32, 32, 8
CONFIG = {
    "model_type": "bart",
    "vocab_size": VOCAB,
    "d_model": D,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "encoder_attention_heads": 4,
    "decoder_attention_heads": 4,
    "encoder_ffn_dim": FF,
    "decoder_ffn_dim": FF,
    "max_position_embeddings": POSITIONS,
}


def _write_config(model_dir: Path, **updates) -> ModelConfig:
    (model_dir / "config.json").write_text(json.dumps({**CONFIG, **updates}), encoding="utf-8")
    return ModelConfig.from_dir(model_dir)


def _write_checkpoint(model_dir: Path, embedding_key: str) -> np.ndarray:
    tensors = {embedding_key: np.random.default_rng(0).random((VOCAB, D), dtype=np.float32)}
    for side in ("encoder", "decoder"):
        layer = f"model.{side}.layers.0"
        tensors[f"model.{side}.embed_positions.weight"] = np.zeros((POSITIONS + 2, D), np.float32)
        norms = [
            "layernorm_embedding",
            "layers.0.self_attn_layer_norm",
            "layers.0.final_layer_norm",
        ]
        attentions = ["self_attn"]
        if side == "decoder":
            norms.append("layers.0.encoder_attn_layer_norm")
            attentions.append("encoder_attn")
        for name in norms:
            tensors[f"model.{side}.{name}.weight"] = np.ones(D, np.float32)
            tensors[f"model.{side}.{name}.bias"] = np.zeros(D, np.float32)
        linears = {"fc1": (FF, D), "fc2": (D, FF)}
        for attention in attentions:
            linears.update({f"{attention}.{name}_proj": (D, D) for name in ("q", "k", "v", "out")})
        for name, shape in linears.items():
            tensors[f"{layer}.{name}.weight"] = np.zeros(shape, np.float32)
            tensors[f"{layer}.{name}.bias"] = np.zeros(shape[0], np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors[embedding_key]


@pytest.mark.parametrize(
    "embedding_key",
    ["shared.weight", "model.shared.weight", "model.decoder.embed_tokens.weight"],
)
def test_shared_embedding_is_loaded_from_every_checkpoint_name(
    tmp_path: Path, embedding_key: str
) -> None:
    embedding = _write_checkpoint(tmp_path, embedding_key)

    weights = model._BartModel().load_weights(str(tmp_path), _write_config(tmp_path))

    np.testing.assert_array_equal(weights["shared_embedding"], embedding)
    np.testing.assert_array_equal(weights["w_out"], embedding.T)


@pytest.mark.parametrize(
    ("generation_config", "config_updates", "expected"),
    [
        ({"forced_bos_token_id": 0}, {}, 0),
        ({"eos_token_id": 2}, {"forced_bos_token_id": 0}, -1),
        (None, {"forced_bos_token_id": 0}, 0),
        (None, {}, -1),
    ],
)
def test_runtime_config_carries_forced_bos_token_id(
    tmp_path: Path, generation_config: dict | None, config_updates: dict, expected: int
) -> None:
    config = _write_config(tmp_path, **config_updates)
    if generation_config is not None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps(generation_config), encoding="utf-8"
        )

    runtime = model._runtime_config(tmp_path, config, model._BartModel())

    assert runtime["forced_bos_token_id"] == expected
