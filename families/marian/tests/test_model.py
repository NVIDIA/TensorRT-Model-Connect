# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    fake_tensorrt = types.ModuleType("tensorrt")
    fake_tensorrt.__version__ = "11.0.0"
    sys.modules["tensorrt"] = fake_tensorrt

from families.marian.config import ModelConfig
from families.marian.model import _MarianModel, _make_sinusoidal_pos_embed
from families.marian import graph_ops


def _create_minimal_state_dict(
    *,
    vocab_size: int = 32,
    hidden_size: int = 16,
    ffn_dim: int = 32,
    enc_layers: int = 1,
    dec_layers: int = 1,
    shared_embeddings: bool = False,
    include_embed_positions: bool = True,
    max_positions: int = 64,
) -> dict[str, torch.Tensor]:
    state = {}
    if shared_embeddings:
        state["model.shared.weight"] = torch.randn(vocab_size, hidden_size)
    else:
        state["model.encoder.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
        state["model.decoder.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)

    if include_embed_positions:
        state["model.encoder.embed_positions.weight"] = torch.randn(max_positions, hidden_size)
        state["model.decoder.embed_positions.weight"] = torch.randn(max_positions, hidden_size)

    for i in range(enc_layers):
        pfx = f"model.encoder.layers.{i}"
        for proj in ("q", "k", "v"):
            state[f"{pfx}.self_attn.{proj}_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state[f"{pfx}.self_attn.{proj}_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn.out_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state[f"{pfx}.self_attn.out_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn_layer_norm.weight"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn_layer_norm.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.fc1.weight"] = torch.randn(ffn_dim, hidden_size)
        state[f"{pfx}.fc1.bias"] = torch.randn(ffn_dim)
        state[f"{pfx}.fc2.weight"] = torch.randn(hidden_size, ffn_dim)
        state[f"{pfx}.fc2.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.final_layer_norm.weight"] = torch.randn(hidden_size)
        state[f"{pfx}.final_layer_norm.bias"] = torch.randn(hidden_size)

    for i in range(dec_layers):
        pfx = f"model.decoder.layers.{i}"
        for proj in ("q", "k", "v"):
            state[f"{pfx}.self_attn.{proj}_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state[f"{pfx}.self_attn.{proj}_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn.out_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state[f"{pfx}.self_attn.out_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn_layer_norm.weight"] = torch.randn(hidden_size)
        state[f"{pfx}.self_attn_layer_norm.bias"] = torch.randn(hidden_size)

        for proj in ("q", "k", "v"):
            state[f"{pfx}.encoder_attn.{proj}_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state[f"{pfx}.encoder_attn.{proj}_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.encoder_attn.out_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state[f"{pfx}.encoder_attn.out_proj.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.encoder_attn_layer_norm.weight"] = torch.randn(hidden_size)
        state[f"{pfx}.encoder_attn_layer_norm.bias"] = torch.randn(hidden_size)

        state[f"{pfx}.fc1.weight"] = torch.randn(ffn_dim, hidden_size)
        state[f"{pfx}.fc1.bias"] = torch.randn(ffn_dim)
        state[f"{pfx}.fc2.weight"] = torch.randn(hidden_size, ffn_dim)
        state[f"{pfx}.fc2.bias"] = torch.randn(hidden_size)
        state[f"{pfx}.final_layer_norm.weight"] = torch.randn(hidden_size)
        state[f"{pfx}.final_layer_norm.bias"] = torch.randn(hidden_size)

    return state


def test_sinusoidal_pos_embed_shape_and_values():
    n_pos = 128
    dim = 64
    weights = _make_sinusoidal_pos_embed(n_pos, dim)
    assert weights.shape == (n_pos, dim)
    assert weights.dtype == np.float32

    # First position (pos 0) sin values should be 0.0, cos values should be 1.0
    sentinel = dim // 2
    assert np.allclose(weights[0, :sentinel], 0.0)
    assert np.allclose(weights[0, sentinel:], 1.0)


def test_load_weights_standard(tmp_path: Path):
    vocab_size = 32
    hidden_size = 16
    ffn_dim = 32
    max_positions = 64

    state = _create_minimal_state_dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        enc_layers=1,
        dec_layers=1,
        shared_embeddings=False,
        include_embed_positions=True,
        max_positions=max_positions,
    )
    torch.save(state, tmp_path / "pytorch_model.bin")

    raw_config = {
        "model_type": "marian",
        "vocab_size": vocab_size,
        "d_model": hidden_size,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": ffn_dim,
        "decoder_ffn_dim": ffn_dim,
        "max_position_embeddings": max_positions,
        "activation_function": "swish",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = ModelConfig.from_json(json.dumps(raw_config))

    weights = _MarianModel().load_weights(str(tmp_path), config)

    assert "enc_embedding" in weights
    assert "dec_embedding" in weights
    assert "enc_pos_embedding" in weights
    assert "dec_pos_embedding" in weights
    assert weights["_activation_function"] == "silu"


def test_load_weights_tc_big_checkpoint(tmp_path: Path):
    """OPUS-MT tc-big models store embeddings as model.shared.weight, omit static embed_positions,
    and specify activation_function as relu."""
    vocab_size = 32
    hidden_size = 16
    ffn_dim = 32
    max_positions = 64

    state = _create_minimal_state_dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        enc_layers=1,
        dec_layers=1,
        shared_embeddings=True,
        include_embed_positions=False,
        max_positions=max_positions,
    )
    torch.save(state, tmp_path / "pytorch_model.bin")

    raw_config = {
        "model_type": "marian",
        "vocab_size": vocab_size,
        "d_model": hidden_size,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": ffn_dim,
        "decoder_ffn_dim": ffn_dim,
        "max_position_embeddings": max_positions,
        "activation_function": "relu",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = ModelConfig.from_json(json.dumps(raw_config))

    weights = _MarianModel().load_weights(str(tmp_path), config)

    # Shared embeddings should be mapped to both enc_embedding and dec_embedding
    assert "enc_embedding" in weights
    assert "dec_embedding" in weights
    assert np.array_equal(weights["enc_embedding"], state["model.shared.weight"].numpy())
    assert np.array_equal(weights["dec_embedding"], state["model.shared.weight"].numpy())

    # Positional embeddings should be synthesized
    assert "enc_pos_embedding" in weights
    assert "dec_pos_embedding" in weights
    assert weights["enc_pos_embedding"].shape == (max_positions, hidden_size)
    assert weights["dec_pos_embedding"].shape == (max_positions, hidden_size)

    # Activation should be preserved from config
    assert weights["_activation_function"] == "relu"


def test_load_weights_shared_weight_prefix_fallback(tmp_path: Path):
    """Test fallback when shared embedding is named 'shared.weight' without 'model.' prefix."""
    vocab_size = 32
    hidden_size = 16
    ffn_dim = 32
    max_positions = 64

    state = _create_minimal_state_dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        enc_layers=1,
        dec_layers=1,
        shared_embeddings=False,
        include_embed_positions=False,
        max_positions=max_positions,
    )
    # Remove separate embeddings and add un-prefixed shared.weight
    state.pop("model.encoder.embed_tokens.weight")
    state.pop("model.decoder.embed_tokens.weight")
    state["shared.weight"] = torch.randn(vocab_size, hidden_size)

    torch.save(state, tmp_path / "pytorch_model.bin")

    raw_config = {
        "model_type": "marian",
        "vocab_size": vocab_size,
        "d_model": hidden_size,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": ffn_dim,
        "decoder_ffn_dim": ffn_dim,
        "max_position_embeddings": max_positions,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = ModelConfig.from_json(json.dumps(raw_config))

    weights = _MarianModel().load_weights(str(tmp_path), config)
    assert np.array_equal(weights["enc_embedding"], state["shared.weight"].numpy())
    assert np.array_equal(weights["dec_embedding"], state["shared.weight"].numpy())


def test_load_weights_missing_token_embeddings_raises(tmp_path: Path):
    """Verify KeyError when token embeddings cannot be found."""
    state = _create_minimal_state_dict(
        shared_embeddings=False,
        include_embed_positions=True,
    )
    state.pop("model.encoder.embed_tokens.weight")
    state.pop("model.decoder.embed_tokens.weight")
    torch.save(state, tmp_path / "pytorch_model.bin")

    raw_config = {
        "model_type": "marian",
        "vocab_size": 32,
        "d_model": 16,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = ModelConfig.from_json(json.dumps(raw_config))

    with pytest.raises(KeyError, match="Tensor not found: model.encoder.embed_tokens.weight"):
        _MarianModel().load_weights(str(tmp_path), config)


def test_add_activation_unsupported():
    with pytest.raises(ValueError, match="Unsupported activation: invalid_act"):
        graph_ops.add_activation(None, None, "invalid_act")

