# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations

import math

import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


def _make_standard_decoder_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim
    tensors = {}
    tensors["model.embed_tokens.weight"] = _rand(vocab, hidden)
    for i in range(layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = _rand(hidden)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(hidden)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
        tensors[f"{prefix}.mlp.up_proj.weight"] = _rand(mlp, hidden)
        tensors[f"{prefix}.mlp.down_proj.weight"] = _rand(hidden, mlp)
    tensors["model.norm.weight"] = _rand(hidden)
    tensors["lm_head.weight"] = _rand(vocab, hidden)
    return tensors


class TestGemmaPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    def _setup(self, tmp_path, num_layers=2):
        config = {
            "model_type": "gemma",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": num_layers,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = _make_standard_decoder_tensors(
            self.VOCAB, self.HIDDEN, num_layers, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)
        return tensors

    def test_gamma_plus_one(self, tmp_path):
        """RMSNorm weights should have +1.0 added (Gemma offset)."""
        from tensorrt_model_connect.families.gemma import model as plugin

        tensors = self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            raw_input_norm = tensors[f"model.layers.{i}.input_layernorm.weight"]
            np.testing.assert_allclose(
                weights[f"layer.{i}.input_norm"],
                raw_input_norm + 1.0, atol=1e-6)
            raw_post_norm = tensors[f"model.layers.{i}.post_attention_layernorm.weight"]
            np.testing.assert_allclose(
                weights[f"layer.{i}.post_attn_norm"],
                raw_post_norm + 1.0, atol=1e-6)

        raw_final = tensors["model.norm.weight"]
        np.testing.assert_allclose(
            weights["final_norm"], raw_final + 1.0, atol=1e-6)

    def test_embedding_scaling(self, tmp_path):
        """Embedding should be scaled by sqrt(hidden_size)."""
        from tensorrt_model_connect.families.gemma import model as plugin

        tensors = self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        scale = math.sqrt(self.HIDDEN)
        expected_embed = tensors["model.embed_tokens.weight"] * scale
        np.testing.assert_allclose(
            weights["embedding"], expected_embed, atol=1e-5)
