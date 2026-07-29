# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests."""

from __future__ import annotations

import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestGranitePlugin:
    """Granite plugin: decoder with attention_multiplier."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.granite import plugin

        config = {
            "model_type": "granite",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "embedding_multiplier": 12.0,
            "attention_multiplier": 0.015625,
            "residual_multiplier": 0.22,
            "logits_scaling": 8.0,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP
        )
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        np.testing.assert_array_equal(
            weights["embedding"],
            tensors["model.embed_tokens.weight"].astype(np.float32)
            * config["embedding_multiplier"],
        )
        standard_attention_scale = 1.0 / np.sqrt(self.HIDDEN // self.HEADS)
        np.testing.assert_array_equal(
            weights["layer.0.w_q"],
            tensors["model.layers.0.self_attn.q_proj.weight"].T.astype(
                np.float32,
            )
            * (config["attention_multiplier"] / standard_attention_scale),
        )
        np.testing.assert_array_equal(
            weights["layer.0.w_o"],
            tensors["model.layers.0.self_attn.o_proj.weight"].T.astype(
                np.float32,
            )
            * config["residual_multiplier"],
        )
        np.testing.assert_array_equal(
            weights["layer.0.w_down"],
            tensors["model.layers.0.mlp.down_proj.weight"].T.astype(
                np.float32,
            )
            * config["residual_multiplier"],
        )
        np.testing.assert_array_equal(
            weights["w_out"],
            tensors["lm_head.weight"].T.astype(np.float32) / config["logits_scaling"],
        )

    def test_matches(self):
        from tensorrt_model_connect.families.granite import plugin

        assert plugin.matches("granite")
        assert plugin.matches("granitemoeshared")
        assert not plugin.matches("llama")
