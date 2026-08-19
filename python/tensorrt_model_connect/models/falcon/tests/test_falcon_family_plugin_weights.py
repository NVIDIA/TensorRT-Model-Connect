# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tensorrt_model_connect.models.falcon.tests._family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestFalconPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    MLP = 64  # dense_h_to_4h out size

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        kv_dim = self.KV_HEADS * self.HEAD_DIM
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            # LayerNorm with bias
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.input_layernorm.bias"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(self.HIDDEN)
            # Separate Q/K/V/O
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_dim, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_dim, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            # GELU FC MLP with biases
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(self.HIDDEN, self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(self.HIDDEN)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["model.norm.bias"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_bias_weights_loaded(self, tmp_path):
        """Falcon uses LayerNorm with bias — verify bias keys present."""
        from tensorrt_model_connect.models.falcon import model as plugin

        config = {
            "model_type": "falcon",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.input_norm_beta" in weights
            assert f"layer.{i}.post_attn_norm_beta" in weights
            np.testing.assert_allclose(
                weights[f"layer.{i}.input_norm_beta"],
                tensors[f"model.layers.{i}.input_layernorm.bias"], atol=1e-6)

    def test_fc_mlp_keys(self, tmp_path):
        """Falcon uses fc1/fc2 MLP naming (not gate/up/down)."""
        from tensorrt_model_connect.models.falcon import model as plugin

        config = {
            "model_type": "falcon",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.w_fc1" in weights
            assert f"layer.{i}.w_fc2" in weights
            assert f"layer.{i}.fc1_bias" in weights
            assert f"layer.{i}.fc2_bias" in weights
            # fc1 transposed: [hidden, mlp]
            assert weights[f"layer.{i}.w_fc1"].shape == (self.HIDDEN, self.MLP)

    def test_final_norm_beta(self, tmp_path):
        from tensorrt_model_connect.models.falcon import model as plugin

        config = {
            "model_type": "falcon",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "final_norm_beta" in weights
        np.testing.assert_allclose(
            weights["final_norm_beta"], tensors["model.norm.bias"], atol=1e-6)
