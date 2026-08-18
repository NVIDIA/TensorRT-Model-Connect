# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestGlmPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    Q_DIM = HEADS * HEAD_DIM    # 16
    KV_DIM = KV_HEADS * HEAD_DIM  # 8
    MLP_INTER = 24

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            # Separate Q/K/V with biases
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.Q_DIM, self.HIDDEN)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(self.Q_DIM)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(self.KV_DIM)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(self.KV_DIM)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            # Fused gate_up: [2*intermediate, hidden]
            fused_gate_up = _rand(2 * self.MLP_INTER, self.HIDDEN)
            t[f"{p}.mlp.gate_up_proj.weight"] = fused_gate_up
            t[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP_INTER)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_gate_up_split(self, tmp_path):
        """Fused gate_up should be correctly split into gate and up."""
        from tensorrt_model_connect.families.glm import model as plugin

        config = {
            "model_type": "glm",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        fused = tensors["model.layers.0.mlp.gate_up_proj.weight"]
        gate_raw = fused[:self.MLP_INTER, :]
        up_raw = fused[self.MLP_INTER:, :]

        np.testing.assert_allclose(
            weights["layer.0.w_gate"],
            gate_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_up"],
            up_raw.T.astype(np.float32), atol=1e-6)

    def test_qkv_biases_stay_compact(self, tmp_path):
        """Q/K/V biases should be loaded; K/V biases stay compact."""
        from tensorrt_model_connect.families.glm import model as plugin

        config = {
            "model_type": "glm",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        np.testing.assert_allclose(
            weights["layer.0.q_bias"],
            tensors["model.layers.0.self_attn.q_proj.bias"].astype(np.float32))
        np.testing.assert_allclose(
            weights["layer.0.k_bias"],
            tensors["model.layers.0.self_attn.k_proj.bias"].astype(np.float32))
        np.testing.assert_allclose(
            weights["layer.0.v_bias"],
            tensors["model.layers.0.self_attn.v_proj.bias"].astype(np.float32))
        assert weights["layer.0.k_bias"].shape == (self.KV_DIM,)
        assert weights["layer.0.v_bias"].shape == (self.KV_DIM,)
