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


class TestBloomPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 2, 4
    HEAD_DIM = HIDDEN // HEADS  # 4
    MLP = 64

    def _make_tensors(self):
        t = {}
        t["transformer.word_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        t["transformer.word_embeddings_layernorm.weight"] = _rand(self.HIDDEN)
        t["transformer.word_embeddings_layernorm.bias"] = _rand(self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"transformer.h.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.input_layernorm.bias"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(self.HIDDEN)
            # Fused QKV per-head interleaved: [3*hidden, hidden]
            # Layout: for each head h, rows [h*3*hd : h*3*hd + 3*hd] are Q,K,V
            t[f"{p}.self_attention.query_key_value.weight"] = _rand(
                3 * self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attention.query_key_value.bias"] = _rand(
                3 * self.HIDDEN)
            t[f"{p}.self_attention.dense.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attention.dense.bias"] = _rand(self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(self.HIDDEN, self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(self.HIDDEN)
        t["transformer.ln_f.weight"] = _rand(self.HIDDEN)
        t["transformer.ln_f.bias"] = _rand(self.HIDDEN)
        return t

    def test_qkv_interleaved_split(self, tmp_path):
        """BLOOM QKV is per-head interleaved; verify correct split."""
        from tensorrt_model_connect.families.bloom import model as plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        hd = self.HEAD_DIM
        for i in range(self.LAYERS):
            qkv_w = tensors[
                f"transformer.h.{i}.self_attention.query_key_value.weight"]
            # Extract Q per-head manually
            q_parts = []
            for h in range(self.HEADS):
                base = h * 3 * hd
                q_parts.append(qkv_w[base:base + hd])
            q_expected = np.concatenate(q_parts, axis=0)  # [hidden, hidden]
            # Should be transposed: [hidden, hidden]
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_q"],
                q_expected.T.astype(np.float32), atol=1e-6)

    def test_embedding_layernorm(self, tmp_path):
        """BLOOM has an embedding LayerNorm."""
        from tensorrt_model_connect.families.bloom import model as plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding_norm" in weights
        assert "embedding_norm_beta" in weights
        np.testing.assert_allclose(
            weights["embedding_norm"],
            tensors["transformer.word_embeddings_layernorm.weight"],
            atol=1e-6)

    def test_qkv_bias_split(self, tmp_path):
        """QKV biases should be split per-head interleaved just like weights."""
        from tensorrt_model_connect.families.bloom import model as plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "layer.0.q_bias" in weights
        assert "layer.0.k_bias" in weights
        assert "layer.0.v_bias" in weights
        assert weights["layer.0.q_bias"].shape == (self.HIDDEN,)

    def test_all_keys(self, tmp_path):
        from tensorrt_model_connect.families.bloom import model as plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        for i in range(self.LAYERS):
            for key in ("input_norm", "input_norm_beta", "post_attn_norm",
                        "post_attn_norm_beta", "w_q", "w_k", "w_v", "w_o",
                        "q_bias", "k_bias", "v_bias", "o_bias",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
        assert "final_norm" in weights
        assert "final_norm_beta" in weights
        assert "w_out" in weights
        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_mlp_size"] == self.MLP
