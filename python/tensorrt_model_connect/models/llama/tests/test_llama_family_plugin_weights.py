# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tensorrt_model_connect.models.llama.tests._family_plugin_test_support import (
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


class TestLlamaPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    def test_selected_fp32_layers_use_single_engine_layout(self):
        from tensorrt_model_connect.models.llama import model as plugin

        config = ModelConfig(
            hidden_size=self.HIDDEN,
            vocab_size=self.VOCAB,
            num_hidden_layers=self.LAYERS,
            num_attention_heads=self.HEADS,
            num_key_value_heads=self.KV_HEADS,
        )
        assert plugin.supports_split_decoder_roles(config)

        config.raw["_fp32_layers"] = [1]
        assert not plugin.supports_split_decoder_roles(config)

    def test_load_weights(self, tmp_path):
        """LLaMA uses load_standard_weights — verify compact GQA K/V."""
        from tensorrt_model_connect.models.llama import model as plugin

        head_dim = self.HIDDEN // self.HEADS  # 4
        kv_hidden = self.KV_HEADS * head_dim  # 8
        config = {
            "model_type": "llama",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = _make_standard_decoder_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS,
            self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # K/V stay compact at [hidden, kv_hidden].
        for i in range(self.LAYERS):
            assert weights[f"layer.{i}.w_k"].shape == (
                self.HIDDEN, kv_hidden)
            assert weights[f"layer.{i}.w_v"].shape == (
                self.HIDDEN, kv_hidden)

        cfg.raw["_fp32_layers"] = [1]
        mixed_weights = plugin.load_weights(
            str(tmp_path), cfg, precision="fp16")
        assert mixed_weights["embedding"].dtype == np.float16
        assert mixed_weights["layer.0.w_q"].dtype == np.float16
        assert mixed_weights["layer.1.w_q"].dtype == np.float32

    def test_tied_embeddings(self, tmp_path):
        """When lm_head.weight is missing, w_out = transposed embedding."""
        from tensorrt_model_connect.models.llama import model as plugin

        config = {
            "model_type": "llama",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "tie_word_embeddings": True,
        }
        tensors = _make_standard_decoder_tensors(
            self.VOCAB, self.HIDDEN, 1, self.HEADS, self.KV_HEADS, self.MLP)
        # Remove lm_head to test tied embedding fallback
        del tensors["lm_head.weight"]
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)
        embedding = tensors["model.embed_tokens.weight"]
        np.testing.assert_allclose(
            weights["w_out"], embedding.T, atol=1e-6)
