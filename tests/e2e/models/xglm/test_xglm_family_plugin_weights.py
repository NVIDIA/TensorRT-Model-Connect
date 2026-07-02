# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests."""

from __future__ import annotations


from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestXglmPlugin:
    """XGLM plugin: multilingual decoder."""

    VOCAB, HIDDEN, LAYERS, HEADS, MLP = 32, 16, 2, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, mlp):
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{p}.fc1.weight"] = _rand(mlp, hidden)
            t[f"{p}.fc1.bias"] = _rand(mlp)
            t[f"{p}.fc2.weight"] = _rand(hidden, mlp)
            t[f"{p}.fc2.bias"] = _rand(hidden)
            t[f"{p}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.final_layer_norm.bias"] = _rand(hidden)
        t["model.layer_norm.weight"] = _rand(hidden)
        t["model.layer_norm.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.xglm import plugin

        config = {
            "model_type": "xglm",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "ffn_dim": self.MLP,
            "max_position_embeddings": 64,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights

    def test_matches(self):
        from tensorrt_model_connect.families.xglm import plugin
        assert plugin.matches("xglm")
        assert not plugin.matches("gpt2")
