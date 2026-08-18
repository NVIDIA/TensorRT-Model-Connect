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

class TestElectraPlugin:
    """Electra plugin: BERT-like encoder with learned pos + token type embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["electra.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["electra.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["electra.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["electra.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["electra.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"electra.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(hidden)
            t[f"{p}.attention.self.value.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.families.electra.model as plugin

        config = {
            "model_type": "electra",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "token_type_embedding" in weights
        assert "embed_norm" in weights
        assert "embed_norm_beta" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "q_bias", "k_bias", "v_bias",
                        "w_o", "o_bias", "post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.families.electra.model as plugin
        assert plugin.matches("electra")
        assert not plugin.matches("bert")

    def test_runtime_strategy(self):
        import tensorrt_model_connect.families.electra.model as plugin
        assert plugin.runtime_strategy == "electra_encoder_only"
