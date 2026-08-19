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

class TestDebertaPlugin:
    """DeBERTa plugin: disentangled attention, relative position embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["deberta.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["deberta.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["deberta.embeddings.LayerNorm.bias"] = _rand(hidden)

        # Relative position embedding
        t["deberta.encoder.rel_embeddings.weight"] = _rand(2 * max_pos, hidden)

        for i in range(layers):
            p = f"deberta.encoder.layer.{i}"
            # Fused QKV (in_proj)
            t[f"{p}.attention.self.in_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.self.q_bias"] = _rand(hidden)
            t[f"{p}.attention.self.v_bias"] = _rand(hidden)
            # Position projections (c2p and p2c)
            t[f"{p}.attention.self.pos_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.bias"] = _rand(hidden)
            # Output
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            # FFN
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.deberta.model as plugin

        config = {
            "model_type": "deberta",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "pos_att_type": "c2p|p2c",
            "max_relative_positions": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "embed_norm" in weights
        assert "rel_embeddings" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "q_bias", "v_bias",
                        "pos_proj", "pos_q_proj", "w_o", "o_bias",
                        "post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.models.deberta.model as plugin
        assert plugin.matches("deberta")
        assert not plugin.matches("deberta-v2")

    def test_runtime_strategy(self):
        import tensorrt_model_connect.models.deberta.model as plugin
        assert plugin.runtime_strategy == "deberta_encoder_only"
