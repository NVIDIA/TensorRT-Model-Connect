# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests."""

from __future__ import annotations


from tensorrt_model_connect.models.albert.tests._family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestAlbertPlugin:
    """Albert plugin: cross-layer parameter sharing, factored embedding."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64
    EMBEDDING_SIZE = 8  # Albert uses factored embedding (smaller than hidden)

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos, embed_size):
        t = {}
        t["albert.embeddings.word_embeddings.weight"] = _rand(vocab, embed_size)
        t["albert.embeddings.position_embeddings.weight"] = _rand(max_pos, embed_size)
        t["albert.embeddings.token_type_embeddings.weight"] = _rand(2, embed_size)
        t["albert.embeddings.LayerNorm.weight"] = _rand(embed_size)
        t["albert.embeddings.LayerNorm.bias"] = _rand(embed_size)
        # Embedding projection (embed_size -> hidden)
        t["albert.encoder.embedding_hidden_mapping_in.weight"] = _rand(hidden, embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.bias"] = _rand(hidden)

        # Albert uses shared weights across groups
        group_prefix = "albert.encoder.albert_layer_groups.0.albert_layers.0"
        t[f"{group_prefix}.attention.query.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.query.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.key.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.key.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.value.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.value.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.dense.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.dense.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.weight"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.bias"] = _rand(hidden)
        t[f"{group_prefix}.ffn.weight"] = _rand(intermediate, hidden)
        t[f"{group_prefix}.ffn.bias"] = _rand(intermediate)
        t[f"{group_prefix}.ffn_output.weight"] = _rand(hidden, intermediate)
        t[f"{group_prefix}.ffn_output.bias"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.weight"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.bias"] = _rand(hidden)

        # Pooler
        t["albert.pooler.weight"] = _rand(hidden, hidden)
        t["albert.pooler.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.albert.model as plugin

        config = {
            "model_type": "albert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "embedding_size": self.EMBEDDING_SIZE,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "num_hidden_groups": 1,
            "inner_group_num": 1,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.EMBEDDING_SIZE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "embed_norm" in weights

    def test_matches(self):
        import tensorrt_model_connect.models.albert.model as plugin
        assert plugin.matches("albert")
        assert not plugin.matches("bert")
