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

class TestDPRPlugin:
    """DPR plugin: BERT-like encoder for dense retrieval."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        # DPR uses ctx_encoder or question_encoder prefix
        t["ctx_encoder.bert_model.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["ctx_encoder.bert_model.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["ctx_encoder.bert_model.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["ctx_encoder.bert_model.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["ctx_encoder.bert_model.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"ctx_encoder.bert_model.encoder.layer.{i}"
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
        from tensorrt_model_connect.families.dpr import plugin

        config = {
            "model_type": "dpr",
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
        assert "embed_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.dpr import plugin
        assert plugin.matches("dpr")
        assert not plugin.matches("bert")
