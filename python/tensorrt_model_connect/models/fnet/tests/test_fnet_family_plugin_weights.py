# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests."""

from __future__ import annotations


from tensorrt_model_connect.models.fnet.tests._family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestFNetPlugin:
    """FNet plugin: no attention, uses DFT, GELU FFN."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE, MAX_POS = 32, 16, 2, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate, max_pos):
        t = {}
        t["fnet.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["fnet.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["fnet.embeddings.token_type_embeddings.weight"] = _rand(4, hidden)
        t["fnet.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["fnet.embeddings.LayerNorm.bias"] = _rand(hidden)
        t["fnet.embeddings.projection.weight"] = _rand(hidden, hidden)
        t["fnet.embeddings.projection.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"fnet.encoder.layer.{i}"
            t[f"{p}.fourier.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.fourier.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.fnet.model as plugin

        config = {
            "model_type": "fnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "type_vocab_size": 4,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "token_type_embedding" in weights
        assert "embed_norm" in weights
        assert "embed_projection" in weights

        for i in range(self.LAYERS):
            for key in ("post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.models.fnet.model as plugin
        assert plugin.matches("fnet")
        assert not plugin.matches("bert")
