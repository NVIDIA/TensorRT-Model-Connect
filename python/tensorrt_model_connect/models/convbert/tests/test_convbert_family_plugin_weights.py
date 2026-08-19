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

class TestConvBERTPlugin:
    """ConvBERT plugin: mixed attention with span-based convolution."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        # ConvBERT uses head_ratio=2 by default — half heads are conv-based
        new_heads = heads // 2
        attn_head_size = (hidden // new_heads) // 2
        all_head_size = new_heads * attn_head_size
        conv_kernel_size = 9

        t = {}
        t["convbert.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["convbert.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["convbert.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["convbert.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["convbert.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"convbert.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.key.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.value.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(all_head_size)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            # SeparableConv1D: depthwise [channels, 1, kernel] + pointwise [1, channels, 1]
            t[f"{p}.attention.self.key_conv_attn_layer.depthwise.weight"] = _rand(all_head_size, 1, conv_kernel_size)
            t[f"{p}.attention.self.key_conv_attn_layer.pointwise.weight"] = _rand(1, all_head_size, 1)
            t[f"{p}.attention.self.key_conv_attn_layer.bias"] = _rand(all_head_size, 1)
            t[f"{p}.attention.self.conv_kernel_layer.weight"] = _rand(new_heads * conv_kernel_size, all_head_size)
            t[f"{p}.attention.self.conv_kernel_layer.bias"] = _rand(new_heads * conv_kernel_size)
            t[f"{p}.attention.self.conv_out_layer.weight"] = _rand(hidden, all_head_size)
            t[f"{p}.attention.self.conv_out_layer.bias"] = _rand(hidden)
            # FFN
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.convbert.model as plugin

        config = {
            "model_type": "convbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "head_ratio": 2,
            "conv_kernel_size": 9,
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
            for key in ("w_q", "w_k", "w_v", "w_o"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.models.convbert.model as plugin
        assert plugin.matches("convbert")
        assert not plugin.matches("bert")
