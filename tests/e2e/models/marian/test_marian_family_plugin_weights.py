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

class TestMarianPlugin:
    """Marian plugin: encoder-decoder for machine translation."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn, max_pos):
        t = {}
        # Marian uses separate encoder/decoder embeddings and position embeddings
        t["model.encoder.embed_tokens.weight"] = _rand(vocab, hidden)
        t["model.encoder.embed_positions.weight"] = _rand(max_pos, hidden)
        t["model.decoder.embed_tokens.weight"] = _rand(vocab, hidden)
        t["model.decoder.embed_positions.weight"] = _rand(max_pos, hidden)

        # Encoder
        for i in range(layers):
            pfx = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        # Decoder
        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["final_logits_bias"] = _rand(1, vocab)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.families.marian.model as plugin

        config = {
            "model_type": "marian",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "encoder_layers": self.LAYERS,
            "decoder_layers": self.LAYERS,
            "encoder_attention_heads": self.HEADS,
            "decoder_attention_heads": self.HEADS,
            "encoder_ffn_dim": self.FFN,
            "decoder_ffn_dim": self.FFN,
            "max_position_embeddings": self.MAX_POS,
            "scale_embedding": True,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "enc_embedding" in weights
        assert "enc_pos_embedding" in weights

    def test_matches(self):
        import tensorrt_model_connect.families.marian.model as plugin
        assert plugin.matches("marian")
        assert not plugin.matches("bart")
