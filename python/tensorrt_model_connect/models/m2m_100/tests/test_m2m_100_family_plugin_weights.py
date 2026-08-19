# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests."""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.models.m2m_100.tests._family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestM2M100SinusoidalPosEmbed:
    """Test the _make_sinusoidal_pos_embed utility function."""

    def test_output_shape(self):
        from tensorrt_model_connect.models.m2m_100.model import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16)
        assert result.shape == (10, 16)
        assert result.dtype == np.float32

    def test_padding_idx_zeroed(self):
        from tensorrt_model_connect.models.m2m_100.model import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16, padding_idx=1)
        np.testing.assert_array_equal(result[1], np.zeros(16))

    def test_first_position_pattern(self):
        from tensorrt_model_connect.models.m2m_100.model import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16, padding_idx=None)
        # Position 0 should have specific sin/cos pattern
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)  # sin(0) = 0

    def test_odd_dimension(self):
        from tensorrt_model_connect.models.m2m_100.model import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 17)
        assert result.shape == (10, 17)


class TestM2M100Plugin:
    """M2M-100 plugin: shared embedding, sinusoidal pos, encoder+decoder."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)

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

        t["model.encoder.layer_norm.weight"] = _rand(hidden)
        t["model.encoder.layer_norm.bias"] = _rand(hidden)

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

        t["model.decoder.layer_norm.weight"] = _rand(hidden)
        t["model.decoder.layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.m2m_100.model as plugin

        config = {
            "model_type": "m2m_100",
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
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert "sinusoidal_pos_embed" in weights
        assert "enc_final_norm" in weights
        assert weights["sinusoidal_pos_embed"].shape[1] == self.HIDDEN

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o",
                        "attn_norm", "attn_norm_beta", "w_fc1", "b_fc1", "w_fc2",
                        "b_fc2", "ffn_norm", "ffn_norm_beta"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.models.m2m_100.model as plugin
        assert plugin.matches("m2m_100")
        assert plugin.matches("nllb")
        assert not plugin.matches("bart")
        assert plugin.runtime_strategy == "m2m_100_seq2seq_encoder_decoder"
