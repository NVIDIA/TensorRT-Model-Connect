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

class TestT5Plugin:
    """T5 plugin: shared embedding, relative attention biases, encoder+decoder layers."""

    VOCAB, HIDDEN, LAYERS, HEADS, DKV, DFF = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, dkv, dff):
        t = {}
        t["shared.weight"] = _rand(vocab, hidden)
        # Encoder relative bias
        t["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["encoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"encoder.block.{i}"
            for proj in ("q", "k", "v", "o"):
                t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(dkv * heads if proj in ("q",) else dkv * heads if proj == "o" else dkv * heads, hidden) if proj != "o" else _rand(hidden, dkv * heads)
            # Fix: T5 uses [d_kv*heads, hidden] for q,k,v and [hidden, d_kv*heads] for o
            t[f"{pfx}.layer.0.SelfAttention.q.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.k.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.v.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.o.weight"] = _rand(hidden, dkv * heads)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)

        # Decoder relative biases
        t["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"decoder.block.{i}"
            # Self-attention
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            # Cross-attention
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)
            # FFN
            t[f"{pfx}.layer.2.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.2.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.2.layer_norm.weight"] = _rand(hidden)

        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.t5.model as plugin

        config = {
            "model_type": "t5",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_heads": self.HEADS,
            "d_kv": self.DKV,
            "d_ff": self.DFF,
            "num_layers": self.LAYERS,
            "num_encoder_layers": self.LAYERS,
            "num_decoder_layers": self.LAYERS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.DKV, self.DFF)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert weights["shared_embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "enc_final_norm" in weights
        assert "final_norm" in weights
        assert "w_out" in weights
        assert "enc_rel_attn_bias" in weights
        assert "dec_self_rel_attn_bias" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "attn_norm", "w_fc1", "w_fc2", "ffn_norm"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"
            for key in ("w_q", "w_k", "w_v", "w_o", "input_norm",
                        "cross_w_q", "cross_w_k", "cross_w_v", "cross_w_o",
                        "cross_attn_norm", "w_fc1", "w_fc2", "post_attn_norm"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        import tensorrt_model_connect.models.t5.model as plugin
        assert plugin.matches("t5")
        assert not plugin.matches("bart")

    def test_runtime_strategy(self):
        import tensorrt_model_connect.models.t5.model as plugin
        assert plugin.runtime_strategy == "t5_text_to_text"
