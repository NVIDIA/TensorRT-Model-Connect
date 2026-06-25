"""Family-owned plugin weight tests."""

from __future__ import annotations


from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestBartPlugin:
    """BART plugin: shared embedding, encoder+decoder with cross-attention."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn, max_pos):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)

        # Encoder position embeddings (offset=2 -> max_pos+2)
        t["model.encoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        # Encoder layernorm_embedding
        t["model.encoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.encoder.layernorm_embedding.bias"] = _rand(hidden)

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
        t["model.decoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        t["model.decoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.decoder.layernorm_embedding.bias"] = _rand(hidden)

        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            # Self-attention
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            # Cross-attention
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            # FFN
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.bart import plugin

        config = {
            "model_type": "bart",
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
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert "enc_pos_embedding" in weights
        assert "dec_pos_embedding" in weights
        assert "enc_embed_norm" in weights
        assert "dec_embed_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o",
                        "attn_norm", "attn_norm_beta", "w_fc1", "b_fc1", "w_fc2",
                        "b_fc2", "ffn_norm", "ffn_norm_beta"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "q_bias", "k_bias", "v_bias",
                        "o_bias", "input_norm", "input_norm_beta",
                        "cross_w_q", "cross_w_k", "cross_w_v", "cross_w_o",
                        "cross_b_q", "cross_b_k", "cross_b_v", "cross_b_o",
                        "cross_attn_norm", "cross_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "post_attn_norm", "post_attn_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.bart import plugin
        assert plugin.matches("bart")
        assert plugin.matches("mbart")
        assert not plugin.matches("t5")
        assert plugin.runtime_strategy == "bart_seq2seq_encoder_decoder"
