"""Family-owned plugin weight tests."""

from __future__ import annotations


from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestStarcoder2Plugin:
    """Starcoder2 plugin: standard decoder for code generation."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.input_layernorm.bias"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.o_proj.bias"] = _rand(hidden)
            t[f"{p}.mlp.c_fc.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.c_fc.bias"] = _rand(mlp)
            t[f"{p}.mlp.c_proj.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.c_proj.bias"] = _rand(hidden)
        t["model.norm.weight"] = _rand(hidden)
        t["model.norm.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.starcoder2 import plugin

        config = {
            "model_type": "starcoder2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "final_norm" in weights
        kv_hidden = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["_kv_attention_size"] == kv_hidden
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.k_bias"].shape == (kv_hidden,)
        assert weights["layer.0.v_bias"].shape == (kv_hidden,)

    def test_matches(self):
        from tensorrt_model_connect.families.starcoder2 import plugin
        assert plugin.matches("starcoder2")
        assert not plugin.matches("gpt2")
