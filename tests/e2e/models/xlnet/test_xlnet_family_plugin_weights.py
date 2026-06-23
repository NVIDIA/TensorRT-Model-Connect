"""Family-owned plugin weight tests."""

from __future__ import annotations


from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestXLNetPlugin:
    """XLNet plugin: two-stream attention, relative segment embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE = 32, 16, 2, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate):
        head_dim = hidden // heads
        t = {}
        t["transformer.word_embedding.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            p = f"transformer.layer.{i}"
            # XLNet stores projections as [hidden, num_heads, d_head] (not [out, in])
            for proj in ("q", "k", "v", "o", "r"):
                t[f"{p}.rel_attn.{proj}"] = _rand(hidden, heads, head_dim)
            t[f"{p}.rel_attn.r_w_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_r_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_s_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.seg_embed"] = _rand(2, heads, head_dim)
            t[f"{p}.rel_attn.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.rel_attn.layer_norm.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_1.weight"] = _rand(intermediate, hidden)
            t[f"{p}.ff.layer_1.bias"] = _rand(intermediate)
            t[f"{p}.ff.layer_2.weight"] = _rand(hidden, intermediate)
            t[f"{p}.ff.layer_2.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.xlnet import plugin

        config = {
            "model_type": "xlnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "d_inner": self.INTERMEDIATE,
            "n_layer": self.LAYERS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)

    def test_matches(self):
        from tensorrt_model_connect.families.xlnet import plugin
        assert plugin.matches("xlnet")
        assert not plugin.matches("bert")
