"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations



from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestEagleVLMPlugin:
    """Eagle VLM plugin — Llama backbone for embedding/reranking."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp,
                      *, add_score_head=False):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        if add_score_head:
            t["score.weight"] = _rand(1, hidden)
            t["score.bias"] = _rand(1)
        return t

    def test_load_weights_embedding(self, tmp_path):
        """Embedding mode: loads Llama backbone weights, no score head."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForEmbedding"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)
        _write_safetensors(
            tmp_path,
            self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS,
                               self.HEADS, self.KV_HEADS, self.MLP))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Core keys
        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "layer.0.w_q" in weights
        assert "layer.0.w_k" in weights
        assert "layer.0.w_v" in weights
        assert "layer.0.w_o" in weights
        assert "layer.0.w_gate" in weights
        assert "layer.0.input_norm" in weights
        assert "final_norm" in weights
        kv_hidden = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["_kv_attention_size"] == kv_hidden
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_hidden)
        # No score head for embedding
        assert "score_weight" not in weights

    def test_load_weights_reranking(self, tmp_path):
        """Reranking mode: loads score head weights."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForReranking"],
            "is_reranker": True,
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)
        _write_safetensors(
            tmp_path,
            self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS,
                               self.HEADS, self.KV_HEADS, self.MLP,
                               add_score_head=True))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "score_weight" in weights
        assert "score_bias" in weights

    def test_get_vl_config(self, tmp_path):
        """get_vl_config returns embedding config with vision info."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "image_size": 384,
                "patch_size": 14,
                "hidden_size": 64,
            },
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)

        assert vl_cfg is not None
        assert vl_cfg["embedding_dim"] == self.HIDDEN
        assert vl_cfg["preprocessor_type"] == "simple_chw"
        # After 2x2 pixel_shuffle merge: (384//14//2)^2 = 13^2 = 169
        assert vl_cfg["num_vision_tokens"] == (384 // 14 // 2) ** 2

    def test_bundle_config_overrides_embedding(self, tmp_path):
        """Bundle config overrides: embedding mode."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForEmbedding"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "eagle_vlm_embedding"

    def test_bundle_config_overrides_reranking(self, tmp_path):
        """Bundle config overrides: reranking mode."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForReranking"],
            "is_reranker": True,
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "eagle_vlm_reranking"
