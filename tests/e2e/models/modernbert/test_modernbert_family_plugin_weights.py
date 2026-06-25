"""Family-owned plugin weight tests."""

from __future__ import annotations

import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)

class TestModernbertPlugin:
    """ModernBERT plugin: fused QKV, GeGLU MLP, RoPE, no bias."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE = 32, 16, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate):
        t = {}
        t["model.embeddings.tok_embeddings.weight"] = _rand(vocab, hidden)
        t["model.embeddings.norm.weight"] = _rand(hidden)
        t["model.final_norm.weight"] = _rand(hidden)

        for i in range(layers):
            p = f"model.layers.{i}"
            if i > 0:
                t[f"{p}.attn_norm.weight"] = _rand(hidden)
            t[f"{p}.attn.Wqkv.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.Wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp_norm.weight"] = _rand(hidden)
            t[f"{p}.mlp.Wi.weight"] = _rand(2 * intermediate, hidden)
            t[f"{p}.mlp.Wo.weight"] = _rand(hidden, intermediate)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        tensors = self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "embed_norm" in weights
        assert "final_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "mlp_norm",
                        "w_mlp_input", "w_mlp_gate", "w_down"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
            # Layer 0 has no attn_norm
            if i > 0:
                assert f"layer.{i}.attn_norm" in weights

    def test_fused_qkv_split(self, tmp_path):
        """Verify the fused QKV weight is split correctly into Q, K, V."""
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        # Build known QKV
        q_raw = _rand(self.HIDDEN, self.HIDDEN)
        k_raw = _rand(self.HIDDEN, self.HIDDEN)
        v_raw = _rand(self.HIDDEN, self.HIDDEN)
        wqkv = np.concatenate([q_raw, k_raw, v_raw], axis=0)

        tensors = {}
        tensors["model.embeddings.tok_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        tensors["model.embeddings.norm.weight"] = _rand(self.HIDDEN)
        tensors["model.final_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.attn.Wqkv.weight"] = wqkv
        tensors["model.layers.0.attn.Wo.weight"] = _rand(self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.mlp_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.mlp.Wi.weight"] = _rand(2 * self.INTERMEDIATE, self.HIDDEN)
        tensors["model.layers.0.mlp.Wo.weight"] = _rand(self.HIDDEN, self.INTERMEDIATE)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Q, K, V should be transposed versions of the raw splits
        np.testing.assert_array_equal(weights["layer.0.w_q"],
                                       np.ascontiguousarray(q_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_k"],
                                       np.ascontiguousarray(k_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_v"],
                                       np.ascontiguousarray(v_raw.T.astype(np.float32)))

    def test_geglu_split(self, tmp_path):
        """Verify the fused GeGLU Wi weight is split into input and gate."""
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        input_raw = _rand(self.INTERMEDIATE, self.HIDDEN)
        gate_raw = _rand(self.INTERMEDIATE, self.HIDDEN)
        wi = np.concatenate([input_raw, gate_raw], axis=0)

        tensors = {}
        tensors["model.embeddings.tok_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        tensors["model.embeddings.norm.weight"] = _rand(self.HIDDEN)
        tensors["model.final_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.attn.Wqkv.weight"] = _rand(3 * self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.attn.Wo.weight"] = _rand(self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.mlp_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.mlp.Wi.weight"] = wi
        tensors["model.layers.0.mlp.Wo.weight"] = _rand(self.HIDDEN, self.INTERMEDIATE)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        np.testing.assert_array_equal(weights["layer.0.w_mlp_input"],
                                       np.ascontiguousarray(input_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_mlp_gate"],
                                       np.ascontiguousarray(gate_raw.T.astype(np.float32)))

    def test_matches(self):
        from tensorrt_model_connect.families.modernbert import plugin
        assert plugin.matches("modernbert")
        assert plugin.matches("ModernBert")
        assert not plugin.matches("bert")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.modernbert import plugin
        assert plugin.runtime_strategy == "modernbert_encoder_only"
