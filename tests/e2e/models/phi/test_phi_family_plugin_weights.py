"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestPhiPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    Q_DIM = HEADS * HEAD_DIM    # 16
    KV_DIM = KV_HEADS * HEAD_DIM  # 8
    MLP_INTER = 24  # intermediate size

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            # Fused QKV: [q_dim + 2*kv_dim, hidden]
            fused_qkv = _rand(self.Q_DIM + 2 * self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.qkv_proj.weight"] = fused_qkv
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            # Fused gate_up: [2*intermediate, hidden]
            fused_gate_up = _rand(2 * self.MLP_INTER, self.HIDDEN)
            t[f"{p}.mlp.gate_up_proj.weight"] = fused_gate_up
            t[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP_INTER)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_qkv_split(self, tmp_path):
        """Fused QKV should be correctly split into Q, K, V."""
        from tensorrt_model_connect.families.phi import plugin

        config = {
            "model_type": "phi3",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            fused = tensors[f"model.layers.{i}.self_attn.qkv_proj.weight"]
            q_raw = fused[:self.Q_DIM, :]
            k_raw = fused[self.Q_DIM:self.Q_DIM + self.KV_DIM, :]
            v_raw = fused[self.Q_DIM + self.KV_DIM:, :]

            # Q should be transposed: [hidden, q_dim]
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_q"],
                q_raw.T.astype(np.float32), atol=1e-6)

            np.testing.assert_allclose(
                weights[f"layer.{i}.w_k"],
                k_raw.T.astype(np.float32), atol=1e-6)
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_v"],
                v_raw.T.astype(np.float32), atol=1e-6)

    def test_gate_up_split(self, tmp_path):
        """Fused gate_up should be correctly split into gate and up."""
        from tensorrt_model_connect.families.phi import plugin

        config = {
            "model_type": "phi3",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        fused = tensors["model.layers.0.mlp.gate_up_proj.weight"]
        gate_raw = fused[:self.MLP_INTER, :]
        up_raw = fused[self.MLP_INTER:, :]

        # gate should be transposed: [hidden, intermediate]
        np.testing.assert_allclose(
            weights["layer.0.w_gate"],
            gate_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_up"],
            up_raw.T.astype(np.float32), atol=1e-6)

    def test_all_keys_present(self, tmp_path):
        from tensorrt_model_connect.families.phi import plugin

        config = {
            "model_type": "phi3",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        for i in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm", "w_q", "w_k", "w_v",
                        "w_o", "w_gate", "w_up", "w_down"):
                assert f"layer.{i}.{key}" in weights
        assert "final_norm" in weights
        assert "w_out" in weights
        assert weights["_attention_size"] == self.Q_DIM
        assert weights["_mlp_size"] == self.MLP_INTER
