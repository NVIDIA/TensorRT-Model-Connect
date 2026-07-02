# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


class TestQwenPlugin:
    """Qwen plugin delegates to load_standard_weights; verify standard mapping."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
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
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.qwen import plugin

        config = {
            "model_type": "qwen3",
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
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm", "w_q", "w_k", "w_v",
                        "w_o", "w_gate", "w_up", "w_down"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
        assert "final_norm" in weights
        assert "w_out" in weights
        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_kv_attention_size"] == self.KV_HEADS * (
            self.HIDDEN // self.HEADS)
        assert weights["_mlp_size"] == self.MLP

    def test_transpose_applied(self, tmp_path):
        """Projections are transposed from [out, in] to [in, out]."""
        from tensorrt_model_connect.families.qwen import plugin

        config = {
            "model_type": "qwen3",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, 1, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # w_q original [hidden, hidden] transposed to [hidden, hidden]
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        # w_gate original [mlp, hidden] transposed to [hidden, mlp]
        assert weights["layer.0.w_gate"].shape == (self.HIDDEN, self.MLP)
        # w_out original [vocab, hidden] transposed to [hidden, vocab]
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_tensor_parallel_shards_qwen_projection_weights(self, tmp_path):
        """TP shards attention/MLP inner dims and leaves replicated weights intact."""
        from tensorrt_model_connect.families.qwen import plugin
        from tensorrt_model_connect.parallel_config import (
            ParallelConfig,
            shard_standard_decoder_weights,
        )

        config = {
            "model_type": "qwen3",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, 1, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        tp_size = 4
        rank = 3
        hidden_start = rank * self.HIDDEN // tp_size
        hidden_end = (rank + 1) * self.HIDDEN // tp_size
        mlp_start = rank * self.MLP // tp_size
        mlp_end = (rank + 1) * self.MLP // tp_size
        shard = shard_standard_decoder_weights(
            cfg, weights,
            ParallelConfig(mode="tensor_parallel", tp_size=tp_size, rank=rank))

        np.testing.assert_allclose(
            shard["layer.0.w_q"], weights["layer.0.w_q"][:, hidden_start:hidden_end])
        np.testing.assert_allclose(
            shard["layer.0.w_k"], weights["layer.0.w_k"][:, hidden_start:hidden_end])
        np.testing.assert_allclose(
            shard["layer.0.w_o"], weights["layer.0.w_o"][hidden_start:hidden_end, :])
        np.testing.assert_allclose(
            shard["layer.0.w_gate"], weights["layer.0.w_gate"][:, mlp_start:mlp_end])
        np.testing.assert_allclose(
            shard["layer.0.w_down"], weights["layer.0.w_down"][mlp_start:mlp_end, :])
        np.testing.assert_allclose(shard["w_out"], weights["w_out"])
        assert shard["_attention_size"] == self.HIDDEN // tp_size
        assert shard["_kv_attention_size"] == self.HIDDEN // tp_size
        assert shard["_mlp_size"] == self.MLP // tp_size
