# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tensorrt_model_connect.models.mixtral.tests._family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestMixtralPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    KV_DIM = KV_HEADS * HEAD_DIM  # 8
    NUM_EXPERTS = 4
    MOE_INTER = 24

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(
                self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(
                self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            # Router
            t[f"{p}.block_sparse_moe.gate.weight"] = _rand(
                self.NUM_EXPERTS, self.HIDDEN)
            # Per-expert weights
            for e in range(self.NUM_EXPERTS):
                ep = f"{p}.block_sparse_moe.experts.{e}"
                t[f"{ep}.w1.weight"] = _rand(self.MOE_INTER, self.HIDDEN)
                t[f"{ep}.w3.weight"] = _rand(self.MOE_INTER, self.HIDDEN)
                t[f"{ep}.w2.weight"] = _rand(self.HIDDEN, self.MOE_INTER)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_expert_weights_mapped(self, tmp_path):
        """Expert gate/up/down weights should be correctly mapped and transposed."""
        from tensorrt_model_connect.models.mixtral import model as plugin

        config = {
            "model_type": "mixtral",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MOE_INTER,
            "num_local_experts": self.NUM_EXPERTS,
            "num_experts_per_tok": 2,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.router" in weights
            # Router transposed: [hidden, num_experts]
            assert weights[f"layer.{i}.router"].shape == (
                self.HIDDEN, self.NUM_EXPERTS)
            for e in range(self.NUM_EXPERTS):
                assert f"layer.{i}.expert.{e}.w_gate" in weights
                assert f"layer.{i}.expert.{e}.w_up" in weights
                assert f"layer.{i}.expert.{e}.w_down" in weights
                # gate transposed: [hidden, moe_inter]
                assert weights[f"layer.{i}.expert.{e}.w_gate"].shape == (
                    self.HIDDEN, self.MOE_INTER)
                # down transposed: [moe_inter, hidden]
                assert weights[f"layer.{i}.expert.{e}.w_down"].shape == (
                    self.MOE_INTER, self.HIDDEN)

    def test_moe_metadata(self, tmp_path):
        from tensorrt_model_connect.models.mixtral import model as plugin

        config = {
            "model_type": "mixtral",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MOE_INTER,
            "num_local_experts": self.NUM_EXPERTS,
            "num_experts_per_tok": 2,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_num_experts"] == self.NUM_EXPERTS
        assert weights["_moe_intermediate_size"] == self.MOE_INTER
        assert weights["_num_experts_per_tok"] == 2

    def test_expert_transpose_values(self, tmp_path):
        """Verify expert weight values are transposed correctly."""
        from tensorrt_model_connect.models.mixtral import model as plugin

        config = {
            "model_type": "mixtral",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MOE_INTER,
            "num_local_experts": self.NUM_EXPERTS,
            "num_experts_per_tok": 2,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        w1_raw = tensors[
            "model.layers.0.block_sparse_moe.experts.0.w1.weight"]
        np.testing.assert_allclose(
            weights["layer.0.expert.0.w_gate"],
            w1_raw.T.astype(np.float32), atol=1e-6)
