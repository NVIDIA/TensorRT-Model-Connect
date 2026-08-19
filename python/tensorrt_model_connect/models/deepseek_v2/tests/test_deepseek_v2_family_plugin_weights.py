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


class TestDeepSeekV2Plugin:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 32, 2, 4
    QK_NOPE_HEAD_DIM = 4
    QK_ROPE_HEAD_DIM = 2
    V_HEAD_DIM = 4
    KV_LORA_RANK = 8
    N_ROUTED_EXPERTS = 4
    N_SHARED_EXPERTS = 1
    MOE_INTER = 16
    DENSE_INTER = 24
    FIRST_K_DENSE = 1  # layer 0 is dense, layer 1 is MoE

    @property
    def k_head_dim(self):
        return self.QK_NOPE_HEAD_DIM + self.QK_ROPE_HEAD_DIM  # 6

    @property
    def q_total(self):
        return self.HEADS * self.k_head_dim  # 24

    @property
    def kv_a_dim(self):
        return self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM  # 10

    @property
    def kv_b_out_dim(self):
        return self.HEADS * (self.QK_NOPE_HEAD_DIM + self.V_HEAD_DIM)  # 32

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)

        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)

            # MLA attention: direct Q (V2-Lite, q_lora_rank=null)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(
                self.q_total, self.HIDDEN)
            # KV-A with MQA: [kv_lora_rank + rope_dim, hidden]
            t[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = _rand(
                self.kv_a_dim, self.HIDDEN)
            # KV-A LayerNorm
            t[f"{p}.self_attn.kv_a_layernorm.weight"] = _rand(
                self.KV_LORA_RANK)
            # KV-B: [num_heads * (nope + v_head), kv_lora_rank]
            t[f"{p}.self_attn.kv_b_proj.weight"] = _rand(
                self.kv_b_out_dim, self.KV_LORA_RANK)
            # O projection: [hidden, num_heads * v_head_dim]
            t[f"{p}.self_attn.o_proj.weight"] = _rand(
                self.HIDDEN, self.HEADS * self.V_HEAD_DIM)

            if i < self.FIRST_K_DENSE:
                # Dense MLP for layer 0
                t[f"{p}.mlp.gate_proj.weight"] = _rand(
                    self.DENSE_INTER, self.HIDDEN)
                t[f"{p}.mlp.up_proj.weight"] = _rand(
                    self.DENSE_INTER, self.HIDDEN)
                t[f"{p}.mlp.down_proj.weight"] = _rand(
                    self.HIDDEN, self.DENSE_INTER)
            else:
                # MoE for layer 1+
                t[f"{p}.mlp.gate.weight"] = _rand(
                    self.N_ROUTED_EXPERTS, self.HIDDEN)
                for e in range(self.N_ROUTED_EXPERTS):
                    ep = f"{p}.mlp.experts.{e}"
                    t[f"{ep}.gate_proj.weight"] = _rand(
                        self.MOE_INTER, self.HIDDEN)
                    t[f"{ep}.up_proj.weight"] = _rand(
                        self.MOE_INTER, self.HIDDEN)
                    t[f"{ep}.down_proj.weight"] = _rand(
                        self.HIDDEN, self.MOE_INTER)
                # Shared experts
                sp = f"{p}.mlp.shared_experts"
                shared_inter = self.MOE_INTER * self.N_SHARED_EXPERTS
                t[f"{sp}.gate_proj.weight"] = _rand(
                    shared_inter, self.HIDDEN)
                t[f"{sp}.up_proj.weight"] = _rand(
                    shared_inter, self.HIDDEN)
                t[f"{sp}.down_proj.weight"] = _rand(
                    self.HIDDEN, shared_inter)

        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_mla_keys_present(self, tmp_path):
        """MLA-specific weight keys should be present."""
        from tensorrt_model_connect.models.deepseek_v2 import model as plugin

        config = {
            "model_type": "deepseek_v2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.HEADS,
            "intermediate_size": self.DENSE_INTER,
            "qk_nope_head_dim": self.QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": self.QK_ROPE_HEAD_DIM,
            "v_head_dim": self.V_HEAD_DIM,
            "kv_lora_rank": self.KV_LORA_RANK,
            "q_lora_rank": None,
            "n_routed_experts": self.N_ROUTED_EXPERTS,
            "n_shared_experts": self.N_SHARED_EXPERTS,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": self.FIRST_K_DENSE,
            "moe_layer_freq": 1,
            "moe_intermediate_size": self.MOE_INTER,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            # MLA weights
            assert f"layer.{i}.w_q" in weights
            assert f"layer.{i}.w_kv_a" in weights
            assert f"layer.{i}.kv_a_norm" in weights
            assert f"layer.{i}.w_kv_b" in weights
            assert f"layer.{i}.w_o" in weights

        # Dense MLP in layer 0
        assert "layer.0.w_gate" in weights
        assert "layer.0.w_up" in weights
        assert "layer.0.w_down" in weights

        # MoE in layer 1
        assert "layer.1.router" in weights
        for e in range(self.N_ROUTED_EXPERTS):
            assert f"layer.1.expert.{e}.w_gate" in weights
            assert f"layer.1.expert.{e}.w_up" in weights
            assert f"layer.1.expert.{e}.w_down" in weights
        assert "layer.1.shared.w_gate" in weights
        assert "layer.1.shared.w_up" in weights
        assert "layer.1.shared.w_down" in weights

    def test_mla_metadata(self, tmp_path):
        """DeepSeek-V2 metadata keys should be stored."""
        from tensorrt_model_connect.models.deepseek_v2 import model as plugin

        config = {
            "model_type": "deepseek_v2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.HEADS,
            "intermediate_size": self.DENSE_INTER,
            "qk_nope_head_dim": self.QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": self.QK_ROPE_HEAD_DIM,
            "v_head_dim": self.V_HEAD_DIM,
            "kv_lora_rank": self.KV_LORA_RANK,
            "q_lora_rank": None,
            "n_routed_experts": self.N_ROUTED_EXPERTS,
            "n_shared_experts": self.N_SHARED_EXPERTS,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": self.FIRST_K_DENSE,
            "moe_layer_freq": 1,
            "moe_intermediate_size": self.MOE_INTER,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        attention_size = self.HEADS * self.k_head_dim
        assert weights["_attention_size"] == attention_size
        assert weights["_qk_nope_head_dim"] == self.QK_NOPE_HEAD_DIM
        assert weights["_qk_rope_head_dim"] == self.QK_ROPE_HEAD_DIM
        assert weights["_v_head_dim"] == self.V_HEAD_DIM
        assert weights["_kv_lora_rank"] == self.KV_LORA_RANK
        assert weights["_q_lora_rank"] is None
        assert weights["_n_routed_experts"] == self.N_ROUTED_EXPERTS
        assert weights["_n_shared_experts"] == self.N_SHARED_EXPERTS

    def test_kv_a_transpose(self, tmp_path):
        """KV-A projection should be transposed."""
        from tensorrt_model_connect.models.deepseek_v2 import model as plugin

        config = {
            "model_type": "deepseek_v2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.HEADS,
            "intermediate_size": self.DENSE_INTER,
            "qk_nope_head_dim": self.QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": self.QK_ROPE_HEAD_DIM,
            "v_head_dim": self.V_HEAD_DIM,
            "kv_lora_rank": self.KV_LORA_RANK,
            "q_lora_rank": None,
            "n_routed_experts": self.N_ROUTED_EXPERTS,
            "n_shared_experts": self.N_SHARED_EXPERTS,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
            "moe_intermediate_size": self.MOE_INTER,
        }
        # Need to create tensors for a single MoE layer (first_k_dense_replace=0)
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        p = "model.layers.0"
        t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
        t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
        t[f"{p}.self_attn.q_proj.weight"] = _rand(
            self.q_total, self.HIDDEN)
        kv_a_raw = _rand(self.kv_a_dim, self.HIDDEN)
        t[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = kv_a_raw
        t[f"{p}.self_attn.kv_a_layernorm.weight"] = _rand(self.KV_LORA_RANK)
        t[f"{p}.self_attn.kv_b_proj.weight"] = _rand(
            self.kv_b_out_dim, self.KV_LORA_RANK)
        t[f"{p}.self_attn.o_proj.weight"] = _rand(
            self.HIDDEN, self.HEADS * self.V_HEAD_DIM)
        t[f"{p}.mlp.gate.weight"] = _rand(
            self.N_ROUTED_EXPERTS, self.HIDDEN)
        for e in range(self.N_ROUTED_EXPERTS):
            ep = f"{p}.mlp.experts.{e}"
            t[f"{ep}.gate_proj.weight"] = _rand(self.MOE_INTER, self.HIDDEN)
            t[f"{ep}.up_proj.weight"] = _rand(self.MOE_INTER, self.HIDDEN)
            t[f"{ep}.down_proj.weight"] = _rand(self.HIDDEN, self.MOE_INTER)
        sp = f"{p}.mlp.shared_experts"
        shared_inter = self.MOE_INTER * self.N_SHARED_EXPERTS
        t[f"{sp}.gate_proj.weight"] = _rand(shared_inter, self.HIDDEN)
        t[f"{sp}.up_proj.weight"] = _rand(shared_inter, self.HIDDEN)
        t[f"{sp}.down_proj.weight"] = _rand(self.HIDDEN, shared_inter)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, t)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # w_kv_a: original [kv_a_dim, hidden] transposed to [hidden, kv_a_dim]
        assert weights["layer.0.w_kv_a"].shape == (
            self.HIDDEN, self.kv_a_dim)
        np.testing.assert_allclose(
            weights["layer.0.w_kv_a"],
            kv_a_raw.T.astype(np.float32), atol=1e-6)
