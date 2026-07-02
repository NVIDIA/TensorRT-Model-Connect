# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3 MoE family plugin — verifies weight key mapping and transforms.

Creates synthetic model directories with config.json and mock safetensors,
calls plugin.load_weights(), and verifies the returned WeightDict has
expected keys, shapes, and transforms. Also tests matches() and metadata.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-QWEN-MOE
Intent: Validate Qwen3 MoE family plugin expert weight mapping, router weights, and shared expert handling
Preconditions: Synthetic safetensors with MoE expert weight naming and config with expert count are available
Postconditions: Plugin produces correct per-expert weight keys, router weights, and shared expert projections
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="Qwen-MoE builder tests require TensorRT")

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


class TestQwen3MoePlugin:
    """Tests for qwen_moe family plugin."""

    VOCAB = 32
    HIDDEN = 16
    LAYERS = 2
    HEADS = 4
    KV_HEADS = 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    KV_DIM = KV_HEADS * HEAD_DIM  # 8
    NUM_EXPERTS = 4
    NUM_EXPERTS_PER_TOK = 2
    MOE_INTER = 24
    SHARED_INTER = 16
    DENSE_INTER = 32
    MLP_ONLY_LAYERS = [0]  # layer 0 is dense, layer 1 is MoE

    @classmethod
    def _make_config(cls) -> dict:
        return {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "vocab_size": cls.VOCAB,
            "hidden_size": cls.HIDDEN,
            "num_hidden_layers": cls.LAYERS,
            "num_attention_heads": cls.HEADS,
            "num_key_value_heads": cls.KV_HEADS,
            "intermediate_size": cls.DENSE_INTER,
            "num_experts": cls.NUM_EXPERTS,
            "num_experts_per_tok": cls.NUM_EXPERTS_PER_TOK,
            "moe_intermediate_size": cls.MOE_INTER,
            "shared_expert_intermediate_size": cls.SHARED_INTER,
            "mlp_only_layers": cls.MLP_ONLY_LAYERS,
            "decoder_sparse_step": 1,
        }

    @classmethod
    def _make_tensors(cls) -> dict[str, np.ndarray]:
        t = {}
        t["model.embed_tokens.weight"] = _rand(cls.VOCAB, cls.HIDDEN)

        for i in range(cls.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(cls.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(cls.HIDDEN)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(cls.HIDDEN, cls.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(
                cls.KV_DIM, cls.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(
                cls.KV_DIM, cls.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(
                cls.HIDDEN, cls.HIDDEN)

            is_dense = i in cls.MLP_ONLY_LAYERS

            if is_dense:
                # Dense MLP
                t[f"{p}.mlp.gate_proj.weight"] = _rand(
                    cls.DENSE_INTER, cls.HIDDEN)
                t[f"{p}.mlp.up_proj.weight"] = _rand(
                    cls.DENSE_INTER, cls.HIDDEN)
                t[f"{p}.mlp.down_proj.weight"] = _rand(
                    cls.HIDDEN, cls.DENSE_INTER)
            else:
                # MoE layer (Qwen3-MoE: no shared experts)
                t[f"{p}.mlp.gate.weight"] = _rand(
                    cls.NUM_EXPERTS, cls.HIDDEN)
                for e in range(cls.NUM_EXPERTS):
                    ep = f"{p}.mlp.experts.{e}"
                    t[f"{ep}.gate_proj.weight"] = _rand(
                        cls.MOE_INTER, cls.HIDDEN)
                    t[f"{ep}.up_proj.weight"] = _rand(
                        cls.MOE_INTER, cls.HIDDEN)
                    t[f"{ep}.down_proj.weight"] = _rand(
                        cls.HIDDEN, cls.MOE_INTER)

        t["model.norm.weight"] = _rand(cls.HIDDEN)
        t["lm_head.weight"] = _rand(cls.VOCAB, cls.HIDDEN)
        return t

    def test_matches(self):
        from tensorrt_model_connect.families.qwen_moe import plugin
        assert plugin.matches("qwen3_moe")
        assert plugin.matches("Qwen3_moe")
        assert not plugin.matches("qwen3")
        assert not plugin.matches("qwen2")
        assert not plugin.matches("mixtral")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.qwen_moe import plugin
        assert plugin.runtime_strategy == "qwen_moe_decoder_moe"

    def test_qwen_plugin_excludes_moe(self):
        """The standard Qwen plugin should NOT match qwen3_moe."""
        from tensorrt_model_connect.families.qwen import plugin
        assert not plugin.matches("qwen3_moe")

    def test_load_weights_dense_layer_keys(self, tmp_path):
        """Dense MLP layer (layer 0) should have w_gate/w_up/w_down."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, self._make_tensors())

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Layer 0 is dense
        assert "layer.0.w_gate" in weights
        assert "layer.0.w_up" in weights
        assert "layer.0.w_down" in weights
        # Should NOT have router or expert keys
        assert "layer.0.router" not in weights
        assert "layer.0.experts.w_gate" not in weights

    def test_load_weights_moe_layer_keys(self, tmp_path):
        """MoE layer (layer 1) should have router and expert keys but no shared expert (Qwen3-MoE)."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, self._make_tensors())

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Layer 1 is MoE
        assert "layer.1.router" in weights
        assert "layer.1.experts.w_gate" in weights
        assert "layer.1.experts.w_up" in weights
        assert "layer.1.experts.w_down" in weights
        assert weights["layer.1.experts.w_gate"].shape == (
            self.NUM_EXPERTS, self.HIDDEN, self.MOE_INTER)
        assert weights["layer.1.experts.w_up"].shape == (
            self.NUM_EXPERTS, self.HIDDEN, self.MOE_INTER)
        assert weights["layer.1.experts.w_down"].shape == (
            self.NUM_EXPERTS, self.MOE_INTER, self.HIDDEN)
        # Qwen3-MoE: no shared experts
        assert "layer.1.shared_expert.w_gate" not in weights
        assert "layer.1.shared_expert.w_up" not in weights
        assert "layer.1.shared_expert.w_down" not in weights
        assert "layer.1.shared_expert_gate" not in weights
        # Should NOT have dense MLP keys
        assert "layer.1.w_gate" not in weights

    def test_load_weights_attention_keys(self, tmp_path):
        """Attention keys should be present for all layers."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, self._make_tensors())

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)

        for i in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm",
                        "w_q", "w_k", "w_v", "w_o"):
                assert f"layer.{i}.{key}" in weights, (
                    f"Missing layer.{i}.{key}")

        assert "final_norm" in weights
        assert "w_out" in weights

    def test_transpose_applied(self, tmp_path):
        """Weight projections should be transposed from [out, in] to [in, out]."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        tensors = self._make_tensors()
        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Q projection: [hidden, hidden] -> [hidden, hidden]
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        q_raw = tensors["model.layers.0.self_attn.q_proj.weight"]
        np.testing.assert_allclose(
            weights["layer.0.w_q"], q_raw.T.astype(np.float32), atol=1e-6)

        # Router: [num_experts, hidden] -> [hidden, num_experts]
        assert weights["layer.1.router"].shape == (
            self.HIDDEN, self.NUM_EXPERTS)

        # Packed expert tensors preserve the same per-expert transpose.
        assert weights["layer.1.experts.w_gate"].shape == (
            self.NUM_EXPERTS, self.HIDDEN, self.MOE_INTER)
        assert weights["layer.1.experts.w_down"].shape == (
            self.NUM_EXPERTS, self.MOE_INTER, self.HIDDEN)

        # Dense layer gate: [dense_inter, hidden] -> [hidden, dense_inter]
        assert weights["layer.0.w_gate"].shape == (
            self.HIDDEN, self.DENSE_INTER)

        # LM head: [vocab, hidden] -> [hidden, vocab]
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_gqa_kv_stays_compact(self, tmp_path):
        """K/V should stay compact at kv_dim."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, self._make_tensors())

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        kv_dim = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        for i in range(self.LAYERS):
            assert weights[f"layer.{i}.w_k"].shape == (
                self.HIDDEN, kv_dim)
            assert weights[f"layer.{i}.w_v"].shape == (
                self.HIDDEN, kv_dim)

    def test_metadata_keys(self, tmp_path):
        """Metadata should be stored correctly."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, self._make_tensors())

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_num_experts"] == self.NUM_EXPERTS
        assert weights["_num_experts_per_tok"] == self.NUM_EXPERTS_PER_TOK
        assert weights["_moe_intermediate_size"] == self.MOE_INTER
        assert weights["_shared_expert_intermediate_size"] == self.SHARED_INTER
        assert weights["_dense_intermediate_size"] == self.DENSE_INTER
        assert weights["_mlp_only_layers"] == sorted(self.MLP_ONLY_LAYERS)

    def test_expert_transpose_values(self, tmp_path):
        """Verify expert weight values match transposed HF originals."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        tensors = self._make_tensors()
        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Check expert 0 gate weight (layer 1)
        gate_raw = tensors[
            "model.layers.1.mlp.experts.0.gate_proj.weight"]
        np.testing.assert_allclose(
            weights["layer.1.experts.w_gate"][0],
            gate_raw.T.astype(np.float32), atol=1e-6)

        # Check expert 2 down weight (layer 1)
        down_raw = tensors[
            "model.layers.1.mlp.experts.2.down_proj.weight"]
        np.testing.assert_allclose(
            weights["layer.1.experts.w_down"][2],
            down_raw.T.astype(np.float32), atol=1e-6)

    def test_no_shared_expert_keys_for_qwen3_moe(self, tmp_path):
        """Qwen3-MoE should NOT have shared expert keys in the weight dict."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        tensors = self._make_tensors()
        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "layer.1.shared_expert_gate" not in weights
        assert "layer.1.shared_expert.w_gate" not in weights
        assert "layer.1.shared_expert.w_up" not in weights
        assert "layer.1.shared_expert.w_down" not in weights
        assert weights["_has_shared_expert"] is False

    def test_tied_embeddings(self, tmp_path):
        """When lm_head.weight is missing, w_out = transposed embedding."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        config = self._make_config()
        config["tie_word_embeddings"] = True
        tensors = self._make_tensors()
        del tensors["lm_head.weight"]

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)
        embedding = tensors["model.embed_tokens.weight"]
        np.testing.assert_allclose(
            weights["w_out"], embedding.T, atol=1e-6)

    def test_all_moe_no_dense(self, tmp_path):
        """With mlp_only_layers=[], all layers should be MoE (Qwen3-MoE, no shared experts)."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        config = self._make_config()
        config["mlp_only_layers"] = []
        # Rebuild tensors with all MoE layers (no shared experts)
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(
                self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(
                self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            # MoE for all layers (no shared experts)
            t[f"{p}.mlp.gate.weight"] = _rand(
                self.NUM_EXPERTS, self.HIDDEN)
            for e in range(self.NUM_EXPERTS):
                ep = f"{p}.mlp.experts.{e}"
                t[f"{ep}.gate_proj.weight"] = _rand(
                    self.MOE_INTER, self.HIDDEN)
                t[f"{ep}.up_proj.weight"] = _rand(
                    self.MOE_INTER, self.HIDDEN)
                t[f"{ep}.down_proj.weight"] = _rand(
                    self.HIDDEN, self.MOE_INTER)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, t)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.router" in weights
            assert f"layer.{i}.shared_expert.w_gate" not in weights
            assert f"layer.{i}.w_gate" not in weights
            assert f"layer.{i}.experts.w_gate" in weights

    def test_fp16_load_uses_packed_fp16_expert_weights(self, tmp_path):
        """Large MoE matrices should honor fp16 load precision."""
        from tensorrt_model_connect.families.qwen_moe import plugin

        tensors = self._make_tensors()
        _write_config(tmp_path, self._make_config())
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg, precision="fp16")

        assert weights["embedding"].dtype == np.float16
        assert weights["layer.0.w_gate"].dtype == np.float16
        assert weights["layer.1.router"].dtype == np.float16
        assert weights["layer.1.experts.w_gate"].dtype == np.float16
        assert weights["layer.1.experts.w_up"].dtype == np.float16
        assert weights["layer.1.experts.w_down"].dtype == np.float16
        assert weights["final_norm"].dtype == np.float32

    def test_plugin_discovery(self):
        """Plugin should be discoverable via find_plugin."""
        from tensorrt_model_connect.families import find_plugin
        p = find_plugin("qwen3_moe")
        assert p is not None
        assert p.name == "qwen_moe"
