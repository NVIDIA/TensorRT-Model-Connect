# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE checkpoint mapping contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Phi-MoE builder tests")
safetensors_numpy = pytest.importorskip(
    "safetensors.numpy",
    reason="safetensors is required for Phi-MoE checkpoint tests",
)

from families.phi_moe import model  # noqa: E402
from families.phi_moe.config import ModelConfig  # noqa: E402


class TestPhiMoEWeightMapping:
    VOCAB = 32
    HIDDEN = 16
    LAYERS = 2
    HEADS = 4
    KV_HEADS = 2
    INTERMEDIATE = 32
    EXPERTS = 2

    @classmethod
    def _config(cls) -> dict:
        return {
            "model_type": "phimoe",
            "vocab_size": cls.VOCAB,
            "hidden_size": cls.HIDDEN,
            "intermediate_size": cls.INTERMEDIATE,
            "num_hidden_layers": cls.LAYERS,
            "num_attention_heads": cls.HEADS,
            "num_key_value_heads": cls.KV_HEADS,
            "num_local_experts": cls.EXPERTS,
            "num_experts_per_tok": 2,
            "router_jitter_noise": 0.01,
        }

    @classmethod
    def _tensors(cls) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        kv_hidden = cls.KV_HEADS * (cls.HIDDEN // cls.HEADS)
        tensors = {"model.embed_tokens.weight": rand(cls.VOCAB, cls.HIDDEN)}
        for index in range(cls.LAYERS):
            prefix = f"model.layers.{index}"
            tensors[f"{prefix}.input_layernorm.weight"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.input_layernorm.bias"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.post_attention_layernorm.bias"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = rand(cls.HIDDEN, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.q_proj.bias"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = rand(kv_hidden, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.k_proj.bias"] = rand(kv_hidden)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = rand(kv_hidden, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.v_proj.bias"] = rand(kv_hidden)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = rand(cls.HIDDEN, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.o_proj.bias"] = rand(cls.HIDDEN)
            tensors[f"{prefix}.block_sparse_moe.gate.weight"] = rand(
                cls.EXPERTS,
                cls.HIDDEN,
            )
            for expert in range(cls.EXPERTS):
                expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert}"
                tensors[f"{expert_prefix}.w1.weight"] = rand(
                    cls.INTERMEDIATE,
                    cls.HIDDEN,
                )
                tensors[f"{expert_prefix}.w3.weight"] = rand(
                    cls.INTERMEDIATE,
                    cls.HIDDEN,
                )
                tensors[f"{expert_prefix}.w2.weight"] = rand(
                    cls.HIDDEN,
                    cls.INTERMEDIATE,
                )
        tensors["model.norm.weight"] = rand(cls.HIDDEN)
        tensors["model.norm.bias"] = rand(cls.HIDDEN)
        tensors["lm_head.weight"] = rand(cls.VOCAB, cls.HIDDEN)
        tensors["lm_head.bias"] = rand(cls.VOCAB)
        return tensors

    @classmethod
    def _load(cls, model_dir: Path):
        (model_dir / "config.json").write_text(json.dumps(cls._config()), encoding="utf-8")
        safetensors_numpy.save_file(cls._tensors(), str(model_dir / "model.safetensors"))
        config = ModelConfig.from_dir(model_dir)
        weights = model._PhiMoEModel().load_weights(str(model_dir), config)
        return config, weights

    def test_layernorm_biases_present(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        for index in range(self.LAYERS):
            prefix = f"layer.{index}"
            assert f"{prefix}.input_norm_beta" in weights
            assert f"{prefix}.post_attn_norm_beta" in weights
        assert "final_norm_beta" in weights

    def test_attention_biases_present(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        for index in range(self.LAYERS):
            prefix = f"layer.{index}"
            for projection in ("q_bias", "k_bias", "v_bias", "o_bias"):
                assert f"{prefix}.{projection}" in weights

    def test_router_shape_transposed(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        assert weights["layer.0.router"].shape == (self.HIDDEN, self.EXPERTS)

    def test_per_expert_weights_shapes(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        for expert in range(self.EXPERTS):
            prefix = f"layer.0.expert.{expert}"
            assert weights[f"{prefix}.w_gate"].shape == (self.HIDDEN, self.INTERMEDIATE)
            assert weights[f"{prefix}.w_up"].shape == (self.HIDDEN, self.INTERMEDIATE)
            assert weights[f"{prefix}.w_down"].shape == (self.INTERMEDIATE, self.HIDDEN)

    def test_lm_head_bias_present(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        assert weights["lm_head_bias"].shape == (self.VOCAB,)

    def test_gqa_kv_stays_compact(self, tmp_path: Path) -> None:
        _, weights = self._load(tmp_path)
        kv_dim = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_dim)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_dim)
