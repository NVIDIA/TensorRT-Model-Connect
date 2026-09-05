# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-MoE checkpoint mapping contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Qwen-MoE builder tests")
safetensors_numpy = pytest.importorskip(
    "safetensors.numpy",
    reason="safetensors is required for Qwen-MoE checkpoint tests",
)

from families.qwen_moe import model  # noqa: E402
from families.qwen_moe.config import ModelConfig  # noqa: E402


RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


class TestQwen3MoeWeightMapping:
    VOCAB = 32
    HIDDEN = 16
    LAYERS = 2
    HEADS = 4
    KV_HEADS = 2
    HEAD_DIM = HIDDEN // HEADS
    KV_DIM = KV_HEADS * HEAD_DIM
    NUM_EXPERTS = 4
    NUM_EXPERTS_PER_TOK = 2
    MOE_INTER = 24
    SHARED_INTER = 16
    DENSE_INTER = 32
    MLP_ONLY_LAYERS = [0]

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
        tensors = {"model.embed_tokens.weight": _rand(cls.VOCAB, cls.HIDDEN)}
        for index in range(cls.LAYERS):
            prefix = f"model.layers.{index}"
            tensors[f"{prefix}.input_layernorm.weight"] = _rand(cls.HIDDEN)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(cls.HIDDEN)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand(cls.HIDDEN, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand(cls.KV_DIM, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand(cls.KV_DIM, cls.HIDDEN)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand(cls.HIDDEN, cls.HIDDEN)

            if index in cls.MLP_ONLY_LAYERS:
                tensors[f"{prefix}.mlp.gate_proj.weight"] = _rand(
                    cls.DENSE_INTER,
                    cls.HIDDEN,
                )
                tensors[f"{prefix}.mlp.up_proj.weight"] = _rand(
                    cls.DENSE_INTER,
                    cls.HIDDEN,
                )
                tensors[f"{prefix}.mlp.down_proj.weight"] = _rand(
                    cls.HIDDEN,
                    cls.DENSE_INTER,
                )
            else:
                tensors[f"{prefix}.mlp.gate.weight"] = _rand(cls.NUM_EXPERTS, cls.HIDDEN)
                for expert in range(cls.NUM_EXPERTS):
                    expert_prefix = f"{prefix}.mlp.experts.{expert}"
                    tensors[f"{expert_prefix}.gate_proj.weight"] = _rand(
                        cls.MOE_INTER,
                        cls.HIDDEN,
                    )
                    tensors[f"{expert_prefix}.up_proj.weight"] = _rand(
                        cls.MOE_INTER,
                        cls.HIDDEN,
                    )
                    tensors[f"{expert_prefix}.down_proj.weight"] = _rand(
                        cls.HIDDEN,
                        cls.MOE_INTER,
                    )

        tensors["model.norm.weight"] = _rand(cls.HIDDEN)
        tensors["lm_head.weight"] = _rand(cls.VOCAB, cls.HIDDEN)
        return tensors

    @classmethod
    def _load(
        cls,
        model_dir: Path,
        *,
        config: dict | None = None,
        tensors: dict[str, np.ndarray] | None = None,
        precision: str = "fp32",
    ):
        selected_config = cls._make_config() if config is None else config
        selected_tensors = cls._make_tensors() if tensors is None else tensors
        (model_dir / "config.json").write_text(json.dumps(selected_config), encoding="utf-8")
        safetensors_numpy.save_file(selected_tensors, str(model_dir / "model.safetensors"))
        parsed = ModelConfig.from_dir(model_dir)
        weights = model._QwenMoeModel().load_weights(
            str(model_dir),
            parsed,
            precision=precision,
        )
        return weights

    def test_load_weights_dense_layer_keys(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        assert "layer.0.w_gate" in weights
        assert "layer.0.w_up" in weights
        assert "layer.0.w_down" in weights
        assert "layer.0.router" not in weights
        assert "layer.0.experts.w_gate" not in weights

    def test_load_weights_moe_layer_keys(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        assert "layer.1.router" in weights
        assert weights["layer.1.experts.w_gate"].shape == (
            self.NUM_EXPERTS,
            self.HIDDEN,
            self.MOE_INTER,
        )
        assert weights["layer.1.experts.w_up"].shape == (
            self.NUM_EXPERTS,
            self.HIDDEN,
            self.MOE_INTER,
        )
        assert weights["layer.1.experts.w_down"].shape == (
            self.NUM_EXPERTS,
            self.MOE_INTER,
            self.HIDDEN,
        )
        assert "layer.1.shared_expert.w_gate" not in weights
        assert "layer.1.shared_expert.w_up" not in weights
        assert "layer.1.shared_expert.w_down" not in weights
        assert "layer.1.shared_expert_gate" not in weights
        assert "layer.1.w_gate" not in weights

    def test_load_weights_attention_keys(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        for index in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm", "w_q", "w_k", "w_v", "w_o"):
                assert f"layer.{index}.{key}" in weights
        assert "final_norm" in weights
        assert "w_out" in weights

    def test_transpose_applied(self, tmp_path: Path) -> None:
        tensors = self._make_tensors()
        weights = self._load(tmp_path, tensors=tensors)

        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        np.testing.assert_allclose(
            weights["layer.0.w_q"],
            tensors["model.layers.0.self_attn.q_proj.weight"].T.astype(np.float32),
            atol=1e-6,
        )
        assert weights["layer.1.router"].shape == (self.HIDDEN, self.NUM_EXPERTS)
        assert weights["layer.1.experts.w_gate"].shape == (
            self.NUM_EXPERTS,
            self.HIDDEN,
            self.MOE_INTER,
        )
        assert weights["layer.1.experts.w_down"].shape == (
            self.NUM_EXPERTS,
            self.MOE_INTER,
            self.HIDDEN,
        )
        assert weights["layer.0.w_gate"].shape == (self.HIDDEN, self.DENSE_INTER)
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_gqa_kv_stays_compact(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        kv_dim = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        for index in range(self.LAYERS):
            assert weights[f"layer.{index}.w_k"].shape == (self.HIDDEN, kv_dim)
            assert weights[f"layer.{index}.w_v"].shape == (self.HIDDEN, kv_dim)

    def test_metadata_keys(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_num_experts"] == self.NUM_EXPERTS
        assert weights["_num_experts_per_tok"] == self.NUM_EXPERTS_PER_TOK
        assert weights["_moe_intermediate_size"] == self.MOE_INTER
        assert weights["_shared_expert_intermediate_size"] == self.SHARED_INTER
        assert weights["_dense_intermediate_size"] == self.DENSE_INTER
        assert weights["_mlp_only_layers"] == sorted(self.MLP_ONLY_LAYERS)

    def test_expert_transpose_values(self, tmp_path: Path) -> None:
        tensors = self._make_tensors()
        weights = self._load(tmp_path, tensors=tensors)

        gate_raw = tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]
        np.testing.assert_allclose(
            weights["layer.1.experts.w_gate"][0],
            gate_raw.T.astype(np.float32),
            atol=1e-6,
        )
        down_raw = tensors["model.layers.1.mlp.experts.2.down_proj.weight"]
        np.testing.assert_allclose(
            weights["layer.1.experts.w_down"][2],
            down_raw.T.astype(np.float32),
            atol=1e-6,
        )

    def test_no_shared_expert_keys_for_qwen3_moe(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path)

        assert "layer.1.shared_expert_gate" not in weights
        assert "layer.1.shared_expert.w_gate" not in weights
        assert "layer.1.shared_expert.w_up" not in weights
        assert "layer.1.shared_expert.w_down" not in weights
        assert weights["_has_shared_expert"] is False

    def test_tied_embeddings(self, tmp_path: Path) -> None:
        config = self._make_config()
        config["tie_word_embeddings"] = True
        tensors = self._make_tensors()
        del tensors["lm_head.weight"]

        weights = self._load(tmp_path, config=config, tensors=tensors)

        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)
        np.testing.assert_allclose(
            weights["w_out"],
            tensors["model.embed_tokens.weight"].T,
            atol=1e-6,
        )

    def test_all_moe_no_dense(self, tmp_path: Path) -> None:
        config = self._make_config()
        config["mlp_only_layers"] = []
        tensors = {"model.embed_tokens.weight": _rand(self.VOCAB, self.HIDDEN)}
        for index in range(self.LAYERS):
            prefix = f"model.layers.{index}"
            tensors[f"{prefix}.input_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            tensors[f"{prefix}.mlp.gate.weight"] = _rand(self.NUM_EXPERTS, self.HIDDEN)
            for expert in range(self.NUM_EXPERTS):
                expert_prefix = f"{prefix}.mlp.experts.{expert}"
                tensors[f"{expert_prefix}.gate_proj.weight"] = _rand(
                    self.MOE_INTER,
                    self.HIDDEN,
                )
                tensors[f"{expert_prefix}.up_proj.weight"] = _rand(
                    self.MOE_INTER,
                    self.HIDDEN,
                )
                tensors[f"{expert_prefix}.down_proj.weight"] = _rand(
                    self.HIDDEN,
                    self.MOE_INTER,
                )
        tensors["model.norm.weight"] = _rand(self.HIDDEN)
        tensors["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)

        weights = self._load(tmp_path, config=config, tensors=tensors)

        for index in range(self.LAYERS):
            assert f"layer.{index}.router" in weights
            assert f"layer.{index}.shared_expert.w_gate" not in weights
            assert f"layer.{index}.w_gate" not in weights
            assert f"layer.{index}.experts.w_gate" in weights

    def test_fp16_load_uses_packed_fp16_expert_weights(self, tmp_path: Path) -> None:
        weights = self._load(tmp_path, precision="fp16")

        assert weights["embedding"].dtype == np.float16
        assert weights["layer.0.w_gate"].dtype == np.float16
        assert weights["layer.1.router"].dtype == np.float16
        assert weights["layer.1.experts.w_gate"].dtype == np.float16
        assert weights["layer.1.experts.w_up"].dtype == np.float16
        assert weights["layer.1.experts.w_down"].dtype == np.float16
        assert weights["final_norm"].dtype == np.float32
