# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BERT family plugin — verifies weight key mapping and transforms.

Creates a synthetic BERT model directory with config.json and mock safetensors
files containing random tensors of the correct shapes, then calls
plugin.load_weights() and verifies the returned WeightDict.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-BERT
Intent: Validate BERT family plugin weight key mapping and transform correctness
Preconditions: Synthetic safetensors with BERT weight naming convention are available
Postconditions: Loaded WeightDict contains expected keys with correct shapes for encoder-only architecture
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


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


class TestBertPlugin:
    """BERT plugin: load_weights correctness with synthetic weights."""

    VOCAB = 32
    HIDDEN = 16
    LAYERS = 2
    HEADS = 4
    INTERMEDIATE = 64
    MAX_POS = 32
    TYPE_VOCAB = 2

    @staticmethod
    def _make_config(vocab, hidden, layers, heads, intermediate, max_pos,
                     type_vocab):
        return {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": intermediate,
            "max_position_embeddings": max_pos,
            "type_vocab_size": type_vocab,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
        }

    @classmethod
    def _make_tensors(cls, vocab, hidden, layers, heads, intermediate,
                      max_pos, type_vocab):
        t = {}
        # Embeddings
        t["bert.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["bert.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["bert.embeddings.token_type_embeddings.weight"] = _rand(type_vocab, hidden)
        t["bert.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["bert.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"bert.encoder.layer.{i}"
            # Attention
            t[f"{p}.attention.self.query.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(hidden)
            t[f"{p}.attention.self.value.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            # FFN
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        # Pooler
        t["bert.pooler.dense.weight"] = _rand(hidden, hidden)
        t["bert.pooler.dense.bias"] = _rand(hidden)

        return t

    def test_matches(self):
        from tensorrt_model_connect.families.bert import plugin

        assert plugin.matches("bert")
        assert not plugin.matches("gpt2")
        assert not plugin.matches("llama")
        assert not plugin.matches("roberta")

    def test_name(self):
        from tensorrt_model_connect.families.bert import plugin

        assert plugin.name == "bert"

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.bert import plugin

        assert plugin.runtime_strategy == "bert_encoder_only"

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.bert import plugin

        config = self._make_config(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Embedding keys
        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "position_embedding" in weights
        assert weights["position_embedding"].shape == (self.MAX_POS, self.HIDDEN)
        assert "token_type_embedding" in weights
        assert weights["token_type_embedding"].shape == (self.TYPE_VOCAB, self.HIDDEN)
        assert "embed_norm" in weights
        assert "embed_norm_beta" in weights

        # Layer keys
        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o",
                        "q_bias", "k_bias", "v_bias", "o_bias",
                        "post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

        # Pooler keys
        assert "pooler_w" in weights
        assert "pooler_bias" in weights

    def test_weight_shapes(self, tmp_path):
        from tensorrt_model_connect.families.bert import plugin

        config = self._make_config(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Projections are transposed to [in, out]
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, self.HIDDEN)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, self.HIDDEN)
        assert weights["layer.0.w_o"].shape == (self.HIDDEN, self.HIDDEN)
        assert weights["layer.0.w_fc1"].shape == (self.HIDDEN, self.INTERMEDIATE)
        assert weights["layer.0.w_fc2"].shape == (self.INTERMEDIATE, self.HIDDEN)
        assert weights["pooler_w"].shape == (self.HIDDEN, self.HIDDEN)

    def test_weight_transpose_correctness(self, tmp_path):
        """Verify that HF [out, in] weights are correctly transposed to [in, out]."""
        from tensorrt_model_connect.families.bert import plugin

        config = self._make_config(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # The HF q_proj.weight is [out, in] = [hidden, hidden].
        # After transpose it should be [in, out] = [hidden, hidden].
        hf_q = tensors["bert.encoder.layer.0.attention.self.query.weight"]
        np.testing.assert_allclose(
            weights["layer.0.w_q"], hf_q.T, atol=1e-6,
            err_msg="Q weight not properly transposed")

    def test_no_pooler(self, tmp_path):
        """BERT without pooler should not have pooler weights."""
        from tensorrt_model_connect.families.bert import plugin

        config = self._make_config(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.TYPE_VOCAB)
        # Remove pooler tensors
        del tensors["bert.pooler.dense.weight"]
        del tensors["bert.pooler.dense.bias"]
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "pooler_w" not in weights
        assert "pooler_bias" not in weights
