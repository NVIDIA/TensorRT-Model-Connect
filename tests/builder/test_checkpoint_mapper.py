# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Qwen-owned checkpoint mapper.

Uses synthetic safetensors files. No TRT needed.

Trace: ARCH-CHK-001, ARCH-MODPLUG-001, UD-CHK-01
Intent: Validate weight transform helpers (transpose, compact GQA/MQA K/V, head norm repeat) and full weight loading from synthetic safetensors.
Preconditions: tensorrt_model_connect and safetensors are importable; no TRT or GPU required.
Postconditions: Weight shapes, dtypes, and values are correct after transpose, compact K/V load, head norm repeat, and end-to-end load_standard_weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.families.qwen.checkpoint_mapper import (
    _transpose_2d,
    _repeat_head_norm,
    _has_tensor,
    _load_tensor,
    load_standard_weights,
)
from tensorrt_model_connect.families.qwen.config import ModelConfig


class TestTranspose2d:
    def test_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = _transpose_2d(arr, "test")
        expected = arr.T.astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 2)

    def test_square(self):
        arr = np.eye(4, dtype=np.float32)
        result = _transpose_2d(arr, "identity")
        np.testing.assert_array_equal(result, arr)

    def test_rank1_raises(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="Expected rank-2"):
            _transpose_2d(arr, "bad")

    def test_rank3_raises(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected rank-2"):
            _transpose_2d(arr, "bad")

    def test_contiguous_float32(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = _transpose_2d(arr, "test")
        assert result.dtype == np.float32
        assert result.flags["C_CONTIGUOUS"]


class TestRepeatHeadNorm:
    def test_basic(self):
        norm = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = _repeat_head_norm(norm, num_heads=3)
        expected = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32

    def test_single_head(self):
        norm = np.array([5.0, 6.0], dtype=np.float32)
        result = _repeat_head_norm(norm, num_heads=1)
        np.testing.assert_array_equal(result, norm)


class TestReaderLookup:
    class _FakeReader:
        def __init__(self, tensors):
            self.tensors = tensors
            self.key_calls = 0

        def keys(self):
            self.key_calls += 1
            return list(self.tensors.keys())

        def get_tensor(self, name):
            return self.tensors[name]

    class _FakeReaders(list):
        pass

    def test_has_tensor_uses_cached_tensor_map(self):
        r1 = self._FakeReader({"a": np.array([1], dtype=np.float32)})
        r2 = self._FakeReader({"b": np.array([2], dtype=np.float32)})
        readers = self._FakeReaders([r1, r2])
        readers.tensor_map = {"a": r1, "b": r2}

        assert _has_tensor(readers, "b")
        assert not _has_tensor(readers, "missing")
        assert r1.key_calls == 0
        assert r2.key_calls == 0

    def test_load_tensor_uses_cached_tensor_map(self):
        target = np.arange(4, dtype=np.float32)
        r1 = self._FakeReader({"a": np.array([1], dtype=np.float32)})
        r2 = self._FakeReader({"b": target})
        readers = self._FakeReaders([r1, r2])
        readers.tensor_map = {"a": r1, "b": r2}

        loaded = _load_tensor(readers, "b")

        np.testing.assert_array_equal(loaded, target)
        assert np.shares_memory(loaded, target)
        assert r1.key_calls == 0
        assert r2.key_calls == 0


class TestLoadStandardWeights:
    """Test full weight loading with synthetic safetensors."""

    def _create_model_dir(self, tmp_path: Path, config: dict,
                          num_layers: int = 2,
                          hidden: int = 16,
                          vocab: int = 32,
                          num_heads: int = 4,
                          num_kv_heads: int = 4,
                          mlp_size: int = 32) -> Path:
        """Create a minimal model directory with safetensors."""
        from safetensors.numpy import save_file

        (tmp_path / "config.json").write_text(json.dumps(config))

        head_dim = hidden // num_heads
        kv_hidden = num_kv_heads * head_dim
        tensors = {}

        # Embedding
        tensors["model.embed_tokens.weight"] = np.random.randn(
            vocab, hidden).astype(np.float32)

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = np.random.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = \
                np.random.randn(hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(
                hidden, mlp_size).astype(np.float32)

        tensors["model.norm.weight"] = np.random.randn(hidden).astype(np.float32)
        tensors["lm_head.weight"] = np.random.randn(vocab, hidden).astype(np.float32)

        save_file(tensors, str(tmp_path / "model.safetensors"))
        return tmp_path

    def test_basic_loading(self, tmp_path):
        """Load standard weights from a minimal model dir."""
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # Check all expected keys exist
        assert "embedding" in weights
        assert weights["embedding"].shape == (32, 16)
        assert "layer.0.input_norm" in weights
        assert "layer.0.w_q" in weights
        assert "layer.0.w_k" in weights
        assert "layer.0.w_v" in weights
        assert "layer.0.w_o" in weights
        assert "layer.0.w_gate" in weights
        assert "layer.0.w_up" in weights
        assert "layer.0.w_down" in weights
        assert "layer.1.input_norm" in weights
        assert "final_norm" in weights
        assert "w_out" in weights

        fp16_weights = load_standard_weights(model_dir, cfg, precision="fp16")
        assert fp16_weights["embedding"].dtype == np.float16
        assert fp16_weights["layer.0.w_q"].dtype == np.float16
        assert fp16_weights["layer.0.input_norm"].dtype == np.float32

    def test_transpose_applied(self, tmp_path):
        """Verify projections are transposed from [out, in] to [in, out]."""
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config, num_layers=1)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_q should be [hidden, attention_size] = [16, 16] (transposed from [16, 16])
        assert weights["layer.0.w_q"].shape == (16, 16)
        # w_o should be [attention_size, hidden] = [16, 16]
        assert weights["layer.0.w_o"].shape == (16, 16)
        # w_gate: [hidden, mlp_size] = [16, 32]
        assert weights["layer.0.w_gate"].shape == (16, 32)
        # w_out: [hidden, vocab] = [16, 32]
        assert weights["w_out"].shape == (16, 32)

    def test_gqa_kv_stays_compact(self, tmp_path):
        """Verify GQA K/V projections stay at compact KV width."""
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,  # GQA
        }
        model_dir = self._create_model_dir(
            tmp_path, config, num_layers=1, num_kv_heads=2)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert weights["layer.0.w_k"].shape == (16, 8)
        assert weights["layer.0.w_v"].shape == (16, 8)
        assert weights["_kv_attention_size"] == 8

    def test_tied_embeddings_fallback(self, tmp_path):
        """When lm_head.weight is missing, w_out = transposed embedding."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        tensors = {}
        embedding = np.random.randn(32, 16).astype(np.float32)
        tensors["model.embed_tokens.weight"] = embedding

        prefix = "model.layers.0"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(16, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(32, 16).astype(np.float32)
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(32, 16).astype(np.float32)
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(16, 32).astype(np.float32)
        tensors["model.norm.weight"] = np.ones(16, dtype=np.float32)
        # No lm_head.weight!

        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = load_standard_weights(tmp_path, cfg)

        # w_out should be transposed embedding: [16, 32]
        assert weights["w_out"].shape == (16, 32)
        np.testing.assert_allclose(
            weights["w_out"], embedding.T, atol=1e-6)

    def test_metadata_keys(self, tmp_path):
        """Verify attention, KV attention, and MLP metadata."""
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config, num_layers=1)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert weights["_attention_size"] == 16
        assert weights["_kv_attention_size"] == 16
        assert weights["_mlp_size"] == 32


class TestLoadStandardWeightsExtended:
    """Extended tests for load_standard_weights edge cases."""

    def _create_model_dir(self, tmp_path, config, tensors):
        """Create a model dir with config.json and custom tensors."""
        from safetensors.numpy import save_file

        (tmp_path / "config.json").write_text(json.dumps(config))
        save_file(tensors, str(tmp_path / "model.safetensors"))
        return tmp_path

    def _make_standard_tensors(self, num_layers, hidden, vocab,
                               num_heads, num_kv_heads, mlp_size,
                               include_lm_head=True,
                               include_biases=False):
        """Build the standard set of tensors for a model."""
        head_dim = hidden // num_heads
        kv_hidden = num_kv_heads * head_dim
        rng = np.random.RandomState(42)
        tensors = {}

        tensors["model.embed_tokens.weight"] = rng.randn(
            vocab, hidden).astype(np.float32)

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = rng.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = rng.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = rng.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = rng.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = rng.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = rng.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = rng.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = rng.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = rng.randn(
                hidden, mlp_size).astype(np.float32)

            if include_biases:
                tensors[f"{prefix}.self_attn.q_proj.bias"] = rng.randn(
                    hidden).astype(np.float32)
                tensors[f"{prefix}.self_attn.k_proj.bias"] = rng.randn(
                    kv_hidden).astype(np.float32)
                tensors[f"{prefix}.self_attn.v_proj.bias"] = rng.randn(
                    kv_hidden).astype(np.float32)

        tensors["model.norm.weight"] = rng.randn(hidden).astype(np.float32)
        if include_lm_head:
            tensors["lm_head.weight"] = rng.randn(
                vocab, hidden).astype(np.float32)

        return tensors

    # --- Compact GQA value correctness ---

    def test_gqa_compact_values_correct(self, tmp_path):
        """Verify compact GQA K/V values are only transposed, not repeated."""
        hidden = 16
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden // num_heads  # 2

        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        kv_hidden = num_kv_heads * head_dim
        assert weights["layer.0.w_k"].shape == (hidden, kv_hidden)
        assert weights["layer.0.w_v"].shape == (hidden, kv_hidden)
        np.testing.assert_array_equal(
            weights["layer.0.w_k"],
            tensors["model.layers.0.self_attn.k_proj.weight"].T)

    # --- Tied embeddings ---

    def test_tied_embeddings_no_lm_head(self, tmp_path):
        """When lm_head.weight is absent, w_out = transposed embedding."""
        hidden = 16
        vocab = 32
        config = {
            "model_type": "standard_decoder",
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=vocab,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_lm_head=False)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_out should be transposed embedding: [hidden, vocab]
        assert weights["w_out"].shape == (hidden, vocab)
        embedding = tensors["model.embed_tokens.weight"]
        np.testing.assert_allclose(weights["w_out"], embedding.T, atol=1e-6)

    def test_lm_head_present_ignores_tie(self, tmp_path):
        """When lm_head.weight exists, it is used even if tie_word_embeddings=True."""
        hidden = 16
        vocab = 32
        config = {
            "model_type": "standard_decoder",
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=vocab,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_lm_head=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_out should be transposed lm_head.weight, NOT embedding
        lm_head_raw = tensors["lm_head.weight"]
        np.testing.assert_allclose(weights["w_out"], lm_head_raw.T, atol=1e-6)

    # --- Optional biases ---

    def test_biases_loaded_when_present(self, tmp_path):
        """QKV biases are loaded when present in safetensors."""
        hidden = 16
        config = {
            "model_type": "standard_decoder_with_bias",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_biases=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert "layer.0.q_bias" in weights
        assert "layer.0.k_bias" in weights
        assert "layer.0.v_bias" in weights
        assert weights["layer.0.q_bias"].shape == (hidden,)
        assert weights["layer.0.k_bias"].shape == (hidden,)
        assert weights["layer.0.v_bias"].shape == (hidden,)

    def test_no_biases_when_absent(self, tmp_path):
        """QKV biases are absent from weights when not in safetensors."""
        hidden = 16
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_biases=False)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert "layer.0.q_bias" not in weights
        assert "layer.0.k_bias" not in weights
        assert "layer.0.v_bias" not in weights

    def test_gqa_biases_stay_compact(self, tmp_path):
        """K/V biases stay at compact KV width when num_kv_heads < num_heads."""
        hidden = 16
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden // num_heads  # 2
        kv_hidden = num_kv_heads * head_dim

        config = {
            "model_type": "standard_decoder_with_bias",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32,
            include_biases=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # q_bias stays at query attention size.
        assert weights["layer.0.q_bias"].shape == (hidden,)
        assert weights["layer.0.k_bias"].shape == (kv_hidden,)
        assert weights["layer.0.v_bias"].shape == (kv_hidden,)
        k_bias_raw = tensors["model.layers.0.self_attn.k_proj.bias"]
        np.testing.assert_array_equal(weights["layer.0.k_bias"], k_bias_raw)

    # --- Error paths ---

    def test_missing_embed_tokens_raises(self, tmp_path):
        """Missing embed_tokens.weight raises KeyError with descriptive message."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Create safetensors WITHOUT embed_tokens
        tensors = {
            "model.layers.0.input_layernorm.weight":
                np.ones(16, dtype=np.float32),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(KeyError, match="model.embed_tokens.weight"):
            load_standard_weights(tmp_path, cfg)

    def test_wrong_embedding_shape_raises(self, tmp_path):
        """Embedding with wrong shape raises AssertionError."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Embedding with wrong vocab dimension
        tensors = {
            "model.embed_tokens.weight":
                np.random.randn(64, 16).astype(np.float32),  # vocab=64 != 32
            "model.norm.weight": np.ones(16, dtype=np.float32),
            "lm_head.weight": np.random.randn(32, 16).astype(np.float32),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(AssertionError, match="Embedding shape"):
            load_standard_weights(tmp_path, cfg)

    def test_no_safetensors_raises(self, tmp_path):
        """Model dir without any safetensors/bin files raises FileNotFoundError."""
        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        # No safetensors file created

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_standard_weights(tmp_path, cfg)

    def test_single_kv_head_stays_compact(self, tmp_path):
        """MQA keeps the single KV head compact."""
        hidden = 16
        num_heads = 8
        num_kv_heads = 1
        head_dim = hidden // num_heads  # 2
        kv_hidden = num_kv_heads * head_dim  # 2

        config = {
            "model_type": "standard_decoder",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert weights["layer.0.w_k"].shape == (hidden, kv_hidden)
        assert weights["layer.0.w_v"].shape == (hidden, kv_hidden)

        k_raw_transposed = tensors[
            "model.layers.0.self_attn.k_proj.weight"].T.astype(np.float32)
        assert k_raw_transposed.shape == (hidden, kv_hidden)
        np.testing.assert_array_equal(weights["layer.0.w_k"], k_raw_transposed)

        v_raw_transposed = tensors[
            "model.layers.0.self_attn.v_proj.weight"].T.astype(np.float32)
        np.testing.assert_array_equal(weights["layer.0.w_v"], v_raw_transposed)
