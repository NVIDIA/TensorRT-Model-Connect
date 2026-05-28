"""Tests for family plugin load_weights — verifies weight key mapping and transforms.

Each test creates a synthetic model directory with config.json and mock safetensors
files containing random tensors of the correct shapes, then calls plugin.load_weights()
and verifies the returned WeightDict has the expected keys and transforms applied.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-WEIGHTS
Intent: Validate load_weights correctness for core family plugins (Qwen, LLaMA, Gemma, Phi, etc.)
Preconditions: Synthetic safetensors with family-specific HF weight naming are available in temp directories
Postconditions: Each plugin produces expected canonical weight keys with correct shapes and family-specific transforms
"""

from __future__ import annotations

import json
import math
import importlib
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig

# ---- helpers shared across all tests ----

RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# 1. Qwen — standard decoder baseline
# =========================================================================

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


# =========================================================================
# 1b. Nemotron Labs Diffusion — encoder.* checkpoint aliases
# =========================================================================

class TestNemotronLabsDiffusionPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["encoder.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"encoder.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["encoder.norm.weight"] = _rand(hidden)
        t["diffusion_head.weight"] = _rand(vocab, hidden)
        return t

    def _setup(self, tmp_path):
        config = {
            "model_type": "nemotron_labs_diffusion",
            "architectures": ["NemotronLabsDiffusionModel"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "intermediate_size": self.MLP,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "head_dim": self.HIDDEN // self.HEADS,
            "mask_token_id": 100,
            "block_size": 32,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)
        return tensors

    def test_load_weights_uses_encoder_prefix_and_diffusion_head(self, tmp_path):
        from tensorrt_model_connect.families.nemotron_labs_diffusion import plugin

        tensors = self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        np.testing.assert_allclose(weights["embedding"], tensors["encoder.embed_tokens.weight"])
        np.testing.assert_allclose(weights["final_norm"], tensors["encoder.norm.weight"])
        np.testing.assert_allclose(weights["w_out"], tensors["diffusion_head.weight"].T)
        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_kv_attention_size"] == self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["_mlp_size"] == self.MLP

    def test_build_engine_requests_full_logits_runtime(self, tmp_path, monkeypatch):
        plugin_mod = importlib.import_module(
            "tensorrt_model_connect.families.nemotron_labs_diffusion.plugin")
        from tensorrt_model_connect.families.nemotron_labs_diffusion import plugin

        self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        captured = {}

        def fake_build(config, weights, max_cache_length, **kwargs):
            captured.update(kwargs)
            captured["runtime_strategy"] = config.raw.get("runtime_strategy")
            captured["decoder_engine_role"] = config.raw.get("_decoder_engine_role")
            captured["full_logits_raw"] = config.raw.get("_decoder_full_logits_output")
            return b"plan"

        monkeypatch.setattr(plugin_mod, "build_standard_decoder_engine", fake_build)
        assert plugin.build_engine(cfg, {}, 64, precision="bf16") == b"plan"
        assert captured["full_logits_output"] is True
        assert captured["runtime_strategy"] == "nemotron_labs_diffusion"
        assert captured["decoder_engine_role"] == "dual_profile"
        assert captured["full_logits_raw"] is True

    def test_build_extra_engines_merges_linear_spec_lora(self, tmp_path, monkeypatch):
        plugin_mod = importlib.import_module(
            "tensorrt_model_connect.families.nemotron_labs_diffusion.plugin")
        from tensorrt_model_connect.families.nemotron_labs_diffusion import plugin

        self._setup(tmp_path)
        lora_dir = tmp_path / "linear_spec_lora"
        lora_dir.mkdir()
        (lora_dir / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "target_modules": ["o_proj"],
            "r": 2,
            "lora_alpha": 4,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
        }))

        adapters = {}
        expected_deltas = {}
        for layer_idx in range(self.LAYERS):
            prefix = f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj"
            lora_a = (
                np.arange(2 * self.HIDDEN, dtype=np.float32).reshape(2, self.HIDDEN)
                + layer_idx
            )
            lora_b = (
                np.arange(self.HIDDEN * 2, dtype=np.float32).reshape(self.HIDDEN, 2)
                + 0.25 + layer_idx
            )
            adapters[f"{prefix}.lora_A.weight"] = lora_a
            adapters[f"{prefix}.lora_B.weight"] = lora_b
            expected_deltas[layer_idx] = ((lora_b @ lora_a) * 2.0).T
        save_file(adapters, str(lora_dir / "adapter_model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        cfg.raw["_model_dir"] = str(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        assert plugin.get_lora_config(cfg) == {
            "linear_spec_lora_engine_section": plugin.lora_engine_section
        }
        base_w_o = {i: weights[f"layer.{i}.w_o"].copy() for i in range(self.LAYERS)}
        captured = {}

        def fake_build(config, merged_weights, max_cache_length, **kwargs):
            captured["weights"] = merged_weights
            captured["kwargs"] = kwargs
            captured["runtime_strategy"] = config.raw.get("runtime_strategy")
            captured["full_logits_raw"] = config.raw.get("_decoder_full_logits_output")
            return b"lora-plan"

        monkeypatch.setattr(plugin_mod, "build_standard_decoder_engine", fake_build)
        extra = plugin.build_extra_engines(cfg, weights, 64, precision="fp32")

        assert extra == {plugin.lora_engine_section: b"lora-plan"}
        assert captured["kwargs"]["full_logits_output"] is True
        assert captured["runtime_strategy"] == "nemotron_labs_diffusion"
        assert captured["full_logits_raw"] is True
        for layer_idx in range(self.LAYERS):
            np.testing.assert_allclose(
                captured["weights"][f"layer.{layer_idx}.w_o"],
                base_w_o[layer_idx] + expected_deltas[layer_idx],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(weights[f"layer.{layer_idx}.w_o"], base_w_o[layer_idx])


# =========================================================================
# 2. Gemma — gamma +1.0 offset, embedding scaling
# =========================================================================

class TestGemmaPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    def _setup(self, tmp_path, num_layers=2):
        config = {
            "model_type": "gemma",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": num_layers,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = TestQwenPlugin._make_tensors(
            self.VOCAB, self.HIDDEN, num_layers, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)
        return tensors

    def test_gamma_plus_one(self, tmp_path):
        """RMSNorm weights should have +1.0 added (Gemma offset)."""
        from tensorrt_model_connect.families.gemma import plugin

        tensors = self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            raw_input_norm = tensors[f"model.layers.{i}.input_layernorm.weight"]
            np.testing.assert_allclose(
                weights[f"layer.{i}.input_norm"],
                raw_input_norm + 1.0, atol=1e-6)
            raw_post_norm = tensors[f"model.layers.{i}.post_attention_layernorm.weight"]
            np.testing.assert_allclose(
                weights[f"layer.{i}.post_attn_norm"],
                raw_post_norm + 1.0, atol=1e-6)

        raw_final = tensors["model.norm.weight"]
        np.testing.assert_allclose(
            weights["final_norm"], raw_final + 1.0, atol=1e-6)

    def test_embedding_scaling(self, tmp_path):
        """Embedding should be scaled by sqrt(hidden_size)."""
        from tensorrt_model_connect.families.gemma import plugin

        tensors = self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        scale = math.sqrt(self.HIDDEN)
        expected_embed = tensors["model.embed_tokens.weight"] * scale
        np.testing.assert_allclose(
            weights["embedding"], expected_embed, atol=1e-5)


# =========================================================================
# 3. Phi — fused QKV split, fused gate_up split
# =========================================================================

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


# =========================================================================
# 4. Falcon — LayerNorm with bias, GELU FC MLP
# =========================================================================

class TestFalconPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    MLP = 64  # dense_h_to_4h out size

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        kv_dim = self.KV_HEADS * self.HEAD_DIM
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            # LayerNorm with bias
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.input_layernorm.bias"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(self.HIDDEN)
            # Separate Q/K/V/O
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_dim, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_dim, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            # GELU FC MLP with biases
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(self.HIDDEN, self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(self.HIDDEN)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["model.norm.bias"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_bias_weights_loaded(self, tmp_path):
        """Falcon uses LayerNorm with bias — verify bias keys present."""
        from tensorrt_model_connect.families.falcon import plugin

        config = {
            "model_type": "falcon",
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
            assert f"layer.{i}.input_norm_beta" in weights
            assert f"layer.{i}.post_attn_norm_beta" in weights
            np.testing.assert_allclose(
                weights[f"layer.{i}.input_norm_beta"],
                tensors[f"model.layers.{i}.input_layernorm.bias"], atol=1e-6)

    def test_fc_mlp_keys(self, tmp_path):
        """Falcon uses fc1/fc2 MLP naming (not gate/up/down)."""
        from tensorrt_model_connect.families.falcon import plugin

        config = {
            "model_type": "falcon",
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
            assert f"layer.{i}.w_fc1" in weights
            assert f"layer.{i}.w_fc2" in weights
            assert f"layer.{i}.fc1_bias" in weights
            assert f"layer.{i}.fc2_bias" in weights
            # fc1 transposed: [hidden, mlp]
            assert weights[f"layer.{i}.w_fc1"].shape == (self.HIDDEN, self.MLP)

    def test_final_norm_beta(self, tmp_path):
        from tensorrt_model_connect.families.falcon import plugin

        config = {
            "model_type": "falcon",
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

        assert "final_norm_beta" in weights
        np.testing.assert_allclose(
            weights["final_norm_beta"], tensors["model.norm.bias"], atol=1e-6)


# =========================================================================
# 5. Mamba — SSM-specific weights
# =========================================================================

class TestMambaPlugin:
    VOCAB, HIDDEN, LAYERS = 32, 16, 2
    D_INNER = 32
    STATE_SIZE = 8
    CONV_KERNEL = 4
    DT_RANK = 6

    def _make_tensors(self):
        t = {}
        t["backbone.embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"backbone.layers.{i}"
            t[f"{p}.norm.weight"] = _rand(self.HIDDEN)
            # in_proj: [2*d_inner, hidden]
            t[f"{p}.mixer.in_proj.weight"] = _rand(2 * self.D_INNER, self.HIDDEN)
            # conv1d: [d_inner, 1, conv_kernel]
            t[f"{p}.mixer.conv1d.weight"] = _rand(
                self.D_INNER, 1, self.CONV_KERNEL)
            t[f"{p}.mixer.conv1d.bias"] = _rand(self.D_INNER)
            # x_proj: [dt_rank + 2*state_size, d_inner]
            t[f"{p}.mixer.x_proj.weight"] = _rand(
                self.DT_RANK + 2 * self.STATE_SIZE, self.D_INNER)
            # dt_proj: [d_inner, dt_rank]
            t[f"{p}.mixer.dt_proj.weight"] = _rand(self.D_INNER, self.DT_RANK)
            t[f"{p}.mixer.dt_proj.bias"] = _rand(self.D_INNER)
            # A_log: [d_inner, state_size]
            t[f"{p}.mixer.A_log"] = _rand(self.D_INNER, self.STATE_SIZE)
            # D: [d_inner]
            t[f"{p}.mixer.D"] = _rand(self.D_INNER)
            # out_proj: [hidden, d_inner]
            t[f"{p}.mixer.out_proj.weight"] = _rand(self.HIDDEN, self.D_INNER)
        t["backbone.norm_f.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_a_log_transform(self, tmp_path):
        """A_log should be transformed to A = -exp(A_log)."""
        from tensorrt_model_connect.families.mamba import plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            A_log = tensors[f"backbone.layers.{i}.mixer.A_log"]
            expected_A = -np.exp(A_log.astype(np.float32))
            np.testing.assert_allclose(
                weights[f"layer.{i}.A"], expected_A, atol=1e-5)

    def test_in_proj_split(self, tmp_path):
        """in_proj should be split into w_in_x and w_in_z."""
        from tensorrt_model_connect.families.mamba import plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            in_proj = tensors[f"backbone.layers.{i}.mixer.in_proj.weight"]
            x_raw = in_proj[:self.D_INNER, :]
            z_raw = in_proj[self.D_INNER:, :]
            # Transposed: [hidden, d_inner]
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_in_x"],
                x_raw.T.astype(np.float32), atol=1e-6)
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_in_z"],
                z_raw.T.astype(np.float32), atol=1e-6)

    def test_x_proj_split(self, tmp_path):
        """x_proj should be split into dt, B, C projections."""
        from tensorrt_model_connect.families.mamba import plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        x_proj = tensors["backbone.layers.0.mixer.x_proj.weight"]
        dt_raw = x_proj[:self.DT_RANK, :]
        B_raw = x_proj[self.DT_RANK:self.DT_RANK + self.STATE_SIZE, :]
        C_raw = x_proj[self.DT_RANK + self.STATE_SIZE:, :]

        # All transposed
        np.testing.assert_allclose(
            weights["layer.0.w_dt_in"],
            dt_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_B"],
            B_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_C"],
            C_raw.T.astype(np.float32), atol=1e-6)

    def test_conv1d_reshaped(self, tmp_path):
        """conv1d weight [d_inner, 1, conv_kernel] should be reshaped to [d_inner, conv_kernel]."""
        from tensorrt_model_connect.families.mamba import plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["layer.0.conv1d_weight"].shape == (
            self.D_INNER, self.CONV_KERNEL)

    def test_metadata_keys(self, tmp_path):
        """Mamba-specific dimension metadata should be stored."""
        from tensorrt_model_connect.families.mamba import plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_d_inner"] == self.D_INNER
        assert weights["_state_size"] == self.STATE_SIZE
        assert weights["_conv_kernel"] == self.CONV_KERNEL
        assert weights["_dt_rank"] == self.DT_RANK


# =========================================================================
# 6. Mixtral — MoE expert weights
# =========================================================================

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
        from tensorrt_model_connect.families.mixtral import plugin

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
        from tensorrt_model_connect.families.mixtral import plugin

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
        from tensorrt_model_connect.families.mixtral import plugin

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


# =========================================================================
# 7. Bloom — ALiBi, fused QKV per-head interleaved, embedding LayerNorm
# =========================================================================

class TestBloomPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 2, 4
    HEAD_DIM = HIDDEN // HEADS  # 4
    MLP = 64

    def _make_tensors(self):
        t = {}
        t["transformer.word_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        t["transformer.word_embeddings_layernorm.weight"] = _rand(self.HIDDEN)
        t["transformer.word_embeddings_layernorm.bias"] = _rand(self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"transformer.h.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.input_layernorm.bias"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(self.HIDDEN)
            # Fused QKV per-head interleaved: [3*hidden, hidden]
            # Layout: for each head h, rows [h*3*hd : h*3*hd + 3*hd] are Q,K,V
            t[f"{p}.self_attention.query_key_value.weight"] = _rand(
                3 * self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attention.query_key_value.bias"] = _rand(
                3 * self.HIDDEN)
            t[f"{p}.self_attention.dense.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attention.dense.bias"] = _rand(self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(self.HIDDEN, self.MLP)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(self.HIDDEN)
        t["transformer.ln_f.weight"] = _rand(self.HIDDEN)
        t["transformer.ln_f.bias"] = _rand(self.HIDDEN)
        return t

    def test_qkv_interleaved_split(self, tmp_path):
        """BLOOM QKV is per-head interleaved; verify correct split."""
        from tensorrt_model_connect.families.bloom import plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        hd = self.HEAD_DIM
        for i in range(self.LAYERS):
            qkv_w = tensors[
                f"transformer.h.{i}.self_attention.query_key_value.weight"]
            # Extract Q per-head manually
            q_parts = []
            for h in range(self.HEADS):
                base = h * 3 * hd
                q_parts.append(qkv_w[base:base + hd])
            q_expected = np.concatenate(q_parts, axis=0)  # [hidden, hidden]
            # Should be transposed: [hidden, hidden]
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_q"],
                q_expected.T.astype(np.float32), atol=1e-6)

    def test_embedding_layernorm(self, tmp_path):
        """BLOOM has an embedding LayerNorm."""
        from tensorrt_model_connect.families.bloom import plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding_norm" in weights
        assert "embedding_norm_beta" in weights
        np.testing.assert_allclose(
            weights["embedding_norm"],
            tensors["transformer.word_embeddings_layernorm.weight"],
            atol=1e-6)

    def test_qkv_bias_split(self, tmp_path):
        """QKV biases should be split per-head interleaved just like weights."""
        from tensorrt_model_connect.families.bloom import plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "layer.0.q_bias" in weights
        assert "layer.0.k_bias" in weights
        assert "layer.0.v_bias" in weights
        assert weights["layer.0.q_bias"].shape == (self.HIDDEN,)

    def test_all_keys(self, tmp_path):
        from tensorrt_model_connect.families.bloom import plugin

        config = {
            "model_type": "bloom",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        for i in range(self.LAYERS):
            for key in ("input_norm", "input_norm_beta", "post_attn_norm",
                        "post_attn_norm_beta", "w_q", "w_k", "w_v", "w_o",
                        "q_bias", "k_bias", "v_bias", "o_bias",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
        assert "final_norm" in weights
        assert "final_norm_beta" in weights
        assert "w_out" in weights
        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_mlp_size"] == self.MLP


# =========================================================================
# 8. LLaMA — standard decoder baseline reference
# =========================================================================

class TestLlamaPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    def test_load_weights(self, tmp_path):
        """LLaMA uses load_standard_weights — verify compact GQA K/V."""
        from tensorrt_model_connect.families.llama import plugin

        head_dim = self.HIDDEN // self.HEADS  # 4
        kv_hidden = self.KV_HEADS * head_dim  # 8
        config = {
            "model_type": "llama",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = TestQwenPlugin._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS,
            self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # K/V stay compact at [hidden, kv_hidden].
        for i in range(self.LAYERS):
            assert weights[f"layer.{i}.w_k"].shape == (
                self.HIDDEN, kv_hidden)
            assert weights[f"layer.{i}.w_v"].shape == (
                self.HIDDEN, kv_hidden)

    def test_tied_embeddings(self, tmp_path):
        """When lm_head.weight is missing, w_out = transposed embedding."""
        from tensorrt_model_connect.families.llama import plugin

        config = {
            "model_type": "llama",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "tie_word_embeddings": True,
        }
        tensors = TestQwenPlugin._make_tensors(
            self.VOCAB, self.HIDDEN, 1, self.HEADS, self.KV_HEADS, self.MLP)
        # Remove lm_head to test tied embedding fallback
        del tensors["lm_head.weight"]
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)
        embedding = tensors["model.embed_tokens.weight"]
        np.testing.assert_allclose(
            weights["w_out"], embedding.T, atol=1e-6)


# =========================================================================
# 9. DeepSeek-V2 — MLA attention decomposition, MoE + shared experts
# =========================================================================

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
        from tensorrt_model_connect.families.deepseek_v2 import plugin

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
        from tensorrt_model_connect.families.deepseek_v2 import plugin

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
        from tensorrt_model_connect.families.deepseek_v2 import plugin

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


# =========================================================================
# 10. Qwen VL — text weights loaded separately for Qwen3-VL variant
# =========================================================================

class TestQwenVLPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    def test_qwen25_vl_delegates_to_standard(self, tmp_path):
        """Qwen2.5-VL (no deepstack) delegates to load_standard_weights."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen2_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
        }
        tensors = TestQwenPlugin._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS,
            self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        for i in range(self.LAYERS):
            assert f"layer.{i}.w_q" in weights
        assert "final_norm" in weights
        assert "w_out" in weights

    def test_qwen3_vl_language_model_prefix(self, tmp_path):
        """Qwen3-VL uses model.language_model.layers.{i} prefix."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen3_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0, 1],
            },
        }
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        tensors = {}
        tensors["model.language_model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.language_model.layers.{i}"
            tensors[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.self_attn.q_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.self_attn.k_proj.weight"] = _rand(
                kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.v_proj.weight"] = _rand(
                kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.o_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.down_proj.weight"] = _rand(
                self.HIDDEN, self.MLP)
        tensors["model.language_model.norm.weight"] = _rand(self.HIDDEN)
        tensors["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)

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
        assert weights["_mlp_size"] == self.MLP

    def test_qwen3_vl_vision_weights_not_in_text(self, tmp_path):
        """Vision weights (visual.*) should NOT appear in text weight dict."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen3_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0],
            },
        }
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        tensors = {}
        tensors["model.language_model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        p = "model.language_model.layers.0"
        tensors[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
        tensors[f"{p}.self_attn.q_proj.weight"] = _rand(
            self.HIDDEN, self.HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = _rand(
            kv_hidden, self.HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = _rand(
            kv_hidden, self.HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = _rand(
            self.HIDDEN, self.HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP)
        tensors["model.language_model.norm.weight"] = _rand(self.HIDDEN)
        tensors["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        # Add some vision weights that should be ignored by text loader
        tensors["model.visual.patch_embed.weight"] = _rand(16, 3, 14, 14)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Vision keys should not appear in text weights
        for key in weights:
            assert not key.startswith("visual."), f"Vision key leaked: {key}"
            assert not key.startswith("model.visual."), \
                f"Vision key leaked: {key}"


# =========================================================================
# 11. InternVL — VL model with language_model.model.layers prefix
# =========================================================================

class TestInternVLPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    def _make_text_tensors(self):
        """Create synthetic text decoder weights with InternVL3 key naming."""
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        t = {}
        t["language_model.model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"language_model.model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP)
        t["language_model.model.norm.weight"] = _rand(self.HIDDEN)
        t["language_model.lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_load_text_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "patch_size": 14,
            },
        }
        tensors = self._make_text_tensors()
        # Add vision weights that should be ignored
        tensors["visual.patch_embed.proj.weight"] = _rand(64, 3, 14, 14)
        tensors["mlp1.0.weight"] = _rand(self.HIDDEN, 64)

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
        assert weights["_mlp_size"] == self.MLP

    def test_qkv_biases_loaded(self, tmp_path):
        """InternVL3 (Qwen2) has q/k biases."""
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.q_bias" in weights
            assert f"layer.{i}.k_bias" in weights
            kv_dim = self.KV_HEADS * (self.HIDDEN // self.HEADS)
            assert weights[f"layer.{i}.k_bias"].shape == (kv_dim,)

    def test_vision_weights_not_in_text(self, tmp_path):
        """Vision and projector keys should NOT appear in text weight dict."""
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        tensors["visual.patch_embed.proj.weight"] = _rand(64, 3, 14, 14)
        tensors["mlp1.0.weight"] = _rand(self.HIDDEN, 64)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for key in weights:
            assert not key.startswith("visual."), f"Vision key leaked: {key}"
            assert not key.startswith("mlp1."), f"Projector key leaked: {key}"

    def test_transpose_applied(self, tmp_path):
        """Projections should be transposed from [out, in] to [in, out]."""
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": 1,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # w_q: [hidden, hidden] transposed
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        # w_gate: [mlp, hidden] transposed to [hidden, mlp]
        assert weights["layer.0.w_gate"].shape == (self.HIDDEN, self.MLP)
        # w_out: [vocab, hidden] transposed to [hidden, vocab]
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_get_vl_config(self, tmp_path):
        """get_vl_config should return correct VL config for InternVL."""
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {
                "hidden_size": 64,
                "patch_size": 14,
            },
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)

        assert vl_cfg is not None
        assert vl_cfg["preprocessor_type"] == "simple_chw"
        assert vl_cfg["interpolation"] == "bicubic"
        assert vl_cfg["fixed_image_size"] == 448
        # num_patches = (448/14)^2 = 1024
        assert vl_cfg["num_image_pad_tokens"] == 256
        assert vl_cfg["vision_output_dim"] == self.HIDDEN
        assert "image_token_id" in vl_cfg
        assert "vl_prompt_template" in vl_cfg

    def test_no_vl_config_without_vision(self, tmp_path):
        """get_vl_config returns None when no vision_config present."""
        from tensorrt_model_connect.families.internvl import plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)
        assert vl_cfg is None


# =========================================================================
# Eagle VLM — embedding and reranking
# =========================================================================

class TestEagleVLMPlugin:
    """Eagle VLM plugin — Llama backbone for embedding/reranking."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp,
                      *, add_score_head=False):
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
        if add_score_head:
            t["score.weight"] = _rand(1, hidden)
            t["score.bias"] = _rand(1)
        return t

    def test_load_weights_embedding(self, tmp_path):
        """Embedding mode: loads Llama backbone weights, no score head."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForEmbedding"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)
        _write_safetensors(
            tmp_path,
            self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS,
                               self.HEADS, self.KV_HEADS, self.MLP))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Core keys
        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "layer.0.w_q" in weights
        assert "layer.0.w_k" in weights
        assert "layer.0.w_v" in weights
        assert "layer.0.w_o" in weights
        assert "layer.0.w_gate" in weights
        assert "layer.0.input_norm" in weights
        assert "final_norm" in weights
        kv_hidden = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["_kv_attention_size"] == kv_hidden
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_hidden)
        # No score head for embedding
        assert "score_weight" not in weights

    def test_load_weights_reranking(self, tmp_path):
        """Reranking mode: loads score head weights."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForReranking"],
            "is_reranker": True,
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)
        _write_safetensors(
            tmp_path,
            self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS,
                               self.HEADS, self.KV_HEADS, self.MLP,
                               add_score_head=True))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "score_weight" in weights
        assert "score_bias" in weights

    def test_get_vl_config(self, tmp_path):
        """get_vl_config returns embedding config with vision info."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "image_size": 384,
                "patch_size": 14,
                "hidden_size": 64,
            },
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)

        assert vl_cfg is not None
        assert vl_cfg["embedding_dim"] == self.HIDDEN
        assert vl_cfg["preprocessor_type"] == "simple_chw"
        # After 2x2 pixel_shuffle merge: (384//14//2)^2 = 13^2 = 169
        assert vl_cfg["num_vision_tokens"] == (384 // 14 // 2) ** 2

    def test_bundle_config_overrides_embedding(self, tmp_path):
        """Bundle config overrides: embedding mode."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForEmbedding"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "embedding"

    def test_bundle_config_overrides_reranking(self, tmp_path):
        """Bundle config overrides: reranking mode."""
        from tensorrt_model_connect.families.eagle_vlm import plugin

        config = {
            "model_type": "eagle",
            "architectures": ["EagleForReranking"],
            "is_reranker": True,
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "reranking"


# =========================================================================
# GLM-4 — fused gate_up_proj split, compact Q/K/V biases
# =========================================================================

class TestGlmPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 2, 4, 2
    HEAD_DIM = HIDDEN // HEADS  # 4
    Q_DIM = HEADS * HEAD_DIM    # 16
    KV_DIM = KV_HEADS * HEAD_DIM  # 8
    MLP_INTER = 24

    def _make_tensors(self):
        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            # Separate Q/K/V with biases
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.Q_DIM, self.HIDDEN)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(self.Q_DIM)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(self.KV_DIM)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(self.KV_DIM, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(self.KV_DIM)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            # Fused gate_up: [2*intermediate, hidden]
            fused_gate_up = _rand(2 * self.MLP_INTER, self.HIDDEN)
            t[f"{p}.mlp.gate_up_proj.weight"] = fused_gate_up
            t[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP_INTER)
        t["model.norm.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_gate_up_split(self, tmp_path):
        """Fused gate_up should be correctly split into gate and up."""
        from tensorrt_model_connect.families.glm import plugin

        config = {
            "model_type": "glm",
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

        np.testing.assert_allclose(
            weights["layer.0.w_gate"],
            gate_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_up"],
            up_raw.T.astype(np.float32), atol=1e-6)

    def test_qkv_biases_stay_compact(self, tmp_path):
        """Q/K/V biases should be loaded; K/V biases stay compact."""
        from tensorrt_model_connect.families.glm import plugin

        config = {
            "model_type": "glm",
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

        np.testing.assert_allclose(
            weights["layer.0.q_bias"],
            tensors["model.layers.0.self_attn.q_proj.bias"].astype(np.float32))
        np.testing.assert_allclose(
            weights["layer.0.k_bias"],
            tensors["model.layers.0.self_attn.k_proj.bias"].astype(np.float32))
        np.testing.assert_allclose(
            weights["layer.0.v_bias"],
            tensors["model.layers.0.self_attn.v_proj.bias"].astype(np.float32))
        assert weights["layer.0.k_bias"].shape == (self.KV_DIM,)
        assert weights["layer.0.v_bias"].shape == (self.KV_DIM,)

class TestCanaryPlugin:
    """Canary encoder-decoder ASR plugin loads from synthetic .nemo archive."""

    VOCAB, HIDDEN, ENC_LAYERS, DEC_LAYERS = 64, 16, 2, 2
    HEADS, HEAD_DIM, FFN = 2, 8, 32
    MEL_BINS, CONV_KERNEL, SUB_CH = 8, 3, 4

    @staticmethod
    def _make_tp_weights(
        *,
        hidden: int = HIDDEN,
        dec_layers: int = DEC_LAYERS,
        dec_heads: int = HEADS,
        ffn: int = FFN,
    ) -> WeightDict:
        rng = np.random.RandomState(123)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        weights = WeightDict({
            "_dec_layers": dec_layers,
            "_dec_heads": dec_heads,
            "_dec_ffn": ffn,
            "_enc_seq": 16,
            "dec_emb": rand(TestCanaryPlugin.VOCAB, hidden),
            "dec_pos": rand(128, hidden),
            "emb_ln": rand(hidden),
            "emb_ln_b": rand(hidden),
            "final_norm": rand(hidden),
            "final_norm_b": rand(hidden),
            "w_out": rand(hidden, TestCanaryPlugin.VOCAB),
            "out_bias": rand(TestCanaryPlugin.VOCAB),
        })
        for i in range(dec_layers):
            pfx = f"layer.{i}"
            for key in ("w_q", "w_k", "w_v", "xw_q", "xw_k", "xw_v"):
                weights[f"{pfx}.{key}"] = rand(hidden, hidden)
            for key in ("q_bias", "k_bias", "v_bias", "xb_q", "xb_k", "xb_v"):
                weights[f"{pfx}.{key}"] = rand(hidden)
            for key in ("w_o", "xw_o"):
                weights[f"{pfx}.{key}"] = rand(hidden, hidden)
            weights[f"{pfx}.o_bias"] = rand(hidden)
            weights[f"{pfx}.xb_o"] = rand(hidden)
            weights[f"{pfx}.w_fc1"] = rand(hidden, ffn)
            weights[f"{pfx}.fc1_bias"] = rand(ffn)
            weights[f"{pfx}.w_fc2"] = rand(ffn, hidden)
            weights[f"{pfx}.fc2_bias"] = rand(hidden)
            for key in (
                "input_norm", "input_norm_b",
                "xattn_norm", "xattn_norm_b",
                "ffn_norm", "ffn_norm_b",
            ):
                weights[f"{pfx}.{key}"] = rand(hidden)
        return weights

    @staticmethod
    def _tp_builder_module():
        return pytest.importorskip(
            "tensorrt_model_connect.families.canary.decoder_tp_builder",
            reason="TensorRT is required for Canary TP builder tests",
        )

    @staticmethod
    def _make_nemo_state_dict(vocab, hidden, enc_layers, dec_layers,
                              heads, head_dim, ffn, mel_bins, conv_kernel,
                              sub_ch):
        """Create synthetic NeMo state dict matching canary-1b-v2."""
        import torch

        sd = {}
        # Subsampling
        sd["encoder.pre_encode.conv.0.weight"] = torch.randn(sub_ch, 1, 3, 3)
        sd["encoder.pre_encode.conv.0.bias"] = torch.randn(sub_ch)
        for dw, pw in [(2, 3), (5, 6)]:
            sd[f"encoder.pre_encode.conv.{dw}.weight"] = torch.randn(sub_ch, 1, 3, 3)
            sd[f"encoder.pre_encode.conv.{dw}.bias"] = torch.randn(sub_ch)
            sd[f"encoder.pre_encode.conv.{pw}.weight"] = torch.randn(sub_ch, sub_ch, 1, 1)
            sd[f"encoder.pre_encode.conv.{pw}.bias"] = torch.randn(sub_ch)
        feat_after = mel_bins
        for _ in range(3):
            feat_after = (feat_after + 2 - 3) // 2 + 1
        sd["encoder.pre_encode.out.weight"] = torch.randn(hidden, sub_ch * feat_after)
        sd["encoder.pre_encode.out.bias"] = torch.randn(hidden)

        # Encoder layers (with biases)
        for i in range(enc_layers):
            p = f"encoder.layers.{i}"
            for proj in ("linear_q", "linear_k", "linear_v", "linear_out"):
                sd[f"{p}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
                sd[f"{p}.self_attn.{proj}.bias"] = torch.randn(hidden)
            sd[f"{p}.self_attn.linear_pos.weight"] = torch.randn(hidden, hidden)
            sd[f"{p}.self_attn.pos_bias_u"] = torch.randn(heads, head_dim)
            sd[f"{p}.self_attn.pos_bias_v"] = torch.randn(heads, head_dim)
            for norm in ("norm_self_att", "norm_feed_forward1",
                         "norm_feed_forward2", "norm_conv", "norm_out"):
                sd[f"{p}.{norm}.weight"] = torch.randn(hidden)
                sd[f"{p}.{norm}.bias"] = torch.randn(hidden)
            for fn in ("feed_forward1", "feed_forward2"):
                sd[f"{p}.{fn}.linear1.weight"] = torch.randn(ffn, hidden)
                sd[f"{p}.{fn}.linear1.bias"] = torch.randn(ffn)
                sd[f"{p}.{fn}.linear2.weight"] = torch.randn(hidden, ffn)
                sd[f"{p}.{fn}.linear2.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.pointwise_conv1.weight"] = torch.randn(2*hidden, hidden, 1)
            sd[f"{p}.conv.pointwise_conv1.bias"] = torch.randn(2*hidden)
            sd[f"{p}.conv.depthwise_conv.weight"] = torch.randn(hidden, 1, conv_kernel)
            sd[f"{p}.conv.depthwise_conv.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.weight"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.running_mean"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.running_var"] = torch.abs(torch.randn(hidden))
            sd[f"{p}.conv.pointwise_conv2.weight"] = torch.randn(hidden, hidden, 1)
            sd[f"{p}.conv.pointwise_conv2.bias"] = torch.randn(hidden)

        # Decoder (note underscore prefixes _embedding, _decoder)
        sd["transf_decoder._embedding.token_embedding.weight"] = torch.randn(vocab, hidden)
        sd["transf_decoder._embedding.position_embedding.pos_enc"] = torch.randn(128, hidden)
        sd["transf_decoder._embedding.layer_norm.weight"] = torch.randn(hidden)
        sd["transf_decoder._embedding.layer_norm.bias"] = torch.randn(hidden)
        sd["transf_decoder._decoder.final_layer_norm.weight"] = torch.randn(hidden)
        sd["transf_decoder._decoder.final_layer_norm.bias"] = torch.randn(hidden)
        for i in range(dec_layers):
            p = f"transf_decoder._decoder.layers.{i}"
            for sub in ("first_sub_layer", "second_sub_layer"):
                for pn in ("query_net", "key_net", "value_net", "out_projection"):
                    sd[f"{p}.{sub}.{pn}.weight"] = torch.randn(hidden, hidden)
                    sd[f"{p}.{sub}.{pn}.bias"] = torch.randn(hidden)
            sd[f"{p}.third_sub_layer.dense_in.weight"] = torch.randn(ffn, hidden)
            sd[f"{p}.third_sub_layer.dense_in.bias"] = torch.randn(ffn)
            sd[f"{p}.third_sub_layer.dense_out.weight"] = torch.randn(hidden, ffn)
            sd[f"{p}.third_sub_layer.dense_out.bias"] = torch.randn(hidden)
            for ln in ("layer_norm_1", "layer_norm_2", "layer_norm_3"):
                sd[f"{p}.{ln}.weight"] = torch.randn(hidden)
                sd[f"{p}.{ln}.bias"] = torch.randn(hidden)

        sd["log_softmax.mlp.layer0.weight"] = torch.randn(vocab, hidden)
        sd["log_softmax.mlp.layer0.bias"] = torch.randn(vocab)
        return sd

    @staticmethod
    def _make_nemo_archive(tmp_path, state_dict, nemo_cfg):
        """Create a synthetic .nemo tar archive."""
        import io
        import tarfile
        import torch
        import yaml

        nemo_path = tmp_path / "canary.nemo"
        with tarfile.open(str(nemo_path), "w") as tar:
            # Write model_config.yaml
            cfg_bytes = yaml.dump(nemo_cfg).encode("utf-8")
            cfg_info = tarfile.TarInfo(name="model_config.yaml")
            cfg_info.size = len(cfg_bytes)
            tar.addfile(cfg_info, io.BytesIO(cfg_bytes))

            # Write model_weights.ckpt
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            buf.seek(0)
            ckpt_info = tarfile.TarInfo(name="model_weights.ckpt")
            ckpt_info.size = len(buf.getvalue())
            tar.addfile(ckpt_info, buf)

        return nemo_path

    def test_load_weights_keys(self, tmp_path):
        """Canary load_weights extracts correct keys from .nemo archive."""
        pytest.importorskip("torch", reason="torch required for canary test")
        pytest.importorskip("yaml", reason="yaml required for canary test")

        from tensorrt_model_connect.families.canary import plugin

        sd = self._make_nemo_state_dict(
            self.VOCAB, self.HIDDEN, self.ENC_LAYERS, self.DEC_LAYERS,
            self.HEADS, self.HEAD_DIM, self.FFN, self.MEL_BINS,
            self.CONV_KERNEL, self.SUB_CH)

        nemo_cfg = {
            "target": "EncDecMultiTaskModel",
            "encoder": {
                "d_model": self.HIDDEN,
                "n_layers": self.ENC_LAYERS,
                "n_heads": self.HEADS,
                "ff_expansion_factor": self.FFN // self.HIDDEN,
                "conv_kernel_size": self.CONV_KERNEL,
                "feat_in": self.MEL_BINS,
                "subsampling_conv_channels": self.SUB_CH,
            },
            "transf_decoder": {
                "config_dict": {
                    "num_layers": self.DEC_LAYERS,
                    "num_attention_heads": self.HEADS,
                    "inner_size": self.FFN,
                },
            },
            "preprocessor": {"features": self.MEL_BINS},
        }

        self._make_nemo_archive(tmp_path, sd, nemo_cfg)

        config = {
            "model_type": "canary",
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.DEC_LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.FFN,
            "vocab_size": self.VOCAB,
            "rms_norm_eps": 1e-5,
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Encoder subsampling
        assert "enc_sub_conv0_w" in weights
        assert "enc_sub_dw0_w" in weights
        assert "enc_sub_dw1_w" in weights

        # Encoder layers
        for i in range(self.ENC_LAYERS):
            pfx = f"el.{i}"
            assert f"{pfx}.w_q" in weights
            assert f"{pfx}.pos_bias_u" in weights
            assert f"{pfx}.rpe_proj" in weights
            assert f"{pfx}.cpw1_w" in weights
            assert f"{pfx}.bn_w" in weights
            assert f"{pfx}.ff1.w1" in weights
            assert f"{pfx}.ff2.w1" in weights
            assert f"{pfx}.norm_out" in weights

        # Decoder
        assert "dec_emb" in weights
        assert weights["dec_emb"].shape == (self.VOCAB, self.HIDDEN)
        assert "emb_ln" in weights
        for i in range(self.DEC_LAYERS):
            pfx = f"layer.{i}"
            assert f"{pfx}.w_q" in weights
            assert f"{pfx}.xw_q" in weights
            assert f"{pfx}.w_fc1" in weights
            assert f"{pfx}.input_norm" in weights

        # Head
        assert "w_out" in weights
        assert "out_bias" in weights
        assert "final_norm" in weights

        assert weights["_enc_layers"] == self.ENC_LAYERS
        assert weights["_dec_layers"] == self.DEC_LAYERS
        assert weights["_hidden"] == self.HIDDEN
        assert weights["_vocab"] == self.VOCAB

    def test_tp_build_rejects_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()

        with pytest.raises(ValueError, match="requires tensor_parallel mode"):
            decoder_tp_builder.build_canary_tp_decoder_engine(
                object(),
                WeightDict(),
                max_cache_length=4,
                parallel_config=ParallelConfig(),
            )

    def test_tp_validation_ignores_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()

        decoder_tp_builder._validate_canary_tp(
            WeightDict(),
            hidden=self.HIDDEN,
            num_heads=self.HEADS,
            ffn_dim=self.FFN,
            parallel=ParallelConfig(),
        )

    @pytest.mark.parametrize(
        ("parallel", "overrides", "message"),
        [
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=-1),
                {},
                "concrete rank",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"hidden": HIDDEN + 1},
                "hidden size divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"num_heads": HEADS + 1},
                "decoder_attention_heads divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"ffn_dim": FFN + 1},
                "decoder_ffn_dim divisible",
            ),
        ],
    )
    def test_tp_validation_rejects_bad_config_dimensions(
        self,
        parallel,
        overrides,
        message,
    ):
        decoder_tp_builder = self._tp_builder_module()
        kwargs = {
            "hidden": self.HIDDEN,
            "num_heads": self.HEADS,
            "ffn_dim": self.FFN,
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_canary_tp(
                self._make_tp_weights(),
                parallel=parallel,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("key", "shape", "message"),
        [
            ("layer.0.w_q", (HIDDEN, HIDDEN - 1), "output dim"),
            ("layer.0.w_o", (HIDDEN - 1, HIDDEN), "input dim"),
            ("layer.0.w_fc1", (HIDDEN, FFN - 1), "w_fc1 output dim"),
        ],
    )
    def test_tp_validation_rejects_unshardable_weight_shapes(
        self,
        key,
        shape,
        message,
    ):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        weights[key] = np.zeros(shape, dtype=np.float32)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_canary_tp(
                weights,
                hidden=self.HIDDEN,
                num_heads=self.HEADS,
                ffn_dim=self.FFN,
                parallel=ParallelConfig(
                    mode="tensor_parallel",
                    tp_size=2,
                    rank=0,
                ),
            )

    def test_tp_sharding_returns_original_for_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        assert decoder_tp_builder.shard_canary_decoder_weights(
            weights,
            parallel=ParallelConfig(),
        ) is weights

    def test_tp_shards_rank_local_decoder_weights(self):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        shard = decoder_tp_builder.shard_canary_decoder_weights(
            weights,
            parallel=ParallelConfig(
                mode="tensor_parallel",
                tp_size=2,
                rank=1,
            ),
        )

        assert isinstance(shard, WeightDict)
        assert shard["_tensor_parallel_size"] == 2
        assert shard["_tensor_parallel_rank"] == 1
        assert shard["_dec_layers"] == self.DEC_LAYERS

        np.testing.assert_array_equal(
            shard["layer.0.w_q"],
            weights["layer.0.w_q"][:, self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xw_v"],
            weights["layer.0.xw_v"][:, self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.q_bias"],
            weights["layer.0.q_bias"][self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xb_k"],
            weights["layer.0.xb_k"][self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_o"],
            weights["layer.0.w_o"][self.HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xw_o"],
            weights["layer.0.xw_o"][self.HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc1"],
            weights["layer.0.w_fc1"][:, self.FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.fc1_bias"],
            weights["layer.0.fc1_bias"][self.FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc2"],
            weights["layer.0.w_fc2"][self.FFN // 2:, :],
        )
        assert shard["final_norm"] is weights["final_norm"]
