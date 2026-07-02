# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations

import json
import importlib

import numpy as np
import pytest

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
    save_file,
)


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

    def test_default_decoder_threads_full_logits_to_dual_profile(self, tmp_path, monkeypatch):
        pytest.importorskip("tensorrt")
        from tensorrt_model_connect.families.nemotron_labs_diffusion import default_decoder

        self._setup(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        captured = {}

        def fake_build(config, weights, max_cache_length, **kwargs):
            captured.update(kwargs)
            return b"plan"

        monkeypatch.setattr(default_decoder, "build_dual_profile_decoder_engine", fake_build)
        assert default_decoder.build_standard_decoder_engine(
            cfg, {}, 64, precision="bf16", full_logits_output=True) == b"plan"
        assert captured["full_logits_output"] is True

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
