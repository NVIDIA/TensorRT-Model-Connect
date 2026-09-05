# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned weight, logits, and linear-spec LoRA contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.nemotron_labs_diffusion import default_decoder
from families.nemotron_labs_diffusion import model as model_module
from families.nemotron_labs_diffusion.config import ModelConfig


VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32


def _rand(*shape: int) -> np.ndarray:
    return np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)


def _make_tensors() -> dict[str, np.ndarray]:
    head_dim = HIDDEN // HEADS
    kv_hidden = KV_HEADS * head_dim
    tensors = {"encoder.embed_tokens.weight": _rand(VOCAB, HIDDEN)}
    for layer in range(LAYERS):
        prefix = f"encoder.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = _rand(HIDDEN)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(HIDDEN)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand(HIDDEN, HIDDEN)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand(kv_hidden, HIDDEN)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand(kv_hidden, HIDDEN)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand(HIDDEN, HIDDEN)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = _rand(MLP, HIDDEN)
        tensors[f"{prefix}.mlp.up_proj.weight"] = _rand(MLP, HIDDEN)
        tensors[f"{prefix}.mlp.down_proj.weight"] = _rand(HIDDEN, MLP)
    tensors["encoder.norm.weight"] = _rand(HIDDEN)
    tensors["diffusion_head.weight"] = _rand(VOCAB, HIDDEN)
    return tensors


def _setup(tmp_path) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    config = {
        "model_type": "nemotron_labs_diffusion",
        "architectures": ["NemotronLabsDiffusionModel"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HIDDEN // HEADS,
        "mask_token_id": 100,
        "block_size": 32,
    }
    tensors = _make_tensors()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return ModelConfig.from_dir(tmp_path), tensors


def test_load_weights_uses_encoder_prefix_and_diffusion_head(tmp_path) -> None:
    config, tensors = _setup(tmp_path)
    model = model_module._NemotronLabsDiffusionModel()

    weights = model.load_weights(str(tmp_path), config)

    np.testing.assert_allclose(weights["embedding"], tensors["encoder.embed_tokens.weight"])
    np.testing.assert_allclose(weights["final_norm"], tensors["encoder.norm.weight"])
    np.testing.assert_allclose(weights["w_out"], tensors["diffusion_head.weight"].T)
    assert weights["_attention_size"] == HIDDEN
    assert weights["_kv_attention_size"] == KV_HEADS * (HIDDEN // HEADS)
    assert weights["_mlp_size"] == MLP


def test_build_engine_requests_full_logits(monkeypatch, tmp_path) -> None:
    config, _ = _setup(tmp_path)
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        del weights, max_cache_length
        captured.update(kwargs)
        captured["role"] = config.raw.get("_decoder_engine_role")
        captured["full_logits_raw"] = config.raw.get("_decoder_full_logits_output")
        return b"plan"

    monkeypatch.setattr(model_module, "build_standard_decoder_engine", fake_build)
    plan = model_module._NemotronLabsDiffusionModel().build_engine(
        config,
        {},
        64,
        precision="bf16",
    )

    assert plan == b"plan"
    assert captured["full_logits_output"] is True
    assert captured["role"] == "dual_profile"
    assert captured["full_logits_raw"] is True


def test_default_decoder_threads_full_logits_to_dual_profile(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tensorrt")
    config, _ = _setup(tmp_path)
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        del config, weights, max_cache_length
        captured.update(kwargs)
        return b"plan"

    monkeypatch.setattr(default_decoder, "build_dual_profile_decoder_engine", fake_build)
    assert (
        default_decoder.build_standard_decoder_engine(
            config,
            {},
            64,
            precision="bf16",
            full_logits_output=True,
        )
        == b"plan"
    )
    assert captured["full_logits_output"] is True


def test_extra_engine_merges_linear_spec_lora(monkeypatch, tmp_path) -> None:
    config, _ = _setup(tmp_path)
    lora_dir = tmp_path / "linear_spec_lora"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "target_modules": ["o_proj"],
                "r": 2,
                "lora_alpha": 4,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    adapters = {}
    expected_deltas = {}
    for layer in range(LAYERS):
        prefix = f"base_model.model.encoder.layers.{layer}.self_attn.o_proj"
        lora_a = np.arange(2 * HIDDEN, dtype=np.float32).reshape(2, HIDDEN) + layer
        lora_b = np.arange(HIDDEN * 2, dtype=np.float32).reshape(HIDDEN, 2) + 0.25 + layer
        adapters[f"{prefix}.lora_A.weight"] = lora_a
        adapters[f"{prefix}.lora_B.weight"] = lora_b
        expected_deltas[layer] = ((lora_b @ lora_a) * 2.0).T
    save_file(adapters, str(lora_dir / "adapter_model.safetensors"))

    config.raw["_model_dir"] = str(tmp_path)
    model = model_module._NemotronLabsDiffusionModel()
    weights = model.load_weights(str(tmp_path), config)
    assert model.get_lora_config(config) == {
        "linear_spec_lora_engine_section": model.lora_engine_section
    }
    base_outputs = {layer: weights[f"layer.{layer}.w_o"].copy() for layer in range(LAYERS)}
    captured = {}

    def fake_build(config, merged_weights, max_cache_length, **kwargs):
        del max_cache_length
        captured["weights"] = merged_weights
        captured["kwargs"] = kwargs
        captured["role"] = config.raw.get("_decoder_engine_role")
        captured["full_logits_raw"] = config.raw.get("_decoder_full_logits_output")
        return b"lora-plan"

    monkeypatch.setattr(model_module, "build_standard_decoder_engine", fake_build)
    extra = model.build_extra_engines(config, weights, 64, precision="fp32")

    assert extra == {model.lora_engine_section: b"lora-plan"}
    assert captured["kwargs"]["full_logits_output"] is True
    assert captured["role"] == "dual_profile"
    assert captured["full_logits_raw"] is True
    for layer in range(LAYERS):
        np.testing.assert_allclose(
            captured["weights"][f"layer.{layer}.w_o"],
            base_outputs[layer] + expected_deltas[layer],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(weights[f"layer.{layer}.w_o"], base_outputs[layer])
