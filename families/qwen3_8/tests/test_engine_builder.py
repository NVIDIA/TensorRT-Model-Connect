# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 hybrid graph and runtime-config contracts."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from families.qwen3_8 import engine_builder
from families.qwen3_8.config import ModelConfig


def test_layer_type_aliases_are_normalized() -> None:
    assert engine_builder._parse_layer_types(
        ["linear", "FULL", "linear_attention", "full_attention", "Custom"]
    ) == ["deltanet", "attention", "deltanet", "attention", "custom"]


def test_fp16_runtime_inputs_keep_recurrent_state_in_fp32() -> None:
    class Tensor:
        def __init__(self, name: str, dtype) -> None:
            self.name = name
            self.dtype = dtype

    class Layer:
        def __init__(self, output: Tensor) -> None:
            self.output = output

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.cast_inputs: list[Tensor] = []

        def add_cast(self, tensor: Tensor, dtype) -> Layer:
            self.cast_inputs.append(tensor)
            return Layer(Tensor(f"{tensor.name}_cast", dtype))

    network = Network()
    attention_mask = Tensor("attention_mask", trt.float32)
    conv_state = Tensor("conv_state", trt.float32)
    recurrent_state = Tensor("recurrent_state", trt.float32)
    cache_k = Tensor("cache_k", trt.float16)
    cache_v = Tensor("cache_v", trt.float16)

    prepared = engine_builder._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [recurrent_state],
        [cache_k],
        [cache_v],
    )

    prepared_mask, prepared_conv, prepared_recurrent, prepared_k, prepared_v = prepared
    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_k[0].dtype == trt.float16
    assert prepared_v[0].dtype == trt.float16
    assert prepared_recurrent == [recurrent_state]
    assert recurrent_state not in network.cast_inputs


def test_runtime_config_publishes_flat_hybrid_dimensions() -> None:
    layer_types = ["linear_attention", "full_attention", "unknown"]
    config = ModelConfig(
        model_type="qwen3_8",
        vocab_size=32,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        raw={
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 12,
                "intermediate_size": 16,
                "num_hidden_layers": 3,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "layer_types": layer_types,
                "linear_num_value_heads": 3,
                "linear_num_key_heads": 1,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 5,
            }
        },
    )

    runtime = engine_builder.Qwen38Model().get_bundle_config_overrides(config)

    assert runtime["layer_types"] == ["deltanet", "attention", "unknown"]
    assert runtime["num_mamba_layers"] == 1
    assert runtime["num_attention_layers"] == 1
    assert runtime["hidden_size"] == 12
    assert runtime["num_attention_heads"] == 3
    assert runtime["num_key_value_heads"] == 1
    assert runtime["head_dim"] == 4
    assert runtime["d_inner"] == 12
    assert runtime["mamba_d_state"] == 4
    assert runtime["mamba_d_conv"] == 5
    assert runtime["mamba_nheads"] == 3
    assert runtime["mamba_head_dim"] == 4
    assert runtime["conv_dim"] == 20


def _tiny_load_weights_config() -> ModelConfig:
    """A minimal 2-layer (deltanet + attention) config exercising every
    load_weights() branch, with dimensions small enough to hand-write a
    fake tensor catalog for."""
    layer_types = ["deltanet", "attention"]
    return ModelConfig(
        model_type="qwen3_8",
        vocab_size=6,
        hidden_size=4,
        intermediate_size=3,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        raw={
            "text_config": {
                "vocab_size": 6,
                "hidden_size": 4,
                "intermediate_size": 3,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "layer_types": layer_types,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_value_head_dim": 2,
                "linear_conv_kernel_dim": 3,
            }
        },
    )


def _tiny_tensor_catalog() -> dict[str, np.ndarray]:
    """HF-style tensor name -> small deterministic array, sized to match
    `_tiny_load_weights_config()` (hidden=4, vocab=6, mlp_size=3, attn_size=4,
    kv_size=2, d_inner=4, deltanet num_heads=2, conv_dim=8, d_conv=3)."""

    def arr(*shape: int) -> np.ndarray:
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    catalog = {
        "model.language_model.embed_tokens.weight": arr(6, 4),
        "model.language_model.norm.weight": arr(4),
        "lm_head.weight": arr(6, 4),
        # layer 0: deltanet
        "model.language_model.layers.0.input_layernorm.weight": arr(4),
        "model.language_model.layers.0.post_attention_layernorm.weight": arr(4),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": arr(8, 4),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": arr(4, 4),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": arr(2, 4),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": arr(2, 4),
        "model.language_model.layers.0.linear_attn.A_log": arr(2),
        "model.language_model.layers.0.linear_attn.dt_bias": arr(2),
        "model.language_model.layers.0.linear_attn.conv1d.weight": arr(8, 1, 3),
        "model.language_model.layers.0.linear_attn.norm.weight": arr(2),
        "model.language_model.layers.0.linear_attn.out_proj.weight": arr(4, 4),
        "model.language_model.layers.0.mlp.gate_proj.weight": arr(3, 4),
        "model.language_model.layers.0.mlp.up_proj.weight": arr(3, 4),
        "model.language_model.layers.0.mlp.down_proj.weight": arr(4, 3),
        # layer 1: attention
        "model.language_model.layers.1.input_layernorm.weight": arr(4),
        "model.language_model.layers.1.post_attention_layernorm.weight": arr(4),
        "model.language_model.layers.1.self_attn.q_proj.weight": arr(8, 4),
        "model.language_model.layers.1.self_attn.k_proj.weight": arr(2, 4),
        "model.language_model.layers.1.self_attn.v_proj.weight": arr(2, 4),
        "model.language_model.layers.1.self_attn.o_proj.weight": arr(4, 4),
        "model.language_model.layers.1.self_attn.q_norm.weight": arr(2),
        "model.language_model.layers.1.self_attn.k_norm.weight": arr(2),
        "model.language_model.layers.1.mlp.gate_proj.weight": arr(3, 4),
        "model.language_model.layers.1.mlp.up_proj.weight": arr(3, 4),
        "model.language_model.layers.1.mlp.down_proj.weight": arr(4, 3),
    }
    return catalog


class _FakeProfile:
    def __init__(self, owned: set[str]) -> None:
        self._owned = owned
        self.scales: dict = {}

    def should_quantize(self, name: str) -> bool:
        return name in self._owned


class _FakeQuantCtx:
    def __init__(self, owned: set[str]) -> None:
        self.profile = _FakeProfile(owned)


def _patch_fake_checkpoint(monkeypatch) -> tuple[dict[str, np.ndarray], list[str]]:
    """Monkeypatch engine_builder's checkpoint_mapper imports to read from a
    small in-memory catalog instead of real safetensors files, and record
    every tensor name that gets loaded."""
    catalog = _tiny_tensor_catalog()
    loaded: list[str] = []

    def fake_has_tensor(readers, name: str) -> bool:
        return name in catalog

    def fake_load_tensor(readers, name: str) -> np.ndarray:
        loaded.append(name)
        return catalog[name]

    monkeypatch.setattr(engine_builder, "_open_safetensors", lambda model_dir: object())
    monkeypatch.setattr(engine_builder, "_has_tensor", fake_has_tensor)
    monkeypatch.setattr(engine_builder, "_load_tensor", fake_load_tensor)
    return catalog, loaded


def test_load_weights_without_quant_ctx_loads_everything(monkeypatch) -> None:
    """Regression guard: quant_ctx=None must behave exactly like before this
    fix -- every projection is loaded and dequantized."""
    _catalog, loaded = _patch_fake_checkpoint(monkeypatch)
    config = _tiny_load_weights_config()

    weights = engine_builder.Qwen38Model().load_weights(
        "unused", config, precision="fp32", quant_ctx=None)

    for name in ("layer.0.w_gate", "layer.0.w_up", "layer.0.w_down",
                 "layer.0.deltanet_in_proj_qkv", "layer.0.deltanet_z_proj",
                 "layer.0.deltanet_out_proj",
                 "layer.1.w_q", "layer.1.w_gate_attn", "layer.1.w_k",
                 "layer.1.w_v", "layer.1.w_o", "layer.1.w_gate",
                 "w_lm_head"):
        assert name in weights, f"{name} missing with quant_ctx=None"

    assert "model.language_model.layers.0.mlp.gate_proj.weight" in loaded
    assert "model.language_model.layers.1.self_attn.q_proj.weight" in loaded
    assert "lm_head.weight" in loaded


def test_load_weights_skips_quantized_projections(monkeypatch) -> None:
    """Weights quant_ctx already owns must be neither loaded/dequantized nor
    present in the returned WeightDict -- maybe_quantized_matmul() sources
    them from the checkpoint's own packed bytes instead."""
    _catalog, loaded = _patch_fake_checkpoint(monkeypatch)
    config = _tiny_load_weights_config()

    owned = {
        "layer.0.w_gate", "layer.0.w_up", "layer.0.w_down",
        "layer.0.deltanet_in_proj_qkv", "layer.0.deltanet_z_proj",
        "layer.0.deltanet_out_proj",
        "layer.1.w_q", "layer.1.w_gate_attn", "layer.1.w_k",
        "layer.1.w_v", "layer.1.w_o",
        "w_lm_head",
    }
    quant_ctx = _FakeQuantCtx(owned)

    weights = engine_builder.Qwen38Model().load_weights(
        "unused", config, precision="fp32", quant_ctx=quant_ctx)

    # Owned names: absent from the WeightDict, and their checkpoint tensors
    # were never even loaded.
    for name in owned:
        assert name not in weights, f"{name} should have been skipped"
    for hf_key in (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.layers.1.self_attn.v_proj.weight",
        "model.language_model.layers.1.self_attn.o_proj.weight",
        "lm_head.weight",
    ):
        assert hf_key not in loaded, f"{hf_key} should not have been loaded"

    # Non-owned names still load normally: the checkpoint's decay/beta
    # projections are never quantized by this scheme, and layer 1's MLP was
    # not marked owned in this test.
    for name in ("layer.0.deltanet_a_proj", "layer.0.deltanet_b_proj",
                 "layer.1.w_gate", "layer.1.w_up", "layer.1.w_down",
                 "embedding", "final_norm"):
        assert name in weights, f"{name} should still have loaded"
    assert "model.language_model.layers.0.linear_attn.in_proj_a.weight" in loaded
    assert "model.language_model.layers.1.mlp.gate_proj.weight" in loaded
