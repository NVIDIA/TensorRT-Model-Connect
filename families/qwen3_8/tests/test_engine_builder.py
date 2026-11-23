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


def test_required_native_mtp_cannot_publish_base_only_bundle(monkeypatch, tmp_path) -> None:
    import pytest
    from families.qwen3_8 import model
    from families.qwen3_8.build_request import BuildRequest

    config = _tiny_load_weights_config()
    config.raw["text_config"]["output_gate_type"] = "swish"
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _: config)
    monkeypatch.setattr(model.Qwen38Model, "load_weights", lambda *_args, **_kwargs: {})
    request = BuildRequest(tmp_path, tmp_path / "out", "qwen3_8", "text_generation",
                           "fp16", mtp_seq_len=4)
    with pytest.raises(ValueError, match="embedded draft weights"):
        model._build_native(request, None, require_mtp=True)


def test_native_prepared_format_does_not_silently_disable_activation_quantization(tmp_path) -> None:
    import json
    from families.qwen3_8 import model

    raw = {"quantization_config": {"quant_method": "modelopt", "config_groups": {
        "attention": {"weights": {"num_bits": 8, "type": "float"}},
        "mlp": {"weights": {"num_bits": 4, "type": "float"}},
    }}}
    assert model._native_quantization(None, tmp_path, raw) == "nvfp4"
    assert model._native_quantization("none", tmp_path, raw) == "none"
    assert model._native_quantization(None, tmp_path, {}) is None
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(
        {"quantization": {"quant_algo": "NVFP4", "group_size": 16}}))
    assert model._native_quantization(None, tmp_path, {}) == "nvfp4"
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(
        {"quantization": {"quant_algo": "W4A16_AWQ", "group_size": 128}}))
    assert model._native_quantization(None, tmp_path, {}) == "int4_awq"


def test_ordinary_native_build_does_not_add_unrequested_mtp_engines(monkeypatch, tmp_path) -> None:
    from unittest.mock import Mock
    from families.qwen3_8 import model
    from families.qwen3_8.build_request import BuildRequest

    config = _tiny_load_weights_config()
    config.raw["text_config"]["output_gate_type"] = "swish"
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _: config)
    monkeypatch.setattr(model.Qwen38Model, "load_weights",
                        lambda *_args, **_kwargs: {"mtp_layer.fc": object()})
    monkeypatch.setattr(model.Qwen38Model, "build_engine", lambda *_args, **_kwargs: b"target")
    prefill = Mock(return_value=b"prefill")
    mtp_prefill = Mock(return_value=b"mtp-prefill")
    monkeypatch.setattr(model.Qwen38Model, "build_engine_prefill", prefill)
    monkeypatch.setattr(model.Qwen38Model, "build_mtp_prefill_engine", mtp_prefill)
    monkeypatch.setattr(model, "_runtime_config", lambda *_args, **kwargs: kwargs)
    writer = Mock()
    request = BuildRequest(tmp_path, tmp_path / "out", "qwen3_8", "text_generation", "fp16")
    model._build_native(request, writer)
    assert {call.args for call in writer.add_bytes.call_args_list} == {
        ("engine.plan", b"target"), ("prefill_engine.plan", b"prefill")}
    prefill.assert_called_once()
    mtp_prefill.assert_not_called()
    runtime = writer.add_json.call_args.args[1]
    assert runtime["has_mtp"] is False and runtime["mtp_num_draft_tokens"] == 0

    # Explicit MTP retains all three proposals, but no longer asks TensorRT
    # to compile the rejected in-graph draft chain. Its scratch suffix fits
    # even when the main target is at its last admitted cache position.
    from dataclasses import replace
    step = Mock(return_value=b"single-step-draft")
    verify = Mock(return_value=b"four-position-verification")
    chain = Mock(side_effect=AssertionError("New native bundles must orchestrate draft steps in C++"))
    monkeypatch.setattr(model.Qwen38Model, "build_mtp_engine", step)
    monkeypatch.setattr(model.Qwen38Model, "build_engine_multi_token", verify)
    monkeypatch.setattr(model.Qwen38Model, "build_mtp_draft_chain_engine", chain)
    writer.reset_mock()
    model._build_native(replace(request, max_sequence_length=32, mtp_seq_len=4), writer)
    assert step.call_args.args[2] == 34
    assert verify.call_args.args[2:] == (32, 4)
    chain.assert_not_called()
    assert {call.args[0] for call in writer.add_bytes.call_args_list} == {
        "engine.plan", "mtp_engine.plan", "multi_token_engine.plan",
        "prefill_engine.plan", "mtp_prefill_engine.plan"}
    assert mtp_prefill.call_args.args[2:] == (34, 32)
    assert writer.add_json.call_args.args[1]["mtp_num_draft_tokens"] == 3

    # Media MTP preserves the target's visual task and supplies the same media
    # contract to both speculative graphs; external drafts remain text-only.
    from types import SimpleNamespace
    from families.qwen3_8 import native_media, vision
    monkeypatch.setattr(native_media, "native_media_metadata",
                        lambda *_args: (SimpleNamespace(merge=2), {"max_tokens": 8}))
    monkeypatch.setattr(vision, "load_vision_weights", lambda *_args, **_kwargs: ({}, None))
    monkeypatch.setattr(vision, "build_vision_engine", lambda *_args, **_kwargs: b"visual")
    target = Mock(return_value=b"target")
    monkeypatch.setattr(model.Qwen38Model, "build_engine", target)
    writer.reset_mock()
    prefill.reset_mock()
    mtp_prefill.reset_mock()
    model._build_native(replace(request, task="vision_language_generation",
                               max_sequence_length=32, mtp_seq_len=4), writer)
    prefill.assert_not_called()
    mtp_prefill.assert_not_called()
    assert writer.add_json.call_args.args[1]["prefill_chunk_len"] == 0
    assert target.call_args.kwargs["media_inputs"] is True
    assert step.call_args.kwargs["media_inputs"] is True
    assert verify.call_args.kwargs["media_inputs"] is True
    assert step.call_args.args[2] == 34 and verify.call_args.args[2:] == (32, 4)
    assert writer.set_header.call_args.kwargs["task"] == "images_text_to_text"
    assert {call.args[0] for call in writer.add_bytes.call_args_list} == {
        "engine.plan", "native_visual.plan", "mtp_engine.plan", "multi_token_engine.plan"}
    chain.assert_not_called()


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
    # E4M3 storage must reach the attention graph unchanged; add_cast would
    # decode bytes without the exported K/V scale and silently change semantics.
    fp8_k, fp8_v = Tensor("fp8_k", trt.fp8), Tensor("fp8_v", trt.fp8)
    prepared = engine_builder._prepare_runtime_inputs(
        network, trt.float16, attention_mask, [conv_state], [recurrent_state],
        [fp8_k], [fp8_v])
    assert prepared[3] == [fp8_k] and prepared[4] == [fp8_v]
    assert fp8_k not in network.cast_inputs and fp8_v not in network.cast_inputs


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
    assert weights["_fp8_kv_cache"] is False
    # Quantization metadata (including visual-only metadata) must not enable
    # LLM FP8 storage. Actual exported LLM scales do, with Edge defaults for
    # layers/projections lacking a scale tensor.
    config.raw["quantization_config"] = {"kv_cache_quant_algo": "FP8"}
    assert not engine_builder.Qwen38Model().load_weights("unused", config)["_fp8_kv_cache"]
    key = "model.language_model.layers.1.self_attn.k_proj.k_scale"
    _catalog[key] = np.array([0.125], dtype=np.float32)
    _catalog[key.replace("k_proj.k_scale", "v_proj.v_scale")] = np.array([0.25])
    loaded_weights = engine_builder.Qwen38Model().load_weights("unused", config)
    assert loaded_weights["_fp8_kv_cache"] is True
    assert loaded_weights["layer.1.kv_scales"] == (1.0, 0.125, 0.25)
    assert loaded_weights["mtp_layer.kv_scales"] == (1.0, 1.0, 1.0)
    import pytest
    for invalid in (np.array([0.0]), np.array([float("nan")]), np.array([1.0, 2.0])):
        _catalog[key] = invalid
        with pytest.raises(ValueError, match="FP8 attention scale"):
            engine_builder.Qwen38Model().load_weights("unused", config)


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


def test_native_tp2_preserves_packed_qkv_and_row_projection_sum() -> None:
    from families.qwen3_8.parallel import shard_weights

    config = ModelConfig(hidden_size=8, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=12, _head_dim=2)
    def array(*shape):
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    weights = {
        "_mlp_size": 12, "_d_inner": 8, "_conv_dim": 16,
        "_deltanet_num_heads": 4, "_deltanet_num_kv_heads": 2,
        "_deltanet_head_dim": 2, "_attn_size": 8,
        "layer.0.deltanet_in_proj_qkv": array(8, 16),
        "layer.0.conv1d_weight": array(16, 3), "layer.0.conv1d_bias": array(16),
        "layer.0.deltanet_out_proj": array(8, 8),
        "layer.0.deltanet_z_proj": array(8, 8),
        "layer.0.deltanet_a_proj": array(8, 4),
        "layer.0.deltanet_b_proj": array(8, 4),
        "layer.0.A": array(4), "layer.0.dt_bias": array(4),
        "layer.0.deltanet_norm": array(8),
        "layer.1.w_q": array(8, 8), "layer.1.w_gate_attn": array(8, 8),
        "layer.1.w_k": array(8, 4), "layer.1.w_v": array(8, 4),
        "layer.1.w_o": array(8, 8), "layer.1.q_norm": array(8),
        "layer.1.k_norm": array(4), "layer.1.w_gate": array(8, 12),
        "layer.1.w_up": array(8, 12), "layer.1.w_down": array(12, 8),
        "embedding": array(16, 8), "w_lm_head": array(8, 16),
        "layer.1.input_norm": array(8),
    }
    shards = [shard_weights(config, weights, rank) for rank in range(2)]
    for rank, (local, shard) in enumerate(shards):
        assert local.num_attention_heads == 2 and local.num_key_value_heads == 1
        assert local.head_dim == config.head_dim and local.hidden_size == config.hidden_size
        assert shard["_conv_dim"] == 8 and shard["_d_inner"] == 4
        # Correct Q | K | V partition, not one contiguous half of their concat.
        indices = np.r_[rank * 2:rank * 2 + 2,
                        4 + rank * 2:6 + rank * 2,
                        8 + rank * 4:12 + rank * 4]
        np.testing.assert_array_equal(shard["layer.0.deltanet_in_proj_qkv"],
                                      weights["layer.0.deltanet_in_proj_qkv"][:, indices])
        for key in ("layer.0.conv1d_weight", "layer.0.conv1d_bias"):
            np.testing.assert_array_equal(shard[key], weights[key][indices])
        for key in ("embedding", "w_lm_head", "layer.1.input_norm"):
            assert shard[key] is weights[key]
    # Row parallel contribution sums reproduce the complete projection exactly
    # for these integer-representable FP32 fixtures, without summing residuals.
    for key in ("layer.0.deltanet_out_proj", "layer.1.w_o", "layer.1.w_down"):
        x = np.arange(weights[key].shape[0], dtype=np.float32)[None]
        actual = sum(np.split(x, 2, axis=1)[rank] @ shards[rank][1][key] for rank in range(2))
        np.testing.assert_array_equal(actual, x @ weights[key])
    assert config.num_attention_heads == 4 and weights["_conv_dim"] == 16


def test_native_tp2_rejects_incompatible_geometry_and_uses_trt_collective() -> None:
    from families.qwen3_8.parallel import shard_weights, add_all_reduce_sum
    from unittest.mock import Mock
    import pytest

    weights = {"_mlp_size": 8, "_deltanet_num_heads": 4,
               "_deltanet_num_kv_heads": 2}
    with pytest.raises(ValueError, match="divisible"):
        shard_weights(ModelConfig(num_attention_heads=4, num_key_value_heads=1), weights, 0)
    with pytest.raises(ValueError, match="two ranks"):
        shard_weights(ModelConfig(), weights, 2)
    with pytest.raises(ValueError, match="unquantized KV"):
        shard_weights(ModelConfig(), {"_fp8_kv_cache": True}, 0)
    network = Mock()
    value = object()
    assert add_all_reduce_sum(network, value, {}) is value
    network.add_dist_collective.assert_not_called()
    result = add_all_reduce_sum(network, value, {"_tensor_parallel_size": 2})
    network.add_dist_collective.assert_called_once_with(
        value, trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1, [])
    assert network.add_dist_collective.return_value.num_ranks == 2
    assert result is network.add_dist_collective.return_value.get_output.return_value


def test_native_tp_broadcast_requires_none_enum_and_root_zero() -> None:
    from families.qwen3_8.parallel import broadcast_from_rank0
    from unittest.mock import Mock

    network, value = Mock(), object()
    assert broadcast_from_rank0(network, value, {}) is value
    network.add_dist_collective.assert_not_called()
    broadcast_from_rank0(network, value, {"_tensor_parallel_size": 2})
    network.add_dist_collective.assert_called_once_with(
        value, trt.CollectiveOperation.BROADCAST, trt.ReduceOperation.NONE, 0, [])
    assert network.add_dist_collective.return_value.num_ranks == 2


def test_default_prefill_engine_is_dropped_only_when_two_weight_copies_do_not_fit(tmp_path) -> None:
    from dataclasses import replace

    from families.qwen3_8 import model
    from families.qwen3_8.build_request import BuildRequest

    request = BuildRequest(tmp_path, tmp_path / "out", "qwen3_8", "text_generation", "fp16")
    gib = 1024**3
    # A 27B FP16 checkpoint (about 54 GiB) on a 121 GiB unified-memory GPU: decode engine only.
    assert model._fit_prefill_engine(request, 128, 54 * gib, 121 * gib) == 0
    # An 8B checkpoint fits twice, as does the 27B checkpoint on a large discrete GPU.
    assert model._fit_prefill_engine(request, 128, 16 * gib, 121 * gib) == 128
    assert model._fit_prefill_engine(request, 128, 54 * gib, 275 * gib) == 128
    # No device to query, a single-token build, or an explicit chunk length is kept as is.
    assert model._fit_prefill_engine(request, 128, 54 * gib, None) == 128
    assert model._fit_prefill_engine(request, 0, 54 * gib, 121 * gib) == 0
    explicit = replace(request, prefill_chunk_len=128)
    assert model._fit_prefill_engine(explicit, 128, 54 * gib, 121 * gib) == 128
