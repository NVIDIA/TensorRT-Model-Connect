# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for DeepSeek-OCR tensor-parallel support."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    deepseek_ocr_module = importlib.import_module(
        "tensorrt_model_connect.families.deepseek_ocr.plugin")
    from tensorrt_model_connect.checkpoint_mapper import WeightDict
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.families.deepseek_ocr import (
        graph_ops,
        norm_utils,
        tp_builder,
    )
    from tensorrt_model_connect.families.deepseek_ocr.prefill_config import (
        sequence_prefill_profile_lengths,
    )
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def test_native_builders_do_not_import_legacy_decoder_paths() -> None:
    family_prefix = "tensorrt_model_connect.families.deepseek_ocr."
    legacy_modules = {
        family_prefix + "default_decoder",
        family_prefix + "default_dual_profile_decoder",
        family_prefix + "graph_blocks",
        family_prefix + "standard_decoder_builder",
    }
    assert legacy_modules.isdisjoint(sys.modules)
    assert norm_utils.__all__ == ["_apply_norm"]
    assert not hasattr(graph_ops, "add_decoder_attention_ffi")
    assert not hasattr(graph_ops, "add_tvm_ffi_kernel")


def _config(num_heads: int = 4, num_kv_heads: int | None = None) -> SimpleNamespace:
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    return SimpleNamespace(
        raw={},
        model_type="deepseek_vl_v2",
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=kv_heads,
        head_dim=4,
        attention_size=num_heads * 4,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=4096,
    )


def _disable_native_contract_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        deepseek_ocr_module, "validate_native_kv_build",
        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        deepseek_ocr_module, "validate_native_kv_weights",
        lambda *args, **kwargs: None)


def _weights() -> WeightDict:
    hidden = 8
    attention = 16
    kv_attention = 16
    inter = 16
    shared = 32
    weights = WeightDict({
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_n_routed_experts": 2,
        "_n_shared_experts": 2,
        "_num_experts_per_tok": 2,
        "_first_k_dense_replace": 1,
        "_moe_intermediate_size": inter,
        "_shared_intermediate_size": shared,
        "_norm_topk_prob": False,
        "_routed_scaling_factor": 1.0,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
    })
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.w_q"] = np.arange(
            hidden * attention, dtype=np.float32).reshape(hidden, attention)
        weights[f"{prefix}.w_k"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_v"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_o"] = np.arange(
            attention * hidden, dtype=np.float32).reshape(attention, hidden)

    for key in ("w_gate", "w_up"):
        weights[f"layer.0.{key}"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
    weights["layer.0.w_down"] = np.arange(
        inter * hidden, dtype=np.float32).reshape(inter, hidden)

    weights["layer.1.router"] = np.zeros((hidden, 2), dtype=np.float32)
    for expert_idx in range(2):
        prefix = f"layer.1.expert.{expert_idx}"
        for key in ("w_gate", "w_up"):
            weights[f"{prefix}.{key}"] = np.arange(
                hidden * inter, dtype=np.float32).reshape(hidden, inter)
        weights[f"{prefix}.w_down"] = np.arange(
            inter * hidden, dtype=np.float32).reshape(inter, hidden)
    for key in ("w_gate", "w_up"):
        weights[f"layer.1.shared.{key}"] = np.arange(
            hidden * shared, dtype=np.float32).reshape(hidden, shared)
    weights["layer.1.shared.w_down"] = np.arange(
        shared * hidden, dtype=np.float32).reshape(shared, hidden)
    return weights


def test_deepseek_ocr_tp_slices_attention_and_moe_weights() -> None:
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    weights = _weights()

    sharded = tp_builder.shard_deepseek_ocr_weights(
        _config(), weights, parallel=parallel)

    np.testing.assert_array_equal(
        sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_k"], weights["layer.0.w_k"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_v"], weights["layer.0.w_v"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_o"], weights["layer.0.w_o"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.0.w_gate"], weights["layer.0.w_gate"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_down"], weights["layer.0.w_down"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.expert.0.w_up"],
        weights["layer.1.expert.0.w_up"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.1.expert.0.w_down"],
        weights["layer.1.expert.0.w_down"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.shared.w_gate"],
        weights["layer.1.shared.w_gate"][:, 16:32])
    assert sharded["_attention_size"] == 8
    assert sharded["_kv_attention_size"] == 8
    assert sharded["_moe_intermediate_size"] == 8
    assert sharded["_shared_intermediate_size"] == 16


def test_deepseek_ocr_tp_validation_rejects_non_divisible_attention_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads"):
        tp_builder._validate_deepseek_ocr_tp(
            _config(num_heads=10),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_deepseek_ocr_plugin_routes_parallel_builds(monkeypatch) -> None:
    _disable_native_contract_validation(monkeypatch)
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"deepseek-ocr-tp-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "require_tensorrt_11_for_tensor_parallel",
        fake_require,
    )
    monkeypatch.setattr(tp_builder, "build_deepseek_ocr_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    plugin = deepseek_ocr_module.DeepSeekOCRPlugin()
    result = plugin.build_engine(
        _config(), _weights(), 4096,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"deepseek-ocr-tp-plan"
    assert calls["require"][0] == parallel
    assert "DeepSeek-OCR tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 4096
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["profile_mode"] == "decode"
    assert "native_kv_cache" not in kwargs
    assert kwargs["precision"] == "bf16"


def test_deepseek_ocr_defaults_to_full_context_bf16_split() -> None:
    plugin = deepseek_ocr_module.DeepSeekOCRPlugin()
    config = _config()
    assert plugin.default_build_precision(config) == "bf16"
    assert plugin.default_max_cache_length(config) == 4096
    assert plugin.supports_split_decoder_roles(config) is True
    assert plugin.runtime_capabilities == {"decoder_kv"}


def test_deepseek_ocr_tp_extra_engines_are_rank_local_prefill(
    monkeypatch,
) -> None:
    _disable_native_contract_validation(monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls.append(kwargs)
        return f"prefill-{kwargs['parallel_config'].rank}".encode()

    monkeypatch.setattr(
        tp_builder, "build_deepseek_ocr_tp_engine", fake_build)
    plans = deepseek_ocr_module.DeepSeekOCRPlugin().build_extra_engines(
        _config(), _weights(), 4096, precision="bf16",
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2))

    assert plans == {
        "prefill_engine_tp_rank0_plan": b"prefill-0",
        "prefill_engine_tp_rank1_plan": b"prefill-1",
    }
    assert all(call["profile_mode"] == "prefill" for call in calls)
    assert all("native_kv_cache" not in call for call in calls)


def test_deepseek_ocr_parallel_build_rejects_debug_outputs(monkeypatch) -> None:
    with pytest.raises(ValueError, match="debug layer outputs"):
        deepseek_ocr_module.DeepSeekOCRPlugin().build_engine(
            _config(),
            _weights(),
            max_cache_length=4096,
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
            debug_layer_outputs=True,
        )


@pytest.mark.parametrize(
    ("requested", "expected"), [("fp16", "fp16"), ("bf16", "fp32")])
def test_deepseek_ocr_forwards_supported_precision_to_vision_engine(
    monkeypatch, requested: str, expected: str,
) -> None:
    calls: dict[str, object] = {}

    def fake_build(model_dir, config, *, precision, verbose):
        calls.update(
            model_dir=model_dir,
            config=config,
            precision=precision,
            verbose=verbose,
        )
        return b"vision-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "_build_deepseek_ocr_vision_engine",
        fake_build,
    )
    config = _config()

    plan = deepseek_ocr_module.DeepSeekOCRPlugin().build_vision_engine(
        "/model", config, _weights(), precision=requested, verbose=True)

    assert plan == b"vision-plan"
    assert calls == {
        "model_dir": "/model",
        "config": config,
        "precision": expected,
        "verbose": True,
    }


def test_deepseek_ocr_vision_mask_keeps_image_attention_bidirectional() -> None:
    mask = deepseek_ocr_module._make_qwen2_vision_attention_mask(3)

    assert mask.shape == (6, 6)
    np.testing.assert_array_equal(mask[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(mask[:3, 3:], np.full((3, 3), -10000.0))
    np.testing.assert_array_equal(
        mask[3], [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0])
    np.testing.assert_array_equal(
        mask[4], [0.0, 0.0, 0.0, 0.0, 0.0, -10000.0])
    np.testing.assert_array_equal(mask[5], np.zeros(6))


def test_deepseek_ocr_vision_mask_rejects_empty_image_sequence() -> None:
    with pytest.raises(ValueError, match="image_tokens must be positive"):
        deepseek_ocr_module._make_qwen2_vision_attention_mask(0)


def test_deepseek_ocr_resizes_sam_position_embedding_for_768_view() -> None:
    position_embedding = np.ones((1, 64, 64, 4), dtype=np.float32)

    resized = deepseek_ocr_module._resize_sam_position_embedding(
        position_embedding, 48)

    assert resized.shape == (1, 48, 48, 4)
    assert resized.dtype == np.float32
    np.testing.assert_allclose(resized, 1.0, atol=1e-6)


def test_deepseek_ocr_uses_official_768_single_view_contract() -> None:
    vl_config = deepseek_ocr_module.DeepSeekOCRPlugin().get_vl_config(_config())

    assert vl_config is not None
    assert vl_config["fixed_image_size"] == 768
    assert vl_config["num_image_pad_tokens"] == 145
    assert vl_config["prefill_max_length"] == 256
    assert vl_config["preprocessor_type"] == "simple_chw"


@pytest.mark.parametrize(
    ("max_cache_length", "expected"),
    [(4096, (64, 256)), (128, (64, 128)), (32, (32, 32))],
)
def test_deepseek_ocr_bounds_sequence_prefill_profile(
    max_cache_length: int, expected: tuple[int, int]
) -> None:
    assert sequence_prefill_profile_lengths(max_cache_length) == expected


def test_deepseek_ocr_native_attention_updates_cache_and_stays_fused() -> None:
    trt = trt_compat.get_trt()
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    head_dim = 128
    hidden_size = 2 * head_dim
    q = network.add_input("q", trt.bfloat16, (2, hidden_size))
    k = network.add_input("k", trt.bfloat16, (2, hidden_size))
    v = network.add_input("v", trt.bfloat16, (2, hidden_size))
    cache_k = network.add_input(
        "cache_k", trt.bfloat16, (1, 2, 16, head_dim))
    cache_v = network.add_input(
        "cache_v", trt.bfloat16, (1, 2, 16, head_dim))
    write_index = network.add_input("write_index", trt.int32, (1,))
    kv_length = network.add_input("kv_length", trt.int32, (1,))

    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network, q, k, v, cache_k, cache_v, write_index, kv_length,
        num_heads=2, num_kv_heads=2, head_dim=head_dim, q_seq=2,
        tag="layer.0.attn")

    assert result["context"].shape == (2, hidden_size)
    assert result["present_k"].shape == (1, 2, 16, head_dim)
    named_layers = {
        network.get_layer(index).name: network.get_layer(index)
        for index in range(network.num_layers)
    }
    assert "layer.0.attn.cache_k_update" in named_layers
    assert "layer.0.attn.cache_v_update" in named_layers
    layer_types = {
        network.get_layer(index).type for index in range(network.num_layers)
    }
    assert trt.LayerType.ATTENTION_INPUT in layer_types
    assert trt.LayerType.ATTENTION_OUTPUT in layer_types
    assert trt.LayerType.SOFTMAX not in layer_types
