# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned decoder tensor-parallel tests for codegen."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    shard_standard_decoder_weights,
)


FAMILY = 'codegen'
MODEL_TYPE = 'codegen'
TP_SIZE = 4
RAW = {'rotary_dim': 2}
EXPECTED_KWARGS = {'fp32_lm_head': True,
 'fp32_qk_attention': True,
 'fp32_rope': True,
 'partial_rotary_factor': 0.5}
SPECIALIZED_SOURCE_MARKERS = (
    "eps_tensor, 'layernorm', work_np_dtype",
    "add_activation(network, fc1, 'gelu_new'",
    "interleaved=True",
    "mlp_out = _gelu_fc_mlp(",
    "sum_attn = network.add_elementwise(hidden_state, attn_out",
)
RETIRED_FIXED_PARAMETERS = {
    "activation",
    "interleaved_rope",
    "mlp_type",
    "norm_type",
    "parallel_residual",
    "position_type",
}


def _config(model_type: str, tp_size: int, raw: dict[str, object]) -> ModelConfig:
    kv_heads = 2 if tp_size == 2 else 4
    return ModelConfig(
        model_type=model_type,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw=raw,
    )


def test_codegen_tp_builder_owns_fixed_family_contract() -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.codegen.dual_profile_decoder_tp_builder"
    )
    parameters = inspect.signature(
        builder_module.build_dual_profile_tp_decoder_engine
    ).parameters
    assert RETIRED_FIXED_PARAMETERS.isdisjoint(parameters)
    source = inspect.getsource(builder_module)
    for marker in SPECIALIZED_SOURCE_MARKERS:
        assert marker in source


def test_codegen_model_routes_tp_build(monkeypatch) -> None:
    model_module = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.model")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_cache_length"] = max_cache_length
        captured["kwargs"] = kwargs
        return b"tp-plan"

    monkeypatch.setattr(
        model_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(
        model_module,
        "build_dual_profile_tp_decoder_engine",
        fake_build,
    )

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=TP_SIZE, rank=1)
    plan = model_module.build_engine(
        _config(MODEL_TYPE, TP_SIZE, RAW),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
        verbose=True,
        parallel_config=parallel,
    )

    assert plan == b"tp-plan"
    assert captured["max_cache_length"] == 17
    kwargs = captured["kwargs"]
    assert kwargs["precision"] == "fp16"
    assert kwargs["quant_ctx"] is None
    assert kwargs["verbose"] is True
    assert kwargs["parallel_config"] == parallel
    for key, expected in EXPECTED_KWARGS.items():
        assert kwargs[key] == expected


def test_codegen_model_routes_accuracy_precision_boundaries(monkeypatch) -> None:
    model_module = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.model")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"single-device-plan"

    monkeypatch.setattr(
        model_module,
        "build_standard_decoder_engine",
        fake_build,
    )

    plan = model_module.build_engine(
        _config(MODEL_TYPE, 1, RAW),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
    )

    assert plan == b"single-device-plan"
    kwargs = captured["kwargs"]
    assert kwargs["fp32_rope"] is True
    assert kwargs["fp32_qk_attention"] is True
    assert kwargs["fp32_lm_head"] is True


def test_codegen_model_rejects_quantized_tp(monkeypatch) -> None:
    model_module = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.model")
    monkeypatch.setattr(
        model_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        model_module.build_engine(
            _config(MODEL_TYPE, TP_SIZE, RAW),
            WeightDict(),
            max_cache_length=17,
            quant_ctx=object(),
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=TP_SIZE, rank=0),
        )


def test_standard_decoder_tp_shards_gelu_fc_input_bias_only() -> None:
    weights = WeightDict({
        "_attention_size": 16,
        "_kv_attention_size": 16,
        "_mlp_size": 32,
        "layer.0.w_fc1": np.zeros((16, 32), dtype=np.float32),
        "layer.0.fc1_bias": np.arange(32, dtype=np.float32),
        "layer.0.w_fc2": np.zeros((32, 16), dtype=np.float32),
        "layer.0.fc2_bias": np.arange(16, dtype=np.float32),
    })

    shard = shard_standard_decoder_weights(
        _config("codegen", 4, {}),
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    np.testing.assert_array_equal(
        shard["layer.0.fc1_bias"],
        np.arange(16, 24, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        shard["layer.0.fc2_bias"],
        weights["layer.0.fc2_bias"],
    )
