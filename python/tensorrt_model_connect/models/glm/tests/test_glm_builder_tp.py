# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned decoder tensor-parallel tests for glm."""

from __future__ import annotations

import importlib
import inspect

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
)


FAMILY = 'glm'
MODEL_TYPE = 'glm'
TP_SIZE = 2
RAW = {'partial_rotary_factor': 0.5}
EXPECTED_KWARGS = {'partial_rotary_factor': 0.5}
SPECIALIZED_DEFAULTS = {
    "activation": "silu",
    "mlp_type": "swiglu",
    "norm_type": "rmsnorm",
    "position_type": "rope",
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


def test_glm_tp_builder_owns_fixed_family_contract() -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.models.glm.dual_profile_decoder_tp_builder"
    )
    parameters = inspect.signature(
        builder_module.build_dual_profile_tp_decoder_engine
    ).parameters
    for name, expected in SPECIALIZED_DEFAULTS.items():
        assert parameters[name].default == expected
    assert "interleaved=True" in inspect.getsource(builder_module)


def test_glm_model_routes_tp_build(monkeypatch) -> None:
    model_module = importlib.import_module(
        f"tensorrt_model_connect.models.{FAMILY}.model")
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


def test_glm_model_rejects_quantized_tp(monkeypatch) -> None:
    model_module = importlib.import_module(
        f"tensorrt_model_connect.models.{FAMILY}.model")
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
