# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for OLMo tensor-parallel dispatch."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        attention_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_olmo_plugin_routes_parallel_builds(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.olmo.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"olmo-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"olmo-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs == {
        "precision": "fp32",
        "quant_ctx": None,
        "verbose": True,
        "parallel_config": parallel,
    }


def test_olmo_builders_use_fixed_layernorm_specialization() -> None:
    modules_and_builders = (
        (
            "tensorrt_model_connect.models.olmo.default_decoder",
            "build_standard_decoder_engine",
            "graph_ops.add_layer_norm_native(",
            None,
        ),
        (
            "tensorrt_model_connect.models.olmo.default_dual_profile_decoder",
            "build_dual_profile_decoder_engine",
            "_norm_multi(",
            "'layernorm'",
        ),
        (
            "tensorrt_model_connect.models.olmo.dual_profile_decoder_tp_builder",
            "build_dual_profile_tp_decoder_engine",
            "_norm_multi(",
            "'layernorm'",
        ),
    )

    for module_name, builder_name, layernorm_call, fixed_argument in modules_and_builders:
        builder = getattr(importlib.import_module(module_name), builder_name)
        assert "norm_type" not in inspect.signature(builder).parameters
        source = inspect.getsource(builder)
        assert layernorm_call in source
        if fixed_argument is not None:
            assert fixed_argument in source
        assert "'rmsnorm'" not in source
