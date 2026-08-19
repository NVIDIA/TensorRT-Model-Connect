# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for OPT tensor-parallel dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, call

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


def test_opt_plugin_routes_parallel_builds(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.opt.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"opt-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"opt-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs == {
        "precision": "fp32",
        "quant_ctx": None,
        "verbose": True,
        "parallel_config": parallel,
    }


def test_opt_tp_mlp_uses_fixed_relu(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.opt.default_dual_profile_decoder_tp"
    )
    matmul = Mock(side_effect=["fc1", "output"])
    activation = Mock(return_value="activated")
    monkeypatch.setattr(module.graph_ops, "add_activation", activation)
    weights = {
        "layer.0.w_fc1": "fc1-weight",
        "layer.0.w_fc2": "fc2-weight",
    }

    result = module._gelu_fc_mlp(
        "network",
        "input",
        matmul=matmul,
        weights=weights,
        prefix="layer.0",
        hidden=16,
        mlp_size=32,
        work_np_dtype="work-dtype",
    )

    assert result == "output"
    activation.assert_called_once_with(
        "network", "fc1", "relu", dtype="work-dtype"
    )
    assert matmul.call_args_list == [
        call("input", 16, 32, "fc1-weight", "layer.0.w_fc1"),
        call("activated", 32, 16, "fc2-weight", "layer.0.w_fc2"),
    ]
