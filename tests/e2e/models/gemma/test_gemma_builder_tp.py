# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Gemma tensor-parallel dispatch."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
        },
        hidden_act="gelu_pytorch_tanh",
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=26,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        attention_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_gemma_plugin_routes_parallel_builds(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.gemma.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"gemma-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.GemmaPlugin().build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"gemma-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["activation"] == "gelu_pytorch_tanh"


def test_gemma_plugin_forwards_checkpoint_activation(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.gemma.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"gemma-plan"

    monkeypatch.setattr(module, "build_standard_decoder_engine", fake_build)

    result = module.GemmaPlugin().build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
    )

    assert result == b"gemma-plan"
    config, _, _, kwargs = calls["build"]
    assert config.num_hidden_layers == 26
    assert kwargs["activation"] == "gelu_pytorch_tanh"


class _FakeLayer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index):
        assert index == 0
        return self._output


class _FakeNetwork:
    def __init__(self, trt):
        self._trt = trt

    def add_elementwise(self, lhs, rhs, operation):
        if operation == self._trt.ElementWiseOperation.PROD:
            return _FakeLayer(lhs * rhs)
        if operation == self._trt.ElementWiseOperation.SUM:
            return _FakeLayer(lhs + rhs)
        raise AssertionError(f"unexpected elementwise operation: {operation}")

    def add_activation(self, tensor, activation):
        assert activation == self._trt.ActivationType.TANH
        return _FakeLayer(np.tanh(tensor))


def test_gemma_dual_profile_mlp_uses_checkpoint_gelu(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.gemma.dual_profile_decoder_builder")
    gate = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    up = np.ones_like(gate)

    def fake_matmul(inp, lhs_width, rhs_width, weights, name):
        del lhs_width, rhs_width, weights
        if name.endswith("w_gate"):
            return gate
        if name.endswith("w_up"):
            return up
        return inp

    def fake_constant(network, shape, values, dtype):
        del network
        return np.asarray(values, dtype=dtype).reshape(shape)

    monkeypatch.setattr(module.graph_ops, "add_constant", fake_constant)
    monkeypatch.setattr(
        module.graph_ops,
        "_cast_back_to_trt_dtype",
        lambda network, tensor, dtype: tensor,
    )

    actual = module._swiglu_mlp(
        _FakeNetwork(module.trt),
        np.zeros_like(gate),
        matmul=fake_matmul,
        weights={
            "layer.0.w_gate": None,
            "layer.0.w_up": None,
            "layer.0.w_down": None,
        },
        prefix="layer.0",
        hidden=4,
        mlp_size=4,
        activation="gelu_pytorch_tanh",
        work_np_dtype=np.float32,
    )
    expected = 0.5 * gate * (
        1.0
        + np.tanh(
            math.sqrt(2.0 / math.pi)
            * (gate + 0.044715 * np.power(gate, 3))
        )
    ) * up
    silu = gate / (1.0 + np.exp(-gate))

    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0.0)
    assert not np.allclose(actual, silu, atol=1e-3, rtol=0.0)


def test_gemma_rejects_missing_checkpoint_activation() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.gemma.plugin")
    config = _config()
    config.hidden_act = ""
    config.raw = {}

    with pytest.raises(
        ValueError,
        match="requires a supported checkpoint gated activation",
    ):
        module.GemmaPlugin().build_engine(config, {}, 23)
