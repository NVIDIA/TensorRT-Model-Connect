# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Falcon tensor-parallel dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(*, alibi: bool) -> SimpleNamespace:
    return SimpleNamespace(
        raw={"alibi": alibi},
        hidden_size=16,
        hidden_act="gelu",
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


@pytest.mark.parametrize(
    ("alibi", "position_type", "alibi_bias_scale"),
    [
        (False, "rope", 1.0),
        (True, "alibi", 0.5),
    ],
)
def test_falcon_plugin_routes_parallel_builds(
    monkeypatch, alibi: bool, position_type: str, alibi_bias_scale: float
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.falcon.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"falcon-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.build_engine(
        _config(alibi=alibi),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"falcon-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["position_type"] == position_type
    assert kwargs["alibi_bias_scale"] == pytest.approx(alibi_bias_scale)
    assert kwargs["activation"] == "gelu"
    assert kwargs["mlp_type"] == "gelu_fc"
    assert kwargs["verbose"] is True
