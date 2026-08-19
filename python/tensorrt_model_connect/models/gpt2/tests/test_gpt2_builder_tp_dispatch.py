# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned tensor-parallel dispatch tests for gpt2."""

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
        raw={"rotary_pct": 0.25, "use_parallel_residual": True},
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


def test_gpt2_plugin_routes_parallel_builds(monkeypatch) -> None:
    module = importlib.import_module("tensorrt_model_connect.models.gpt2.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"gpt-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"gpt-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs == {
        "precision": "fp32",
        "quant_ctx": None,
        "verbose": True,
        "parallel_config": parallel,
    }


def test_gpt2_tp_builder_keeps_fixed_gelu_new_fc_mlp() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.gpt2.default_dual_profile_decoder_tp"
    )

    assert "mlp_type" not in inspect.signature(
        module.build_dual_profile_tp_decoder_engine
    ).parameters
    source = inspect.getsource(module._gelu_fc_mlp)
    assert ".w_fc1" in source and ".w_fc2" in source
    assert "add_activation" in source and "'gelu_new'" in source
