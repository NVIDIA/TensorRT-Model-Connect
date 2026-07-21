# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for InternVL tensor-parallel text decoder support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.families.internvl.weights import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig


class _Config:
    model_type = "internvl_chat"
    raw = {}


def test_internvl_decoder_dispatches_to_dual_profile_builder(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.model.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"internvl-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = _Config()
    config.raw = {"_decoder_engine_role": "decode"}
    result = module.build_standard_decoder_engine(config, {}, 31, precision="fp16")

    assert result == b"internvl-dual-profile-plan"
    assert calls["build"][3]["profile_mode"] == "dual_profile"


def test_internvl_tp_builder_rejects_single_device_mode() -> None:
    from tensorrt_model_connect.families.internvl.model import parallel

    weights = WeightDict({
        "embedding": np.zeros((4, 4), dtype=np.float32),
        "final_norm": np.ones((4,), dtype=np.float32),
    })

    with pytest.raises(ValueError, match="requires .*tensor_parallel"):
        parallel.build_dual_profile_tp_decoder_engine(
            _Config(),
            weights,
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


def test_internvl_parallel_build_routes_to_tp_builder(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    from tensorrt_model_connect.families.internvl.model import parallel

    calls = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = {
            "config": config,
            "weights": weights,
            "max_cache_length": max_cache_length,
            "kwargs": kwargs,
        }
        return b"internvl-tp-plan"

    monkeypatch.setattr(
        plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(
        parallel, "build_dual_profile_tp_decoder_engine", fake_build)

    config = object()
    weights = WeightDict()
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)

    result = plugin_module.plugin.build_engine(
        config,
        weights,
        max_cache_length=384,
        parallel_config=parallel,
        precision="fp16",
    )

    assert result == b"internvl-tp-plan"
    assert calls["require"] == (parallel, "InternVL tensor-parallel builds")
    assert calls["build"]["config"] is config
    assert calls["build"]["weights"] is weights
    assert calls["build"]["max_cache_length"] == 384

    kwargs = calls["build"]["kwargs"]
    assert kwargs == {
        "parallel_config": parallel,
        "precision": "fp16",
        "quant_ctx": None,
        "verbose": False,
    }


def test_internvl_parallel_build_rejects_debug_outputs(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")

    monkeypatch.setattr(
        plugin_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="debug layer outputs"):
        plugin_module.plugin.build_engine(
            object(),
            WeightDict(),
            max_cache_length=384,
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
            debug_layer_outputs=True,
        )


def test_internvl_non_parallel_build_forwards_precision(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    calls = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = {
            "config": config,
            "weights": weights,
            "max_cache_length": max_cache_length,
            "kwargs": kwargs,
        }
        return b"internvl-plan"

    monkeypatch.setattr(plugin_module, "build_standard_decoder_engine", fake_build)

    config = object()
    weights = WeightDict()
    result = plugin_module.plugin.build_engine(
        config,
        weights,
        max_cache_length=512,
        precision="bf16",
        verbose=True,
    )

    assert result == b"internvl-plan"
    assert calls["build"]["config"] is config
    assert calls["build"]["weights"] is weights
    assert calls["build"]["max_cache_length"] == 512
    kwargs = calls["build"]["kwargs"]
    assert kwargs == {
        "debug_layer_outputs": False,
        "precision": "bf16",
        "quant_ctx": None,
        "verbose": True,
    }
