"""Focused tests for InternVL tensor-parallel text decoder support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig


class _Config:
    model_type = "internvl_chat"


def test_internvl_tp_builder_rejects_single_device_mode() -> None:
    from tensorrt_model_connect.families.internvl import tp_builder

    weights = WeightDict({
        "embedding": np.zeros((4, 4), dtype=np.float32),
        "final_norm": np.ones((4,), dtype=np.float32),
    })

    with pytest.raises(ValueError, match="requires .*tensor_parallel"):
        tp_builder.build_dual_profile_tp_decoder_engine(
            _Config(),
            weights,
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


def test_internvl_parallel_build_routes_to_tp_builder(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    from tensorrt_model_connect.families.internvl import tp_builder

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
        tp_builder, "build_dual_profile_tp_decoder_engine", fake_build)

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
    assert kwargs["precision"] == "fp16"
    assert kwargs["parallel_config"] == parallel
    assert kwargs["embed_input"] is True
    assert kwargs["norm_type"] == "rmsnorm"
    assert kwargs["mlp_type"] == "swiglu"
    assert kwargs["position_type"] == "rope"
    assert kwargs["activation"] == "silu"


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
