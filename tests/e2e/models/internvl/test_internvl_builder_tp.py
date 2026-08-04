# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused routing tests for InternVL native KV and tensor parallelism."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig


class _Config:
    model_type = "internvl"
    raw: dict[str, object]

    def __init__(self, role: str | None = None) -> None:
        self.raw = {}
        if role is not None:
            self.raw["_decoder_engine_role"] = role


def _disable_contract_validation(monkeypatch, plugin_module) -> None:
    monkeypatch.setattr(plugin_module, "validate_native_kv_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(plugin_module, "validate_native_kv_weights", lambda *args, **kwargs: None)


def test_internvl_tp_builder_rejects_single_device_mode() -> None:
    from tensorrt_model_connect.families.internvl import tp_builder

    weights = WeightDict({
        "embedding": np.zeros((4, 4), dtype=np.float32),
        "final_norm": np.ones((4,), dtype=np.float32),
    })
    with pytest.raises(ValueError, match="requires .*tensor_parallel"):
        tp_builder.build_dual_profile_tp_decoder_engine(
            _Config(), weights, max_cache_length=4,
            parallel_config=ParallelConfig())


def test_internvl_parallel_primary_build_is_native_decode(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    from tensorrt_model_connect.families.internvl import tp_builder

    _disable_contract_validation(monkeypatch, plugin_module)
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        plugin_module, "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: calls.setdefault("require", (parallel, feature)))

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = kwargs
        return b"internvl-tp-decode"

    monkeypatch.setattr(tp_builder, "build_dual_profile_tp_decoder_engine", fake_build)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    config = _Config()
    result = plugin_module.plugin.build_engine(
        config, WeightDict(), 32768, parallel_config=parallel, precision="bf16")

    assert result == b"internvl-tp-decode"
    kwargs = calls["build"]
    assert kwargs["precision"] == "bf16"
    assert kwargs["parallel_config"] == parallel
    assert kwargs["profile_mode"] == "decode"
    assert kwargs["embed_input"] is True
    assert config.raw["_native_kv_cache_metadata"]["native_kv_tp_rank_local"] is True


def test_internvl_tp_extra_engines_are_rank_local_prefill(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    from tensorrt_model_connect.families.internvl import tp_builder

    _disable_contract_validation(monkeypatch, plugin_module)
    calls: list[dict[str, object]] = []

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls.append(kwargs)
        return f"prefill-{kwargs['parallel_config'].rank}".encode()

    monkeypatch.setattr(tp_builder, "build_dual_profile_tp_decoder_engine", fake_build)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2)
    plans = plugin_module.plugin.build_extra_engines(
        _Config(), WeightDict(), 32768, precision="bf16", parallel_config=parallel)

    assert plans == {
        "prefill_engine_tp_rank0_plan": b"prefill-0",
        "prefill_engine_tp_rank1_plan": b"prefill-1",
    }
    assert all(call["profile_mode"] == "prefill" for call in calls)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_internvl_single_gpu_routes_each_split_role_to_native_builder(
    monkeypatch, role: str,
) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    _disable_contract_validation(monkeypatch, plugin_module)
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls.update(kwargs)
        return role.encode()

    monkeypatch.setattr(plugin_module, "build_dual_profile_decoder_engine", fake_build)
    config = _Config(role)
    assert plugin_module.plugin.build_engine(
        config, WeightDict(), 32768, precision="bf16") == role.encode()
    assert calls["profile_mode"] == role
    assert calls["embed_input"] is True


def test_internvl_single_gpu_rejects_unsplit_role(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internvl.plugin")
    _disable_contract_validation(monkeypatch, plugin_module)
    with pytest.raises(ValueError, match="explicit split engine role"):
        plugin_module.plugin.build_engine(
            _Config(), WeightDict(), 32768, precision="bf16")
