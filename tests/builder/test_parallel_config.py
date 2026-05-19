"""Tests for tensor-parallel build config and sharding helpers."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    require_tensorrt_11_for_tensor_parallel,
    shard_standard_decoder_weights,
)


def test_parallel_config_accepts_supported_tp_sizes() -> None:
    cfg = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    cfg.validate()
    assert cfg.enabled
    assert cfg.tp_size == 4
    assert cfg.rank == 2


def test_parallel_config_rejects_unsupported_tp_size() -> None:
    with pytest.raises(ValueError, match="parallel.tp_size"):
        ParallelConfig(mode="tensor_parallel", tp_size=3).validate()


def test_standard_decoder_weight_sharding_preserves_single_device() -> None:
    cfg = ModelConfig.create_tiny("qwen3")
    weights = {"_mlp_size": 32, "_attention_size": 16, "_kv_attention_size": 16}

    out = shard_standard_decoder_weights(cfg, weights, ParallelConfig())

    assert out is weights


def test_tensor_parallel_requires_trt11(monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "10.13.3")

    with pytest.raises(RuntimeError, match="TensorRT 11\\.0\\+"):
        require_tensorrt_11_for_tensor_parallel(
            ParallelConfig(mode="tensor_parallel", tp_size=2))


def test_tensor_parallel_trt11_guard_ignores_single_device(monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "10.13.3")

    require_tensorrt_11_for_tensor_parallel(ParallelConfig())
