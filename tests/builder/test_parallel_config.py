"""Tests for tensor-parallel build config and sharding helpers."""

from __future__ import annotations

import pytest
import numpy as np

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    rank_denoiser_section,
    require_tensorrt_11_for_tensor_parallel,
    shard_standard_decoder_weights,
    validate_dit_tp,
    validate_flux_dit_tp,
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


def test_standard_decoder_weight_sharding_slices_gelu_fc_bias() -> None:
    cfg = ModelConfig.create_tiny("opt")
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 32
    weights = {
        "_attention_size": 16,
        "_kv_attention_size": 16,
        "_mlp_size": 32,
        "layer.0.w_fc1": np.zeros((16, 32)),
        "layer.0.fc1_bias": np.arange(32),
        "layer.0.w_fc2": np.zeros((32, 16)),
        "layer.0.fc2_bias": np.arange(16),
    }

    out = shard_standard_decoder_weights(
        cfg, weights, ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2))

    np.testing.assert_array_equal(
        out["layer.0.fc1_bias"], weights["layer.0.fc1_bias"][16:24])
    np.testing.assert_array_equal(
        out["layer.0.fc2_bias"], weights["layer.0.fc2_bias"])
    assert out["_mlp_size"] == 8


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


def test_rank_denoiser_section_names_flux_tp_sections() -> None:
    assert rank_denoiser_section(3) == "denoiser_plan_tp_rank3"


def test_flux_dit_tp_validation_requires_concrete_rank() -> None:
    with pytest.raises(ValueError, match="concrete rank"):
        validate_flux_dit_tp(
            dim=3072,
            num_heads=24,
            ffn_dim=12288,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4),
        )


def test_flux_dit_tp_validation_rejects_undivisible_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads divisible"):
        validate_flux_dit_tp(
            dim=3072,
            num_heads=22,
            ffn_dim=12288,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_generic_dit_tp_validation_allows_pixart_tp4() -> None:
    validate_dit_tp(
        dim=1152,
        num_heads=16,
        ffn_dim=4608,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        feature="PixArt tensor parallel",
    )


def test_generic_dit_tp_validation_rejects_zimage_tp4_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads divisible"):
        validate_dit_tp(
            dim=3840,
            num_heads=30,
            ffn_dim=10240,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
            feature="Z-Image tensor parallel",
        )
