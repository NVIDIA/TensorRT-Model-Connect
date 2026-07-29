# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor-parallel build config and sharding helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    context_denoiser_section,
    rank_denoiser_section,
    require_tensorrt_11_for_distributed,
    require_tensorrt_11_for_tensor_parallel,
    shard_standard_decoder_weights,
    validate_dit_tp,
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


def test_parallel_config_accepts_context_parallel_cp4() -> None:
    cfg = ParallelConfig(mode="context_parallel", cp_size=4, rank=2)

    cfg.validate()

    assert not cfg.enabled
    assert cfg.cp_enabled
    assert cfg.distributed
    assert cfg.world_size == 4
    assert cfg.to_bundle_config_fields() == {
        "parallelism": {
            "mode": "context_parallel",
            "tp_size": 1,
            "cp_size": 4,
            "rank": 2,
            "require_mpirun": True,
        },
        "parallel_mode": "context_parallel",
        "context_parallel_mode": "context_parallel",
        "context_parallel_size": 4,
        "context_parallel_require_mpirun": 1,
    }


def test_parallel_config_rejects_mixed_tp_and_cp() -> None:
    with pytest.raises(ValueError, match="requires parallel.tp_size=1"):
        ParallelConfig(
            mode="context_parallel", tp_size=2, cp_size=4).validate()


def test_standard_decoder_weight_sharding_preserves_single_device() -> None:
    cfg = ModelConfig.create_tiny("standard_decoder")
    weights = {"_mlp_size": 32, "_attention_size": 16, "_kv_attention_size": 16}

    out = shard_standard_decoder_weights(cfg, weights, ParallelConfig())

    assert out is weights


def test_standard_decoder_weight_sharding_slices_gelu_fc_bias() -> None:
    cfg = ModelConfig.create_tiny("gelu_decoder")
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


def test_context_parallel_requires_trt11(monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "10.13.3")

    with pytest.raises(RuntimeError, match="TensorRT 11\\.0\\+"):
        require_tensorrt_11_for_distributed(
            ParallelConfig(mode="context_parallel", cp_size=4))


def test_rank_denoiser_section_names_tp_sections() -> None:
    assert rank_denoiser_section(3) == "denoiser_plan_tp_rank3"


def test_context_denoiser_section_names_shared_cp_plan() -> None:
    assert context_denoiser_section() == "denoiser_plan_cp"


def test_dit_tp_validation_requires_concrete_rank() -> None:
    with pytest.raises(ValueError, match="concrete rank"):
        validate_dit_tp(
            dim=3072,
            num_heads=24,
            ffn_dim=12288,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4),
        )


def test_dit_tp_validation_rejects_undivisible_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads divisible"):
        validate_dit_tp(
            dim=3072,
            num_heads=22,
            ffn_dim=12288,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_generic_dit_tp_validation_allows_tp4() -> None:
    validate_dit_tp(
        dim=1152,
        num_heads=16,
        ffn_dim=4608,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
    )


def test_generic_dit_tp_validation_rejects_tp4_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads divisible"):
        validate_dit_tp(
            dim=3840,
            num_heads=30,
            ffn_dim=10240,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
