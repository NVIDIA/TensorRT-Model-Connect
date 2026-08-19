# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SAM tensor-parallel encoder support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    sam_plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.sam.model")
    from tensorrt_model_connect.models.sam import sam_tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _raw_config(mlp_dim: int = 64) -> dict:
    return {
        "model_type": "sam",
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 64,
            "patch_size": 16,
            "mlp_dim": mlp_dim,
            "window_size": 2,
            "global_attn_indexes": [1],
        },
        "prompt_encoder_config": {
            "hidden_size": 16,
            "image_embedding_size": 4,
        },
        "mask_decoder_config": {
            "hidden_size": 16,
            "num_multimask_outputs": 3,
            "num_attention_heads": 4,
            "depth": 2,
        },
    }


def _model_config(mlp_dim: int = 64) -> SimpleNamespace:
    return SimpleNamespace(raw=_raw_config(mlp_dim))


def test_sam_tp_slices_encoder_mlp_columns_and_rows():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    columns = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    rows = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)

    np.testing.assert_array_equal(
        sam_tp_builder._slice_mlp_columns(columns, 64, parallel),
        columns[:, 32:48],
    )
    np.testing.assert_array_equal(
        sam_tp_builder._slice_mlp_rows(rows, 64, parallel),
        rows[32:48, :],
    )


def test_sam_tp_validation_rejects_non_divisible_mlp_dim():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0)
    with pytest.raises(ValueError, match="mlp_dim divisible"):
        sam_tp_builder._validate_sam_encoder_tp(_model_config(66), parallel)


def test_sam_tp_validation_requires_concrete_rank():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1)
    with pytest.raises(ValueError, match="concrete rank"):
        sam_tp_builder._validate_sam_encoder_tp(_model_config(), parallel)


def test_sam_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"sam-tp-plan"

    monkeypatch.setattr(
        sam_plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(sam_tp_builder, "build_sam_tp_encoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = sam_plugin_module.build_engine(
        _model_config(), {"encoder.layer0.mlp.fc1.weight": np.zeros((32, 64))}, 7,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"sam-tp-plan"
    assert calls["require"][0] == parallel
    assert "SAM encoder tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 7
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True


def test_sam_plugin_rejects_parallel_quantization(monkeypatch):
    monkeypatch.setattr(
        sam_plugin_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="quantization"):
        sam_plugin_module.build_engine(
            _model_config(), {}, 1,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
