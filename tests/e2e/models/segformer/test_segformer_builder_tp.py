# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel tests for SegFormer Mix-FFN support."""

from __future__ import annotations

from types import SimpleNamespace
import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.parallel_config import ParallelConfig
from tensorrt_model_connect.families.segformer.segformer_tp_builder import (
    _slice_mlp_columns,
    _slice_mlp_rows,
    _validate_segformer_tp,
)

segformer_plugin = importlib.import_module(
    "tensorrt_model_connect.families.segformer.model")


def _config(*, hidden_sizes=None, mlp_ratios=None):
    return SimpleNamespace(raw={
        "hidden_sizes": hidden_sizes or [32, 64, 160, 256],
        "mlp_ratios": mlp_ratios or [4, 4, 4, 4],
    })


def test_slice_mlp_columns_uses_rank_local_ffn_range() -> None:
    arr = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)

    sliced = _slice_mlp_columns(arr, 16, parallel)

    np.testing.assert_array_equal(sliced, arr[:, 8:12])
    assert sliced.flags.c_contiguous


def test_slice_mlp_rows_uses_rank_local_ffn_range() -> None:
    arr = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)

    sliced = _slice_mlp_rows(arr, 16, parallel)

    np.testing.assert_array_equal(sliced, arr[4:8, :])
    assert sliced.flags.c_contiguous


def test_validate_segformer_tp_rejects_non_divisible_ffn() -> None:
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0)

    with pytest.raises(ValueError, match="FFN width divisible"):
        _validate_segformer_tp(
            _config(hidden_sizes=[30, 64, 160, 256], mlp_ratios=[3, 4, 4, 4]),
            parallel,
        )


def test_plugin_routes_parallel_builds_to_tp_builder(monkeypatch) -> None:
    calls = {}

    def fake_require(parallel, *, feature):
        calls["feature"] = feature
        assert parallel.tp_size == 4

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["config"] = config
        calls["weights"] = weights
        calls["max_cache_length"] = max_cache_length
        calls["kwargs"] = kwargs
        return b"segformer-tp"

    monkeypatch.setattr(
        segformer_plugin, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(segformer_plugin, "build_segformer_tp_engine", fake_build)

    out = segformer_plugin.build_engine(
        _config(),
        {"dummy": np.zeros(1, dtype=np.float32)},
        max_cache_length=1,
        debug_layer_outputs=True,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=3),
    )

    assert out == b"segformer-tp"
    assert calls["feature"] == "SegFormer tensor-parallel Mix-FFN builds"
    assert calls["weights"]["dummy"].shape == (1,)
    assert calls["max_cache_length"] == 1
    assert calls["kwargs"]["debug_layer_outputs"] is True
    assert calls["kwargs"]["parallel_config"].rank == 3


def test_plugin_rejects_quantized_tp(monkeypatch) -> None:
    monkeypatch.setattr(
        segformer_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda *_, **__: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        segformer_plugin.build_engine(
            _config(),
            {},
            max_cache_length=1,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
