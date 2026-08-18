# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt", reason="Nemotron Speech Streaming TP builder tests require TensorRT"
)

from tensorrt_model_connect.parallel_config import ParallelConfig
from tensorrt_model_connect.families.nemotron_speech_streaming.predictor_tp_builder import (
    _slice_lstm_gate_columns,
    _validate_predictor_tp,
)

nss = importlib.import_module("tensorrt_model_connect.families.nemotron_speech_streaming.model")


def _weights(hidden: int = 8) -> dict:
    weights = {
        "_pred_hidden": hidden,
        "_pred_layers": 1,
        "_vocab_total": 16,
        "pred_embedding": np.zeros((16, hidden), dtype=np.float32),
        "pred.0.w_ih_t": np.zeros((hidden, 4 * hidden), dtype=np.float32),
        "pred.0.w_hh_t": np.zeros((hidden, 4 * hidden), dtype=np.float32),
        "pred.0.bias": np.zeros((1, 4 * hidden), dtype=np.float32),
    }
    return weights


def test_slice_lstm_gate_columns_preserves_ifgo_gate_order() -> None:
    hidden = 8
    arr = np.arange(hidden * 4 * hidden, dtype=np.float32).reshape(hidden, 4 * hidden)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)

    sliced = _slice_lstm_gate_columns(arr, hidden, parallel)

    expected = np.concatenate(
        [
            arr[:, 4:6],
            arr[:, 12:14],
            arr[:, 20:22],
            arr[:, 28:30],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(sliced, expected)
    assert sliced.flags.c_contiguous


def test_validate_predictor_tp_rejects_non_divisible_hidden() -> None:
    weights = _weights(hidden=10)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0)

    with pytest.raises(ValueError, match="pred_hidden divisible"):
        _validate_predictor_tp(weights, parallel)


def test_plugin_routes_parallel_builds_to_tp_predictor(monkeypatch) -> None:
    calls = {}

    def fake_require(parallel, *, feature):
        calls["feature"] = feature
        assert parallel.tp_size == 4

    def fake_build(weights, *, verbose, parallel_config):
        calls["rank"] = parallel_config.rank
        calls["weights"] = weights
        calls["verbose"] = verbose
        return b"tp-predictor"

    monkeypatch.setattr(nss, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(nss, "build_nemotron_streaming_tp_predictor", fake_build)

    out = nss.build_engine(
        SimpleNamespace(),
        _weights(),
        128,
        verbose=True,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=3),
    )

    assert out == b"tp-predictor"
    assert calls["feature"] == "Nemotron Speech Streaming TP predictor builds"
    assert calls["rank"] == 3
    assert calls["weights"]["_pred_hidden"] == 8
    assert calls["verbose"] is True


def test_plugin_rejects_tp_quantization(monkeypatch) -> None:
    monkeypatch.setattr(nss, "require_tensorrt_11_for_tensor_parallel", lambda *_, **__: None)

    with pytest.raises(ValueError, match="do not support quantization"):
        nss.build_engine(
            SimpleNamespace(),
            _weights(),
            128,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
