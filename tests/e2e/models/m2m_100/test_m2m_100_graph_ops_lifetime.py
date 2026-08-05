# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for M2M-100 TensorRT constant buffer lifetime."""

from __future__ import annotations

import gc
import importlib
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from tensorrt_model_connect.families.m2m_100 import graph_ops


plugin = importlib.import_module("tensorrt_model_connect.families.m2m_100.plugin")


class _FakeLayer:
    def get_output(self, _index: int) -> object:
        return object()


class _FakeNetwork:
    def __init__(self) -> None:
        self.weights: list[object] = []

    def add_constant(self, _shape: tuple[int, ...], weights: object) -> _FakeLayer:
        self.weights.append(weights)
        return _FakeLayer()


def test_fp16_constant_buffer_lives_until_engine_build_finishes(monkeypatch) -> None:
    buffer_refs: list[weakref.ReferenceType[np.ndarray]] = []

    class _WeakWeights:
        def __init__(self, values: np.ndarray) -> None:
            buffer_refs.append(weakref.ref(values))

    monkeypatch.setattr(
        graph_ops,
        "trt",
        SimpleNamespace(Weights=_WeakWeights),
    )
    network = _FakeNetwork()
    values = np.arange(4, dtype=np.float32)

    @graph_ops.retain_constant_buffers
    def _build_engine() -> bytes:
        graph_ops.add_constant(network, (4,), values, dtype=np.float16)
        gc.collect()
        assert buffer_refs[0]() is not None
        return b"plan"

    assert _build_engine() == b"plan"

    gc.collect()

    assert buffer_refs[0]() is None


def test_constant_buffer_is_released_when_engine_build_raises(monkeypatch) -> None:
    buffer_refs: list[weakref.ReferenceType[np.ndarray]] = []

    class _WeakWeights:
        def __init__(self, values: np.ndarray) -> None:
            buffer_refs.append(weakref.ref(values))

    monkeypatch.setattr(
        graph_ops,
        "trt",
        SimpleNamespace(Weights=_WeakWeights),
    )
    network = _FakeNetwork()
    values = np.arange(4, dtype=np.float32)

    @graph_ops.retain_constant_buffers
    def _build_engine() -> None:
        graph_ops.add_constant(network, (4,), values, dtype=np.float16)
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        _build_engine()

    gc.collect()

    assert buffer_refs[0]() is None


def test_process_logger_is_reused_and_retained(monkeypatch) -> None:
    created: list[object] = []

    class _FakeLogger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, severity: int) -> None:
            self.severity = severity
            created.append(self)

    monkeypatch.setattr(plugin, "trt", SimpleNamespace(Logger=_FakeLogger))
    monkeypatch.setattr(plugin, "_PROCESS_LOGGERS", {})

    first = plugin._get_process_logger(verbose=False)
    logger_ref = weakref.ref(first)
    del first
    gc.collect()

    second = plugin._get_process_logger(verbose=True)
    third = plugin._get_process_logger(verbose=False)

    assert logger_ref() is third
    assert created == [third, second]
    assert third.severity == _FakeLogger.WARNING
    assert second.severity == _FakeLogger.VERBOSE
