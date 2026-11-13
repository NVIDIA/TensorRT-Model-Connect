# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock
import weakref

import numpy as np

from families.musicgen.graph import Graph


def test_lstm_trip_count_remains_live_until_build(monkeypatch):
    # Model TensorRT's non-owning weights without retaining the array in the mock.
    graph = Graph.__new__(Graph)
    graph.net = MagicMock()
    graph.weights = []
    graph.linear = MagicMock()
    graph.op = MagicMock()
    constants = []

    def constant(shape, weights):
        tensor = SimpleNamespace(shape=shape, weights=weights)
        constants.append(tensor)
        return SimpleNamespace(get_output=lambda index: tensor)

    monkeypatch.setattr("families.musicgen.graph.trt.Weights", weakref.ref)
    graph.net.add_constant.side_effect = constant
    weights = np.zeros((8, 2), dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    graph.lstm(SimpleNamespace(shape=(1, 3, 2)), weights, weights, bias, bias)
    gc.collect()
    count = constants[0]
    assert count.shape == ()
    assert count.weights() is not None
    assert count.weights().dtype == np.int32
    assert count.weights().item() == 3
    assert any(value is count.weights() for value in graph.weights)
    loop = graph.net.add_loop.return_value
    assert loop.add_trip_limit.call_args.args[0] is count
    loop.add_loop_output.return_value.set_input.assert_called_once_with(1, count)
