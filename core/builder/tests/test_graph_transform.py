# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextvars import Context
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.graph_transform import graph_transform


class _Builder:
    calls: list[tuple[object, object]] = []

    def __init__(self, logger: object) -> None:
        self.logger = logger

    def create_network(self) -> object:
        return SimpleNamespace(name="network")

    def build_serialized_network(self, network: object, config: object) -> bytes:
        self.calls.append((network, config))
        return b"engine"


def test_transform_runs_before_each_serialization_and_builder_still_delegates(
    monkeypatch,
) -> None:
    _Builder.calls = []
    fake_trt = SimpleNamespace(Builder=_Builder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    events: list[tuple[object, ...]] = []

    def transform(network: object, engine_index: int) -> None:
        events.append(("transform", network, engine_index, len(_Builder.calls)))

    with graph_transform(transform):
        first_builder = fake_trt.Builder("logger")
        first_network = first_builder.create_network()
        assert first_builder.logger == "logger"
        assert first_builder.build_serialized_network(first_network, "config-0") == b"engine"

        second_builder = fake_trt.Builder("logger")
        second_network = second_builder.create_network()
        assert second_builder.build_serialized_network(second_network, "config-1") == b"engine"

    assert events == [
        ("transform", first_network, 0, 0),
        ("transform", second_network, 1, 1),
    ]
    assert _Builder.calls == [
        (first_network, "config-0"),
        (second_network, "config-1"),
    ]
    assert fake_trt.Builder is _Builder


def test_transform_failure_prevents_serialization_and_restores_builder(monkeypatch) -> None:
    _Builder.calls = []
    fake_trt = SimpleNamespace(Builder=_Builder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    def fail(_network: object, _engine_index: int) -> None:
        raise ValueError("invalid replacement")

    with pytest.raises(ValueError, match="invalid replacement"):
        with graph_transform(fail):
            fake_trt.Builder("logger").build_serialized_network("network", "config")

    assert _Builder.calls == []
    assert fake_trt.Builder is _Builder


def test_transform_can_replace_a_selected_region_by_rewiring_its_consumer(
    monkeypatch,
) -> None:
    class Tensor:
        def __init__(self, name: str) -> None:
            self.name = name

    class Layer:
        def __init__(self, input_tensor: Tensor, output_tensor: Tensor) -> None:
            self.input = input_tensor
            self.output = output_tensor

        def get_input(self, _index: int) -> Tensor:
            return self.input

        def get_output(self, _index: int) -> Tensor:
            return self.output

        def set_input(self, _index: int, tensor: Tensor) -> None:
            self.input = tensor

    class Network:
        def __init__(self) -> None:
            source = Tensor("source")
            selected = Tensor("selected-output")
            final = Tensor("final")
            self.layers = [Layer(source, selected), Layer(selected, final)]

        def get_layer(self, index: int) -> Layer:
            return self.layers[index]

    class Builder:
        def __init__(self, _logger: object) -> None:
            pass

        def build_serialized_network(self, network: Network, _config: object) -> bytes:
            return network.get_layer(1).get_input(0).name.encode()

    fake_trt = SimpleNamespace(Builder=Builder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    network = Network()

    def replace(live_network: Network, engine_index: int) -> None:
        assert engine_index == 0
        consumer = live_network.get_layer(1)
        consumer.set_input(0, Tensor("replacement-output"))

    with graph_transform(replace):
        plan = fake_trt.Builder("logger").build_serialized_network(network, "config")

    assert plan == b"replacement-output"


def test_graph_transform_cannot_be_nested(monkeypatch) -> None:
    fake_trt = SimpleNamespace(Builder=_Builder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    with graph_transform(lambda network, engine_index: None):
        with pytest.raises(RuntimeError, match="cannot run concurrently"):
            with graph_transform(lambda network, engine_index: None):
                pass


def test_transform_does_not_leak_into_an_unrelated_context(monkeypatch) -> None:
    _Builder.calls = []
    fake_trt = SimpleNamespace(Builder=_Builder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    transformed: list[object] = []

    with graph_transform(lambda network, engine_index: transformed.append(network)):
        plan = Context().run(
            lambda: fake_trt.Builder("logger").build_serialized_network(
                "other-network", "config"
            )
        )

    assert plan == b"engine"
    assert transformed == []
