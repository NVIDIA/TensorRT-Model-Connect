# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for MoGe's exact selective zero-padding optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from families.moge import model


EXPECTED_ZERO_PAD_SELECTION = frozenset(
    {
        "mask_head.res_blocks.1.0.layers.5",
        "mask_head.res_blocks.2.0.layers.2",
        "mask_head.res_blocks.2.0.layers.5",
        "mask_head.res_blocks.3.0.layers.2",
        "mask_head.res_blocks.3.0.layers.5",
        "mask_head.resamplers.1.1",
        "mask_head.resamplers.2.1",
        "neck.res_blocks.2.0.layers.2",
        "neck.res_blocks.2.0.layers.5",
        "neck.res_blocks.2.1.layers.2",
        "neck.res_blocks.2.1.layers.5",
        "neck.res_blocks.3.0.layers.2",
        "neck.res_blocks.3.0.layers.5",
        "neck.res_blocks.3.1.layers.2",
        "neck.res_blocks.3.1.layers.5",
        "points_head.res_blocks.1.0.layers.2",
        "points_head.res_blocks.1.0.layers.5",
        "points_head.res_blocks.2.0.layers.2",
        "points_head.res_blocks.2.0.layers.5",
        "points_head.res_blocks.3.0.layers.2",
        "points_head.res_blocks.3.0.layers.5",
        "points_head.resamplers.0.1",
        "points_head.resamplers.1.1",
        "points_head.resamplers.2.1",
    }
)


@dataclass
class _Tensor:
    dtype: Any
    shape: tuple[int, ...]
    label: str


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self.output = output
        self.name = ""
        self.stride_nd = None
        self.padding_nd = None

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self.output


class _Network:
    def __init__(self) -> None:
        self.calls = []

    def add_convolution_nd(self, tensor, output_channels, kernel_shape, weight, bias):
        layer = _Layer(_Tensor(tensor.dtype, (1, output_channels, 8, 8), "output"))
        self.calls.append((tensor, kernel_shape, layer))
        return layer


class _Trt:
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def Weights(array: np.ndarray) -> np.ndarray:
        return array


def _graph(monkeypatch: pytest.MonkeyPatch) -> tuple[model._NativeMogeGraph, _Network, list]:
    network = _Network()
    graph = model._NativeMogeGraph(_Trt, network, {}, fast_path=True)
    weight = np.zeros((4, 4, 3, 3), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)
    monkeypatch.setattr(
        graph,
        "_array",
        lambda name, expected=None: weight if name.endswith(".weight") else bias,
    )
    monkeypatch.setattr(graph, "cast", lambda tensor, dtype, name: tensor)
    pad_calls = []

    def replicate_pad(tensor: _Tensor, padding: int, name: str) -> _Tensor:
        pad_calls.append((padding, name))
        return _Tensor(tensor.dtype, (1, 4, 10, 10), "replicate-padded")

    monkeypatch.setattr(graph, "replicate_pad", replicate_pad)
    return graph, network, pad_calls


def test_zero_pad_selection_is_exact_and_family_owned() -> None:
    assert model._ZERO_PAD_SELECTION == EXPECTED_ZERO_PAD_SELECTION
    assert len(model._ZERO_PAD_SELECTION) == 24


def test_only_selected_modules_use_native_zero_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, network, pad_calls = _graph(monkeypatch)
    selected = next(iter(EXPECTED_ZERO_PAD_SELECTION))
    unselected = "neck.resamplers.0.1"
    source = _Tensor(_Trt.float32, (1, 4, 8, 8), "source")

    graph.convolution(source, selected, "selected", replicate_padding=1)
    assert pad_calls == []
    assert network.calls[-1][0] is source
    assert network.calls[-1][2].padding_nd == (1, 1)

    graph.convolution(source, unselected, "unselected", replicate_padding=1)
    assert pad_calls == [(1, "unselected.pad")]
    assert network.calls[-1][0].label == "replicate-padded"
    assert network.calls[-1][2].padding_nd == (0, 0)


def test_selected_module_rejects_an_unexpected_convolution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, _, _ = _graph(monkeypatch)
    source = _Tensor(_Trt.float32, (1, 4, 8, 8), "source")
    with pytest.raises(ValueError, match="must be stride-1 3x3"):
        graph.convolution(
            source,
            next(iter(EXPECTED_ZERO_PAD_SELECTION)),
            "invalid",
            replicate_padding=0,
        )
