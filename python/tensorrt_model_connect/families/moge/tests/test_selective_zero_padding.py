# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the family-owned selective zero-padding lane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensorrt_model_connect.families.moge import model as model_module


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
FUSED_LEVEL3_RESAMPLERS = frozenset(
    {
        "neck.resamplers.3.1",
        "points_head.resamplers.3.1",
        "mask_head.resamplers.3.1",
    }
)


def _decoder_replicate_modules() -> frozenset[str]:
    modules = set()
    for prefix, block_counts in (
        ("neck", (0, 2, 2, 2, 0)),
        ("points_head", (0, 1, 1, 1, 0)),
        ("mask_head", (0, 1, 1, 1, 0)),
    ):
        modules.update(f"{prefix}.resamplers.{level}.1" for level in range(4))
        for level, count in enumerate(block_counts):
            for block in range(count):
                modules.add(f"{prefix}.res_blocks.{level}.{block}.layers.2")
                modules.add(f"{prefix}.res_blocks.{level}.{block}.layers.5")
    return frozenset(modules)


@dataclass
class _FakeTensor:
    dtype: Any
    shape: tuple[int, ...]
    label: str


class _FakeLayer:
    def __init__(self, output: _FakeTensor) -> None:
        self.output = output
        self.name = ""
        self.stride_nd = None
        self.padding_nd = None

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeNetwork:
    def __init__(self) -> None:
        self.calls = []

    def add_convolution_nd(
        self,
        tensor: _FakeTensor,
        output_channels: int,
        kernel_shape: tuple[int, int],
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> _FakeLayer:
        layer = _FakeLayer(_FakeTensor(tensor.dtype, (1, output_channels, 8, 8), "output"))
        self.calls.append(
            {
                "tensor": tensor,
                "kernel_shape": kernel_shape,
                "weight": weight,
                "bias": bias,
                "layer": layer,
            }
        )
        return layer


class _FakeTrt:
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def Weights(array: np.ndarray) -> np.ndarray:
        return array


def test_zero_pad_selection_is_exact_unique_and_topology_owned() -> None:
    replicate_modules = _decoder_replicate_modules()

    assert len(replicate_modules) == 36
    assert model_module._ZERO_PAD_SELECTION == EXPECTED_ZERO_PAD_SELECTION
    assert len(model_module._ZERO_PAD_SELECTION) == 24
    assert model_module._ZERO_PAD_SELECTION < replicate_modules
    assert model_module._ZERO_PAD_SELECTION.isdisjoint(FUSED_LEVEL3_RESAMPLERS)
    assert FUSED_LEVEL3_RESAMPLERS < replicate_modules


def test_only_selected_modules_use_native_zero_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _FakeNetwork()
    graph = model_module._NativeMogeGraph(_FakeTrt, network, {}, fast_path=True)
    weight = np.zeros((4, 4, 3, 3), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)
    monkeypatch.setattr(
        graph,
        "_array",
        lambda name, expected=None: weight if name.endswith(".weight") else bias,
    )
    monkeypatch.setattr(graph, "cast", lambda tensor, dtype, name: tensor)
    pad_calls = []

    def fake_replicate_pad(tensor: _FakeTensor, padding: int, name: str) -> _FakeTensor:
        pad_calls.append((padding, name))
        return _FakeTensor(tensor.dtype, (1, 4, 10, 10), "replicate-padded")

    monkeypatch.setattr(graph, "replicate_pad", fake_replicate_pad)

    for module in sorted(_decoder_replicate_modules()):
        pad_calls.clear()
        network.calls.clear()
        tensor = _FakeTensor(_FakeTrt.float32, (1, 4, 8, 8), "source")
        graph.convolution(
            tensor,
            module,
            f"graph.{module}",
            replicate_padding=1,
            compute_dtype=_FakeTrt.float32,
        )
        assert len(network.calls) == 1
        call = network.calls[0]
        if module in EXPECTED_ZERO_PAD_SELECTION:
            assert pad_calls == []
            assert call["tensor"] is tensor
            assert call["layer"].padding_nd == (1, 1)
        else:
            assert pad_calls == [(1, f"graph.{module}.pad")]
            assert call["tensor"].label == "replicate-padded"
            assert call["layer"].padding_nd == (0, 0)


def test_selected_module_rejects_an_unexpected_convolution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _FakeNetwork()
    graph = model_module._NativeMogeGraph(_FakeTrt, network, {}, fast_path=True)
    weight = np.zeros((4, 4, 3, 3), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)
    monkeypatch.setattr(
        graph,
        "_array",
        lambda name, expected=None: weight if name.endswith(".weight") else bias,
    )
    monkeypatch.setattr(graph, "cast", lambda tensor, dtype, name: tensor)
    tensor = _FakeTensor(_FakeTrt.float32, (1, 4, 8, 8), "source")

    with pytest.raises(ValueError, match="must be stride-1 3x3"):
        graph.convolution(
            tensor,
            next(iter(EXPECTED_ZERO_PAD_SELECTION)),
            "invalid",
            replicate_padding=0,
            compute_dtype=_FakeTrt.float32,
        )


def test_selective_zero_patch_contains_no_linear_or_level4_fusion() -> None:
    source = Path(model_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "_ENABLE_LEVEL12_LINEAR_FUSION",
        "_ENABLE_LEVEL4_HEAD_FUSION",
        "_compose_deconv2_replicate_conv3",
    ):
        assert forbidden not in source
