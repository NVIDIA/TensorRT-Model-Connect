# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only algebra and graph contracts for the exact level-3 resample fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensorrt_model_connect.families.moge import model as model_module


def _half_pixel_resize_x2(tensor: np.ndarray) -> np.ndarray:
    batch, channels, height, width = tensor.shape
    output = np.empty((batch, channels, 2 * height, 2 * width), dtype=tensor.dtype)
    for output_y in range(2 * height):
        source_y = (output_y + 0.5) * 0.5 - 0.5
        lower_y_raw = int(np.floor(source_y))
        fraction_y = source_y - lower_y_raw
        lower_y = min(height - 1, max(0, lower_y_raw))
        upper_y = min(height - 1, max(0, lower_y_raw + 1))
        for output_x in range(2 * width):
            source_x = (output_x + 0.5) * 0.5 - 0.5
            lower_x_raw = int(np.floor(source_x))
            fraction_x = source_x - lower_x_raw
            lower_x = min(width - 1, max(0, lower_x_raw))
            upper_x = min(width - 1, max(0, lower_x_raw + 1))
            output[:, :, output_y, output_x] = (
                tensor[:, :, lower_y, lower_x] * (1.0 - fraction_y) * (1.0 - fraction_x)
                + tensor[:, :, lower_y, upper_x] * (1.0 - fraction_y) * fraction_x
                + tensor[:, :, upper_y, lower_x] * fraction_y * (1.0 - fraction_x)
                + tensor[:, :, upper_y, upper_x] * fraction_y * fraction_x
            )
    return output


def _replicate_conv3x3(tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    output = np.empty(
        (tensor.shape[0], weight.shape[0], tensor.shape[2], tensor.shape[3]),
        dtype=tensor.dtype,
    )
    for output_y in range(tensor.shape[2]):
        for output_x in range(tensor.shape[3]):
            patch = padded[:, :, output_y : output_y + 3, output_x : output_x + 3]
            output[:, :, output_y, output_x] = np.einsum("nchw,ochw->no", patch, weight) + bias
    return output


def _deconvolution_stride2_padding4(
    tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    full_height = (tensor.shape[2] - 1) * 2 + 6
    full_width = (tensor.shape[3] - 1) * 2 + 6
    full = np.zeros(
        (tensor.shape[0], weight.shape[1], full_height, full_width),
        dtype=tensor.dtype,
    )
    for input_y in range(tensor.shape[2]):
        for input_x in range(tensor.shape[3]):
            contribution = np.einsum("ni,iohw->nohw", tensor[:, :, input_y, input_x], weight)
            full[
                :,
                :,
                2 * input_y : 2 * input_y + 6,
                2 * input_x : 2 * input_x + 6,
            ] += contribution
    output_height = (tensor.shape[2] - 1) * 2 - 8 + 6
    output_width = (tensor.shape[3] - 1) * 2 - 8 + 6
    return full[:, :, 4 : 4 + output_height, 4 : 4 + output_width] + bias[None, :, None, None]


def _reference(tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return _replicate_conv3x3(_half_pixel_resize_x2(tensor), weight, bias)


def _fused(tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    fused_weight = model_module._fuse_half_pixel_x2_conv_weight(weight)
    return _deconvolution_stride2_padding4(padded, fused_weight, bias)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((np.float64, 1.0e-11), (np.float32, 5.0e-5)),
)
@pytest.mark.parametrize("height,width", ((1, 1), (1, 3), (2, 1), (2, 2), (3, 5), (7, 4)))
def test_fused_resample_matches_half_pixel_replicate_reference(
    dtype: Any, tolerance: float, height: int, width: int
) -> None:
    random = np.random.default_rng(1000 + 10 * height + width)
    tensor = random.standard_normal((1, 3, height, width)).astype(dtype)
    weight = random.standard_normal((2, 3, 3, 3)).astype(dtype)
    bias = random.standard_normal((2,)).astype(dtype)

    reference = _reference(tensor, weight, bias)
    fused = _fused(tensor, weight, bias)

    assert reference.shape == fused.shape == (1, 2, 2 * height, 2 * width)
    np.testing.assert_allclose(fused, reference, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("height,width", ((1, 1), (2, 3), (5, 4)))
def test_fused_resample_preserves_corner_edge_and_center_impulses(height: int, width: int) -> None:
    random = np.random.default_rng(2000 + 10 * height + width)
    weight = random.standard_normal((2, 2, 3, 3)).astype(np.float64)
    bias = np.zeros((2,), dtype=np.float64)
    positions = {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (height // 2, width // 2),
    }
    for input_y, input_x in positions:
        tensor = np.zeros((1, 2, height, width), dtype=np.float64)
        tensor[0, 0, input_y, input_x] = 1.0
        tensor[0, 1, input_y, input_x] = -0.5
        np.testing.assert_allclose(
            _fused(tensor, weight, bias),
            _reference(tensor, weight, bias),
            rtol=0.0,
            atol=1.0e-11,
        )


@dataclass
class _FakeTensor:
    dtype: Any
    shape: tuple[int, ...]


class _FakeLayer:
    def __init__(self, output: _FakeTensor) -> None:
        self.output = output
        self.name = ""
        self.stride_nd: tuple[int, ...] | None = None
        self.padding_nd: tuple[int, ...] | None = None

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeNetwork:
    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype
        self.calls: list[dict[str, Any]] = []

    def add_deconvolution_nd(
        self,
        tensor: _FakeTensor,
        output_channels: int,
        kernel_size: tuple[int, int],
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> _FakeLayer:
        height = (tensor.shape[2] - 1) * 2 + kernel_size[0] - 8
        width = (tensor.shape[3] - 1) * 2 + kernel_size[1] - 8
        layer = _FakeLayer(_FakeTensor(self.dtype, (1, output_channels, height, width)))
        self.calls.append(
            {
                "tensor": tensor,
                "output_channels": output_channels,
                "kernel_size": kernel_size,
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


@pytest.mark.parametrize("height,width", ((1, 1), (3, 5), (32, 57), (57, 32)))
def test_fused_resample_graph_contract_is_dynamic_and_native(
    monkeypatch: pytest.MonkeyPatch, height: int, width: int
) -> None:
    network = _FakeNetwork(_FakeTrt.float32)
    graph = model_module._NativeMogeGraph(_FakeTrt, network, {}, fast_path=True)
    source_weight = np.arange(2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3)
    source_bias = np.asarray((0.25, -0.5), dtype=np.float32)
    arrays = {"resampler.weight": source_weight, "resampler.bias": source_bias}
    monkeypatch.setattr(graph, "_array", lambda name, expected=None: arrays[name])
    pad_calls: list[tuple[int, str]] = []

    def fake_pad(tensor: _FakeTensor, padding: int, name: str) -> _FakeTensor:
        pad_calls.append((padding, name))
        return _FakeTensor(
            tensor.dtype,
            (tensor.shape[0], tensor.shape[1], tensor.shape[2] + 2, tensor.shape[3] + 2),
        )

    monkeypatch.setattr(graph, "replicate_pad", fake_pad)
    tensor = _FakeTensor(_FakeTrt.float32, (1, 3, height, width))

    output = graph.fused_half_pixel_resample(
        tensor,
        "resampler",
        "level3",
        compute_dtype=_FakeTrt.float32,
    )

    assert output.shape == (1, 2, 2 * height, 2 * width)
    assert pad_calls == [(1, "level3.input_pad")]
    assert len(network.calls) == 1
    call = network.calls[0]
    assert call["output_channels"] == 2
    assert call["kernel_size"] == (6, 6)
    assert call["layer"].stride_nd == (2, 2)
    assert call["layer"].padding_nd == (4, 4)
    np.testing.assert_array_equal(
        call["weight"], model_module._fuse_half_pixel_x2_conv_weight(source_weight)
    )
    np.testing.assert_array_equal(call["bias"], source_bias)


def test_level3_fast_path_uses_fused_native_deconvolution_only() -> None:
    source = Path(model_module.__file__).read_text(encoding="utf-8")
    assert "elif level == 3 and self.fast_path:" in source
    assert 'f"{name}.fused_deconvolution"' in source
    assert "self.network.add_deconvolution_nd" in source
    assert "layer.stride_nd = (2, 2)" in source
    assert "layer.padding_nd = (4, 4)" in source
    for stack in ('"neck"', '"points_head"', '"mask_head"'):
        assert stack in source
    assert "add_plugin" not in source.lower()
