# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.fast_foundation_stereo import native_plugin_builder, native_post

@pytest.mark.parametrize(
    ("plugin_name", "expected_version"),
    (
        (native_plugin_builder._PLUGIN_NAME, "2"),
        (native_plugin_builder._GEOMETRY_VOLUME_CONVC1_PLUGIN_NAME, "1"),
        (native_plugin_builder._SPATIAL_ATTENTION_REDUCE_PLUGIN_NAME, "1"),
        (native_plugin_builder._POST8_SUM_PLUGIN_NAME, "1"),
        (native_plugin_builder._FULL_VOLUME_LEAKY_PLUGIN_NAME, "1"),
        ("FastFoundationStereoFuturePlugin", "1"),
    ),
)
def test_native_plugin_creator_uses_per_plugin_version(
    plugin_name: str,
    expected_version: str,
    monkeypatch,
) -> None:
    creator = object()
    calls: list[tuple[str, str, str]] = []

    class Registry:
        def get_creator(self, name: str, version: str, namespace: str):
            calls.append((name, version, namespace))
            return creator

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda: None)
    trt_module = SimpleNamespace(get_plugin_registry=lambda: Registry())

    assert native_plugin_builder._plugin_version(plugin_name) == expected_version
    assert native_plugin_builder._plugin_creator(trt_module, plugin_name) is creator
    assert calls == [(plugin_name, expected_version, "")]


def test_native_plugin_creator_error_reports_selected_version(monkeypatch) -> None:
    class Registry:
        def get_creator(self, _name: str, _version: str, _namespace: str):
            return None

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda: None)
    trt_module = SimpleNamespace(get_plugin_registry=lambda: Registry())

    with pytest.raises(
        RuntimeError,
        match=r"TensorRT plugin creator FastFoundationStereoCombinedVolume v2 is missing",
    ):
        native_plugin_builder._plugin_creator(trt_module)



def test_native_plugin_load_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "libnative.so"
    library.write_bytes(b"plugin")
    calls = []
    handle = object()
    monkeypatch.setattr(native_plugin_builder, "ensure_native_plugin", lambda **_kwargs: library)
    monkeypatch.setattr(
        native_plugin_builder.ctypes,
        "CDLL",
        lambda path, mode: calls.append((path, mode)) or handle,
    )
    native_plugin_builder._PLUGIN_HANDLE = None
    native_plugin_builder._PLUGIN_PATH = None

    assert native_plugin_builder.load_native_plugin() == library
    assert native_plugin_builder.load_native_plugin() == library
    assert calls == [(str(library), native_plugin_builder.ctypes.RTLD_GLOBAL)]
    assert native_plugin_builder._PLUGIN_HANDLE is handle
    assert native_plugin_builder._PLUGIN_PATH == library
    native_plugin_builder._PLUGIN_HANDLE = None
    native_plugin_builder._PLUGIN_PATH = None


def test_native_combined_volume_layer_has_four_inputs_and_one_named_output(
    monkeypatch,
) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields):
            assert name == "combined_volume"
            assert fields == []
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v2(self, inputs, selected_plugin):
            self.inputs = inputs
            self.plugin = selected_plugin
            return self.layer

    monkeypatch.setattr(
        native_plugin_builder,
        "_plugin_creator",
        lambda _trt, plugin_name: (
            Creator() if plugin_name == native_plugin_builder._PLUGIN_NAME else None
        ),
    )
    reference = object()
    target = object()
    left_projected = object()
    right_projected = object()
    network = Network()
    trt_module = SimpleNamespace(PluginFieldCollection=lambda fields: fields)

    output = native_plugin_builder.add_combined_volume_plugin(
        network,
        reference,
        target,
        left_projected,
        right_projected,
        trt_module=trt_module,
    )

    assert network.inputs == [reference, target, left_projected, right_projected]
    assert network.plugin is plugin
    assert network.layer.name == "combined_volume"
    assert output.name == "combined_volume"


def test_native_spatial_attention_reduce_layer_has_one_input_and_two_named_outputs(
    monkeypatch,
) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields):
            assert name == "spatial_attention_reduce"
            assert fields == []
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.outputs = (Output(), Output())

        def get_output(self, index: int):
            return self.outputs[index]

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v2(self, inputs, selected_plugin):
            self.inputs = inputs
            self.plugin = selected_plugin
            return self.layer

    def plugin_creator(_trt, plugin_name):
        assert plugin_name == "FastFoundationStereoSpatialAttentionReduce"
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", plugin_creator)
    tensor = object()
    network = Network()
    trt_module = SimpleNamespace(PluginFieldCollection=lambda fields: fields)

    average, maximum = native_plugin_builder.add_spatial_attention_reduce_plugin(
        network,
        tensor,
        trt_module=trt_module,
    )

    assert network.inputs == [tensor]
    assert network.plugin is plugin
    assert network.layer.name == "spatial_attention_reduce"
    assert average.name == "spatial_attention_reduce_average"
    assert maximum.name == "spatial_attention_reduce_maximum"


def test_native_geometry_volume_convc1_layer_has_six_direct_inputs(monkeypatch) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields):
            assert name == "geometry_volume_convc1_3"
            assert fields == []
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v2(self, inputs, selected_plugin):
            self.inputs = inputs
            self.plugin = selected_plugin
            return self.layer

    def plugin_creator(_trt, plugin_name):
        assert plugin_name == "FastFoundationStereoGeometryVolumeConvc1"
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", plugin_creator)
    tensors = [object() for _ in range(6)]
    network = Network()
    trt_module = SimpleNamespace(PluginFieldCollection=lambda fields: fields)

    output = native_plugin_builder.add_geometry_volume_convc1_plugin(
        network,
        *tensors,
        trt_module=trt_module,
        name="geometry_volume_convc1_3",
    )

    assert network.inputs == tensors
    assert network.plugin is plugin
    assert network.layer.name == "geometry_volume_convc1_3"
    assert output.name == "geometry_volume_convc1_3"



def test_direct_volume_convc1_packs_the_distilled_checkpoint_for_wmma() -> None:
    generator = np.random.default_rng(20260816)
    weight = generator.standard_normal((56, 522, 1, 1), dtype=np.float32)
    bias = generator.standard_normal((56,), dtype=np.float32)
    module = SimpleNamespace(
        in_channels=522,
        out_channels=56,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        weight=weight,
        bias=bias,
    )

    packed_weight, packed_bias = native_post._pack_geometry_convc1_parameters(module)

    assert packed_weight.shape == (64, 528)
    assert packed_bias.shape == (64,)
    assert packed_weight.dtype == np.float16
    assert packed_bias.dtype == np.float16
    np.testing.assert_array_equal(packed_weight[:56, :522], weight[:, :, 0, 0].astype(np.float16))
    np.testing.assert_array_equal(packed_bias[:56], bias.astype(np.float16))
    assert not np.any(packed_weight[56:, :])
    assert not np.any(packed_weight[:, 522:])
    assert not np.any(packed_bias[56:])


def test_direct_volume_convc1_rejects_the_unpruned_constructor_shape() -> None:
    module = SimpleNamespace(
        in_channels=522,
        out_channels=256,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        weight=np.zeros((256, 522, 1, 1), dtype=np.float32),
        bias=np.zeros((256,), dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match=r"convc1\.out_channels.*56"):
        native_post._pack_geometry_convc1_parameters(module)


def _fill_linear_sample(values: np.ndarray, coordinate: np.float32) -> np.float32:
    if not np.isfinite(coordinate):
        return np.float32(0)
    coordinate_floor = int(np.floor(coordinate))
    fraction = np.float32(coordinate - coordinate_floor)
    left = (
        np.float32(values[coordinate_floor])
        if 0 <= coordinate_floor < values.size
        else np.float32(0)
    )
    right_index = coordinate_floor + 1
    right = np.float32(values[right_index]) if 0 <= right_index < values.size else np.float32(0)
    return np.float32(left * np.float32(1.0 - fraction) + right * fraction)


def _direct_volume_reference(
    logical_volume: np.ndarray,
    disparity: np.ndarray,
    width_indices: np.ndarray,
    correlations: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    level0 = logical_volume.transpose(1, 2, 0).astype(np.float32)
    level1 = np.float32(
        np.float32(level0[:, :, 0::2]) + np.float32(level0[:, :, 1::2])
    ) * np.float32(0.5)
    geometries = (level0, level1)
    output = np.empty((disparity.size, 522), dtype=np.float32)
    for pixel in range(disparity.size):
        for level in range(2):
            inverse_scale = np.float32(1.0 if level == 0 else 0.5)
            disparity_level = np.float32(disparity[pixel] * inverse_scale)
            bases = (
                disparity_level,
                np.float32(np.float32(width_indices[pixel]) * inverse_scale - disparity_level),
            )
            for source_channel in range(29):
                is_correlation = source_channel == 28
                values = (
                    correlations[level][pixel]
                    if is_correlation
                    else geometries[level][pixel, source_channel]
                )
                output_base = level * 261 + (252 if is_correlation else source_channel * 9)
                for sample in range(9):
                    coordinate = np.float32(bases[int(is_correlation)] + np.float32(sample - 4))
                    output[pixel, output_base + sample] = _fill_linear_sample(values, coordinate)
    return output.astype(np.float16)


def _direct_volume_windowed(
    packed_volume: np.ndarray,
    disparity: np.ndarray,
    width_indices: np.ndarray,
    correlations: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    output = np.empty((disparity.size, 522), dtype=np.float32)
    for pixel in range(disparity.size):
        for level in range(2):
            inverse_scale = np.float32(1.0 if level == 0 else 0.5)
            disparity_level = np.float32(disparity[pixel] * inverse_scale)
            bases = (
                disparity_level,
                np.float32(np.float32(width_indices[pixel]) * inverse_scale - disparity_level),
            )
            for source_channel in range(29):
                is_correlation = source_channel == 28
                source_width = (176, 88)[level] if is_correlation else (48, 24)[level]
                coordinate = bases[int(is_correlation)]
                finite = bool(
                    np.isfinite(coordinate) and -5.0 <= coordinate < float(source_width + 4)
                )
                coordinate_floor = int(np.floor(coordinate)) if finite else 0
                fraction = np.float32(coordinate - coordinate_floor) if finite else np.float32(0)
                coordinate_start = coordinate_floor - 4

                def load(source_index: int) -> np.float32:
                    if not finite or not 0 <= source_index < source_width:
                        return np.float32(0)
                    if is_correlation:
                        return np.float32(correlations[level][pixel, source_index])
                    disparity0 = source_index if level == 0 else 2 * source_index
                    value0 = np.float32(packed_volume[disparity0, pixel, source_channel])
                    if level == 0:
                        return value0
                    value1 = np.float32(packed_volume[disparity0 + 1, pixel, source_channel])
                    return np.float32(np.float32(value0 + value1) * np.float32(0.5))

                value = load(coordinate_start)
                output_base = level * 261 + (252 if is_correlation else source_channel * 9)
                for sample in range(9):
                    next_value = load(coordinate_start + sample + 1)
                    output[pixel, output_base + sample] = np.float32(
                        value * np.float32(1.0 - fraction) + next_value * fraction
                    )
                    value = next_value
    return output.astype(np.float16)


def _coalesced_correlation_halfwarp(values: np.ndarray, coordinate: np.float32) -> np.ndarray:
    source_width = values.size
    finite = bool(np.isfinite(coordinate) and -5.0 <= coordinate < float(source_width + 4))
    coordinate_floor = int(np.floor(coordinate)) if finite else 0
    fraction = np.float32(coordinate - coordinate_floor) if finite else np.float32(0)
    coordinate_start = coordinate_floor - 4
    endpoints = np.zeros((10,), dtype=np.float32)
    for lane in range(10):
        source_index = coordinate_start + lane
        if finite and 0 <= source_index < source_width:
            endpoints[lane] = values[source_index]

    sampled = np.empty((9,), dtype=np.float32)
    for lane in range(9):
        sampled[lane] = np.float32(
            endpoints[lane] * np.float32(1.0 - fraction) + endpoints[lane + 1] * fraction
        )
    return sampled.astype(np.float16)


def test_direct_volume_sampling_matches_materialized_oracle_at_boundaries() -> None:
    generator = np.random.default_rng(20260817)
    disparity = np.array(
        [
            -np.inf,
            -np.finfo(np.float32).max,
            -5.001,
            -5.0,
            -4.999,
            -0.25,
            0.0,
            0.25,
            47.75,
            52.0,
            175.25,
            np.finfo(np.float32).max,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    width_indices = np.array([0, 1, 2, 3, 4, 47, 48, 87, 88, 174, 175, 5, 6, 7])
    pixels = disparity.size
    logical_volume = generator.standard_normal((48, pixels, 28), dtype=np.float32).astype(
        np.float16
    )
    packed_volume = np.full((48, pixels, 32), np.float16(np.nan), dtype=np.float16)
    packed_volume[:, :, :28] = logical_volume
    correlations = (
        generator.standard_normal((pixels, 176), dtype=np.float32),
        generator.standard_normal((pixels, 88), dtype=np.float32),
    )

    expected = _direct_volume_reference(logical_volume, disparity, width_indices, correlations)
    actual = _direct_volume_windowed(packed_volume, disparity, width_indices, correlations)

    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source_width", [176, 88])
def test_direct_volume_coalesced_correlation_matches_serial_oracle(source_width: int) -> None:
    generator = np.random.default_rng(20260818 + source_width)
    values = generator.standard_normal((source_width,), dtype=np.float32)
    coordinates = np.array(
        [
            -np.inf,
            -np.finfo(np.float32).max,
            -5.001,
            -5.0,
            -4.999,
            -0.25,
            0.0,
            0.25,
            source_width - 0.25,
            source_width + 3.999,
            source_width + 4.0,
            np.finfo(np.float32).max,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    coordinates = np.concatenate(
        (coordinates, generator.uniform(-5.0, source_width + 4.0, 128).astype(np.float32))
    )

    for coordinate in coordinates:
        expected = np.array(
            [
                _fill_linear_sample(values, np.float32(coordinate + np.float32(sample - 4)))
                for sample in range(9)
            ],
            dtype=np.float32,
        ).astype(np.float16)
        actual = _coalesced_correlation_halfwarp(values, coordinate)
        np.testing.assert_array_equal(actual, expected)


def test_direct_volume_convc1_phased_k_staging_preserves_channel_order() -> None:
    sampled = np.arange(522, dtype=np.float16)
    staged: list[np.ndarray] = []
    source_group_base = 0
    for source_groups, mma_channels in ((16, 144), (16, 144), (16, 144), (10, 96)):
        valid_channels = source_groups * 9
        phase = np.zeros((mma_channels,), dtype=np.float16)
        phase[:valid_channels] = sampled[
            source_group_base * 9 : (source_group_base + source_groups) * 9
        ]
        staged.append(phase)
        source_group_base += source_groups

    actual = np.concatenate(staged)
    expected = np.pad(sampled, (0, 6))

    assert source_group_base == 58
    assert actual.shape == (528,)
    np.testing.assert_array_equal(actual, expected)
