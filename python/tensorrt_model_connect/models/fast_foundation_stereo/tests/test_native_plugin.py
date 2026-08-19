# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.fast_foundation_stereo import (
    native_plugin_builder,
    native_post,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_RUNTIME_MODEL_DIR = (
    _REPOSITORY_ROOT
    / "python/tensorrt_model_connect/models/fast_foundation_stereo/runtime"
)
_PLUGIN_DIR = _RUNTIME_MODEL_DIR.parent / "native_plugins"


def test_native_plugin_builder_caches_the_standalone_dso(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tensorrt_model_connect import trt_compat

    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[1] == "--build":
            output = Path(command[2]) / "libtrtmc_fast_foundation_stereo_native_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv(native_plugin_builder._PLUGIN_ENV, raising=False)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    cuda_architectures = "89-real;103-virtual"
    monkeypatch.setenv("CMAKE_CUDA_ARCHITECTURES", cuda_architectures)
    monkeypatch.setattr(native_plugin_builder, "_source_digest", lambda _path: "key")
    monkeypatch.setattr(native_plugin_builder, "_active_tensorrt_cmake_hints", lambda: [])
    monkeypatch.setattr(trt_compat, "load_module", lambda: object())
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.2.0.113")
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    first = native_plugin_builder.ensure_native_plugin()
    second = native_plugin_builder.ensure_native_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert [command[1] for command in calls] == ["-S", "--build"]
    assert Path(calls[0][2]) == _PLUGIN_DIR
    assert f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}" in calls[0]


def test_native_plugin_builder_defaults_to_l4_cuda_architectures(monkeypatch) -> None:
    monkeypatch.delenv("CMAKE_CUDA_ARCHITECTURES", raising=False)

    assert native_plugin_builder._cuda_architectures() == "89-real;89-virtual"


def test_native_plugin_builder_requires_model_owned_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        native_plugin_builder,
        "__file__",
        str(tmp_path / "native_plugin_builder.py"),
    )

    with pytest.raises(FileNotFoundError, match="Model-owned native plugin sources"):
        native_plugin_builder._native_plugin_source_dir()


def test_native_plugin_cmake_requires_exact_header_runtime_release() -> None:
    cmake = (_PLUGIN_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "NvInferVersion.h" in cmake
    assert "FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION" in cmake
    assert "_trt_detected_version STREQUAL FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION" in cmake
    assert "TensorRT header/runtime mismatch" in cmake
    assert "TRT_${suffix}_ENTERPRISE" in cmake
    assert "NV_TENSORRT_${suffix}" in cmake
    assert cmake.index("TensorRT header/runtime mismatch") < cmake.index(
        "add_library(trtmc_fast_foundation_stereo_native_plugin"
    )


def test_native_plugin_builder_pins_loaded_tensorrt_abi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tensorrt_model_connect import trt_compat

    library = tmp_path / "tensorrt_libs/libnvinfer.so.11.2.0"
    library.parent.mkdir()
    library.write_bytes(b"runtime")
    include = tmp_path / "include"
    include.mkdir()
    (include / "NvInferRuntime.h").write_text("// header\n", encoding="utf-8")
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(library)])
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.2.0.113")
    monkeypatch.delenv("TRTMC_TRT_INCLUDE_DIR", raising=False)
    monkeypatch.setenv("TRT_INC_DIR", str(include))

    assert native_plugin_builder._active_tensorrt_cmake_hints() == [
        "-DFAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION=11.2.0.113",
        f"-DFAST_FOUNDATION_STEREO_TRT_LIBRARY={library}",
        f"-DFAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR={include}",
    ]


def test_native_plugin_builder_accepts_loaded_soname_aliases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tensorrt_model_connect import trt_compat

    library_dir = tmp_path / "tensorrt_libs"
    library_dir.mkdir()
    major = library_dir / "libnvinfer.so.11"
    exact = library_dir / "libnvinfer.so.11.2.0"
    major.write_bytes(b"runtime")
    exact.write_bytes(b"runtime")
    monkeypatch.setattr(
        trt_compat,
        "loaded_libnvinfer_paths",
        lambda: [str(exact), str(major)],
    )
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.2.0.113")
    monkeypatch.delenv("TRTMC_TRT_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("TRT_INC_DIR", raising=False)

    assert native_plugin_builder._active_tensorrt_cmake_hints() == [
        "-DFAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION=11.2.0.113",
        f"-DFAST_FOUNDATION_STEREO_TRT_LIBRARY={exact}",
    ]


@pytest.mark.parametrize("version", ("10.8.0.43", "11.2.0.113"))
def test_native_plugin_builder_normalizes_exact_tensorrt_release(version: str) -> None:
    assert native_plugin_builder._normalized_tensorrt_version(version) == version


@pytest.mark.parametrize("version", ("", "11.2", "11.2.0", "11.2.0.113.post1"))
def test_native_plugin_builder_rejects_non_exact_tensorrt_release(version: str) -> None:
    with pytest.raises(RuntimeError, match="major.minor.patch.build"):
        native_plugin_builder._normalized_tensorrt_version(version)


def test_native_plugin_cache_key_includes_tensorrt_abi(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plugin.cu").write_text("// kernel\n", encoding="utf-8")

    trt10 = native_plugin_builder._plugin_cache_key(
        source,
        ["-DFAST_FOUNDATION_STEREO_TRT_LIBRARY=/sdk/libnvinfer.so.10"],
        cuda_architectures="89-real;89-virtual",
    )
    trt11 = native_plugin_builder._plugin_cache_key(
        source,
        ["-DFAST_FOUNDATION_STEREO_TRT_LIBRARY=/sdk/libnvinfer.so.11"],
        cuda_architectures="89-real;89-virtual",
    )

    assert trt10 != trt11


def test_native_plugin_cache_key_includes_cuda_architectures(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plugin.cu").write_text("// kernel\n", encoding="utf-8")

    l4 = native_plugin_builder._plugin_cache_key(
        source, [], cuda_architectures="89-real;89-virtual"
    )
    blackwell = native_plugin_builder._plugin_cache_key(
        source, [], cuda_architectures="103-real;103-virtual"
    )

    assert l4 != blackwell


@pytest.mark.parametrize(
    ("plugin_name", "expected_version"),
    (
        (native_plugin_builder._PLUGIN_NAME, "2"),
        (native_plugin_builder._GEOMETRY_VOLUME_CONVC1_PLUGIN_NAME, "1"),
        (native_plugin_builder._SPATIAL_ATTENTION_REDUCE_PLUGIN_NAME, "1"),
        (native_plugin_builder._POST8_SUM_PLUGIN_NAME, "1"),
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
        def get_plugin_creator(self, name: str, version: str, namespace: str):
            calls.append((name, version, namespace))
            return creator

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda: None)
    trt_module = SimpleNamespace(get_plugin_registry=lambda: Registry())

    assert native_plugin_builder._plugin_version(plugin_name) == expected_version
    assert native_plugin_builder._plugin_creator(trt_module, plugin_name) is creator
    assert calls == [(plugin_name, expected_version, "")]


def test_native_plugin_creator_error_reports_selected_version(monkeypatch) -> None:
    class Registry:
        def get_plugin_creator(self, _name: str, _version: str, _namespace: str):
            return None

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda: None)
    trt_module = SimpleNamespace(get_plugin_registry=lambda: Registry())

    with pytest.raises(
        RuntimeError,
        match=r"FastFoundationStereoCombinedVolume v2 was not registered",
    ):
        native_plugin_builder._plugin_creator(trt_module)


def test_native_plugin_v3_creator_skips_v2_only_registry_api(monkeypatch) -> None:
    creator = object()
    calls = []

    class Registry:
        def get_plugin_creator(self, *_args):
            raise AssertionError("V3 lookup must not probe the V2-only registry API")

        def get_creator(self, name, version, namespace):
            calls.append((name, version, namespace))
            return creator

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda: None)
    trt_module = SimpleNamespace(get_plugin_registry=lambda: Registry())

    assert (
        native_plugin_builder._plugin_v3_creator(
            trt_module,
            native_plugin_builder._POST8_SUM_PLUGIN_NAME,
        )
        is creator
    )
    assert calls == [("FastFoundationStereoPost8Sum", "1", "")]


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
    native_plugin_builder._PLUGIN_HANDLES.clear()

    assert native_plugin_builder.load_native_plugin() == library
    assert native_plugin_builder.load_native_plugin() == library
    assert calls == [(str(library), native_plugin_builder.ctypes.RTLD_GLOBAL)]
    assert native_plugin_builder._PLUGIN_HANDLES == {library: handle}
    native_plugin_builder._PLUGIN_HANDLES.clear()


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


def test_cpp_runtime_loads_embedded_plugin_before_deserializing_post_engine() -> None:
    model_dir = _RUNTIME_MODEL_DIR
    plugin_source = (model_dir / "plugin.cpp").read_text(encoding="utf-8")
    pipeline_source = (model_dir / "stereo_pipeline.cpp").read_text(encoding="utf-8")
    creator_source = (_PLUGIN_DIR / "plugin_creators.cpp").read_text(encoding="utf-8")
    identity_symbol = "fast_foundation_stereo_combined_volume_plugin_force_link"

    assert 'find_section(ctx.bundle, "fast_foundation_stereo_native_plugin_so")' in plugin_source
    assert plugin_source.index("load_native_plugin(ctx)") < plugin_source.index(
        '"post engine plan"'
    )
    assert "launch_fast_foundation_stereo_gwc" not in pipeline_source
    assert 'bind_external("gwc_volume"' not in pipeline_source
    assert identity_symbol in plugin_source
    assert identity_symbol in creator_source
    assert not (model_dir / "gwc_kernel.cu").exists()
    assert "feature_->has_output(name)" in pipeline_source
    assert "post_->has_input(name)" in pipeline_source
    assert "feature_->tensor_shape(name)" in pipeline_source
    assert "feature_->tensor_dtype(name)" in pipeline_source
    assert 'post_->has_output("disp")' in pipeline_source
    assert 'post_->tensor_dtype("disp") != DType::kFloat32' in pipeline_source
    assert 'post_->tensor_shape("disp") != expected_shape' in pipeline_source


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
