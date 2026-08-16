# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tensorrt_model_connect.families.fast_foundation_stereo import (
    native_plugin_builder,
)


def test_native_plugin_builder_caches_the_standalone_dso(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tensorrt_model_connect import trt_compat

    calls: list[str] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command[1])
        if command[1] == "--build":
            output = Path(command[2]) / "libtrtmc_fast_foundation_stereo_native_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv(native_plugin_builder._PLUGIN_ENV, raising=False)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(native_plugin_builder, "_source_digest", lambda _path: "key")
    monkeypatch.setattr(native_plugin_builder, "_active_tensorrt_cmake_hints", lambda: [])
    monkeypatch.setattr(trt_compat, "load_module", lambda: object())
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.2.0.113")
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    first = native_plugin_builder.ensure_native_plugin()
    second = native_plugin_builder.ensure_native_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert calls == ["-S", "--build"]


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
    monkeypatch.delenv("TRTMC_TRT_INCLUDE_DIR", raising=False)
    monkeypatch.setenv("TRT_INC_DIR", str(include))

    assert native_plugin_builder._active_tensorrt_cmake_hints() == [
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
        f"-DFAST_FOUNDATION_STEREO_TRT_LIBRARY={exact}"
    ]


def test_native_plugin_cache_key_includes_tensorrt_abi(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plugin.cu").write_text("// kernel\n", encoding="utf-8")

    trt10 = native_plugin_builder._plugin_cache_key(
        source,
        ["-DFAST_FOUNDATION_STEREO_TRT_LIBRARY=/sdk/libnvinfer.so.10"],
    )
    trt11 = native_plugin_builder._plugin_cache_key(
        source,
        ["-DFAST_FOUNDATION_STEREO_TRT_LIBRARY=/sdk/libnvinfer.so.11"],
    )

    assert trt10 != trt11


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


def test_native_gwc_layer_has_two_feature_inputs_and_one_named_output(monkeypatch) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields):
            assert name == "gwc_volume"
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

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", lambda _trt: Creator())
    reference = object()
    target = object()
    network = Network()
    trt_module = SimpleNamespace(PluginFieldCollection=lambda fields: fields)

    output = native_plugin_builder.add_gwc_plugin(
        network,
        reference,
        target,
        trt_module=trt_module,
    )

    assert network.inputs == [reference, target]
    assert network.plugin is plugin
    assert network.layer.name == "gwc_volume"
    assert output.name == "gwc_volume"


def test_cpp_runtime_loads_embedded_plugin_before_deserializing_post_engine() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    model_dir = repository_root / "src/runtime/models/fast_foundation_stereo"
    plugin_source = (model_dir / "plugin.cpp").read_text(encoding="utf-8")
    pipeline_source = (model_dir / "stereo_pipeline.cpp").read_text(encoding="utf-8")

    assert 'find_section(ctx.bundle, "fast_foundation_stereo_native_plugin_so")' in plugin_source
    assert plugin_source.index("load_native_plugin(ctx)") < plugin_source.index(
        '"post engine plan"'
    )
    assert "launch_fast_foundation_stereo_gwc" not in pipeline_source
    assert 'bind_external("gwc_volume"' not in pipeline_source
    assert not (model_dir / "gwc_kernel.cu").exists()
    assert "feature_->has_output(name)" in pipeline_source
    assert "post_->has_input(name)" in pipeline_source
    assert "feature_->tensor_shape(name)" in pipeline_source
    assert "feature_->tensor_dtype(name)" in pipeline_source
    assert 'post_->has_output("disp")' in pipeline_source
    assert 'post_->tensor_dtype("disp") != DType::kFloat32' in pipeline_source
    assert 'post_->tensor_shape("disp") != expected_shape' in pipeline_source


def test_gwc_plugin_owns_the_fixed_l4_tensor_contract() -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/fast_foundation_stereo/"
        "native_plugins/gwc_plugin.cu"
    ).read_text(encoding="utf-8")

    for contract in (
        "kBatch = 1",
        "kChannels = 224",
        "kGroups = 8",
        "kHeight = 176",
        "kWidth = 176",
        "kDisparities = 48",
        "kWorkspaceBytes = 2 * kNormElements * sizeof(__half)",
    ):
        assert contract in source
