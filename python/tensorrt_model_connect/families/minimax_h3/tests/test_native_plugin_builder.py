# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3 import native_plugin_builder


_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_PLUGIN_DIR = _REPOSITORY_ROOT / "src/runtime/models/minimax_h3/native_plugins"


@pytest.fixture(autouse=True)
def _reset_plugin_handles():
    native_plugin_builder._PLUGIN_HANDLES.clear()
    native_plugin_builder._FAILED_PLUGIN_HANDLES.clear()
    yield
    native_plugin_builder._PLUGIN_HANDLES.clear()
    native_plugin_builder._FAILED_PLUGIN_HANDLES.clear()


def test_native_plugin_sources_define_fixed_v3_row_major_hf_contract() -> None:
    audio_header = (_PLUGIN_DIR / "audio_encoder_plugin.h").read_text(encoding="utf-8")
    audio_source = (_PLUGIN_DIR / "audio_encoder_plugin.cpp").read_text(encoding="utf-8")
    header = (_PLUGIN_DIR / "vision_attention_plugin.h").read_text(encoding="utf-8")
    source = (_PLUGIN_DIR / "vision_attention_plugin.cpp").read_text(encoding="utf-8")
    linear_header = (_PLUGIN_DIR / "linear_plugin.h").read_text(encoding="utf-8")
    linear_source = (_PLUGIN_DIR / "linear_plugin.cpp").read_text(encoding="utf-8")
    norm_header = (_PLUGIN_DIR / "layer_norm_plugin.h").read_text(encoding="utf-8")
    norm_source = (_PLUGIN_DIR / "layer_norm_plugin.cpp").read_text(encoding="utf-8")
    patch_header = (_PLUGIN_DIR / "patch_embed_plugin.h").read_text(encoding="utf-8")
    patch_source = (_PLUGIN_DIR / "patch_embed_plugin.cpp").read_text(encoding="utf-8")
    creator = (_PLUGIN_DIR / "plugin_creator.cpp").read_text(encoding="utf-8")

    assert "IPluginV3" in header
    assert '"MiniMaxH3VisionAttention"' in header
    assert 'kPLUGIN_VERSION = "1"' in header
    assert "kROW_WIDTH = kHEADS * kHEAD_DIM" in header
    assert "0x1.e2b7dddfefa66p-4" in source
    assert "{rows, kHEADS, kHEAD_DIM}" in source
    assert ".permute({1, 0, 2})" in source
    assert "std::nullopt, 0.0, false" in source
    assert "std::optional<double>(kVisionAttentionScale), true" in source
    assert "InferenceMode" in source
    assert "CUDAStreamGuard" in source
    assert "q_prescale=false" in source
    assert '"MiniMaxH3PatchEmbed"' in patch_header
    assert "kPIXEL_ROW_WIDTH" in patch_header
    assert "at::conv3d" in patch_source
    assert (
        "{rows, kINPUT_CHANNELS, kTEMPORAL_PATCH, kSPATIAL_PATCH, kSPATIAL_PATCH}" in patch_source
    )
    assert "{kTEMPORAL_PATCH, kSPATIAL_PATCH, kSPATIAL_PATCH}" in patch_source
    assert "{0, 0, 0}" in patch_source
    assert "{1, 1, 1}" in patch_source
    assert "InferenceMode" in patch_source
    assert "CUDAStreamGuard" in patch_source
    assert '"MiniMaxH3Linear"' in linear_header
    assert "at::linear" in linear_source
    assert "InferenceMode" in linear_source
    assert "CUDAStreamGuard" in linear_source
    assert "x=[rows,in]:bf16:linear" in linear_source
    assert "weight=[out,in]:bf16:linear" in linear_source
    assert "bias=[out]:bf16:linear" in linear_source
    assert '"MiniMaxH3LayerNorm"' in norm_header
    assert "kEPSILON = 1.0e-6" in norm_header
    assert "at::layer_norm" in norm_source
    assert "InferenceMode" in norm_source
    assert "CUDAStreamGuard" in norm_source
    assert "eps=1e-6;cudnn=true" in norm_source
    assert '"MiniMaxH3AudioEncoder"' in audio_header
    assert "kMIN_SAMPLES = 64000" in audio_header
    assert "kOPT_SAMPLES = 165600" in audio_header
    assert "kMAX_SAMPLES = 480000" in audio_header
    assert "torch::jit::load" in audio_source
    assert "NoTF32Guard" in audio_source
    assert "CUDAStreamGuard" in audio_source
    assert "module=torchscript" in audio_source
    assert "python_runtime=false" in audio_source
    assert "IPluginCreatorV3One" in creator
    assert "plugin_registrar_minimax_h3_vision_attention" in creator
    assert "plugin_registrar_minimax_h3_audio_encoder" in creator
    assert "plugin_registrar_minimax_h3_layer_norm" in creator
    assert "plugin_registrar_minimax_h3_linear" in creator
    assert "plugin_registrar_minimax_h3_patch_embed" in creator
    assert "MiniMaxH3LanguageAttention" not in header
    assert "MiniMaxH3LanguageAttention" not in source
    assert "MiniMaxH3LanguageAttention" not in creator
    assert not hasattr(native_plugin_builder, "add_language_attention_plugin")
    assert native_plugin_builder._PLUGIN_IDENTITY in creator
    assert native_plugin_builder._PLUGIN_IDENTITY_SYMBOL in creator
    assert native_plugin_builder._PLUGIN_ABI_SYMBOL in creator
    assert native_plugin_builder._PLUGIN_REGISTRY_SYMBOL in creator
    assert "native_plugin_registry_matches" in creator


def test_native_plugin_source_files_cover_complete_standalone_target() -> None:
    assert {path.name for path in native_plugin_builder.native_plugin_source_files()} == {
        "CMakeLists.txt",
        "audio_encoder_plugin.cpp",
        "audio_encoder_plugin.h",
        "layer_norm_plugin.cpp",
        "layer_norm_plugin.h",
        "linear_plugin.cpp",
        "linear_plugin.h",
        "patch_embed_plugin.cpp",
        "patch_embed_plugin.h",
        "plugin_creator.cpp",
        "vision_attention_plugin.cpp",
        "vision_attention_plugin.h",
    }


def test_native_plugin_cmake_pins_trt_torch_cuda_and_cxx_abi() -> None:
    cmake = (_PLUGIN_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "MINIMAX_H3_TRT_EXPECTED_VERSION" in cmake
    assert "_trt_detected_version STREQUAL MINIMAX_H3_TRT_EXPECTED_VERSION" in cmake
    assert "MINIMAX_H3_TORCH_EXPECTED_VERSION" in cmake
    assert "_torch_detected_version STREQUAL MINIMAX_H3_TORCH_EXPECTED_VERSION" in cmake
    assert "MINIMAX_H3_TORCH_EXPECTED_CUDA_VERSION" in cmake
    assert "MINIMAX_H3_CUDA_TOOLKIT_EXPECTED_VERSION" in cmake
    assert "MINIMAX_H3_TORCH_EXPECTED_CXX11_ABI" in cmake
    assert "find_package(Torch CONFIG REQUIRED" in cmake
    assert "find_package(CUDAToolkit REQUIRED)" in cmake


def test_native_plugin_cache_key_covers_every_binary_dependency(tmp_path: Path) -> None:
    source = tmp_path / "plugin.cpp"
    source.write_text("// source\n", encoding="utf-8")
    baseline = {
        "tensorrt_version": "11.1.0.106",
        "torch_version": "2.12.0+cu130",
        "torch_cuda_version": "13.0",
        "cuda_toolkit_version": "13.3.73",
        "torch_cxx11_abi": "1",
        "cxx_compiler_identity": "/usr/bin/c++|13.3.0|c++ 13.3.0",
        "cmake_hints": ("-DMINIMAX_H3_TRT_LIBRARY=/sdk/libnvinfer.so.11",),
    }
    original = native_plugin_builder._plugin_cache_key([source], **baseline)
    variants = (
        {"tensorrt_version": "11.2.0.113"},
        {"torch_version": "2.13.0+cu130"},
        {"torch_cuda_version": "13.1"},
        {"cuda_toolkit_version": "13.4.1"},
        {"torch_cxx11_abi": "0"},
        {"cxx_compiler_identity": "/usr/bin/clang++|19.1.0|clang 19.1.0"},
        {"cmake_hints": ("-DMINIMAX_H3_TRT_LIBRARY=/other/libnvinfer.so.11",)},
    )
    for changed in variants:
        assert native_plugin_builder._plugin_cache_key([source], **(baseline | changed)) != original

    source.write_text("// changed\n", encoding="utf-8")
    assert native_plugin_builder._plugin_cache_key([source], **baseline) != original


def test_native_plugin_build_is_serialized_and_cached(tmp_path: Path, monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    calls: list[str] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(2)
    fake_torch = SimpleNamespace(
        __version__="2.12.0+cu130",
        __file__=str(tmp_path / "torch/__init__.py"),
        version=SimpleNamespace(cuda="13.0"),
        utils=SimpleNamespace(cmake_prefix_path=str(tmp_path / "torch/share/cmake")),
        compiled_with_cxx11_abi=lambda: True,
    )
    Path(fake_torch.__file__).parent.mkdir(parents=True)
    Path(fake_torch.__file__).write_text("", encoding="utf-8")

    def source_files() -> tuple[Path, ...]:
        barrier.wait(timeout=5)
        return native_plugin_builder.native_plugin_source_files.__wrapped__()  # type: ignore[attr-defined]

    original_source_files = native_plugin_builder.native_plugin_source_files
    source_files.__wrapped__ = original_source_files  # type: ignore[attr-defined]

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        action = command[1]
        with calls_lock:
            calls.append(action)
        if action == "-S":
            time.sleep(0.1)
        elif action == "--build":
            output = Path(command[2]) / "libtrtmc_minimax_h3_native_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv(native_plugin_builder._PLUGIN_ENV, raising=False)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(tmp_path / "build"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(trt_compat, "load_module", lambda: object())
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.1.0.106")
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [])
    monkeypatch.setattr(native_plugin_builder, "_cuda_toolkit_version", lambda: "13.3.73")
    monkeypatch.setattr(
        native_plugin_builder,
        "_cxx_compiler_identity",
        lambda: ("/usr/bin/c++", "13.3.0", "c++ 13.3.0"),
    )
    monkeypatch.setattr(native_plugin_builder, "native_plugin_source_files", source_files)
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _index: native_plugin_builder.ensure_native_plugin(), range(2))
        )

    assert results == [results[0], results[0]]
    assert results[0].read_bytes() == b"plugin"
    assert calls == ["-S", "--build"]
    assert stat_mode(results[0].parent) == 0o700
    assert (results[0].parent / ".complete").is_file()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


class _FakeExport:
    def __init__(self, value):
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


def _valid_fake_handle() -> SimpleNamespace:
    return SimpleNamespace(
        **{
            native_plugin_builder._PLUGIN_IDENTITY_SYMBOL: _FakeExport(
                native_plugin_builder._PLUGIN_IDENTITY.encode("utf-8")
            ),
            native_plugin_builder._PLUGIN_ABI_SYMBOL: _FakeExport(
                native_plugin_builder._PLUGIN_ABI_VERSION
            ),
            native_plugin_builder._PLUGIN_BUILD_IDENTITY_SYMBOL: _FakeExport(b"cache-key"),
            native_plugin_builder._PLUGIN_REGISTRY_SYMBOL: _FakeExport(True),
        }
    )


def test_native_plugin_load_validates_identity_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    library = tmp_path / "libnative.so"
    library.write_bytes(b"plugin")
    handle = _valid_fake_handle()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(native_plugin_builder, "ensure_native_plugin", lambda **_kwargs: library)
    monkeypatch.setattr(
        native_plugin_builder.ctypes,
        "CDLL",
        lambda path, mode: calls.append((path, mode)) or handle,
    )
    native_plugin_builder._PLUGIN_HANDLES.clear()
    try:
        assert native_plugin_builder.load_native_plugin() == library
        assert native_plugin_builder.load_native_plugin() == library
        assert calls == [(str(library), native_plugin_builder.ctypes.RTLD_GLOBAL)]
        assert native_plugin_builder._PLUGIN_HANDLES == {library: handle}
    finally:
        native_plugin_builder._PLUGIN_HANDLES.clear()


def test_native_plugin_load_rejects_wrong_abi(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "libnative.so"
    library.write_bytes(b"plugin")
    handle = _valid_fake_handle()
    getattr(handle, native_plugin_builder._PLUGIN_ABI_SYMBOL).value = 2
    monkeypatch.setattr(native_plugin_builder, "ensure_native_plugin", lambda **_kwargs: library)
    monkeypatch.setattr(native_plugin_builder.ctypes, "CDLL", lambda *_args, **_kwargs: handle)
    native_plugin_builder._PLUGIN_HANDLES.clear()
    try:
        with pytest.raises(RuntimeError, match="ABI mismatch"):
            native_plugin_builder.load_native_plugin()
        assert not native_plugin_builder._PLUGIN_HANDLES
        assert native_plugin_builder._FAILED_PLUGIN_HANDLES == {library: handle}
    finally:
        native_plugin_builder._PLUGIN_HANDLES.clear()


def test_native_plugin_load_rejects_creator_registry_collision(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "libnative.so"
    library.write_bytes(b"plugin")
    handle = _valid_fake_handle()
    getattr(handle, native_plugin_builder._PLUGIN_REGISTRY_SYMBOL).value = False
    monkeypatch.setattr(native_plugin_builder, "ensure_native_plugin", lambda **_kwargs: library)
    monkeypatch.setattr(native_plugin_builder.ctypes, "CDLL", lambda *_args, **_kwargs: handle)
    native_plugin_builder._PLUGIN_HANDLES.clear()
    try:
        with pytest.raises(RuntimeError, match="do not own their TensorRT registry entries"):
            native_plugin_builder.load_native_plugin()
        assert not native_plugin_builder._PLUGIN_HANDLES
        assert native_plugin_builder._FAILED_PLUGIN_HANDLES == {library: handle}
    finally:
        native_plugin_builder._PLUGIN_HANDLES.clear()


@pytest.mark.parametrize(
    ("adder", "plugin_name", "layer_name"),
    (
        (
            native_plugin_builder.add_vision_attention_plugin,
            "MiniMaxH3VisionAttention",
            "vision_block_0_attention",
        ),
        (
            native_plugin_builder.add_patch_embed_plugin,
            "MiniMaxH3PatchEmbed",
            "vision_patch_embed",
        ),
        (
            native_plugin_builder.add_linear_plugin,
            "MiniMaxH3Linear",
            "vision_qkv",
        ),
        (
            native_plugin_builder.add_layer_norm_plugin,
            "MiniMaxH3LayerNorm",
            "vision_norm1",
        ),
    ),
)
def test_add_fixed_plugin_uses_v3_with_three_inputs(
    monkeypatch, adder, plugin_name: str, layer_name: str
) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields, phase):
            assert name == layer_name
            assert fields == []
            assert phase == "build"
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""
        metadata = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.shape_inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v3(self, inputs, shape_inputs, selected_plugin):
            self.inputs = inputs
            self.shape_inputs = shape_inputs
            self.plugin = selected_plugin
            return self.layer

    def plugin_creator(_trt, requested_name: str):
        assert requested_name == plugin_name
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", plugin_creator)
    tensors = [object(), object(), object()]
    network = Network()
    trt_module = SimpleNamespace(
        PluginFieldCollection=lambda fields: fields,
        TensorRTPhase=SimpleNamespace(BUILD="build"),
    )

    output = adder(
        network,
        *tensors,
        trt_module=trt_module,
        name=layer_name,
    )

    assert network.inputs == tensors
    assert network.shape_inputs == []
    assert network.plugin is plugin
    assert network.layer.name == layer_name
    assert network.layer.metadata == f"trtmc.native_op={plugin_name};source={layer_name}"
    assert output.name == layer_name


def test_add_audio_encoder_plugin_embeds_one_constant_and_uses_two_v3_inputs(monkeypatch) -> None:
    selected_plugin = object()
    observed = {}

    class Creator:
        def create_plugin(self, name, fields, phase):
            observed.update(name=name, fields=fields, phase=phase)
            return selected_plugin

    class Output:
        def __init__(self) -> None:
            self.name = ""

    class ConstantLayer:
        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Layer:
        name = ""
        metadata = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.layer = Layer()

        def add_constant(self, shape, values):
            observed.update(constant_shape=shape, constant_values=values)
            self.constant_layer = ConstantLayer()
            return self.constant_layer

        def add_plugin_v3(self, inputs, shape_inputs, plugin):
            observed.update(inputs=inputs, shape_inputs=shape_inputs, plugin=plugin)
            return self.layer

    def plugin_creator(_trt, plugin_name):
        assert plugin_name == "MiniMaxH3AudioEncoder"
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", plugin_creator)
    monkeypatch.setattr(native_plugin_builder, "_AUDIO_ENCODER_MODULE_MIN_BYTES", 4)
    monkeypatch.setattr(native_plugin_builder, "_AUDIO_ENCODER_MODULE_MAX_BYTES", 32)
    trt_module = SimpleNamespace(
        PluginFieldCollection=lambda fields: fields,
        TensorRTPhase=SimpleNamespace(BUILD="build"),
    )
    network = Network()
    audio_samples = object()
    module_bytes = b"PK\x03\x04torchscript"

    output = native_plugin_builder.add_audio_encoder_plugin(
        network,
        audio_samples,
        module_bytes,
        trt_module=trt_module,
        name="audio_encoder",
    )

    assert observed["name"] == "audio_encoder"
    assert observed["phase"] == "build"
    assert observed["fields"] == []
    assert observed["constant_shape"] == (len(module_bytes),)
    assert observed["constant_values"].dtype == np.int8
    assert observed["constant_values"].tobytes() == module_bytes
    assert observed["inputs"] == [audio_samples, network.constant_layer.output]
    assert observed["shape_inputs"] == []
    assert observed["plugin"] is selected_plugin
    assert network.layer.metadata == (
        "trtmc.native_op=MiniMaxH3AudioEncoder;source=audio_encoder;module_bytes=15;"
        "module_sha256=a1e8a27d0f091b572ac3dadb8361d57b33f53a0696c13f6f6d6c3e7358efdd7d"
    )
    assert output.name == "audio_encoder"
    assert id(network) in native_plugin_builder._AUDIO_ENCODER_MODULE_KEEPALIVE
    native_plugin_builder.release_audio_encoder_module_storage(network)
    assert id(network) not in native_plugin_builder._AUDIO_ENCODER_MODULE_KEEPALIVE


def test_add_audio_encoder_plugin_rejects_empty_module() -> None:
    with pytest.raises(ValueError, match="between 300 and 400 MiB"):
        native_plugin_builder.add_audio_encoder_plugin(object(), object(), b"", trt_module=object())


def test_add_audio_encoder_plugin_releases_keepalive_on_late_layer_error(monkeypatch) -> None:
    class Output:
        pass

    class ConstantLayer:
        def get_output(self, index):
            assert index == 0
            return Output()

    class RaisingLayer:
        @property
        def name(self):
            return ""

        @name.setter
        def name(self, _value):
            raise RuntimeError("late layer failure")

    class Network:
        def add_constant(self, _shape, _values):
            return ConstantLayer()

        def add_plugin_v3(self, _inputs, _shape_inputs, _plugin):
            return RaisingLayer()

    class Creator:
        def create_plugin(self, _name, _fields, _phase):
            return object()

    monkeypatch.setattr(native_plugin_builder, "_AUDIO_ENCODER_MODULE_MIN_BYTES", 4)
    monkeypatch.setattr(native_plugin_builder, "_AUDIO_ENCODER_MODULE_MAX_BYTES", 8)
    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", lambda *_args: Creator())
    network = Network()
    trt_module = SimpleNamespace(
        PluginFieldCollection=lambda fields: fields,
        TensorRTPhase=SimpleNamespace(BUILD="build"),
    )
    with pytest.raises(RuntimeError, match="late layer failure"):
        native_plugin_builder.add_audio_encoder_plugin(
            network,
            object(),
            b"PK\x03\x04",
            trt_module=trt_module,
        )
    assert id(network) not in native_plugin_builder._AUDIO_ENCODER_MODULE_KEEPALIVE


def test_audio_encoder_plugin_caches_one_stream_ordered_module_load_per_instance() -> None:
    source = next(
        path
        for path in native_plugin_builder.native_plugin_source_files()
        if path.name == "audio_encoder_plugin.cpp"
    ).read_text()
    assert "std::lock_guard<std::mutex> lock(mutex)" in source
    assert "if (module.has_value())" in source
    assert "cudaMemcpyAsync" in source
    assert "cudaMemcpyDeviceToHost, stream" in source
    assert "cudaStreamSynchronize(stream)" in source
    assert "std::vector<std::uint8_t>().swap(host_module_bytes)" in source
    assert source.index("loaded_module->eval()") < source.index(
        "module.emplace(std::move(*loaded_module))"
    )
    assert "GraphOptimizerEnabledGuard optimizer_guard(false)" in source
