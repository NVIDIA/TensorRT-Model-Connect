# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import BuildRequest


build_core = importlib.import_module("tensorrt_model_connect.build")


def _request(tmp_path: Path, *, family: str = "example") -> BuildRequest:
    return BuildRequest(
        model_dir=tmp_path / "model",
        output_path=tmp_path / "model.bundle",
        precision="fp16",
        family=family,
        task="text_generation",
        tensor_parallel_size=2,
        context_parallel_size=3,
    )


def test_build_request_is_a_plain_frozen_dataclass(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.precision = "fp32"  # type: ignore[misc]

    assert BuildRequest.__bases__ == (object,)
    assert request.tensor_parallel_size == 2
    assert request.context_parallel_size == 3
    assert request.backend == "trt"
    assert request.dynamic_kv_cache is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_sequence_length", 0),
        ("image_height", 0),
        ("image_width", 0),
        ("video_num_frames", 0),
        ("max_batch_size", 0),
        ("tensor_parallel_size", 0),
        ("context_parallel_size", 0),
        ("quantization", ""),
        ("fp32_layers", (-1,)),
        ("dynamic_kv_cache", 1),
        ("graph_transform", object()),
        ("backend", "unknown"),
    ],
)
def test_build_request_rejects_invalid_direct_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    kwargs = {
        "model_dir": tmp_path / "model",
        "output_path": tmp_path / "model.bundle",
        "family": "example",
        "task": "text_generation",
        "precision": "fp16",
        field: value,
    }
    with pytest.raises(ValueError):
        BuildRequest(**kwargs)  # type: ignore[arg-type]


def test_resolver_returns_only_the_explicit_family(tmp_path: Path) -> None:
    assert build_core._resolve_family(_request(tmp_path, family="exact_family")) == "exact_family"


@pytest.mark.parametrize("family", ["", "../other", "foo.bar", "MixedCase", "a-b"])
def test_family_rejects_names_that_are_not_safe_directories(tmp_path: Path, family: str) -> None:
    with pytest.raises(ValueError, match="lowercase identifier"):
        build_core._resolve_family(_request(tmp_path, family=family))


def test_load_family_imports_only_the_exact_model_module(monkeypatch) -> None:
    imported: list[str] = []
    expected = SimpleNamespace(build=lambda request, writer: None)

    def fake_import(name: str):
        imported.append(name)
        return expected

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert build_core._load_family("exact_family") is expected
    assert imported == ["families.exact_family.model"]


def test_rtx_backend_is_bound_before_family_import(monkeypatch) -> None:
    standard = object()
    rtx = object()
    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setitem(sys.modules, "tensorrt_rtx", rtx)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: rtx if name == "tensorrt_rtx" else standard,
    )

    build_core._select_backend("trt_rtx")

    assert sys.modules["tensorrt"] is rtx


def test_backend_cannot_switch_after_tensor_rt_is_loaded(monkeypatch) -> None:
    standard = object()
    rtx = object()
    monkeypatch.setitem(sys.modules, "tensorrt", standard)
    monkeypatch.setitem(sys.modules, "tensorrt_rtx", rtx)
    monkeypatch.setattr(importlib, "import_module", lambda _name: rtx)

    with pytest.raises(RuntimeError, match="already loaded"):
        build_core._select_backend("trt_rtx")

    monkeypatch.setitem(sys.modules, "tensorrt", rtx)
    with pytest.raises(RuntimeError, match="already loaded"):
        build_core._select_backend("trt")


def test_family_internal_import_error_is_not_wrapped(monkeypatch) -> None:
    internal_error = ImportError("family dependency failed")

    def fail_import(_name: str):
        raise internal_error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as caught:
        build_core._load_family("exact_family")
    assert caught.value is internal_error


def test_family_internal_module_not_found_error_is_not_wrapped(monkeypatch) -> None:
    internal_error = ModuleNotFoundError(
        "No module named 'family_dependency'", name="family_dependency"
    )

    def fail_import(_name: str):
        raise internal_error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError) as caught:
        build_core._load_family("exact_family")
    assert caught.value is internal_error


def test_build_finishes_after_family_returns(monkeypatch, tmp_path: Path) -> None:
    events: list[object] = []

    class FakeWriter:
        def __init__(self, destination: Path) -> None:
            events.append(("writer", destination))

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(request: BuildRequest, writer: FakeWriter) -> None:
        events.append(("build", request, writer))

    request = _request(tmp_path)
    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    assert build_core.build(request) is None
    assert events[0] == ("writer", request.output_path)
    assert events[1][0:2] == ("build", request)
    assert events[2:] == ["finish"]


def test_build_runs_graph_transform_before_family_engine_serialization(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[object] = []

    class FakeTrtBuilder:
        def __init__(self, _logger: object) -> None:
            pass

        def build_serialized_network(self, network: object, _config: object) -> bytes:
            events.append(("serialize", network))
            return b"engine"

    fake_trt = SimpleNamespace(Builder=FakeTrtBuilder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(_request: BuildRequest, _writer: FakeWriter) -> None:
        network = SimpleNamespace(replaced=False)
        fake_trt.Builder("logger").build_serialized_network(network, "config")

    def transform(network: object, engine_index: int) -> None:
        setattr(network, "replaced", True)
        events.append(("transform", network, engine_index))

    request = replace(_request(tmp_path), graph_transform=transform)
    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    build_core.build(request)

    assert events[0][0] == "transform"
    assert events[0][1].replaced is True
    assert events[0][2] == 0
    assert events[1] == ("serialize", events[0][1])
    assert events[2] == "finish"
    assert fake_trt.Builder is FakeTrtBuilder


def test_build_aborts_and_preserves_family_error(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    family_error = RuntimeError("family failed")

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(_request: BuildRequest, _writer: FakeWriter) -> None:
        raise family_error

    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    with pytest.raises(RuntimeError) as caught:
        build_core.build(_request(tmp_path))
    assert caught.value is family_error
    assert events == ["abort"]


def test_build_aborts_if_finish_fails(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")
            raise OSError("publish failed")

        def abort(self) -> None:
            events.append("abort")

    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core,
        "_load_family",
        lambda family: SimpleNamespace(build=lambda request, writer: None),
    )

    with pytest.raises(OSError, match="publish failed"):
        build_core.build(_request(tmp_path))
    assert events == ["finish", "abort"]


@pytest.mark.parametrize("value", ["", "/one", "/one:/two", ":/one::/two:"])
def test_cmake_prefixes_preserve_standard_search_order(monkeypatch, value):
    monkeypatch.setenv("CMAKE_PREFIX_PATH", value)
    monkeypatch.setattr(build_core.sys, "prefix", "/python")
    expected = [Path(item) for item in value.split(build_core.os.pathsep) if item]
    assert build_core.cmake_prefixes() == [*expected, Path("/python")]


def test_cmake_prefixes_without_environment_use_python_prefix(monkeypatch):
    monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
    monkeypatch.setattr(build_core.sys, "prefix", "/python")
    assert build_core.cmake_prefixes() == [Path("/python")]
    # Constructing explicit child-tool settings must not change the caller's
    # package search order or mutate an inherited search path.
    monkeypatch.setenv("TEST_TOOL_SEARCH_PATH", "/original")
    monkeypatch.delenv("TEST_TOOL_NEW_PATH", raising=False)
    child = build_core.subprocess_environment(
        {"CMAKE_PREFIX_PATH": "/child"},
        prepend_paths={"TEST_TOOL_SEARCH_PATH": "/first", "TEST_TOOL_NEW_PATH": "/new"},
    )
    assert child["CMAKE_PREFIX_PATH"] == "/child"
    assert child["TEST_TOOL_SEARCH_PATH"] == "/first" + build_core.os.pathsep + "/original"
    assert child["TEST_TOOL_NEW_PATH"] == "/new"
    assert build_core.cmake_prefixes() == [Path("/python")]
    assert build_core.os.environ["TEST_TOOL_SEARCH_PATH"] == "/original"
    assert "TEST_TOOL_NEW_PATH" not in build_core.os.environ


@pytest.fixture
def native_platform_bindings(monkeypatch):
    from unittest.mock import Mock

    runtime = SimpleNamespace(
        cudaGetDevice=Mock(return_value=(0, 3)),
        cudaGetDeviceProperties=Mock(return_value=(0, SimpleNamespace(major=8, minor=6))),
        cudaRuntimeGetVersion=Mock(return_value=(0, 13000)),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="11.1.0.106"))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(build_core.sys, "platform", "linux")
    monkeypatch.setattr(build_core.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        build_core.platform, "freedesktop_os_release", lambda: {"VERSION_ID": "24.04"}
    )
    monkeypatch.setattr(build_core.platform, "release", lambda: "fallback-release")
    monkeypatch.setattr(build_core, "_cuda_toolkit_version", lambda: "13.3")
    return runtime


@pytest.mark.parametrize("release_available", [True, False])
def test_native_platform_uses_executing_cuda_device_and_full_sdk(
    native_platform_bindings, monkeypatch, release_available
):
    if not release_available:
        from unittest.mock import Mock

        monkeypatch.setattr(
            build_core.platform, "freedesktop_os_release", Mock(side_effect=OSError("missing"))
        )
    assert build_core.detect_local_platform() == {
        "os": "linux",
        "os_version": "24.04" if release_available else "fallback-release",
        "arch": "x86_64",
        "sm": 86,
        "cuda_version": "13.3",
        "tensorrt_version": "11.1.0.106",
    }
    native_platform_bindings.cudaGetDevice.assert_called_once_with()
    native_platform_bindings.cudaGetDeviceProperties.assert_called_once_with(3)
    native_platform_bindings.cudaRuntimeGetVersion.assert_not_called()


@pytest.mark.parametrize(
    "failing", ["cudaGetDevice", "cudaGetDeviceProperties"]
)
def test_native_platform_propagates_cuda_discovery_failure(native_platform_bindings, failing):
    getattr(native_platform_bindings, failing).return_value = (35,)
    with pytest.raises(RuntimeError, match="CUDA device discovery failed: 35"):
        build_core.detect_local_platform()


def test_native_platform_retains_nonlinux_identity(native_platform_bindings, monkeypatch):
    monkeypatch.setattr(build_core.sys, "platform", "win32")
    result = build_core.detect_local_platform()
    assert result["os"] == "win32"
    assert result["os_version"] == "fallback-release"

@pytest.mark.parametrize("source", ["CUDACXX", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH", "PATH"])
def test_cuda_toolkit_version_uses_selected_compiler(monkeypatch, source):
    from unittest.mock import Mock

    for name in ("CUDACXX", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(name, raising=False)
    compiler = "/selected/bin/nvcc"
    monkeypatch.setattr(build_core.shutil, "which", lambda _: compiler)
    if source != "PATH":
        monkeypatch.setenv(source, compiler if source == "CUDACXX" else "/selected")
    run = Mock(return_value=SimpleNamespace(stdout="Cuda compilation tools, release 13.3, V13.3.1"))
    monkeypatch.setattr(build_core.subprocess, "run", run)
    assert build_core._cuda_toolkit_version() == "13.3"
    run.assert_called_once_with(
        [compiler, "--version"], check=True, capture_output=True, text=True, timeout=10,
    )


def test_cuda_toolkit_version_does_not_guess_when_missing(monkeypatch):
    for name in ("CUDACXX", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(build_core.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="CUDA toolkit not found"):
        build_core._cuda_toolkit_version()


def test_cuda_toolkit_version_rejects_unrecognized_output(monkeypatch):
    monkeypatch.setenv("CUDACXX", "/selected/nvcc")
    monkeypatch.setattr(build_core.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="Cannot identify CUDA toolkit"):
        build_core._cuda_toolkit_version()

@pytest.mark.parametrize("source, value, expected", [
    ("CUDACXX", '"/tool kit/nvcc" --allow-unsupported-compiler',
     ["/tool kit/nvcc", "--allow-unsupported-compiler"]),
    ("CUDAToolkit_ROOT", "/tool kit", ["/tool kit/bin/nvcc"]),
])
def test_cuda_toolkit_compiler_arguments_and_spaces(monkeypatch, source, value, expected):
    from unittest.mock import Mock

    for name in ("CUDACXX", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(source, value)
    run = Mock(return_value=SimpleNamespace(stdout="release 13.3, V13.3.1"))
    monkeypatch.setattr(build_core.subprocess, "run", run)
    assert build_core._cuda_toolkit_version() == "13.3"
    run.assert_called_once_with(
        [*expected, "--version"], check=True, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("failure", [
    FileNotFoundError("compiler missing"),
    build_core.subprocess.CalledProcessError(1, ["nvcc", "--version"]),
    build_core.subprocess.TimeoutExpired(["nvcc", "--version"], 10),
])
def test_cuda_toolkit_compiler_failures_preserve_cause(monkeypatch, failure):
    from unittest.mock import Mock

    monkeypatch.setenv("CUDACXX", "/selected/nvcc")
    monkeypatch.setattr(build_core.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match="Cannot query CUDA toolkit") as caught:
        build_core._cuda_toolkit_version()
    assert caught.value.__cause__ is failure


def test_cuda_toolkit_malformed_compiler_command_preserves_cause(monkeypatch):
    monkeypatch.setenv("CUDACXX", '"unclosed compiler path')
    monkeypatch.setattr(build_core.subprocess, "run", lambda *_a, **_k: pytest.fail("compiler ran"))
    with pytest.raises(RuntimeError, match="Invalid CUDACXX command") as caught:
        build_core._cuda_toolkit_version()
    assert isinstance(caught.value.__cause__, ValueError)
