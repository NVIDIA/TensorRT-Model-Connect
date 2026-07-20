# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from tensorrt_model_connect import trt_compat


REPO_ROOT = Path(__file__).resolve().parents[2]
TRTMC_BUILD_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect"
ALLOWED_TRT_BOUNDARY_FILES = {
    TRTMC_BUILD_ROOT / "trt_compat.py",
}


class _FakeNativeFunction:
    def __init__(self, result, calls: list | None = None):
        self.result = result
        self.calls = calls
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self.calls is not None:
            self.calls.append(args)
        return self.result


class _FakeLibnvinfer:
    def __init__(
        self,
        version: tuple[int, int, int],
        *,
        configured: bool = True,
        configure_calls: list | None = None,
    ):
        self.getInferLibMajorVersion = _FakeNativeFunction(version[0])
        self.getInferLibMinorVersion = _FakeNativeFunction(version[1])
        self.getInferLibPatchVersion = _FakeNativeFunction(version[2])
        self.setInternalLibraryPath = _FakeNativeFunction(configured, configure_calls)


def _native_like_trt_module(tmp_path: Path, version: str = "11.2.0.113"):
    module = types.ModuleType("tensorrt")
    module.__version__ = version
    module.__file__ = str(tmp_path / "site-packages/tensorrt/__init__.py")
    module.Builder = lambda *_args, **_kwargs: object()
    return module


def _install_fake_trt_module(monkeypatch, module) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", module)
    monkeypatch.setattr(trt_compat, "_module", None)
    monkeypatch.setattr(trt_compat, "_backend_module_name", "tensorrt")
    monkeypatch.setattr(trt_compat, "_backend_label", "TensorRT")
    monkeypatch.setattr(trt_compat, "_internal_library_path_state", None)
    monkeypatch.setattr(trt_compat, "_internal_library_handle", None)


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_importlib_import_module(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "import_module"
        and isinstance(node.value, ast.Name)
        and node.value.id == "importlib"
    )


def _is_sys_modules_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
        and _constant_string(node.slice) in {"tensorrt", "tensorrt_rtx"}
    )


def test_tensor_rt_python_api_is_imported_only_through_compat_layer():
    """Builder code must route TensorRT Python API access through trt_compat."""
    violations: list[str] = []
    for path in sorted(TRTMC_BUILD_ROOT.rglob("*.py")):
        if path in ALLOWED_TRT_BOUNDARY_FILES:
            continue
        rel = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in {"tensorrt", "tensorrt_rtx"}:
                    violations.append(f"{rel}:{node.lineno} imports from {node.module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    if node.args and _constant_string(node.args[0]) in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} dynamically imports TensorRT")
                elif _is_importlib_import_module(node.func):
                    if node.args and _constant_string(node.args[0]) in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} dynamically imports TensorRT")
            elif _is_sys_modules_subscript(node) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                violations.append(f"{rel}:{node.lineno} mutates sys.modules TensorRT alias")
            elif isinstance(node, ast.Attribute) and node.attr == "EXPLICIT_BATCH":
                violations.append(f"{rel}:{node.lineno} uses EXPLICIT_BATCH directly")

    assert violations == []


def test_wan22_attention_probe_bootstraps_source_tree(tmp_path):
    """The tracked qualification launchers can execute the probe without an install."""
    script = (
        TRTMC_BUILD_ROOT
        / "families"
        / "wan2_2_ti2v"
        / "dit_attention_probe"
        / "build_trt_probe.py"
    )
    wrapper = (
        "import runpy,sys,types;"
        "sys.modules['numpy']=types.ModuleType('numpy');"
        "script=sys.argv[1];"
        "sys.argv=[script,'--help'];"
        "runpy.run_path(script,run_name='__main__')"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", wrapper, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--plugin" in result.stdout


def test_wan22_time_probe_bootstraps_before_compat_import():
    """The time-path probe's existing source bootstrap must precede package imports."""
    script = (
        TRTMC_BUILD_ROOT
        / "families"
        / "wan2_2_ti2v"
        / "dit_time_probe"
        / "qualify_current_trt_time_path.py"
    )
    source = script.read_text(encoding="utf-8")

    assert source.index("sys.path.insert(0, str(PYTHON_ROOT))") < source.index(
        "from tensorrt_model_connect.trt_compat import trt"
    )


def test_trt_compat_configures_internal_builder_library_path_once(monkeypatch, tmp_path):
    lib_dir = tmp_path / "sdk/lib"
    lib_dir.mkdir(parents=True)
    libnvinfer = lib_dir / "libnvinfer.so.11.2.0"
    libnvinfer.touch()
    (lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.0").touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)

    configure_calls: list[tuple[bytes]] = []
    handle = _FakeLibnvinfer((11, 2, 0), configure_calls=configure_calls)
    cdll_calls: list[str] = []
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(
        trt_compat.ctypes,
        "CDLL",
        lambda path: (cdll_calls.append(path), handle)[1],
    )

    trt = trt_compat.get_trt()
    trt.Builder(None)
    trt.Builder(None)

    # The setter is process-global and runs once, but every Builder call must
    # rescan the mapped DSOs so a late-loaded conflicting TensorRT is detected.
    assert cdll_calls == [str(libnvinfer), str(libnvinfer)]
    assert configure_calls == [(str(lib_dir).encode(),)]
    assert handle.setInternalLibraryPath.argtypes == [trt_compat.ctypes.c_char_p]
    assert handle.setInternalLibraryPath.restype is trt_compat.ctypes.c_bool


def test_trt_compat_configures_internal_builder_library_path_once_concurrently(
    monkeypatch, tmp_path
):
    lib_dir = tmp_path / "sdk/lib"
    lib_dir.mkdir(parents=True)
    libnvinfer = lib_dir / "libnvinfer.so.11.2.0"
    libnvinfer.touch()
    (lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.0").touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)

    configure_calls: list[tuple[bytes]] = []

    def slow_configure(*args):
        configure_calls.append(args)
        time.sleep(0.05)
        return True

    handle = _FakeLibnvinfer((11, 2, 0))
    handle.setInternalLibraryPath = slow_configure
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: handle)

    trt = trt_compat.get_trt()
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def create_builder():
        try:
            start.wait()
            trt.Builder(None)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=create_builder) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert configure_calls == [(str(lib_dir).encode(),)]


def test_trt_compat_rejects_loaded_libnvinfer_version_mismatch(monkeypatch, tmp_path):
    libnvinfer = tmp_path / "libnvinfer.so.11.2.1"
    libnvinfer.touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: _FakeLibnvinfer((11, 2, 1)))

    with pytest.raises(
        RuntimeError,
        match=r"TensorRT Python version 11\.2\.0 does not match.*version 11\.2\.1",
    ):
        trt_compat.get_trt().Builder(None)


def test_trt_compat_rejects_mixed_loaded_libnvinfer_versions(monkeypatch, tmp_path):
    matching = tmp_path / "matching/libnvinfer.so.11.2.0"
    mismatched = tmp_path / "mismatched/libnvinfer.so.10.16.1"
    matching.parent.mkdir()
    mismatched.parent.mkdir()
    matching.touch()
    mismatched.touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)
    monkeypatch.setattr(
        trt_compat,
        "loaded_libnvinfer_paths",
        lambda: [str(matching), str(mismatched)],
    )

    def fake_cdll(path):
        version = (11, 2, 0) if Path(path) == matching else (10, 16, 1)
        return _FakeLibnvinfer(version)

    monkeypatch.setattr(trt_compat.ctypes, "CDLL", fake_cdll)

    with pytest.raises(RuntimeError, match="Multiple TensorRT versions are loaded"):
        trt_compat.get_trt().Builder(None)


def test_trt_compat_rejects_ambiguous_matching_libnvinfer_directories(monkeypatch, tmp_path):
    first = tmp_path / "one/libnvinfer.so.11.2.0"
    second = tmp_path / "two/libnvinfer.so.11.2.0"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)
    monkeypatch.setattr(
        trt_compat,
        "loaded_libnvinfer_paths",
        lambda: [str(first), str(second)],
    )
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: _FakeLibnvinfer((11, 2, 0)))

    with pytest.raises(RuntimeError, match="matching libnvinfer DSOs in multiple"):
        trt_compat.get_trt().Builder(None)


def test_trt_compat_reports_missing_or_rejected_builder_resource(monkeypatch, tmp_path):
    lib_dir = tmp_path / "sdk/lib"
    lib_dir.mkdir(parents=True)
    libnvinfer = lib_dir / "libnvinfer.so.11.2.0"
    libnvinfer.touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)
    handle = _FakeLibnvinfer((11, 2, 0), configured=False)
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: handle)

    with pytest.raises(RuntimeError, match=r"resources for version 11\.2\.0 were not found"):
        trt_compat.get_trt().Builder(None)

    (lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.0").touch()
    with pytest.raises(RuntimeError, match="rejected its internal builder library path"):
        trt_compat.get_trt().Builder(None)


@pytest.mark.parametrize(
    "invalid_resource",
    ("wrong_patch", "major_only", "dangling_symlink", "directory"),
)
def test_trt_compat_rejects_non_exact_or_non_file_builder_resource(
    monkeypatch, tmp_path, invalid_resource
):
    lib_dir = tmp_path / "sdk/lib"
    lib_dir.mkdir(parents=True)
    libnvinfer = lib_dir / "libnvinfer.so.11.2.0"
    libnvinfer.touch()
    if invalid_resource == "wrong_patch":
        resource = lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.1"
        resource.touch()
    elif invalid_resource == "major_only":
        resource = lib_dir / "libnvinfer_builder_resource_sm100.so.11"
        resource.touch()
    elif invalid_resource == "dangling_symlink":
        resource = lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.0"
        resource.symlink_to(lib_dir / "missing-builder-resource.so.11.2.0")
    else:
        resource = lib_dir / "libnvinfer_builder_resource_sm100.so.11.2.0"
        resource.mkdir()

    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)
    configure_calls: list[tuple[bytes]] = []
    handle = _FakeLibnvinfer(
        (11, 2, 0),
        configure_calls=configure_calls,
    )
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: handle)

    with pytest.raises(
        RuntimeError,
        match=r"resources for version 11\.2\.0 were not found",
    ):
        trt_compat.get_trt().Builder(None)
    assert configure_calls == []


@pytest.mark.parametrize(
    ("late_version", "expected_error"),
    (
        ((10, 16, 1), "Multiple TensorRT versions are loaded"),
        ((11, 2, 0), "matching libnvinfer DSOs in multiple directories"),
    ),
)
def test_trt_compat_rescans_mapped_libraries_after_first_builder(
    monkeypatch, tmp_path, late_version, expected_error
):
    first = tmp_path / "first/libnvinfer.so.11.2.0"
    second = (
        tmp_path
        / "second"
        / ("libnvinfer.so.11.2.0" if late_version == (11, 2, 0) else "libnvinfer.so.10.16.1")
    )
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    (first.parent / "libnvinfer_builder_resource_sm100.so.11.2.0").touch()
    module = _native_like_trt_module(tmp_path)
    _install_fake_trt_module(monkeypatch, module)

    configure_calls: list[tuple[bytes]] = []
    first_handle = _FakeLibnvinfer(
        (11, 2, 0),
        configure_calls=configure_calls,
    )
    second_handle = _FakeLibnvinfer(late_version)
    mapped = [str(first)]
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: mapped)
    monkeypatch.setattr(
        trt_compat.ctypes,
        "CDLL",
        lambda path: first_handle if Path(path) == first else second_handle,
    )

    trt = trt_compat.get_trt()
    trt.Builder(None)
    mapped.append(str(second))
    with pytest.raises(RuntimeError, match=expected_error):
        trt.Builder(None)

    assert configure_calls == [(str(first.parent).encode(),)]


def test_trt_compat_preserves_older_library_without_internal_path_api(monkeypatch, tmp_path):
    libnvinfer = tmp_path / "libnvinfer.so.10.16.1"
    libnvinfer.touch()
    module = _native_like_trt_module(tmp_path, version="10.16.1.11")
    _install_fake_trt_module(monkeypatch, module)
    handle = _FakeLibnvinfer((10, 16, 1))
    del handle.setInternalLibraryPath
    monkeypatch.setattr(trt_compat, "loaded_libnvinfer_paths", lambda: [str(libnvinfer)])
    monkeypatch.setattr(trt_compat.ctypes, "CDLL", lambda _path: handle)

    trt_compat.get_trt().Builder(None)
    assert trt_compat._internal_library_path_state is None


@pytest.mark.parametrize("backend", ("tensorrt", "tensorrt_rtx"))
def test_trt_compat_preserves_fake_and_rtx_module_loader_behavior(monkeypatch, backend):
    module = types.ModuleType(backend)
    module.__version__ = "11.2.0.113"
    module.__file__ = f"/fake/{backend}/__init__.py"
    module.Builder = lambda *_args, **_kwargs: object()
    module.Runtime = lambda *_args, **_kwargs: object()
    monkeypatch.setitem(sys.modules, backend, module)
    monkeypatch.setattr(trt_compat, "_module", None)
    monkeypatch.setattr(trt_compat, "_backend_module_name", backend)
    monkeypatch.setattr(
        trt_compat,
        "loaded_libnvinfer_paths",
        lambda: (_ for _ in ()).throw(AssertionError("native discovery must not run")),
    )

    trt = trt_compat.get_trt()
    if backend == "tensorrt_rtx":
        trt.Builder(None)
    else:
        trt.Runtime(None)


def test_trt_compat_proxy_wraps_version_sensitive_builder_calls(monkeypatch):
    """The lazy trt proxy intercepts builder/network/config calls with a fake TRT module."""
    calls: list[tuple] = []

    class FakeLogger:
        WARNING = 1
        VERBOSE = 2

        def __init__(self, level):
            self.level = level

    class FakeNetwork:
        def __init__(self, flags):
            self.flags = flags

        def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
            calls.append(("add_matrix_multiply", lhs, lhs_op, rhs, rhs_op))
            return "layer"

    class FakeConfig:
        def set_memory_pool_limit(self, pool, size):
            calls.append(("set_memory_pool_limit", pool, size))

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags=0):
            calls.append(("create_network", flags))
            return FakeNetwork(flags)

        def create_builder_config(self):
            return FakeConfig()

        def build_serialized_network(self, network, config):
            assert isinstance(network, FakeNetwork)
            assert isinstance(config, FakeConfig)
            return b"plan"

    class FakeRuntime:
        pass

    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "10.16.1.11"
    fake_trt.Logger = FakeLogger
    fake_trt.Builder = FakeBuilder
    fake_trt.Runtime = FakeRuntime
    fake_trt.IBuilder = FakeBuilder
    fake_trt.INetworkDefinition = FakeNetwork
    fake_trt.IBuilderConfig = FakeConfig
    fake_trt.IRuntime = FakeRuntime
    fake_trt.MemoryPoolType = types.SimpleNamespace(WORKSPACE="workspace")
    fake_trt.NetworkDefinitionCreationFlag = types.SimpleNamespace(STRONGLY_TYPED=1)

    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", None)
    monkeypatch.setattr(trt_compat, "_backend_module_name", "tensorrt")
    monkeypatch.setattr(trt_compat, "_backend_label", "TensorRT")

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024)
    layer = network.add_matrix_multiply("lhs", "none", "rhs", "transpose")
    plan = builder.build_serialized_network(network, config)

    assert flags == 2
    assert layer == "layer"
    assert plan == b"plan"
    assert calls == [
        ("create_network", 2),
        ("set_memory_pool_limit", "workspace", 1024),
        ("add_matrix_multiply", "lhs", "none", "rhs", "transpose"),
    ]


def test_trt_compat_applies_builder_env_and_persists_timing_cache(monkeypatch, tmp_path):
    calls: list[tuple] = []
    cache_path = tmp_path / "trt.cache"

    class FakeLogger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    class FakeNetwork:
        pass

    class FakeTimingCache:
        def __init__(self, payload=b""):
            self.payload = bytearray(payload)

        def serialize(self):
            return bytes(self.payload)

        def combine(self, input_cache, ignore_mismatch):
            calls.append(("combine", bytes(input_cache.payload), ignore_mismatch))
            self.payload.extend(input_cache.payload)
            return True

    class FakeConfig:
        def __init__(self):
            self.builder_optimization_level = -1
            self.max_num_tactics = -1
            self.avg_timing_iterations = -1
            self.timing_cache = None

        def create_timing_cache(self, payload):
            calls.append(("create_timing_cache", bytes(payload)))
            return FakeTimingCache(payload)

        def set_timing_cache(self, cache, ignore_mismatch):
            calls.append(("set_timing_cache", ignore_mismatch))
            self.timing_cache = cache
            return True

        def get_timing_cache(self):
            calls.append(("get_timing_cache",))
            return self.timing_cache

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags=0):
            return FakeNetwork()

        def create_builder_config(self):
            return FakeConfig()

        def build_serialized_network(self, network, config):
            assert isinstance(network, FakeNetwork)
            assert isinstance(config, FakeConfig)
            calls.append((
                "build_config",
                config.builder_optimization_level,
                config.max_num_tactics,
                config.avg_timing_iterations,
            ))
            config.timing_cache.payload.extend(b"built")
            return b"plan"

    class FakeRuntime:
        pass

    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "10.16.1.11"
    fake_trt.Logger = FakeLogger
    fake_trt.Builder = FakeBuilder
    fake_trt.Runtime = FakeRuntime
    fake_trt.IBuilder = FakeBuilder
    fake_trt.INetworkDefinition = FakeNetwork
    fake_trt.IBuilderConfig = FakeConfig
    fake_trt.IRuntime = FakeRuntime

    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", None)
    monkeypatch.setattr(trt_compat, "_backend_module_name", "tensorrt")
    monkeypatch.setattr(trt_compat, "_backend_label", "TensorRT")
    monkeypatch.setenv("TRTMC_TRT_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "1")
    monkeypatch.setenv("TRTMC_MAX_NUM_TACTICS", "8")
    monkeypatch.setenv("TRTMC_AVG_TIMING_ITERATIONS", "1")

    trt = trt_compat.get_trt()
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    plan = builder.build_serialized_network(
        builder.create_network(), builder.create_builder_config())

    assert plan == b"plan"
    assert cache_path.read_bytes() == b"built"
    assert ("build_config", 1, 8, 1) in calls
    assert ("create_timing_cache", b"") in calls
    assert ("set_timing_cache", True) in calls
    assert ("get_timing_cache",) in calls


def test_trt_compat_scoped_timing_cache_uses_separate_path(monkeypatch, tmp_path):
    cache_path = tmp_path / "tensorrt-opt1.cache"
    monkeypatch.setenv("TRTMC_TRT_TIMING_CACHE_PATH", str(cache_path))

    with trt_compat.scoped_timing_cache("split example/decode"):
        assert (
            trt_compat._timing_cache_path()
            == tmp_path / "tensorrt-opt1.split_example_decode.cache"
        )

    assert trt_compat._timing_cache_path() == cache_path
