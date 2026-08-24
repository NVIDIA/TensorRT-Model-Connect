# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

from tensorrt_model_connect import trt_compat


REPO_ROOT = Path(__file__).resolve().parents[2]
TRTMC_BUILD_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect"
ALLOWED_TRT_BOUNDARY_FILES = {
    TRTMC_BUILD_ROOT / "trt_compat.py",
}


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


def test_tensorrt_abi_contract() -> None:
    assert trt_compat.tensorrt_abi("11.1.0.106") == "11.1"
    assert trt_compat.tensorrt_abi("11.123.0.999") == "11.123"
    assert trt_compat.tensorrt_abi("11") == ""
    assert trt_compat.tensorrt_abi("unknown") == ""
    assert trt_compat.tensorrt_abi("TensorRT 11.2.0") == "11.2"


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
            self.layers: list[object] = []

        @property
        def num_layers(self) -> int:
            return len(self.layers)

        def add_identity(self, value):
            calls.append(("add_identity", value))
            layer = object()
            self.layers.append(layer)
            return layer

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
    identity = network.add_identity("value")
    layer = network.add_matrix_multiply("lhs", "none", "rhs", "transpose")
    plan = builder.build_serialized_network(network, config)

    assert flags == 2
    assert identity is trt_compat.unwrap(network).layers[0]
    assert layer == "layer"
    assert plan == b"plan"
    assert calls == [
        ("create_network", 2),
        ("set_memory_pool_limit", "workspace", 1024),
        ("add_identity", "value"),
        ("add_matrix_multiply", "lhs", "none", "rhs", "transpose"),
    ]


def test_get_plugin_creator_uses_trt11_registry_api(monkeypatch):
    creator = object()
    registry = types.SimpleNamespace(
        get_creator=lambda name, version, namespace: (
            creator if (name, version, namespace) == ("Plugin", "1", "") else None
        )
    )
    monkeypatch.setattr(
        trt_compat,
        "get_trt",
        lambda: types.SimpleNamespace(get_plugin_registry=lambda: registry),
    )

    assert trt_compat.get_plugin_creator("Plugin", "1") is creator


def test_get_plugin_creator_falls_back_to_trt10_registry_api(monkeypatch):
    creator = object()
    registry = types.SimpleNamespace(
        get_plugin_creator=lambda name, version, namespace: (
            creator if (name, version, namespace) == ("Plugin", "1", "") else None
        )
    )
    monkeypatch.setattr(
        trt_compat,
        "get_trt",
        lambda: types.SimpleNamespace(get_plugin_registry=lambda: registry),
    )

    assert trt_compat.get_plugin_creator("Plugin", "1") is creator


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
