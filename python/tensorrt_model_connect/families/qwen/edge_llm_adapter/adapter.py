#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family adapter for qualified TensorRT Edge-LLM profiles."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


CAPSULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]

IMPLEMENTATION_ID = "qwen.tensorrt-edge-llm"
RUNTIME_LIBRARY = "libtrtmc_impl_qwen_tensorrt_edge_llm.so"
EDGE_LLM_PLUGIN = "libNvInfer_edgellm_plugin.so"
PROFILES_ROOT = CAPSULE_ROOT / "profiles"
DEPENDENCY_PATH = CAPSULE_ROOT / "dependency.lock"
IMPLEMENTATION_PATH = CAPSULE_ROOT / "IMPLEMENTATION.toml"
_ENGINE_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_ENGINE_DIR"
_RUNTIME_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_RUNTIME_LIBRARY"
_PLUGIN_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_PLUGIN_LIBRARY"
_EDGE_BUILD_DIR_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_BUILD_DIR"
_TRT_INCLUDE_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_TENSORRT_INCLUDE_DIR"
_TRT_LIBRARY_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_TENSORRT_LIBRARY"
_ONNX_PARSER_INCLUDE_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_ONNX_PARSER_INCLUDE_DIR"
_ONNX_PARSER_LIBRARY_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_ONNX_PARSER_LIBRARY"
_CUDA_INCLUDE_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_CUDA_INCLUDE_DIR"
_CUDART_LIBRARY_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_CUDART_LIBRARY"
_CUDA_DRIVER_LIBRARY_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_CUDA_DRIVER_LIBRARY"
_ALLOW_FAKE_RUNTIME_BUILD_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_ALLOW_FAKE_RUNTIME_BUILD"
_DEPENDENCY_CACHE_ENV = "_TRTMC_INTERNAL_OPTIMIZED_RUNTIME_DEPENDENCY_ROOT"
_DEPENDENCY_CACHE_NAMESPACE = Path("tensorrt-model-connect") / "optimized-runtimes"
_EDGE_SOURCE_CACHE_NAME = "tensorrt-edge-llm"
_EDGE_BUILD_CACHE_NAME = "tensorrt-edge-llm-builds-v1"
_ACTIVE_CUDA_DEVICE_ENV = "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE"
_PYTHON_PROFILE_ROOT_ENV = "TRTMC_PYTHON_PROFILE_ROOT"
_LEGACY_PYTHON_PROFILE_ROOT_ENV = "TRTMC_E2E_PROFILE_ROOT"
_PYTHON_PROFILE_PREBUILT_ONLY_ENV = "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY"
_EXPORTER_PYTHON_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_EXPORTER_PYTHON"
_DEFAULT_PYTHON_PROFILE_ROOT = "/tmp/trtmc-python-profiles"
_EXPORTER_PROFILE_NAME = "qwen-edgellm-exporter"
_EXPORTER_PROFILE_LAYOUT_VERSION = "isolated-hashed-closure-v2"
_EXPORT_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9._+!-]*) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
)
_EDGE_EXPORT_BOOTSTRAP = (
    "import runpy, sys; "
    "source=sys.argv.pop(1); "
    "sys.path.insert(0, source); "
    "runpy.run_module('tensorrt_edgellm.scripts.export', run_name='__main__', alter_sys=True)"
)
_EXPORTER_IMPORTS = {
    "numpy": "numpy",
    "onnx": "onnx",
    "onnx-graphsurgeon": "onnx_graphsurgeon",
    "onnxscript": "onnxscript",
    "safetensors": "safetensors",
    "torch": "torch",
    "transformers": "transformers",
}
_MAX_ARTIFACT_ENTRIES = 65536
_MAX_ARTIFACT_TOTAL_SIZE = 1 << 40
_FORBIDDEN_BUNDLED_RUNTIME_PREFIXES = (
    "libcuda",
    "libcudart",
    "libnvinfer",
    "libnvonnxparser",
)
_SUBPROCESS_FAILURE_TAIL_CHARS = 3500
_PRIVATE_SDK_HEADERS = (
    "trtmc/pipeline.h",
    "runtime/providers/optimized_runtime_factory.h",
)
_RUNTIME_SEMANTIC_FILES = (
    "CMakeLists.txt",
    "adapter.cpp",
    "device_link_stub.cu",
    "exports.map",
)
_BUILD_BINDING_KEYS = {
    "schema_version",
    "implementation_id",
    "manifest_sha256",
    "request_sha256",
    "profile_id",
    "profile_sha256",
}
_PUBLIC_DEPLOYMENT_FIELDS = (
    "precision",
    "max_cache_length",
    "max_batch_size",
)
_MC_DEFAULT_DEPLOYMENT = {
    "precision": "fp32",
    "max_cache_length": 256,
    "max_batch_size": 1,
}
_EDGE_BUILD_STAMP = ".trtmc-edge-build-stamp.json"
_EDGE_BUILD_TARGETS = ("edgellmCore", "NvInfer_edgellm_plugin", "llm_build")
_EDGE_BUILD_ENVIRONMENT = (
    "AR",
    "CC",
    "CFLAGS",
    "CMAKE_ARGS",
    "CMAKE_COMMAND",
    "CMAKE_GENERATOR",
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "CPPFLAGS",
    "CUDAFLAGS",
    "CUDAHOSTCXX",
    "CUDAARCHS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CPLUS_INCLUDE_PATH",
    "CXX",
    "CXXFLAGS",
    "LD",
    "LD_LIBRARY_PATH",
    "LDFLAGS",
    "LIBRARY_PATH",
    "NVCC_APPEND_FLAGS",
    "NVCC_PREPEND_FLAGS",
    "PKG_CONFIG_PATH",
    "SOURCE_DATE_EPOCH",
)


def _runtime_source_root() -> Path:
    """Return this model's runtime adapter from a checkout or installed wheel."""

    packaged = CAPSULE_ROOT / "runtime"
    if packaged.is_dir():
        return packaged
    source = REPOSITORY_ROOT / "src" / "runtime" / "models" / "qwen" / "edge_llm_adapter"
    if source.is_dir():
        return source
    raise AdapterError(
        "Qwen EdgeLLM runtime adapter source is missing from the model-owned Runtime folder"
    )


class AdapterError(RuntimeError):
    """The capsule request or one of its pinned inputs is invalid."""


def _edge_dependency_cache_root() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    root = base / _DEPENDENCY_CACHE_NAMESPACE
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(
            f"Unable to create the Qwen Edge-LLM dependency cache {root}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise AdapterError(f"Qwen Edge-LLM dependency cache is not a directory: {resolved}")
    return resolved


@contextmanager
def _edge_cache_lock(entry: Path) -> Iterator[None]:
    lock = entry.with_name(f".{entry.name}.lock")
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock.open("a+b")
    except OSError as exc:
        raise AdapterError(f"Unable to open Qwen Edge-LLM cache lock {lock}: {exc}") from exc
    with lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError as exc:
            raise AdapterError(f"Unable to lock Qwen Edge-LLM cache entry {entry}: {exc}") from exc
        yield


def _remove_cache_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


@dataclass(frozen=True)
class _ExporterPythonSpec:
    implementation: str
    version: tuple[int, int]
    platform: str
    architecture: str
    abi: str
    wheel_target: str
    glibc_floor: tuple[int, int]
    requirements: str
    packages: Mapping[str, str]

    @property
    def version_text(self) -> str:
        return ".".join(str(component) for component in self.version)


@dataclass(frozen=True)
class _DependencySpec:
    downstream_name: str
    downstream_source: str
    downstream_version: str
    downstream_tag: str
    downstream_commit: str
    tensorrt_version: tuple[int, int, int, int]
    cuda_version: tuple[int, int]
    host_toolchain_family: str
    host_toolchain_major: int
    exporter_python: _ExporterPythonSpec

    @property
    def tensorrt_version_text(self) -> str:
        return ".".join(str(component) for component in self.tensorrt_version)

    @property
    def cuda_version_text(self) -> str:
        return ".".join(str(component) for component in self.cuda_version)

    @property
    def cuda_runtime_version(self) -> int:
        major, minor = self.cuda_version
        return major * 1000 + minor * 10


@dataclass(frozen=True)
class ProfileSpec:
    """One qualified model-platform profile owned by this adapter."""

    path: Path
    profile_id: str
    qualification_state: str
    qualified_semantic_sha256: str | None
    model_id: str
    revisions: tuple[str, ...]
    architectures: tuple[str, ...]
    target: Mapping[str, str]
    operation: str
    precision: str
    quantization: str
    max_input_length: int
    max_cache_length: int
    max_batch_size: int
    minimum_memory_mib: int
    engine: Mapping[str, Any]
    builder: Mapping[str, Any]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class _TensorRtInstallation:
    include_dir: Path
    library: Path
    onnx_parser_include_dir: Path
    onnx_parser_library: Path


@dataclass(frozen=True)
class _CudaInstallation:
    root: Path
    include_dir: Path
    cudart_library: Path
    driver_library: Path
    compiler: Path
    version: str


@dataclass(frozen=True)
class _EdgeBuildToolchain:
    cc: Path
    cxx: Path
    linker: Path
    archiver: Path
    cmake: Path
    sha256: str
    architecture: str


@dataclass(frozen=True)
class _EdgeDependency:
    source_dir: Path
    build_dir: Path
    build_tool: Path
    tensorrt: _TensorRtInstallation
    cuda: _CudaInstallation
    toolchain: _EdgeBuildToolchain


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"Unable to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{description} must contain a JSON object: {path}")
    return value


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as input_file:
            value = tomllib.load(input_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AdapterError(f"Unable to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{description} must contain a TOML table: {path}")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], description: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise AdapterError(f"{description} has invalid fields: {'; '.join(details)}")


def _profile_from_data(path: Path, data: Mapping[str, Any]) -> ProfileSpec:
    required_fields = {
        "schema_version",
        "profile_id",
        "qualification_state",
        "operation",
        "precision",
        "quantization",
        "max_input_length",
        "max_cache_length",
        "max_batch_size",
        "minimum_memory_mib",
        "model",
        "target",
        "engine",
        "builder",
        "artifacts",
    }
    _require_exact_keys(
        data,
        required_fields
        | ({"qualified_semantic_sha256"} if "qualified_semantic_sha256" in data else set()),
        f"profile {path}",
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise AdapterError(f"Profile schema_version must be 1: {path}")

    def required_string(field: str, *, expected: str | None = None) -> str:
        value = data[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise AdapterError(f"Profile {field} must be a non-empty trimmed string: {path}")
        if expected is not None and value != expected:
            raise AdapterError(f"Profile {field} must be {expected!r}: {path}")
        return value

    profile_id = required_string("profile_id")
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?", profile_id) is None:
        raise AdapterError(f"Profile profile_id contains unsafe characters: {path}")
    qualification_state = required_string("qualification_state")
    if qualification_state not in {"candidate", "qualified"}:
        raise AdapterError(
            f"Profile qualification_state must be 'candidate' or 'qualified': {path}"
        )
    qualified_semantic_sha256 = data.get("qualified_semantic_sha256")
    if qualified_semantic_sha256 is not None and (
        not isinstance(qualified_semantic_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", qualified_semantic_sha256) is None
    ):
        raise AdapterError(f"Profile qualified_semantic_sha256 must be a lowercase SHA-256: {path}")
    if qualification_state == "qualified" and qualified_semantic_sha256 is None:
        raise AdapterError(f"Qualified profile must pin qualified_semantic_sha256: {path}")
    operation = required_string("operation", expected="text-generation-v1")
    precision = required_string("precision", expected="fp16")
    quantization = required_string("quantization", expected="none")

    capacities: dict[str, int] = {}
    for field in (
        "max_input_length",
        "max_cache_length",
        "max_batch_size",
        "minimum_memory_mib",
    ):
        value = data[field]
        if type(value) is not int or value <= 0:
            raise AdapterError(f"Profile {field} must be a positive integer: {path}")
        capacities[field] = value

    model = _require_mapping(data["model"], f"profile {path} model")
    _require_exact_keys(model, {"id", "revisions", "architectures"}, f"profile {path} model")
    model_id = model["id"]
    if not isinstance(model_id, str) or model_id.count("/") != 1 or model_id != model_id.strip():
        raise AdapterError(f"Profile model.id must be an organization/model identifier: {path}")

    def string_tuple(value: Any, description: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise AdapterError(f"{description} must be a non-empty string array")
        result = tuple(value)
        if any(not isinstance(item, str) or not item or item != item.strip() for item in result):
            raise AdapterError(f"{description} must contain non-empty trimmed strings")
        if len(set(result)) != len(result):
            raise AdapterError(f"{description} must not contain duplicates")
        return result

    revisions = string_tuple(model["revisions"], f"profile {path} model.revisions")
    if any(re.fullmatch(r"[0-9a-f]{40}", revision) is None for revision in revisions):
        raise AdapterError(f"Profile model.revisions must contain pinned 40-hex commits: {path}")
    architectures = string_tuple(model["architectures"], f"profile {path} model.architectures")

    target = _require_mapping(data["target"], f"profile {path} target")
    target_fields = {
        "os",
        "architecture",
        "platform_kind",
        "gpu_architecture",
        "gpu_name",
    }
    _require_exact_keys(target, target_fields, f"profile {path} target")
    normalized_target: dict[str, str] = {}
    for field in sorted(target_fields):
        value = target[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise AdapterError(f"Profile target.{field} must be a non-empty string: {path}")
        normalized_target[field] = value
    if re.fullmatch(r"sm[0-9]+", normalized_target["gpu_architecture"]) is None:
        raise AdapterError(f"Profile target.gpu_architecture must use smNN form: {path}")

    engine = _require_mapping(data["engine"], f"profile {path} engine")
    engine_fields = {
        "model",
        "spec_decode_type",
        "engine_role",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "partial_rotary_factor",
        "num_deepstack_features",
        "ple_enabled",
        "num_ple_inputs",
        "ple_hidden_size",
        "kv_cache_dtype",
    }
    _require_exact_keys(engine, engine_fields, f"profile {path} engine")
    for field, expected in {
        "model": "qwen3",
        "spec_decode_type": "none",
        "engine_role": "llm",
        "kv_cache_dtype": "fp16",
    }.items():
        if type(engine[field]) is not str or engine[field] != expected:
            raise AdapterError(f"Profile engine.{field} must be {expected!r}: {path}")
    for field in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ):
        if type(engine[field]) is not int or engine[field] <= 0:
            raise AdapterError(f"Profile engine.{field} must be a positive integer: {path}")
    for field in ("num_deepstack_features", "num_ple_inputs", "ple_hidden_size"):
        if type(engine[field]) is not int or engine[field] < 0:
            raise AdapterError(f"Profile engine.{field} must be a non-negative integer: {path}")
    for field in ("rope_theta", "partial_rotary_factor"):
        if (
            type(engine[field]) is not float
            or not math.isfinite(engine[field])
            or engine[field] <= 0
        ):
            raise AdapterError(f"Profile engine.{field} must be a positive finite float: {path}")
    if type(engine["ple_enabled"]) is not bool:
        raise AdapterError(f"Profile engine.ple_enabled must be a boolean: {path}")
    rope_scaling = _require_mapping(engine["rope_scaling"], f"profile {path} engine.rope_scaling")
    _require_exact_keys(
        rope_scaling,
        {"rope_theta", "rope_type", "type"},
        f"profile {path} engine.rope_scaling",
    )
    if (
        type(rope_scaling["rope_theta"]) is not int
        or rope_scaling["rope_theta"] != int(engine["rope_theta"])
        or rope_scaling["rope_type"] != "default"
        or rope_scaling["type"] != "default"
    ):
        raise AdapterError(f"Profile engine.rope_scaling is inconsistent: {path}")

    builder = _require_mapping(data["builder"], f"profile {path} builder")
    builder_fields = {
        "max_input_len",
        "max_kv_cache_capacity",
        "max_batch_size",
        "spec_draft",
        "spec_base",
        "max_lora_rank",
        "trt_native_ops",
    }
    _require_exact_keys(builder, builder_fields, f"profile {path} builder")
    expected_builder = {
        "max_input_len": capacities["max_input_length"],
        "max_kv_cache_capacity": capacities["max_cache_length"],
        "max_batch_size": capacities["max_batch_size"],
        "spec_draft": False,
        "spec_base": False,
        "max_lora_rank": 0,
        "trt_native_ops": False,
    }
    for field, expected in expected_builder.items():
        if type(builder[field]) is not type(expected) or builder[field] != expected:
            raise AdapterError(f"Profile builder.{field} must be {expected!r}: {path}")

    artifacts = _require_mapping(data["artifacts"], f"profile {path} artifacts")
    _require_exact_keys(artifacts, {"required_files"}, f"profile {path} artifacts")
    required_files = string_tuple(
        artifacts["required_files"], f"profile {path} artifacts.required_files"
    )
    expected_files = {
        "config.json",
        "llm.engine",
        "embedding.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processed_chat_template.json",
    }
    if set(required_files) != expected_files or any(
        Path(item).name != item for item in required_files
    ):
        raise AdapterError(f"Profile artifacts.required_files is not the Edge-LLM layout: {path}")

    return ProfileSpec(
        path=path,
        profile_id=profile_id,
        qualification_state=qualification_state,
        qualified_semantic_sha256=qualified_semantic_sha256,
        model_id=model_id,
        revisions=revisions,
        architectures=architectures,
        target=MappingProxyType(normalized_target),
        operation=operation,
        precision=precision,
        quantization=quantization,
        max_input_length=capacities["max_input_length"],
        max_cache_length=capacities["max_cache_length"],
        max_batch_size=capacities["max_batch_size"],
        minimum_memory_mib=capacities["minimum_memory_mib"],
        engine=MappingProxyType(dict(engine)),
        builder=MappingProxyType(dict(builder)),
        required_files=required_files,
    )


def _load_profiles() -> tuple[ProfileSpec, ...]:
    if PROFILES_ROOT.is_symlink() or not PROFILES_ROOT.is_dir():
        raise AdapterError(f"Profile root must be a non-symlink directory: {PROFILES_ROOT}")
    profiles: list[ProfileSpec] = []
    seen: dict[str, Path] = {}
    for path in sorted(PROFILES_ROOT.glob("*.toml"), key=lambda candidate: candidate.name):
        if path.is_symlink() or not path.is_file():
            raise AdapterError(f"Profile must be a non-symlink regular file: {path}")
        profile = _profile_from_data(path, _load_toml(path, "Qwen Edge-LLM profile"))
        previous = seen.get(profile.profile_id)
        if previous is not None:
            raise AdapterError(
                f"Duplicate Qwen Edge-LLM profile_id {profile.profile_id!r}: {previous}, {path}"
            )
        seen[profile.profile_id] = path
        profiles.append(profile)
    if not profiles:
        raise AdapterError(f"No Qwen Edge-LLM profiles found in {PROFILES_ROOT}")
    return tuple(profiles)


def _private_sdk_include_roots(repository_root: Path | None = None) -> tuple[Path, ...]:
    """Resolve the source-tree or base-wheel private SDK for this external pack."""

    root = (REPOSITORY_ROOT if repository_root is None else repository_root).resolve()

    def complete(include_roots: tuple[Path, ...]) -> bool:
        return all(
            any((include_root / relative).is_file() for include_root in include_roots)
            for relative in _PRIVATE_SDK_HEADERS
        )

    source_roots = (root / "src", root / "include")
    if complete(source_roots):
        return source_roots

    candidates = [root / "runtime_provider" / "_sdk" / "include"]
    try:
        specification = importlib.util.find_spec("tensorrt_model_connect.runtime_provider")
    except (ImportError, AttributeError, ValueError):
        specification = None
    if specification is not None:
        package_locations = list(specification.submodule_search_locations or ())
        if specification.origin:
            package_locations.append(str(Path(specification.origin).parent))
        for package_location in package_locations:
            candidates.append(Path(package_location) / "_sdk" / "include")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if complete((resolved,)):
            return (resolved,)
    raise AdapterError(
        "Runtime build requires both private SDK headers from the "
        "repository source tree or the base wheel's runtime_provider/_sdk/include"
    )


def _implementation_manifest_sha256() -> str:
    """Bind the adapter and its DSO to the exact selected manifest bytes."""

    try:
        return hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise AdapterError(f"Unable to hash capsule manifest {IMPLEMENTATION_PATH}: {exc}") from exc


def _semantic_source_sha256(profile: ProfileSpec) -> str:
    """Bind one profile to every model-owned source file that implements it."""

    runtime_root = _runtime_source_root()
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise AdapterError(
            f"Qwen EdgeLLM runtime source root must be a non-symlink directory: {runtime_root}"
        )

    runtime_sources: dict[str, Path] = {}
    for candidate in sorted(
        runtime_root.rglob("*"),
        key=lambda path: path.relative_to(runtime_root).as_posix(),
    ):
        relative = candidate.relative_to(runtime_root).as_posix()
        if candidate.is_symlink():
            raise AdapterError(f"Qwen EdgeLLM semantic source must not be a symlink: {candidate}")
        if candidate.is_file():
            runtime_sources[relative] = candidate
        elif not candidate.is_dir():
            raise AdapterError(
                f"Qwen EdgeLLM semantic source must be a regular file or directory: {candidate}"
            )

    expected_runtime_sources = set(_RUNTIME_SEMANTIC_FILES)
    actual_runtime_sources = set(runtime_sources)
    if actual_runtime_sources != expected_runtime_sources:
        missing = sorted(expected_runtime_sources - actual_runtime_sources)
        unexpected = sorted(actual_runtime_sources - expected_runtime_sources)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AdapterError(
            "Qwen EdgeLLM runtime semantic source inventory changed: " + "; ".join(details)
        )

    profile_data = _load_toml(profile.path, "Qwen EdgeLLM semantic profile")
    profile_data.pop("qualification_state", None)
    profile_data.pop("qualified_semantic_sha256", None)
    profile_payload = json.dumps(
        profile_data,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    def regular_file_bytes(path: Path) -> bytes:
        if path.is_symlink():
            raise AdapterError(f"Qwen EdgeLLM semantic source must not be a symlink: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise AdapterError(
                f"Unable to hash Qwen EdgeLLM semantic source {path}: {exc}"
            ) from exc
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise AdapterError(f"Qwen EdgeLLM semantic source is not a regular file: {path}")
            return source.read()

    sources = [
        ("IMPLEMENTATION.toml", regular_file_bytes(IMPLEMENTATION_PATH)),
        ("adapter.py", regular_file_bytes(CAPSULE_ROOT / "adapter.py")),
        ("dependency.lock", regular_file_bytes(DEPENDENCY_PATH)),
        ("selected-profile.json", profile_payload),
        *(
            (f"runtime/{relative}", regular_file_bytes(runtime_sources[relative]))
            for relative in _RUNTIME_SEMANTIC_FILES
        ),
    ]
    digest = hashlib.sha256(b"trtmc-qwen-edge-llm-semantic-source-v1\0")
    for label, payload in sources:
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _qualified_semantic_source_sha256(profile: ProfileSpec) -> str:
    current = _semantic_source_sha256(profile)
    if profile.qualified_semantic_sha256 != current:
        raise AdapterError(
            f"Profile {profile.profile_id} semantic source does not match its qualification"
        )
    return current


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_string(
    table: Mapping[str, Any],
    field: str,
    description: str,
    *,
    pattern: str | None = None,
) -> str:
    value = table.get(field)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n;")
        or (pattern is not None and re.fullmatch(pattern, value) is None)
    ):
        raise AdapterError(f"{description}.{field} is invalid")
    return value


def _numeric_version(value: str, components: int, description: str) -> tuple[int, ...]:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*", value) is None:
        raise AdapterError(f"{description} must be a canonical numeric version")
    parsed = tuple(int(component) for component in value.split("."))
    if len(parsed) != components:
        raise AdapterError(f"{description} must have exactly {components} components")
    return parsed


def _parse_exporter_python_spec(exporter_python: Mapping[str, Any]) -> _ExporterPythonSpec:
    expected_fields = {
        "implementation",
        "version",
        "platform",
        "architecture",
        "abi",
        "wheel_target",
        "lock_format",
        "requirements",
    }
    if set(exporter_python) != expected_fields:
        raise AdapterError("Capsule exporter Python lock must contain the exact required fields")
    implementation = _lock_string(
        exporter_python, "implementation", "dependency.exporter_python", pattern=r"[A-Za-z0-9._-]+"
    )
    version_text = _lock_string(
        exporter_python, "version", "dependency.exporter_python", pattern=r"[0-9.]+"
    )
    version = _numeric_version(version_text, 2, "dependency.exporter_python.version")
    platform_name = _lock_string(
        exporter_python, "platform", "dependency.exporter_python", pattern=r"[a-z0-9._-]+"
    )
    architecture = _lock_string(
        exporter_python, "architecture", "dependency.exporter_python", pattern=r"[A-Za-z0-9._-]+"
    )
    abi = _lock_string(
        exporter_python, "abi", "dependency.exporter_python", pattern=r"[A-Za-z0-9._-]+"
    )
    wheel_target = _lock_string(
        exporter_python,
        "wheel_target",
        "dependency.exporter_python",
        pattern=r"[A-Za-z0-9._-]+-manylinux_[0-9]+_[0-9]+",
    )
    wheel_match = re.fullmatch(
        r"(?P<architecture>[A-Za-z0-9._-]+)-manylinux_(?P<major>[0-9]+)_(?P<minor>[0-9]+)",
        wheel_target,
    )
    if wheel_match is None or wheel_match.group("architecture") != architecture:
        raise AdapterError("dependency.exporter_python.wheel_target must match its architecture")
    lock_format = _lock_string(exporter_python, "lock_format", "dependency.exporter_python")
    if lock_format != "pip-require-hashes-v1":
        raise AdapterError("dependency.exporter_python.lock_format is unsupported: " + lock_format)
    requirements = exporter_python.get("requirements")
    if not isinstance(requirements, str):
        raise AdapterError("Capsule exporter Python requirements lock must be a string")

    packages: dict[str, str] = {}
    canonical_lines: list[str] = []
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _EXPORT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise AdapterError(f"Invalid hashed exporter Python requirement: {line!r}")
        name = _canonical_package_name(match.group("name"))
        if name in packages:
            raise AdapterError(f"Duplicate exporter Python requirement: {name}")
        packages[name] = match.group("version")
        canonical_lines.append(line)
    if not packages:
        raise AdapterError("Capsule exporter Python requirements lock must not be empty")
    if list(packages) != sorted(packages):
        raise AdapterError("Capsule exporter Python requirements must be sorted by package name")
    return _ExporterPythonSpec(
        implementation=implementation,
        version=version,
        platform=platform_name,
        architecture=architecture,
        abi=abi,
        wheel_target=wheel_target,
        glibc_floor=(int(wheel_match.group("major")), int(wheel_match.group("minor"))),
        requirements="\n".join(canonical_lines) + "\n",
        packages=MappingProxyType(packages),
    )


def _parse_exporter_lock(
    exporter_python: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    spec = _parse_exporter_python_spec(exporter_python)
    return spec.requirements, dict(spec.packages)


def _load_dependency_spec() -> _DependencySpec:
    dependency = _load_toml(DEPENDENCY_PATH, "capsule dependency lock")
    expected_tables = {
        "schema_version",
        "downstream",
        "tensorrt",
        "cuda",
        "host_toolchain",
        "exporter_python",
    }
    if set(dependency) != expected_tables or dependency.get("schema_version") != 1:
        raise AdapterError("Capsule dependency lock must use schema version 1")

    downstream = _require_mapping(dependency["downstream"], "dependency.downstream")
    if set(downstream) != {"name", "source", "version", "tag", "commit", "source_mode"}:
        raise AdapterError("dependency.downstream has invalid fields")
    downstream_name = _lock_string(
        downstream, "name", "dependency.downstream", pattern=r"[a-z0-9][a-z0-9._-]*"
    )
    if downstream_name != "tensorrt-edge-llm":
        raise AdapterError("This adapter requires a TensorRT Edge-LLM dependency")
    downstream_source = _lock_string(downstream, "source", "dependency.downstream")
    if not downstream_source.startswith("https://") or not downstream_source.endswith(".git"):
        raise AdapterError("dependency.downstream.source must be an HTTPS Git URL")
    downstream_version = _lock_string(
        downstream, "version", "dependency.downstream", pattern=r"[0-9]+(?:\.[0-9]+)+"
    )
    downstream_tag = _lock_string(
        downstream, "tag", "dependency.downstream", pattern=r"[A-Za-z0-9][A-Za-z0-9._/-]*"
    )
    downstream_commit = _lock_string(
        downstream, "commit", "dependency.downstream", pattern=r"[0-9a-f]{40}"
    )
    source_mode = _lock_string(downstream, "source_mode", "dependency.downstream")
    if source_mode != "git":
        raise AdapterError("This adapter supports only a pinned Git dependency")

    tensorrt = _require_mapping(dependency["tensorrt"], "dependency.tensorrt")
    cuda = _require_mapping(dependency["cuda"], "dependency.cuda")
    if set(tensorrt) != {"version"} or set(cuda) != {"version"}:
        raise AdapterError("TensorRT and CUDA dependency tables require exactly version")
    tensorrt_components = _numeric_version(
        _lock_string(tensorrt, "version", "dependency.tensorrt", pattern=r"[0-9.]+"),
        4,
        "dependency.tensorrt.version",
    )
    tensorrt_version = (
        tensorrt_components[0],
        tensorrt_components[1],
        tensorrt_components[2],
        tensorrt_components[3],
    )
    cuda_components = _numeric_version(
        _lock_string(cuda, "version", "dependency.cuda", pattern=r"[0-9.]+"),
        2,
        "dependency.cuda.version",
    )
    cuda_version = (cuda_components[0], cuda_components[1])
    if cuda_version[1] > 9:
        raise AdapterError("dependency.cuda.version cannot be encoded as CUDA_VERSION")

    host = _require_mapping(dependency["host_toolchain"], "dependency.host_toolchain")
    if set(host) != {"family", "major"}:
        raise AdapterError("dependency.host_toolchain has invalid fields")
    host_family = _lock_string(
        host, "family", "dependency.host_toolchain", pattern=r"[a-z][a-z0-9._-]*"
    )
    host_major = host.get("major")
    if host_family != "gcc" or type(host_major) is not int or host_major <= 0:
        raise AdapterError("This adapter requires a positive GCC toolchain major")

    exporter = _parse_exporter_python_spec(
        _require_mapping(dependency["exporter_python"], "dependency.exporter_python")
    )
    spec = _DependencySpec(
        downstream_name=downstream_name,
        downstream_source=downstream_source,
        downstream_version=downstream_version,
        downstream_tag=downstream_tag,
        downstream_commit=downstream_commit,
        tensorrt_version=tensorrt_version,
        cuda_version=cuda_version,
        host_toolchain_family=host_family,
        host_toolchain_major=host_major,
        exporter_python=exporter,
    )

    manifest = _load_toml(IMPLEMENTATION_PATH, "capsule implementation manifest")
    expected_manifest = {
        "implementation_id": IMPLEMENTATION_ID,
        "downstream_runtime": spec.downstream_name,
        "downstream_version": spec.downstream_version,
        "downstream_commit": spec.downstream_commit,
    }
    mismatches = {
        field: {"manifest": manifest.get(field), "dependency_lock": expected}
        for field, expected in expected_manifest.items()
        if manifest.get(field) != expected
    }
    if mismatches:
        raise AdapterError(
            "Implementation manifest does not match dependency.lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return spec


def _pinned_host_toolchain() -> tuple[str, int]:
    spec = _load_dependency_spec()
    return spec.host_toolchain_family, spec.host_toolchain_major


def _pinned_edge_dependency() -> tuple[str, str, str, str, str]:
    spec = _load_dependency_spec()
    return (
        spec.downstream_source,
        spec.downstream_tag,
        spec.downstream_commit,
        spec.tensorrt_version_text,
        spec.cuda_version_text,
    )


def _load_exporter_lock() -> tuple[str, dict[str, str]]:
    exporter = _load_dependency_spec().exporter_python
    return exporter.requirements, dict(exporter.packages)


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(f"{description} must be an object")
    return value


def _validated_runtime_compile_binding(
    request: Mapping[str, Any],
    profile: ProfileSpec,
) -> str:
    binding = _require_mapping(request.get("build_binding"), "request.build_binding")
    if set(binding) != _BUILD_BINDING_KEYS:
        raise AdapterError("Capsule build_binding must contain the exact required field set")
    if type(binding.get("schema_version")) is not int or binding["schema_version"] != 1:
        raise AdapterError("Capsule build_binding schema_version must be 1")

    expected_manifest_sha256 = _implementation_manifest_sha256()
    expected = {
        "implementation_id": IMPLEMENTATION_ID,
        "manifest_sha256": expected_manifest_sha256,
        "profile_id": profile.profile_id,
        "profile_sha256": _qualified_semantic_source_sha256(profile),
    }
    for field, expected_value in expected.items():
        value = binding.get(field)
        if not isinstance(value, str) or value != expected_value:
            raise AdapterError(f"Capsule build_binding {field} does not match capsule source")
    request_sha256 = binding.get("request_sha256")
    if not isinstance(request_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise AdapterError("Capsule build_binding request_sha256 must be a lowercase SHA-256")

    return expected_manifest_sha256


def _load_request(path: Path) -> dict[str, Any]:
    request = _load_json(path, "capsule request")
    allowed = {
        "schema_version",
        "implementation_id",
        "model",
        "target",
        "parameters",
        "build_binding",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise AdapterError(f"Capsule request contains unknown fields: {', '.join(unknown)}")
    if type(request.get("schema_version")) is not int or request["schema_version"] != 1:
        raise AdapterError("Capsule request schema_version must be 1")
    if request.get("implementation_id") != IMPLEMENTATION_ID:
        raise AdapterError("Capsule request implementation_id does not match this capsule")
    model = _require_mapping(request.get("model"), "request.model")
    if set(model) != {"id", "revision"}:
        raise AdapterError("Capsule request model must contain exactly id and revision")
    for field in ("id", "revision"):
        if (
            not isinstance(model.get(field), str)
            or not model[field]
            or model[field] != model[field].strip()
        ):
            raise AdapterError(f"Capsule request model.{field} must be a non-empty string")
    target = _require_mapping(request.get("target"), "request.target")
    if not target:
        raise AdapterError("Capsule request target must not be empty")
    for field, value in target.items():
        if not isinstance(field, str) or not field:
            raise AdapterError("Capsule request target keys must be non-empty strings")
        if type(value) not in (str, int, float, bool) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise AdapterError(f"Capsule request target.{field} must be a finite scalar")
    parameters = _require_mapping(request.get("parameters"), "request.parameters")
    if "model_source" in parameters and not isinstance(parameters["model_source"], str):
        raise AdapterError("Capsule parameter model_source must be a string")
    for field in (
        "engine_dir",
        "runtime_library",
        "runtime_plugin",
    ):
        if field in parameters and (
            not isinstance(parameters[field], str) or not parameters[field].strip()
        ):
            raise AdapterError(f"Capsule parameter {field} must be a non-empty string")
    public_options = parameters.get("public_options", {})
    if not isinstance(public_options, dict):
        raise AdapterError("Capsule parameter public_options must be an object")
    runtime_build = parameters.get("runtime_build", {})
    if not isinstance(runtime_build, dict):
        raise AdapterError("Capsule parameter runtime_build must be an object")
    _validate_runtime_build(parameters)
    injected_payloads = sorted(
        field
        for field in ("engine_dir", "runtime_library", "runtime_plugin")
        if field in parameters
    )
    if injected_payloads and not _test_payload_injection_enabled(parameters):
        raise AdapterError(
            "Capsule payload overrides are test-only: " + ", ".join(injected_payloads)
        )
    return request


def _public_option_reason(options: Mapping[str, Any], profile: ProfileSpec) -> str:
    """Return why this family-runtime adapter cannot represent an MC option."""

    unsupported_when_set = (
        "build_timing_json",
        "build_timing_path",
        "config",
        "diffusion_overrides",
        "dynamic_kv_profile_rows",
        "dynamic_kv_profile_rows_override",
        "family_build_options",
        "fp32_layers",
        "fp8_scales",
        "image_height",
        "image_width",
        "num_inference_steps",
        "quant_scales",
        "save_fp8_scales",
        "set_flags",
        "triattention_kv_budget",
        "triattention_stats",
        "triattention_stats_path",
        "video_height",
        "video_num_frames",
        "video_width",
    )
    for field in unsupported_when_set:
        if options.get(field) is not None and options.get(field) not in ("", (), [], {}):
            return f"Qwen Edge-LLM capsule does not support public option {field}"

    unsupported_when_true = (
        "dynamic_kv_cache",
        "rtx",
        "triattention_disable_mlr",
        "triattention_disable_trig",
        "trust_remote_code",
    )
    for field in unsupported_when_true:
        if field in options and options[field] is not False:
            return f"Qwen Edge-LLM capsule requires public option {field}=False"

    exact_defaults = {
        "decoder_engine_layout": "split",
        "quant_calibration_samples": 512,
        "tensor_parallel_size": 1,
        "triattention_count_prompt_tokens": True,
        "triattention_divide_length": 128,
        "triattention_protect_prefill": True,
        "triattention_recent_window": 128,
        "triattention_score_aggregation": "mean",
    }
    recognized = {
        *unsupported_when_set,
        *unsupported_when_true,
        *exact_defaults,
        "fp8",
        "max_batch_size",
        "max_cache_length",
        "method",
        "parallel_config",
        "precision",
        "quantize",
        "verbose",
    }

    def inert(value: Any) -> bool:
        return (
            value is None
            or value is False
            or value == ""
            or value == ()
            or value == []
            or value == {}
        )

    unknown = sorted(field for field in set(options) - recognized if not inert(options[field]))
    if unknown:
        return "Qwen Edge-LLM capsule does not recognize public option(s): " + ", ".join(unknown)

    for field, expected in exact_defaults.items():
        if field in options and (
            type(options[field]) is not type(expected) or options[field] != expected
        ):
            return (
                f"Qwen Edge-LLM capsule requires public option {field}={expected!r}; "
                f"got {options[field]!r}"
            )

    if "verbose" in options and type(options["verbose"]) is not bool:
        return "Qwen Edge-LLM capsule requires public option verbose to be a boolean"

    # The adapter treats the complete, unchanged MC default tuple as the
    # model-ID-only UX for a qualified Edge profile. The qualified tuple is
    # also accepted for callers that state the Edge engine capacities
    # explicitly. Requiring the complete tuple prevents partial or mixed
    # requests from being reinterpreted as this profile.
    present_deployment_fields = tuple(
        field for field in _PUBLIC_DEPLOYMENT_FIELDS if field in options
    )
    if present_deployment_fields != _PUBLIC_DEPLOYMENT_FIELDS:
        return (
            "Qwen Edge-LLM capsule requires precision, max_cache_length, and "
            "max_batch_size as one complete public deployment tuple"
        )

    def matches(deployment: Mapping[str, Any]) -> bool:
        return all(
            type(options[field]) is type(expected) and options[field] == expected
            for field, expected in deployment.items()
        )

    qualified_deployment = {
        "precision": profile.precision,
        "max_cache_length": profile.max_cache_length,
        "max_batch_size": profile.max_batch_size,
    }
    if not (matches(_MC_DEFAULT_DEPLOYMENT) or matches(qualified_deployment)):
        return (
            "Qwen Edge-LLM capsule does not qualify public deployment tuple "
            f"precision={options['precision']!r}, "
            f"max_cache_length={options['max_cache_length']!r}, "
            f"max_batch_size={options['max_batch_size']!r}"
        )

    exact_profile_options = {
        "fp8": False,
        "method": "auto",
        "quantize": None,
    }
    for field, expected in exact_profile_options.items():
        if field in options and (
            type(options[field]) is not type(expected) or options[field] != expected
        ):
            return (
                "Qwen Edge-LLM capsule requires public option "
                f"{field}={expected!r}; got {options[field]!r}"
            )

    parallel = options.get("parallel_config")
    if parallel is not None and not (type(parallel) is dict and not parallel):
        expected_parallel = {
            "mode": "single",
            "rank": -1,
            "require_mpirun": True,
            "tp_size": 1,
        }
        if (
            type(parallel) is not dict
            or set(parallel) != set(expected_parallel)
            or any(
                type(parallel[field]) is not type(expected) or parallel[field] != expected
                for field, expected in expected_parallel.items()
            )
        ):
            return "Qwen Edge-LLM capsule does not support parallel_config"
    return ""


def _profile_parameter_reason(parameters: Mapping[str, Any], profile: ProfileSpec) -> str:
    allowed_parameters = {
        "model_source",
        "precision",
        "quantization",
        "max_input_length",
        "max_cache_length",
        "max_batch_size",
        "engine_dir",
        "runtime_library",
        "runtime_plugin",
        "public_options",
        "runtime_build",
    }
    unknown = sorted(set(parameters) - allowed_parameters)
    if unknown:
        return "unsupported Qwen Edge-LLM parameter(s): " + ", ".join(unknown)
    exact_parameters = {
        "precision": profile.precision,
        "quantization": profile.quantization,
        "max_input_length": profile.max_input_length,
        "max_cache_length": profile.max_cache_length,
        "max_batch_size": profile.max_batch_size,
    }
    for field, expected in exact_parameters.items():
        actual = parameters.get(field)
        if actual is not None and (type(actual) is not type(expected) or actual != expected):
            return f"unsupported capsule parameter {field}={actual!r}; expected {expected!r}"

    public_options = _require_mapping(
        parameters.get("public_options", {}), "request.parameters.public_options"
    )
    public_reason = _public_option_reason(public_options, profile)
    if public_reason:
        return public_reason

    return ""


def _profile_match_reason(profile: ProfileSpec, request: Mapping[str, Any]) -> str:
    if profile.qualification_state != "qualified":
        return f"profile {profile.profile_id} is not qualified"
    model = _require_mapping(request["model"], "request.model")
    if model["id"] != profile.model_id:
        return f"model {model['id']!r} is not {profile.model_id!r}"
    if model["revision"] not in profile.revisions:
        return f"revision {model['revision']!r} is not qualified for {profile.model_id}"
    if _semantic_source_sha256(profile) != profile.qualified_semantic_sha256:
        return f"profile {profile.profile_id} semantic source is not qualified"

    target = _require_mapping(request["target"], "request.target")
    mismatches = {
        field: (target.get(field), expected)
        for field, expected in profile.target.items()
        if type(target.get(field)) is not type(expected) or target.get(field) != expected
    }
    if mismatches:
        return f"target does not match {profile.profile_id}: {mismatches}"
    memory_mib = target.get("gpu_memory_mib")
    if type(memory_mib) is not int or memory_mib < profile.minimum_memory_mib:
        return (
            f"target gpu_memory_mib must be at least {profile.minimum_memory_mib} "
            f"for {profile.profile_id}"
        )
    gpu_count = target.get("gpu_count", 1)
    if type(gpu_count) is not int or gpu_count < 1:
        return "target gpu_count must be a positive integer"

    return _profile_parameter_reason(
        _require_mapping(request["parameters"], "request.parameters"), profile
    )


def _select_profile(
    request: Mapping[str, Any], profiles: tuple[ProfileSpec, ...]
) -> tuple[ProfileSpec | None, str]:
    matches: list[ProfileSpec] = []
    reasons: list[str] = []
    for profile in profiles:
        reason = _profile_match_reason(profile, request)
        if reason:
            reasons.append(reason)
        else:
            matches.append(profile)
    if len(matches) > 1:
        profile_ids = ", ".join(profile.profile_id for profile in matches)
        raise AdapterError(f"Multiple Qwen Edge-LLM profiles match one request: {profile_ids}")
    if matches:
        return matches[0], ""
    model = _require_mapping(request["model"], "request.model")
    detail = reasons[0] if reasons else "no profiles are installed"
    return None, (
        f"No qualified Qwen Edge-LLM profile matches {model['id']}@{model['revision']}: {detail}"
    )


def _parameter_or_environment(
    parameters: Mapping[str, Any], parameter_name: str, environment_name: str
) -> str:
    value = parameters.get(parameter_name, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get(environment_name, "").strip()


def _test_payload_injection_enabled(parameters: Mapping[str, Any]) -> bool:
    runtime_build = _validate_runtime_build(parameters)
    return runtime_build.get("fake") is True and (
        os.environ.get(_ALLOW_FAKE_RUNTIME_BUILD_ENV, "").strip() == "1"
    )


def _require_payload(
    parameters: Mapping[str, Any],
    parameter_name: str,
    environment_name: str,
    description: str,
) -> Path:
    raw = _parameter_or_environment(parameters, parameter_name, environment_name)
    if not raw:
        raise AdapterError(
            f"{description} is required for this capsule build; provide "
            f"an explicit regular-file path in request parameter "
            f"{parameter_name!r} or {environment_name}"
        )
    path = Path(raw).expanduser()
    if path.is_symlink():
        raise AdapterError(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise AdapterError(f"{description} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise AdapterError(f"{description} must be a non-empty regular file: {resolved}")
    return resolved


def _regular_file_size(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AdapterError(f"Unable to inspect capsule artifact {path}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise AdapterError(f"Capsule artifact is not a regular file: {path}")
    return metadata.st_size


def _validate_artifact_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise AdapterError(f"Capsule artifact root must be a non-symlink directory: {root}")
    files: list[tuple[str, Path]] = []
    entry_count = 0
    for candidate in sorted(root.rglob("*"), key=lambda path: path.as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if not relative or "\\" in relative or len(relative.encode("utf-8")) > 4096:
            raise AdapterError(f"Unsafe capsule artifact path: {relative!r}")
        if candidate.is_symlink():
            raise AdapterError(f"Capsule artifact trees cannot contain symlinks: {candidate}")
        if candidate.is_file():
            if candidate.name.startswith(_FORBIDDEN_BUNDLED_RUNTIME_PREFIXES):
                raise AdapterError(
                    "Capsule artifacts must not package TensorRT or CUDA runtime libraries: "
                    f"{candidate}"
                )
            files.append((relative, candidate))
        elif not candidate.is_dir():
            raise AdapterError(f"Unsupported capsule artifact entry: {candidate}")
        entry_count += 1
        if entry_count > _MAX_ARTIFACT_ENTRIES:
            raise AdapterError("Capsule artifact tree exceeds the entry limit")

    total_size = 0
    for _relative, path in files:
        total_size += _regular_file_size(path)
        if total_size > _MAX_ARTIFACT_TOTAL_SIZE:
            raise AdapterError("Capsule artifact tree exceeds the size limit")
    if not files:
        raise AdapterError(f"Capsule artifact directory contains no files: {root}")


def _validate_engine_directory(path: Path, profile: ProfileSpec) -> tuple[Path, int]:
    dependency = _load_dependency_spec()
    if path.is_symlink():
        raise AdapterError(f"Edge-LLM engine directory must not be a symlink: {path}")
    try:
        engine = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"Edge-LLM engine directory is unavailable: {path}: {exc}") from exc
    if not engine.is_dir():
        raise AdapterError(f"Edge-LLM engine artifact is not a directory: {engine}")
    config_path = engine / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise AdapterError(f"Edge-LLM engine is missing non-symlink config.json: {engine}")
    config = _load_json(config_path, "Edge-LLM engine config")
    expected_model = {
        field: dict(value) if isinstance(value, Mapping) else value
        for field, value in profile.engine.items()
    }
    expected_model["edgellm_version"] = dependency.downstream_version
    for field, expected_value in expected_model.items():
        actual_value = config.get(field)
        if (
            field not in config
            or type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise AdapterError(f"Edge-LLM engine config.{field} must be exactly {expected_value!r}")
    if config.get("reduced_vocab_size") is not None:
        raise AdapterError("Edge-LLM engine config.reduced_vocab_size must be absent or null")
    for field in ("tp_size", "tp_rank"):
        if field in config:
            raise AdapterError(
                f"Edge-LLM engine config.{field} must be absent for this single-GPU profile"
            )
    builder = _require_mapping(config.get("builder_config"), "Edge-LLM builder_config")
    for field, value in profile.builder.items():
        if type(builder.get(field)) is not type(value) or builder.get(field) != value:
            raise AdapterError(f"Edge-LLM engine builder_config.{field} must be exactly {value}")
    for filename in profile.required_files:
        artifact = engine / filename
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size <= 0:
            raise AdapterError(f"Edge-LLM engine is missing required artifact {filename}")
    _validate_artifact_tree(engine)
    return engine, expected_model["vocab_size"]


def _snapshot_identity(path: Path) -> tuple[str, str] | None:
    if path.parent.name != "snapshots" or not path.parent.parent.name.startswith("models--"):
        return None
    components = path.parent.parent.name[len("models--") :].split("--")
    if len(components) != 2 or any(not item for item in components):
        return None
    return "/".join(components), path.name.lower()


def _materialize_model_source(model_source: str, profile: ProfileSpec, revision: str) -> Path:
    def validate_architecture(root: Path) -> Path:
        config = _load_json(root / "config.json", "pinned model config")
        architectures = config.get("architectures")
        if (
            not isinstance(architectures, list)
            or not architectures
            or any(not isinstance(value, str) or not value for value in architectures)
            or not set(architectures).intersection(profile.architectures)
        ):
            raise AdapterError(
                f"Pinned model config architectures do not match {profile.architectures}"
            )
        return root

    candidate = Path(model_source).expanduser()
    if candidate.exists():
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir() or _snapshot_identity(resolved) != (profile.model_id, revision):
            raise AdapterError(
                "Local model source must be the exact pinned Hugging Face cache snapshot"
            )
        return validate_architecture(resolved)
    if candidate.is_absolute():
        raise AdapterError(f"Configured model source does not exist: {candidate}")
    if model_source != profile.model_id:
        raise AdapterError(f"Capsule model source must be {profile.model_id!r}")
    try:
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(repo_id=profile.model_id, revision=revision)
    except Exception as exc:
        raise AdapterError(f"Unable to materialize {profile.model_id}@{revision}: {exc}") from exc
    resolved = Path(downloaded).resolve(strict=True)
    if _snapshot_identity(resolved) != (profile.model_id, revision):
        raise AdapterError("Hugging Face did not return the exact pinned snapshot")
    return validate_architecture(resolved)


def _cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise AdapterError("CUDA Python is required to build this A100 capsule") from exc
    return cudart


def _select_parent_active_cuda_device() -> None:
    """Restore the parent's active CUDA ordinal and pin descendant tools to it."""

    raw = os.environ.get(_ACTIVE_CUDA_DEVICE_ENV, "")
    if not raw:
        return
    if len(raw) > 10 or not raw.isascii() or not raw.isdecimal() or str(int(raw)) != raw:
        raise AdapterError(f"{_ACTIVE_CUDA_DEVICE_ENV} must be a canonical CUDA ordinal")
    ordinal = int(raw)
    cudart = _cuda_runtime()
    success = getattr(getattr(cudart, "cudaError_t", None), "cudaSuccess", 0)
    try:
        status = cudart.cudaSetDevice(ordinal)
        if isinstance(status, tuple):
            status = status[0]
        if status not in (success, 0):
            raise AdapterError(f"cudaSetDevice({ordinal}) failed with status {status}")
        status, selected = cudart.cudaGetDevice()
        if status not in (success, 0) or int(selected) != ordinal:
            raise AdapterError(f"active CUDA device did not remain {ordinal} after cudaSetDevice")
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"Unable to select parent active CUDA device {ordinal}: {exc}") from exc

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = [token.strip() for token in visible.split(",")]
        if any(not token for token in tokens) or ordinal >= len(tokens):
            raise AdapterError(
                f"{_ACTIVE_CUDA_DEVICE_ENV}={ordinal} is inconsistent with CUDA_VISIBLE_DEVICES"
            )
        selected_token = tokens[ordinal]
    else:
        selected_token = raw
    # cudaSetDevice affects this adapter process. Descendant Edge export/build
    # tools are new processes, so expose only the same selected GPU to them.
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_token


def _probe_build_device(profile: ProfileSpec) -> None:
    cudart = _cuda_runtime()
    try:
        status, device = cudart.cudaGetDevice()
        if status != 0:
            raise AdapterError(f"cudaGetDevice failed with status {status}")
        status, properties = cudart.cudaGetDeviceProperties(device)
        if status != 0:
            raise AdapterError(f"cudaGetDeviceProperties failed with status {status}")
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"Unable to inspect the active CUDA build device: {exc}") from exc
    name = properties.name
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace").rstrip("\x00")
    memory_mib = int(properties.totalGlobalMem) // (1024 * 1024)
    architecture = int(profile.target["gpu_architecture"].removeprefix("sm"))
    if (
        int(properties.major) * 10 + int(properties.minor) != architecture
        or name != profile.target["gpu_name"]
        or memory_mib < profile.minimum_memory_mib
    ):
        raise AdapterError(f"Active build device does not match {profile.profile_id}")


def _run_tool(
    command: list[str],
    *,
    verbose: bool,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment) if environment is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except OSError as exc:
        raise AdapterError(f"Unable to launch Edge-LLM tool {command[0]!r}: {exc}") from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "no subprocess output"
        raise AdapterError(
            f"Edge-LLM tool failed with exit code {result.returncode}"
            f"\n{detail}"
            f"\nCommand: {' '.join(command)}"
        )
    if verbose:
        # Adapter stdout is a machine-readable, one-JSON-object protocol. Tool
        # logs always belong on stderr, including in verbose mode.
        for stream in (result.stdout, result.stderr):
            if stream:
                print(stream, file=sys.stderr, end="" if stream.endswith("\n") else "\n")


def _subprocess_failure_tail(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    label, detail = ("stderr", stderr) if stderr else ("stdout", stdout)
    if not detail:
        return ""
    omitted = max(0, len(detail) - _SUBPROCESS_FAILURE_TAIL_CHARS)
    detail = detail[-_SUBPROCESS_FAILURE_TAIL_CHARS:]
    if omitted:
        return f"{label} tail ({omitted} leading characters omitted):\n{detail}"
    return f"{label}:\n{detail}"


def _python_profile_root() -> Path:
    configured = (
        os.environ.get(_PYTHON_PROFILE_ROOT_ENV, "").strip()
        or os.environ.get(_LEGACY_PYTHON_PROFILE_ROOT_ENV, "").strip()
        or _DEFAULT_PYTHON_PROFILE_ROOT
    )
    return Path(configured).expanduser()


def _python_profile_prebuilt_only() -> bool:
    return os.environ.get(_PYTHON_PROFILE_PREBUILT_ONLY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _glibc_version() -> tuple[int, int] | None:
    libc_name, libc_version = platform.libc_ver()
    candidates = [f"{libc_name} {libc_version}".strip()]
    if not libc_name or libc_name.lower() == "glibc":
        try:
            candidates.append(os.confstr("CS_GNU_LIBC_VERSION") or "")
        except (AttributeError, OSError, ValueError):
            pass
    for candidate in candidates:
        match = re.fullmatch(r"(?:glibc\s+)?([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", candidate)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    return None


def _validate_exporter_host() -> None:
    exporter = _load_dependency_spec().exporter_python
    glibc = _glibc_version()
    actual = {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "glibc": ".".join(str(component) for component in glibc) if glibc else "unknown",
    }
    expected = {"platform": exporter.platform, "architecture": exporter.architecture}
    if (
        any(actual[field] != value for field, value in expected.items())
        or not glibc
        or glibc < exporter.glibc_floor
    ):
        raise AdapterError(
            "Qwen Edge-LLM exporter host does not match dependency.lock "
            f"({exporter.wheel_target}): " + json.dumps(actual, sort_keys=True)
        )


def _exporter_base_python() -> Path:
    exporter = _load_dependency_spec().exporter_python
    implementation = exporter.implementation.lower()
    version = exporter.version
    configured = os.environ.get(_EXPORTER_PYTHON_ENV, "").strip()
    if configured:
        candidate = Path(configured).expanduser()
    elif sys.implementation.name == implementation and sys.version_info[:2] == version:
        candidate = Path(sys.executable)
    else:
        discovered = shutil.which(f"python{exporter.version_text}")
        if discovered is None:
            raise AdapterError(
                "Qwen Edge-LLM engine export requires "
                f"{exporter.implementation} {exporter.version_text}"
            )
        candidate = Path(discovered)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"Edge-LLM exporter Python is unavailable: {candidate}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AdapterError(f"Edge-LLM exporter Python is not executable: {resolved}")
    try:
        result = subprocess.run(
            [
                str(resolved),
                "-I",
                "-c",
                "import sys; assert sys.implementation.name == "
                f"{implementation!r} and sys.version_info[:2] == {version!r}",
            ],
            capture_output=True,
            text=True,
            env=_isolated_python_environment(),
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(
            f"Edge-LLM exporter Python inspection timed out after 30 seconds: {resolved}"
        ) from exc
    except OSError as exc:
        raise AdapterError(f"Unable to launch Edge-LLM exporter Python: {resolved}: {exc}") from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "interpreter contract mismatch"
        raise AdapterError(
            "Edge-LLM exporter Python must match dependency.lock "
            f"({exporter.implementation} {exporter.version_text}): {resolved}\n{detail}"
        )
    return resolved


def _isolated_python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    # PyTorch 2.12 asks getpass for a cache identity even when the exporter
    # does not compile kernels. Arbitrary container UIDs need not have an
    # /etc/passwd entry, so give this isolated subprocess a stable identity.
    build_user = f"trtmc-edgellm-{os.getuid()}"
    environment["LOGNAME"] = build_user
    environment["USER"] = build_user
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_CONFIG_FILE"] = os.devnull
    return environment


def _exporter_profile_identity(base_python: Path | None = None) -> str:
    dependency = _load_dependency_spec()
    exporter = dependency.exporter_python
    try:
        lock_bytes = DEPENDENCY_PATH.read_bytes()
    except OSError as exc:
        raise AdapterError(
            f"Unable to read capsule dependency lock {DEPENDENCY_PATH}: {exc}"
        ) from exc
    payload = b"\0".join(
        (
            _EXPORTER_PROFILE_LAYOUT_VERSION.encode("utf-8"),
            str(base_python or _exporter_base_python()).encode("utf-8"),
            f"{exporter.implementation.lower()}-{exporter.version_text}".encode("ascii"),
            platform.machine().encode("utf-8"),
            dependency.downstream_commit.encode("ascii"),
            lock_bytes,
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _exporter_profile_paths(base_python: Path | None = None) -> tuple[str, Path, Path, Path]:
    identity = _exporter_profile_identity(base_python)
    root = _python_profile_root()
    environment = root / f"{_EXPORTER_PROFILE_NAME}-{identity[:16]}"
    return (
        identity,
        environment,
        environment / "bin" / "python",
        root / (f"{_EXPORTER_PROFILE_NAME}-{identity[:16]}.lock"),
    )


def _verify_exporter_python(python: Path) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise AdapterError(f"Edge-LLM exporter Python is unavailable: {python}")
    _requirements, expected = _load_exporter_lock()
    version_script = (
        "import importlib.metadata as m, json, re; "
        "canonical=lambda name: re.sub(r'[-_.]+', '-', name).lower(); "
        "actual={canonical(dist.metadata['Name']): dist.version for dist in m.distributions()}; "
        "print(json.dumps(actual, sort_keys=True))"
    )
    environment = _isolated_python_environment()
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", version_script],
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(python.parent),
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(f"Unable to verify Edge-LLM exporter Python {python}: {exc}") from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "no subprocess output"
        raise AdapterError(
            f"Unable to inspect Edge-LLM exporter Python {python} "
            f"(exit code {result.returncode})"
            f"\n{detail}"
        )
    try:
        actual = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AdapterError(
            f"Edge-LLM exporter Python returned invalid package metadata: {result.stdout!r}"
        ) from exc
    if not isinstance(actual, dict):
        raise AdapterError("Edge-LLM exporter Python package metadata must be a JSON object")
    mismatches: dict[str, Any] = {
        name: {"expected": expected[name], "actual": actual.get(name)}
        for name in expected
        if actual.get(name) != expected[name]
    }
    unexpected = {name: version for name, version in actual.items() if name not in expected}
    if unexpected:
        mismatches["unexpected"] = unexpected
    if mismatches:
        raise AdapterError(
            "Edge-LLM exporter Python package inventory does not match the capsule lock: "
            + json.dumps(mismatches, sort_keys=True)
        )

    import_script = (
        "import importlib, json, sys; "
        "modules=json.loads(sys.argv[1]); "
        "[importlib.import_module(module) for module in modules.values()]"
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", import_script, json.dumps(_EXPORTER_IMPORTS, sort_keys=True)],
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(python.parent),
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(
            f"Unable to smoke-test Edge-LLM exporter Python {python}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "no subprocess output"
        raise AdapterError(
            f"Edge-LLM exporter Python import smoke failed for {python} "
            f"(exit code {result.returncode})"
            f"\n{detail}"
        )


def _existing_exporter_python(identity: str, environment: Path, python: Path) -> Path | None:
    ready = environment / ".ready"
    try:
        if ready.read_text(encoding="utf-8").strip() != identity:
            return None
    except (OSError, UnicodeError):
        return None
    try:
        _verify_exporter_python(python)
    except AdapterError:
        return None
    return python


def _materialize_exporter_python() -> Path:
    """Create this leaf's exact exporter environment after profile selection."""

    _validate_exporter_host()
    base_python = _exporter_base_python()
    locked_requirements, _packages = _load_exporter_lock()
    identity, environment, python, lock = _exporter_profile_paths(base_python)
    existing = _existing_exporter_python(identity, environment, python)
    if existing is not None:
        return existing
    if _python_profile_prebuilt_only():
        raise AdapterError(
            f"Edge-LLM exporter Python profile is not prebuilt or is corrupt: {environment}"
        )

    root = environment.parent
    root.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        existing = _existing_exporter_python(identity, environment, python)
        if existing is not None:
            return existing

        staging = Path(tempfile.mkdtemp(prefix=f"{_EXPORTER_PROFILE_NAME}-", dir=str(root)))
        staging_python = staging / "bin" / "python"
        requirements = staging / "requirements.lock.txt"
        requirements.write_text(locked_requirements, encoding="utf-8")
        isolated_environment = _isolated_python_environment()
        try:
            _run_checked(
                [str(base_python), "-I", "-m", "venv", str(staging)],
                "create the Qwen Edge-LLM exporter Python profile",
                environment=isolated_environment,
                cwd=root,
            )
            _run_checked(
                [
                    str(staging_python),
                    "-I",
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--only-binary=:all:",
                    "--require-hashes",
                    "--no-deps",
                    "--no-compile",
                    "-r",
                    str(requirements),
                ],
                "install the Qwen Edge-LLM exporter Python profile",
                environment=isolated_environment,
                cwd=staging,
            )
            _verify_exporter_python(staging_python)
            (staging / ".ready").write_text(identity + "\n", encoding="utf-8")
            if environment.exists():
                shutil.rmtree(environment)
            staging.rename(environment)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return python


def _edge_export_command(
    python: Path,
    edge_source: Path,
    model_source: Path,
    output_root: Path,
) -> list[str]:
    return [
        str(python),
        "-I",
        "-B",
        "-c",
        _EDGE_EXPORT_BOOTSTRAP,
        str(edge_source),
        str(model_source),
        str(output_root),
        "--dtype=float16",
    ]


def _build_or_resolve_engine(
    parameters: Mapping[str, Any],
    output: Path,
    dependency: _EdgeDependency | None,
    plugin_path: Path,
    profile: ProfileSpec,
    revision: str,
) -> tuple[Path, int, Path | None]:
    configured_engine = _parameter_or_environment(parameters, "engine_dir", _ENGINE_ENV)
    if configured_engine:
        if not _test_payload_injection_enabled(parameters):
            raise AdapterError("Edge-LLM engine payload overrides are test-only")
        if dependency is not None:
            _validate_edge_source(dependency.source_dir)
        engine, vocab_size = _validate_engine_directory(Path(configured_engine), profile)
        return engine, vocab_size, None

    _probe_build_device(profile)
    workspace_base = output / ".edge-build-workspace"
    workspace_base.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="qwen3-edge-", dir=workspace_base))
    output_root = attempt / "output_root"
    onnx = output_root / "llm"
    engine = attempt / "engine.dir"
    output_root.mkdir()
    engine.mkdir()
    tool_cwd = attempt / ".tool-cwd"
    tool_cwd.mkdir()
    runtime_build = _validate_runtime_build(parameters)
    if dependency is None:
        dependency = _resolve_edge_dependency(runtime_build, profile)
    export_script = dependency.source_dir / "tensorrt_edgellm" / "scripts" / "export.py"
    if not export_script.is_file():
        raise AdapterError(f"Pinned TensorRT Edge-LLM exporter is missing: {export_script}")
    export_environment = _isolated_python_environment()
    exporter_python = _materialize_exporter_python()
    _validate_edge_source(dependency.source_dir)
    model_source = _materialize_model_source(
        str(parameters.get("model_source") or profile.model_id), profile, revision
    )
    public_options = _require_mapping(
        parameters.get("public_options", {}), "request.parameters.public_options"
    )
    verbose = bool(public_options.get("verbose", False))
    try:
        _run_tool(
            _edge_export_command(
                exporter_python,
                dependency.source_dir,
                model_source,
                output_root,
            ),
            verbose=verbose,
            environment=export_environment,
            cwd=tool_cwd,
        )
        if dependency is not None:
            _validate_edge_source(dependency.source_dir)
        _run_tool(
            [
                str(dependency.build_tool),
                f"--onnxDir={onnx}",
                f"--engineDir={engine}",
                f"--maxInputLen={profile.builder['max_input_len']}",
                f"--maxKVCacheCapacity={profile.builder['max_kv_cache_capacity']}",
                f"--maxBatchSize={profile.builder['max_batch_size']}",
            ],
            verbose=verbose,
            environment={
                **os.environ,
                # Edge-LLM otherwise assumes a process-CWD-relative
                # build/libNvInfer_edgellm_plugin.so. Bind engine creation to
                # the exact plugin payload that this capsule will package.
                "EDGELLM_PLUGIN_PATH": str(plugin_path),
                "LD_LIBRARY_PATH": os.pathsep.join(
                    dict.fromkeys(
                        (
                            str(dependency.tensorrt.library.parent),
                            str(dependency.tensorrt.onnx_parser_library.parent),
                            str(dependency.cuda.cudart_library.parent),
                            *(
                                (os.environ["LD_LIBRARY_PATH"],)
                                if os.environ.get("LD_LIBRARY_PATH")
                                else ()
                            ),
                        )
                    )
                ),
            },
            cwd=tool_cwd,
        )
        if dependency is not None:
            _validate_edge_source(dependency.source_dir)
        validated, vocab_size = _validate_engine_directory(engine, profile)
        return validated, vocab_size, attempt
    except Exception:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def _runtime_setting(
    runtime_build: Mapping[str, Any], field: str, environment: str, default: str = ""
) -> str:
    value = runtime_build.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get(environment, "").strip() or default


def _validate_runtime_build(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_build = _require_mapping(parameters.get("runtime_build", {}), "runtime_build")
    allowed = {
        "fake",
        "parallel",
        "sdk_include_dir",
        "nlohmann_json_include_dir",
        "edge_llm_source_dir",
        "edge_llm_build_dir",
        "tensorrt_include_dir",
        "tensorrt_library",
        "onnx_parser_include_dir",
        "onnx_parser_library",
        "cuda_include_dir",
        "cudart_library",
        "cuda_driver_library",
    }
    unknown = sorted(set(runtime_build) - allowed)
    if unknown:
        raise AdapterError(f"runtime_build has unknown fields: {', '.join(unknown)}")
    fake = runtime_build.get("fake", False)
    if type(fake) is not bool:
        raise AdapterError("runtime_build.fake must be a boolean")
    parallel = runtime_build.get("parallel", 2)
    if type(parallel) is not int or not 1 <= parallel <= 64:
        raise AdapterError("runtime_build.parallel must be an integer from 1 to 64")
    for field in allowed - {"fake", "parallel"}:
        if field in runtime_build and (
            not isinstance(runtime_build[field], str) or not runtime_build[field].strip()
        ):
            raise AdapterError(f"runtime_build.{field} must be a non-empty string")
    return runtime_build


def _run_checked(
    command: list[str],
    description: str,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=dict(environment) if environment is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except OSError as exc:
        raise AdapterError(f"Unable to launch {description}: {exc}") from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "no subprocess output"
        raise AdapterError(f"{description} failed with exit code {result.returncode}\n{detail}")


def _run_capture(command: list[str], description: str) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise AdapterError(f"Unable to launch {description}: {exc}") from exc
    if result.returncode != 0:
        detail = _subprocess_failure_tail(result) or "no subprocess output"
        raise AdapterError(f"{description} failed with exit code {result.returncode}\n{detail}")
    # Preserve a leading status byte. In particular, ``git submodule status``
    # uses a leading space to mean that a submodule is initialized at its
    # pinned commit; stripping both ends would turn that valid state into an
    # apparent error.
    return result.stdout.rstrip()


def _validate_packaged_elf(path: Path) -> None:
    dependency = _load_dependency_spec()
    header = _run_capture(["readelf", "-h", str(path)], f"ELF header inspection for {path.name}")
    if "Class:                             ELF64" not in header:
        raise AdapterError(f"Packaged Edge-LLM payload is not ELF64: {path}")
    if "Machine:                           Advanced Micro Devices X86-64" not in header:
        raise AdapterError(f"Packaged Edge-LLM payload is not x86-64: {path}")

    dynamic = _run_capture(
        ["readelf", "-d", str(path)], f"ELF dynamic-section inspection for {path.name}"
    )
    search_paths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic, flags=re.MULTILINE)
    if search_paths:
        raise AdapterError(
            f"Packaged Edge-LLM payload contains forbidden RPATH/RUNPATH entries: {path}: "
            + ", ".join(search_paths)
        )
    needed = set(re.findall(r"Shared library: \[(.*?)\]", dynamic))
    tensorrt_soname = f"libnvinfer.so.{dependency.tensorrt_version[0]}"
    cudart_soname = f"libcudart.so.{dependency.cuda_version[0]}"
    required = {"libcuda.so.1", cudart_soname, tensorrt_soname}
    missing = sorted(required - needed)
    if missing:
        raise AdapterError(
            f"Packaged Edge-LLM payload is missing required runtime dependencies: {path}: "
            + ", ".join(missing)
        )
    wrong_tensorrt = sorted(
        library
        for library in needed
        if library.startswith("libnvinfer.so.") and library != tensorrt_soname
    )
    if wrong_tensorrt:
        raise AdapterError(
            f"Packaged Edge-LLM payload links an unsupported TensorRT runtime: {path}: "
            + ", ".join(wrong_tensorrt)
        )


def _validate_edge_source(source: Path) -> Path:
    source_url, _, expected_commit, _, _ = _pinned_edge_dependency()
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"TensorRT Edge-LLM source is unavailable: {source}: {exc}") from exc
    required = (
        resolved / "CMakeLists.txt",
        resolved / "cpp" / "common" / "version.h",
        resolved / "3rdParty" / "nlohmannJson" / "include" / "nlohmann" / "json.hpp",
        resolved / "tensorrt_edgellm" / "scripts" / "export.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AdapterError(
            "TensorRT Edge-LLM source or its nested submodules are incomplete: "
            + ", ".join(missing)
        )
    commit = _run_capture(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        "TensorRT Edge-LLM commit inspection",
    )
    if commit != expected_commit:
        raise AdapterError(
            f"TensorRT Edge-LLM must be pinned at {expected_commit}; got {commit or '<none>'}"
        )
    origin = _run_capture(
        ["git", "-C", str(resolved), "remote", "get-url", "origin"],
        "TensorRT Edge-LLM origin inspection",
    )
    if origin != source_url:
        raise AdapterError(
            f"TensorRT Edge-LLM source origin must be {source_url}; got {origin or '<none>'}"
        )
    source_changes = _run_capture(
        [
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=none",
        ],
        "TensorRT Edge-LLM source cleanliness inspection",
    )
    if source_changes:
        raise AdapterError(
            "TensorRT Edge-LLM source must have no tracked, untracked, or ignored "
            "changes; git status reported: " + source_changes[:4000]
        )
    submodule_status = _run_capture(
        ["git", "-C", str(resolved), "submodule", "status", "--recursive"],
        "TensorRT Edge-LLM recursive submodule inspection",
    )
    invalid_submodules = [
        line for line in submodule_status.splitlines() if line and not line.startswith(" ")
    ]
    if invalid_submodules:
        raise AdapterError(
            "TensorRT Edge-LLM recursive submodules must be initialized at their pinned "
            "commits; git submodule status reported: " + "\n".join(invalid_submodules)[:4000]
        )
    submodule_changes = _run_capture(
        [
            "git",
            "-C",
            str(resolved),
            "submodule",
            "--quiet",
            "foreach",
            "--recursive",
            "git status --porcelain=v1 --untracked-files=all --ignored=matching",
        ],
        "TensorRT Edge-LLM recursive submodule cleanliness inspection",
    )
    if submodule_changes:
        raise AdapterError(
            "TensorRT Edge-LLM recursive submodules must have no tracked, untracked, "
            "or ignored changes; git status reported: " + submodule_changes[:4000]
        )
    return resolved


def _resolve_edge_source(runtime_build: Mapping[str, Any]) -> Path:
    source_url, expected_tag, expected_commit, _, _ = _pinned_edge_dependency()
    configured = _runtime_setting(
        runtime_build,
        "edge_llm_source_dir",
        "_TRTMC_INTERNAL_QWEN_EDGE_LLM_SOURCE_DIR",
    )
    if configured:
        return _validate_edge_source(Path(configured))

    dependency_root = os.environ.get(_DEPENDENCY_CACHE_ENV, "").strip()
    if dependency_root:
        staged = Path(dependency_root) / _EDGE_SOURCE_CACHE_NAME / expected_commit
        return _validate_edge_source(staged)

    acquired = _edge_dependency_cache_root() / _EDGE_SOURCE_CACHE_NAME / expected_commit
    if acquired.exists():
        try:
            return _validate_edge_source(acquired)
        except AdapterError:
            pass

    # This remains lazy because source resolution happens only after profile
    # selection. Concurrent builders publish one immutable checkout atomically.
    with _edge_cache_lock(acquired):
        if acquired.exists():
            try:
                return _validate_edge_source(acquired)
            except AdapterError:
                _remove_cache_entry(acquired)
        staging = Path(tempfile.mkdtemp(prefix=f".{expected_commit}.", dir=str(acquired.parent)))
        try:
            _run_checked(
                [
                    "git",
                    "clone",
                    "--branch",
                    expected_tag,
                    "--single-branch",
                    "--no-checkout",
                    source_url,
                    str(staging),
                ],
                "TensorRT Edge-LLM source acquisition",
            )
            _run_checked(
                ["git", "-C", str(staging), "checkout", "--detach", expected_commit],
                "TensorRT Edge-LLM pinned checkout",
            )
            _run_checked(
                ["git", "-C", str(staging), "submodule", "update", "--init", "--recursive"],
                "TensorRT Edge-LLM nested submodule bootstrap",
            )
            _validate_edge_source(staging)
            staging.rename(acquired)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return _validate_edge_source(acquired)


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.abspath(os.path.expanduser(str(path)))
        if key not in seen:
            seen.add(key)
            result.append(Path(key))
    return result


def _dependency_prefixes(environment_names: tuple[str, ...], fixed: tuple[str, ...]) -> list[Path]:
    candidates = [Path(value) for name in environment_names if (value := os.environ.get(name))]
    candidates.extend(Path(value) for value in fixed)
    candidates.append(Path(sys.prefix))
    candidates.extend(Path(value) for value in sys.path if value)
    return _deduplicate_paths(candidates)


def _header_directories(prefixes: list[Path]) -> list[Path]:
    directories: list[Path] = []
    for root in prefixes:
        directories.extend(
            (
                root,
                root / "include",
                root / "include" / "x86_64-linux-gnu",
                root / "include" / "aarch64-linux-gnu",
                root / "tensorrt" / "include",
                root / "targets" / "x86_64-linux" / "include",
                root / "targets" / "aarch64-linux" / "include",
            )
        )
    return _deduplicate_paths(directories)


def _library_directories(prefixes: list[Path]) -> list[Path]:
    directories: list[Path] = []
    for root in prefixes:
        directories.extend(
            (
                root,
                root / "lib",
                root / "lib64",
                root / "lib" / "x86_64-linux-gnu",
                root / "lib" / "aarch64-linux-gnu",
                root / "x86_64-linux-gnu",
                root / "aarch64-linux-gnu",
                root / "tensorrt_libs",
                root / "targets" / "x86_64-linux" / "lib",
                root / "targets" / "aarch64-linux" / "lib",
            )
        )
    return _deduplicate_paths(directories)


def _first_header(directories: list[Path], filename: str) -> Path | None:
    for directory in directories:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve(strict=True)
    return None


def _library_candidates(directories: list[Path], stem: str) -> list[Path]:
    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for pattern in (f"{stem}.so", f"{stem}.so.*"):
            candidates.extend(sorted(directory.glob(pattern)))
    return _deduplicate_paths([path for path in candidates if path.is_file()])


def _header_tensorrt_version(include_dir: Path) -> tuple[int, int, int, int]:
    version_header = include_dir / "NvInferVersion.h"
    try:
        text = version_header.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AdapterError(
            f"Unable to inspect TensorRT version header {version_header}: {exc}"
        ) from exc

    def component(primary: str, enterprise: str) -> int:
        for macro in (enterprise, primary):
            match = re.search(rf"^\s*#\s*define\s+{re.escape(macro)}\s+(\d+)\b", text, re.MULTILINE)
            if match:
                return int(match.group(1))
        raise AdapterError(f"TensorRT version header does not define {primary}: {version_header}")

    return (
        component("NV_TENSORRT_MAJOR", "TRT_MAJOR_ENTERPRISE"),
        component("NV_TENSORRT_MINOR", "TRT_MINOR_ENTERPRISE"),
        component("NV_TENSORRT_PATCH", "TRT_PATCH_ENTERPRISE"),
        component("NV_TENSORRT_BUILD", "TRT_BUILD_ENTERPRISE"),
    )


def _library_tensorrt_version(library: Path) -> tuple[int, int, int, int]:
    try:
        handle = ctypes.CDLL(str(library))
        values: list[int] = []
        for name in (
            "getInferLibMajorVersion",
            "getInferLibMinorVersion",
            "getInferLibPatchVersion",
            "getInferLibBuildVersion",
        ):
            function = getattr(handle, name)
            function.argtypes = []
            function.restype = ctypes.c_int
            values.append(int(function()))
        return tuple(values)  # type: ignore[return-value]
    except (AttributeError, OSError) as exc:
        raise AdapterError(f"Unable to inspect TensorRT library {library}: {exc}") from exc


def _library_onnx_parser_version(library: Path) -> tuple[int, int, int]:
    match = re.search(
        r"^libnvonnxparser\.so\.(\d+)\.(\d+)\.(\d+)(?:\.|$)",
        library.resolve(strict=True).name,
    )
    if match is None:
        raise AdapterError(
            f"TensorRT ONNX parser library does not expose a release-versioned filename: {library}"
        )
    return tuple(int(match.group(index)) for index in range(1, 4))  # type: ignore[return-value]


def _resolve_tensorrt(runtime_build: Mapping[str, Any]) -> _TensorRtInstallation:
    dependency = _load_dependency_spec()
    expected_version = dependency.tensorrt_version
    expected_version_text = dependency.tensorrt_version_text
    prefixes = _dependency_prefixes(
        ("TRT_PACKAGE_DIR", "TENSORRT_ROOT", "TRT_ROOT"),
        ("/usr", "/usr/local", "/opt/tensorrt"),
    )
    prefixes = _deduplicate_paths(
        [
            *prefixes,
            *sorted(Path("/usr/local").glob("TensorRT*")),
            *sorted(Path("/opt").glob("TensorRT*")),
            *(
                Path(value)
                for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                if value
            ),
        ]
    )
    header_dirs = _header_directories(prefixes)
    library_dirs = _library_directories(prefixes)

    configured_include = _runtime_setting(runtime_build, "tensorrt_include_dir", _TRT_INCLUDE_ENV)
    include_candidates = [Path(configured_include)] if configured_include else header_dirs
    include_dir: Path | None = None
    header_errors: list[str] = []
    for candidate in include_candidates:
        if not (candidate / "NvInfer.h").is_file():
            continue
        try:
            version = _header_tensorrt_version(candidate)
        except AdapterError as exc:
            header_errors.append(str(exc))
            continue
        if version == expected_version:
            include_dir = candidate.resolve(strict=True)
            break
        header_errors.append(
            f"TensorRT headers at {candidate} are {'.'.join(map(str, version))}, "
            f"not {expected_version_text}"
        )
    if include_dir is None:
        detail = f" ({'; '.join(header_errors)})" if header_errors else ""
        raise AdapterError(
            f"Unable to find exact TensorRT {expected_version_text} headers in standard locations{detail}"
        )

    configured_library = _runtime_setting(runtime_build, "tensorrt_library", _TRT_LIBRARY_ENV)
    library_candidates = (
        [Path(configured_library)]
        if configured_library
        else _library_candidates(library_dirs, "libnvinfer")
    )
    library: Path | None = None
    library_errors: list[str] = []
    for candidate in library_candidates:
        if not candidate.is_file():
            continue
        try:
            version = _library_tensorrt_version(candidate.resolve(strict=True))
        except AdapterError as exc:
            library_errors.append(str(exc))
            continue
        if version == expected_version:
            library = candidate.resolve(strict=True)
            break
        library_errors.append(
            f"TensorRT library {candidate} is {'.'.join(map(str, version))}, "
            f"not {expected_version_text}"
        )
    if library is None:
        detail = f" ({'; '.join(library_errors)})" if library_errors else ""
        raise AdapterError(
            f"Unable to find exact TensorRT {expected_version_text} libnvinfer in standard locations{detail}"
        )

    configured_onnx_include = _runtime_setting(
        runtime_build, "onnx_parser_include_dir", _ONNX_PARSER_INCLUDE_ENV
    )
    parser_include_candidate = (
        Path(configured_onnx_include).resolve(strict=True)
        if configured_onnx_include
        else include_dir
    )
    onnx_header = _first_header([parser_include_candidate], "NvOnnxParser.h")
    if onnx_header is None:
        raise AdapterError("Unable to find NvOnnxParser.h for the selected TensorRT installation")
    if onnx_header.parent != include_dir:
        raise AdapterError(
            "TensorRT ONNX parser headers must come from the selected exact TensorRT "
            f"include directory: {include_dir}"
        )
    # NV_ONNX_PARSER_* identifies the parser API (currently 0.1.0), not the
    # TensorRT release. Bind compatibility through this shared include tree and
    # the colocated release-versioned DSO.

    configured_onnx_library = _runtime_setting(
        runtime_build, "onnx_parser_library", _ONNX_PARSER_LIBRARY_ENV
    )
    onnx_libraries = (
        [Path(configured_onnx_library)]
        if configured_onnx_library
        else _library_candidates([library.parent], "libnvonnxparser")
    )
    onnx_library = None
    parser_errors: list[str] = []
    for path in onnx_libraries:
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if resolved.parent != library.parent:
            parser_errors.append(
                f"TensorRT ONNX parser {resolved} is outside selected libnvinfer directory "
                f"{library.parent}"
            )
            continue
        try:
            parser_library_version = _library_onnx_parser_version(resolved)
        except AdapterError as exc:
            parser_errors.append(str(exc))
            continue
        if parser_library_version == expected_version[:3]:
            onnx_library = resolved
            break
        parser_errors.append(
            f"TensorRT ONNX parser {resolved} is {'.'.join(map(str, parser_library_version))}, "
            f"not {'.'.join(map(str, expected_version[:3]))}"
        )
    if onnx_library is None:
        detail = f" ({'; '.join(parser_errors)})" if parser_errors else ""
        raise AdapterError(
            f"Unable to find compatible libnvonnxparser beside selected libnvinfer{detail}"
        )
    return _TensorRtInstallation(include_dir, library, onnx_header.parent, onnx_library)


def _cuda_version(root: Path, include_dir: Path) -> str:
    # The headers are the compile input, so prefer their version over a
    # possibly stale toolkit metadata file.
    cuda_header = include_dir / "cuda.h"
    if cuda_header.is_file():
        try:
            text = cuda_header.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            text = ""
        match = re.search(r"^\s*#\s*define\s+CUDA_VERSION\s+(\d+)\b", text, re.MULTILINE)
        if match:
            encoded = int(match.group(1))
            return f"{encoded // 1000}.{(encoded % 1000) // 10}"
    for path in (root / "version.json", root / "version.txt"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        match = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)?\b", text)
        if match:
            return f"{int(match.group(1))}.{int(match.group(2))}"
    raise AdapterError(f"Unable to determine CUDA toolkit version below {root}")


def _cuda_toolkit_root(include_dir: Path) -> Path:
    for candidate in (include_dir.parent, *include_dir.parents):
        compiler = candidate / "bin" / "nvcc"
        if compiler.is_file():
            return candidate.resolve(strict=True)
    raise AdapterError(f"Unable to find a CUDA toolkit root for headers at {include_dir}")


def _cuda_compiler_version(compiler: Path) -> str:
    output = _run_capture([str(compiler), "--version"], "CUDA compiler version inspection")
    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", output)
    if match is None:
        match = re.search(r"\bV(\d+)\.(\d+)(?:\.\d+)?\b", output)
    if match is None:
        raise AdapterError(f"Unable to determine CUDA compiler version from {compiler}")
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _cuda_toolkit_roots(prefixes: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for prefix in prefixes:
        for candidate in (prefix, *prefix.parents):
            if (candidate / "bin" / "nvcc").is_file():
                roots.append(candidate.resolve(strict=True))
                break
    return _deduplicate_paths(roots)


def _cuda_installation(
    root: Path,
    include_dir: Path,
    configured_cudart: Path | None,
    configured_driver: Path | None,
) -> _CudaInstallation:
    expected_version = _load_dependency_spec().cuda_version_text
    root = root.resolve(strict=True)
    include_dir = include_dir.resolve(strict=True)
    try:
        include_dir.relative_to(root)
    except ValueError as exc:
        raise AdapterError(
            "CUDA headers, compiler, libcudart, and libcuda must come from the same exact "
            f"CUDA {expected_version} toolkit"
        ) from exc
    version = _cuda_version(root, include_dir)
    if version != expected_version:
        raise AdapterError(
            f"CUDA toolkit headers at {include_dir} are {version}, not {expected_version}"
        )
    compiler = root / "bin" / "nvcc"
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise AdapterError(
            f"Exact CUDA {expected_version} toolkit compiler is unavailable: {compiler}"
        )
    compiler = compiler.resolve(strict=True)
    compiler_version = _cuda_compiler_version(compiler)
    if compiler_version != expected_version:
        raise AdapterError(
            f"CUDA compiler {compiler} is {compiler_version}, not {expected_version}"
        )
    try:
        compiler.relative_to(root)
    except ValueError as exc:
        raise AdapterError(
            "CUDA headers, compiler, libcudart, and libcuda must come from the same exact "
            f"CUDA {expected_version} toolkit"
        ) from exc

    cudart_candidates = (
        [configured_cudart]
        if configured_cudart is not None
        else _library_candidates(_library_directories([root]), "libcudart")
    )
    cudart: Path | None = None
    for candidate in cudart_candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        cudart = resolved
        break
    if cudart is None and configured_cudart is not None:
        raise AdapterError(
            "CUDA headers, compiler, libcudart, and libcuda must come from the same exact "
            f"CUDA {expected_version} toolkit"
        )
    if cudart is None:
        raise AdapterError(f"Unable to find libcudart below CUDA toolkit {root}")

    driver_candidates = (
        [configured_driver]
        if configured_driver is not None
        else _library_candidates(
            [
                root / "lib64" / "stubs",
                root / "lib" / "stubs",
                root / "targets" / "x86_64-linux" / "lib" / "stubs",
                root / "targets" / "x86_64-linux" / "lib",
                root / "lib64",
                root / "lib",
            ],
            "libcuda",
        )
    )
    driver: Path | None = None
    for candidate in driver_candidates:
        if not candidate.is_file() or not candidate.name.startswith("libcuda.so"):
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        driver = resolved
        break
    if driver is None and configured_driver is not None:
        raise AdapterError(
            "CUDA headers, compiler, libcudart, and libcuda must come from the same exact "
            f"CUDA {expected_version} toolkit"
        )
    if driver is None:
        raise AdapterError(
            f"Unable to find the CUDA driver stub or library below CUDA toolkit {root}"
        )
    return _CudaInstallation(root, include_dir, cudart, driver, compiler, version)


def _resolve_cuda(runtime_build: Mapping[str, Any]) -> _CudaInstallation:
    expected_version = _load_dependency_spec().cuda_version_text
    configured_include = _runtime_setting(runtime_build, "cuda_include_dir", _CUDA_INCLUDE_ENV)
    configured_cudart_value = _runtime_setting(runtime_build, "cudart_library", _CUDART_LIBRARY_ENV)
    configured_driver_value = _runtime_setting(
        runtime_build, "cuda_driver_library", _CUDA_DRIVER_LIBRARY_ENV
    )
    configured_cudart = Path(configured_cudart_value) if configured_cudart_value else None
    configured_driver = Path(configured_driver_value) if configured_driver_value else None
    if configured_cudart is not None and not configured_cudart.is_file():
        raise AdapterError(f"Configured libcudart does not exist: {configured_cudart}")
    if configured_driver is not None and not configured_driver.is_file():
        raise AdapterError(f"Configured CUDA driver library does not exist: {configured_driver}")

    # Capsule-specific overrides are exact inputs, not discovery hints. Validate
    # their single toolkit immediately so an invalid override cannot fall back to
    # an unrelated host installation.
    if configured_include:
        cuda_header = _first_header([Path(configured_include)], "cuda_runtime_api.h")
        if cuda_header is None:
            raise AdapterError(
                f"Configured CUDA include directory has no cuda_runtime_api.h: {configured_include}"
            )
        include_dir = cuda_header.parent
        return _cuda_installation(
            _cuda_toolkit_root(include_dir),
            include_dir,
            configured_cudart,
            configured_driver,
        )

    if configured_cudart is not None:
        cudart = configured_cudart.resolve(strict=True)
        root = _cuda_toolkit_root(cudart.parent)
        cuda_header = _first_header(_header_directories([root]), "cuda_runtime_api.h")
        if cuda_header is None:
            raise AdapterError(f"Unable to find CUDA toolkit headers below {root}")
        return _cuda_installation(root, cuda_header.parent, cudart, configured_driver)

    if configured_driver is not None:
        driver = configured_driver.resolve(strict=True)
        root = _cuda_toolkit_root(driver.parent)
        cuda_header = _first_header(_header_directories([root]), "cuda_runtime_api.h")
        if cuda_header is None:
            raise AdapterError(f"Unable to find CUDA toolkit headers below {root}")
        return _cuda_installation(root, cuda_header.parent, None, driver)

    prefixes = _dependency_prefixes(
        ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"),
        ("/usr/local/cuda", f"/usr/local/cuda-{expected_version}", "/opt/cuda"),
    )
    prefixes = _deduplicate_paths(
        [
            *prefixes,
            *sorted(Path("/usr/local").glob("cuda-*")),
            *(
                Path(value)
                for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                if value
            ),
        ]
    )
    errors: list[str] = []
    for root in _cuda_toolkit_roots(prefixes):
        cuda_header = _first_header(_header_directories([root]), "cuda_runtime_api.h")
        if cuda_header is None:
            errors.append(f"Unable to find CUDA toolkit headers below {root}")
            continue
        try:
            return _cuda_installation(root, cuda_header.parent, None, None)
        except AdapterError as exc:
            errors.append(str(exc))
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise AdapterError(
        f"Unable to find a coherent exact CUDA {expected_version} toolkit "
        f"in standard locations{detail}"
    )


def _cmake_cache_value(build_dir: Path, key: str) -> str:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return ""
    try:
        text = cache.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    match = re.search(rf"^{re.escape(key)}(?::[^=]+)?=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _resolved_compiler(environment: str, default: str) -> Path:
    configured = os.environ.get(environment, "").strip() or default
    if any(character.isspace() for character in configured):
        raise AdapterError(f"{environment} must name one compiler executable")
    candidate = shutil.which(configured)
    if candidate is None:
        raise AdapterError(f"Required host compiler is unavailable: {configured}")
    try:
        return Path(candidate).resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"Unable to resolve host compiler {candidate}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AdapterError(f"Unable to hash build identity file {path}: {exc}") from exc
    return digest.hexdigest()


def _resolved_build_tool(environment: str, compiler: Path, program: str) -> Path:
    configured = os.environ.get(environment, "").strip()
    candidate = configured or _run_capture(
        [str(compiler), f"-print-prog-name={program}"], f"host {program} inspection"
    )
    resolved = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
    if not resolved:
        raise AdapterError(f"Required build tool is unavailable: {candidate}")
    try:
        return Path(resolved).resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"Unable to resolve build tool {resolved}: {exc}") from exc


def _require_host_compiler(compiler: Path, language: str, family: str, major: int) -> str:
    version = _run_capture(
        [str(compiler), "-dumpfullversion"], f"host {language} compiler version inspection"
    )
    if family != "gcc" or re.fullmatch(rf"{major}(?:\.[0-9]+)*", version) is None:
        raise AdapterError(
            f"Qualified Qwen EdgeLLM builds require GCC {major} for {language}; "
            f"{compiler} reported {version or '<none>'}"
        )
    return version


def _host_toolchain_identity(
    source: Path,
    tensorrt: _TensorRtInstallation,
    cuda: _CudaInstallation,
) -> _EdgeBuildToolchain:
    if os.environ.get("CMAKE_TOOLCHAIN_FILE", "").strip():
        raise AdapterError("Qualified Qwen EdgeLLM builds do not allow CMAKE_TOOLCHAIN_FILE")
    compiler_family, compiler_major = _pinned_host_toolchain()
    cc = _resolved_compiler("CC", "cc")
    cxx = _resolved_compiler("CXX", "c++")
    cc_version = _require_host_compiler(cc, "C", compiler_family, compiler_major)
    cxx_version = _require_host_compiler(cxx, "C++", compiler_family, compiler_major)
    linker = _resolved_build_tool("LD", cxx, "ld")
    archiver = _resolved_build_tool("AR", cxx, "ar")
    cmake = _resolved_compiler("CMAKE_COMMAND", "cmake")
    libstdcxx_value = _run_capture(
        [str(cxx), "-print-file-name=libstdc++.so"], "host C++ runtime inspection"
    )
    try:
        libstdcxx = Path(libstdcxx_value).resolve(strict=True)
    except OSError as exc:
        raise AdapterError(
            f"Host C++ compiler did not resolve libstdc++.so: {libstdcxx_value or '<none>'}"
        ) from exc
    architecture = platform.machine()
    files = {
        "archiver": archiver,
        "cmake": cmake,
        "cuda.cuda.h": cuda.include_dir / "cuda.h",
        "cuda.cuda_runtime_api.h": cuda.include_dir / "cuda_runtime_api.h",
        "cuda.driver": cuda.driver_library,
        "cuda.nvcc": cuda.compiler,
        "cuda.runtime": cuda.cudart_library,
        "edge.CMakeLists.txt": source / "CMakeLists.txt",
        "edge.llmInferenceRuntime.h": source / "cpp" / "runtime" / "llmInferenceRuntime.h",
        "edge.version.h": source / "cpp" / "common" / "version.h",
        "host.cc": cc,
        "host.cxx": cxx,
        "host.libstdc++": libstdcxx,
        "linker": linker,
        "tensorrt.NvInfer.h": tensorrt.include_dir / "NvInfer.h",
        "tensorrt.NvInferVersion.h": tensorrt.include_dir / "NvInferVersion.h",
        "tensorrt.NvOnnxParser.h": tensorrt.onnx_parser_include_dir / "NvOnnxParser.h",
        "tensorrt.nvinfer": tensorrt.library,
        "tensorrt.onnx_parser": tensorrt.onnx_parser_library,
    }
    file_identity = {
        name: {"path": str(path.resolve(strict=True)), "sha256": _file_sha256(path)}
        for name, path in files.items()
    }
    versions = {
        "cc": cc_version,
        "cmake": _run_capture([str(cmake), "--version"], "CMake inspection"),
        "cxx": cxx_version,
        "nvcc": _run_capture([str(cuda.compiler), "--version"], "CUDA compiler inspection"),
    }
    payload = {
        "architecture": architecture,
        "build_environment": {name: os.environ.get(name, "") for name in _EDGE_BUILD_ENVIRONMENT},
        "files": file_identity,
        "schema_version": 1,
        "versions": versions,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _EdgeBuildToolchain(cc, cxx, linker, archiver, cmake, identity, architecture)


def _edge_products(build_dir: Path) -> tuple[Path, Path, Path] | None:
    for configuration in ("", "Release", "RelWithDebInfo", "Debug"):
        suffix = Path(configuration) if configuration else Path()
        core = build_dir / "cpp" / suffix / "libedgellmCore.a"
        plugin = build_dir / suffix / "libNvInfer_edgellm_plugin.so.1.0"
        build_tool = build_dir / "examples" / "llm" / suffix / "llm_build"
        if all(path.is_file() and path.stat().st_size > 0 for path in (core, plugin, build_tool)):
            return core, plugin, build_tool
    return None


def _edge_configure_definitions(
    tensorrt: _TensorRtInstallation,
    cuda: _CudaInstallation,
    toolchain: _EdgeBuildToolchain,
    profile: ProfileSpec,
) -> dict[str, str]:
    return {
        "AARCH64_BUILD": "OFF",
        "BUILD_PYTHON_BINDINGS": "OFF",
        "BUILD_UNIT_TESTS": "OFF",
        "CMAKE_AR": str(toolchain.archiver),
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": profile.target["gpu_architecture"].removeprefix("sm"),
        "CMAKE_CUDA_COMPILER": str(cuda.compiler),
        "CMAKE_CUDA_HOST_COMPILER": str(toolchain.cxx),
        "CMAKE_CXX_COMPILER": str(toolchain.cxx),
        "CMAKE_C_COMPILER": str(toolchain.cc),
        "CMAKE_LINKER": str(toolchain.linker),
        "CMAKE_SKIP_RPATH": "ON",
        "CUDA_CTK_VERSION": cuda.version,
        "CUDA_DIR": str(cuda.root),
        "CUDA_DRIVER_LIB": str(cuda.driver_library),
        "CUDA_RUNTIME_API_INCLUDE_DIR": str(cuda.include_dir),
        "CUDART_LIB": str(cuda.cudart_library),
        "ENABLE_CUTE_DSL": "OFF",
        "ENABLE_NVTX_PROFILING": "OFF",
        "NVINFER_LIB": str(tensorrt.library),
        "NV_ONNX_PARSER_LIB": str(tensorrt.onnx_parser_library),
        "ONNX_PARSER_INCLUDE_DIR": str(tensorrt.onnx_parser_include_dir),
        "TRTMC_EDGE_BUILD_HOST_ARCHITECTURE": toolchain.architecture,
        "TRTMC_EDGE_BUILD_TOOLCHAIN_SHA256": toolchain.sha256,
        "TRT_INCLUDE_DIR": str(tensorrt.include_dir),
        "TRT_PACKAGE_DIR": str(tensorrt.library.parent),
        "TensorRT_INCLUDE_DIR": str(tensorrt.include_dir),
        "TensorRT_LIBRARY": str(tensorrt.library),
        "TensorRT_OnnxParser_INCLUDE_DIR": str(tensorrt.onnx_parser_include_dir),
        "TensorRT_OnnxParser_LIBRARY": str(tensorrt.onnx_parser_library),
    }


def _edge_build_recipe(
    source: Path,
    tensorrt: _TensorRtInstallation,
    cuda: _CudaInstallation,
    toolchain: _EdgeBuildToolchain,
    profile: ProfileSpec,
) -> dict[str, Any]:
    dependency = _load_dependency_spec()
    return {
        "configure_definitions": _edge_configure_definitions(tensorrt, cuda, toolchain, profile),
        "edge_commit": dependency.downstream_commit,
        "edge_version": dependency.downstream_version,
        "schema_version": 1,
        "source": str(source.resolve(strict=True)),
        "targets": list(_EDGE_BUILD_TARGETS),
        "toolchain_sha256": toolchain.sha256,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _edge_cache_matches(
    build_dir: Path,
    source: Path,
    definitions: Mapping[str, str],
    recipe_sha256: str,
) -> bool:
    if _cmake_cache_value(build_dir, "CMAKE_HOME_DIRECTORY") != str(source.resolve()):
        return False
    expected = {**definitions, "TRTMC_EDGE_BUILD_RECIPE_SHA256": recipe_sha256}
    path_keys = {
        "CMAKE_AR",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_HOST_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_C_COMPILER",
        "CMAKE_LINKER",
        "CUDA_DIR",
        "CUDA_DRIVER_LIB",
        "CUDA_RUNTIME_API_INCLUDE_DIR",
        "CUDART_LIB",
        "NVINFER_LIB",
        "NV_ONNX_PARSER_LIB",
        "ONNX_PARSER_INCLUDE_DIR",
        "TRT_INCLUDE_DIR",
        "TRT_PACKAGE_DIR",
        "TensorRT_INCLUDE_DIR",
        "TensorRT_LIBRARY",
        "TensorRT_OnnxParser_INCLUDE_DIR",
        "TensorRT_OnnxParser_LIBRARY",
    }
    for key, expected_value in expected.items():
        cached = _cmake_cache_value(build_dir, key)
        if not cached:
            return False
        if key not in path_keys:
            if cached != expected_value:
                return False
            continue
        try:
            if Path(cached).resolve(strict=True) != Path(expected_value).resolve(strict=True):
                return False
        except OSError:
            return False
    return True


def _edge_product_hashes(build_dir: Path, products: tuple[Path, Path, Path]) -> dict[str, str]:
    return {path.relative_to(build_dir).as_posix(): _file_sha256(path) for path in products}


def _write_edge_build_stamp(
    build_dir: Path,
    recipe: Mapping[str, Any],
    products: tuple[Path, Path, Path],
) -> None:
    stamp = build_dir / _EDGE_BUILD_STAMP
    payload = {
        "products": _edge_product_hashes(build_dir, products),
        "recipe": dict(recipe),
        "recipe_sha256": _canonical_sha256(recipe),
        "schema_version": 1,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{_EDGE_BUILD_STAMP}.", dir=build_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, stamp)
    finally:
        temporary.unlink(missing_ok=True)


def _edge_build_stamp_matches(
    build_dir: Path,
    recipe: Mapping[str, Any],
    products: tuple[Path, Path, Path],
) -> bool:
    try:
        stamp = _load_json(build_dir / _EDGE_BUILD_STAMP, "Edge-LLM build stamp")
    except AdapterError:
        return False
    expected = {
        "products": _edge_product_hashes(build_dir, products),
        "recipe": dict(recipe),
        "recipe_sha256": _canonical_sha256(recipe),
        "schema_version": 1,
    }
    return stamp == expected


def _edge_build_matches(
    build_dir: Path,
    source: Path,
    tensorrt: _TensorRtInstallation,
    cuda: _CudaInstallation,
    profile: ProfileSpec,
    toolchain: _EdgeBuildToolchain,
    recipe: Mapping[str, Any],
) -> bool:
    products = _edge_products(build_dir)
    if products is None:
        return False
    definitions = _edge_configure_definitions(tensorrt, cuda, toolchain, profile)
    recipe_sha256 = _canonical_sha256(recipe)
    return _edge_cache_matches(
        build_dir, source, definitions, recipe_sha256
    ) and _edge_build_stamp_matches(build_dir, recipe, products)


def _resolve_edge_dependency(
    runtime_build: Mapping[str, Any],
    profile: ProfileSpec,
) -> _EdgeDependency:
    dependency_spec = _load_dependency_spec()
    source = _resolve_edge_source(runtime_build)
    tensorrt = _resolve_tensorrt(runtime_build)
    cuda = _resolve_cuda(runtime_build)
    parallel = runtime_build.get("parallel", 2)
    toolchain = _host_toolchain_identity(source, tensorrt, cuda)
    definitions = _edge_configure_definitions(tensorrt, cuda, toolchain, profile)
    recipe = _edge_build_recipe(source, tensorrt, cuda, toolchain, profile)
    recipe_sha256 = _canonical_sha256(recipe)

    def resolved_dependency(build_dir: Path) -> _EdgeDependency:
        products = _edge_products(build_dir)
        if products is None:
            raise AdapterError(
                "TensorRT Edge-LLM build did not produce a complete pinned runtime "
                "and llm_build tool"
            )
        return _EdgeDependency(
            source,
            build_dir.resolve(strict=True),
            products[2],
            tensorrt,
            cuda,
            toolchain,
        )

    def build_dependency(build_dir: Path) -> _EdgeDependency:
        cmake_launcher = str(toolchain.cmake)
        configure = [
            cmake_launcher,
            "-S",
            str(source),
            "-B",
            str(build_dir),
            *(f"-D{name}={value}" for name, value in definitions.items()),
            f"-DTRTMC_EDGE_BUILD_RECIPE_SHA256={recipe_sha256}",
        ]
        _run_checked(configure, "pinned TensorRT Edge-LLM CMake configure")
        _validate_edge_source(source)
        _run_checked(
            [
                cmake_launcher,
                "--build",
                str(build_dir),
                "--parallel",
                str(parallel),
                "--target",
                *_EDGE_BUILD_TARGETS,
            ],
            "pinned TensorRT Edge-LLM required-target build",
        )
        _validate_edge_source(source)
        products = _edge_products(build_dir)
        if products is None or not _edge_cache_matches(
            build_dir, source, definitions, recipe_sha256
        ):
            raise AdapterError(
                "TensorRT Edge-LLM build did not produce a complete pinned runtime "
                "and llm_build tool"
            )
        _write_edge_build_stamp(build_dir, recipe, products)
        if not _edge_build_matches(build_dir, source, tensorrt, cuda, profile, toolchain, recipe):
            raise AdapterError("TensorRT Edge-LLM build stamp validation failed after build")
        return resolved_dependency(build_dir)

    configured_build = _runtime_setting(runtime_build, "edge_llm_build_dir", _EDGE_BUILD_DIR_ENV)
    if configured_build:
        candidate = Path(configured_build).expanduser()
        if _edge_build_matches(candidate, source, tensorrt, cuda, profile, toolchain, recipe):
            return resolved_dependency(candidate)
        build_dir = Path(configured_build).expanduser()
        if (build_dir / "CMakeCache.txt").exists():
            raise AdapterError(
                f"Configured TensorRT Edge-LLM build does not match the pinned source, "
                f"TensorRT {dependency_spec.tensorrt_version_text}, "
                f"and CUDA installation: {build_dir}"
            )
        return build_dependency(build_dir)

    build_dir = _edge_dependency_cache_root() / _EDGE_BUILD_CACHE_NAME / recipe_sha256
    if _edge_build_matches(build_dir, source, tensorrt, cuda, profile, toolchain, recipe):
        return resolved_dependency(build_dir)
    with _edge_cache_lock(build_dir):
        if _edge_build_matches(build_dir, source, tensorrt, cuda, profile, toolchain, recipe):
            return resolved_dependency(build_dir)
        _remove_cache_entry(build_dir)
        try:
            return build_dependency(build_dir)
        except Exception:
            _remove_cache_entry(build_dir)
            raise


def _build_runtime_dso(
    output: Path,
    parameters: Mapping[str, Any],
    manifest_sha256: str,
    profile: ProfileSpec,
) -> tuple[Path, Path | None, _EdgeDependency | None]:
    dependency_spec = _load_dependency_spec()
    runtime_build = _validate_runtime_build(parameters)
    fake = runtime_build.get("fake", False)
    if fake and os.environ.get(_ALLOW_FAKE_RUNTIME_BUILD_ENV, "").strip() != "1":
        raise AdapterError(f"Fake runtime builds require {_ALLOW_FAKE_RUNTIME_BUILD_ENV}=1")
    parallel = runtime_build.get("parallel", 2)
    build_directory = output / ".runtime-build"
    configured_sdk_include = _runtime_setting(
        runtime_build,
        "sdk_include_dir",
        "_TRTMC_INTERNAL_QWEN_EDGE_LLM_SDK_INCLUDE_DIR",
    )
    if configured_sdk_include:
        sdk_include_dirs = configured_sdk_include
    else:
        sdk_include_dirs = ";".join(str(root) for root in _private_sdk_include_roots())
    dependency = None if fake else _resolve_edge_dependency(runtime_build, profile)
    cmake_launcher = str(dependency.toolchain.cmake) if dependency is not None else "cmake"
    edge_source = (
        _runtime_setting(
            runtime_build,
            "edge_llm_source_dir",
            "_TRTMC_INTERNAL_QWEN_EDGE_LLM_SOURCE_DIR",
        )
        if dependency is None
        else str(dependency.source_dir)
    )
    json_default = (
        str(Path(edge_source) / "3rdParty" / "nlohmannJson" / "include") if edge_source else ""
    )
    json_include = _runtime_setting(
        runtime_build,
        "nlohmann_json_include_dir",
        "_TRTMC_INTERNAL_QWEN_EDGE_LLM_NLOHMANN_JSON_INCLUDE_DIR",
        json_default,
    )
    if not json_include:
        raise AdapterError(
            "Fake Qwen runtime builds require runtime_build.nlohmann_json_include_dir "
            "or _TRTMC_INTERNAL_QWEN_EDGE_LLM_NLOHMANN_JSON_INCLUDE_DIR"
        )
    configure = [
        cmake_launcher,
        "-S",
        str(_runtime_source_root()),
        "-B",
        str(build_directory),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DTRTMC_SDK_INCLUDE_DIRS={sdk_include_dirs}",
        f"-DTRTMC_NLOHMANN_JSON_INCLUDE_DIR={json_include}",
        f"-DTRTMC_QWEN_EDGE_FAKE_RUNTIME={'ON' if fake else 'OFF'}",
        f"-DTRTMC_QWEN_EDGE_MANIFEST_SHA256={manifest_sha256}",
        f"-DTRTMC_EDGE_LLM_VERSION={dependency_spec.downstream_version}",
        f"-DTRTMC_EDGE_LLM_COMMIT={dependency_spec.downstream_commit}",
        f"-DTRTMC_TENSORRT_VERSION={dependency_spec.tensorrt_version_text}",
        f"-DTRTMC_CUDA_VERSION={dependency_spec.cuda_version_text}",
        f"-DTRTMC_CUDA_RUNTIME_VERSION={dependency_spec.cuda_runtime_version}",
        "-DTRTMC_QWEN_EDGE_CUDA_ARCHITECTURES="
        + profile.target["gpu_architecture"].removeprefix("sm"),
    ]
    if dependency is not None:
        selected_cc = dependency.toolchain.cc
        selected_cxx = dependency.toolchain.cxx
        required = {
            "CMAKE_AR": dependency.toolchain.archiver,
            "CMAKE_C_COMPILER": selected_cc,
            "CMAKE_CXX_COMPILER": selected_cxx,
            "CMAKE_LINKER": dependency.toolchain.linker,
            "TRTMC_EDGE_LLM_SOURCE_DIR": dependency.source_dir,
            "TRTMC_EDGE_LLM_BUILD_DIR": dependency.build_dir,
            "TRTMC_TENSORRT_INCLUDE_DIR": dependency.tensorrt.include_dir,
            "TRTMC_TENSORRT_LIBRARY": dependency.tensorrt.library,
            "TRTMC_CUDA_INCLUDE_DIR": dependency.cuda.include_dir,
            "TRTMC_CUDART_LIBRARY": dependency.cuda.cudart_library,
            "TRTMC_CUDA_DRIVER_LIBRARY": dependency.cuda.driver_library,
            "CMAKE_CUDA_COMPILER": dependency.cuda.compiler,
            "CMAKE_CUDA_HOST_COMPILER": selected_cxx,
        }
        configure.extend(f"-D{name}={value}" for name, value in required.items())
    _run_checked(configure, "capsule runtime CMake configure")
    if dependency is not None:
        _validate_edge_source(dependency.source_dir)
    _run_checked(
        [
            cmake_launcher,
            "--build",
            str(build_directory),
            "--parallel",
            str(parallel),
        ],
        "capsule runtime build",
    )
    if dependency is not None:
        _validate_edge_source(dependency.source_dir)
    runtime_candidates = list(build_directory.rglob(RUNTIME_LIBRARY))
    if len(runtime_candidates) != 1:
        raise AdapterError(
            f"Capsule runtime build produced {len(runtime_candidates)} copies of {RUNTIME_LIBRARY}"
        )
    plugin_candidates = list(build_directory.rglob(EDGE_LLM_PLUGIN))
    plugin = plugin_candidates[0] if len(plugin_candidates) == 1 else None
    if dependency is not None:
        if plugin is None:
            raise AdapterError("Capsule runtime build did not stage exactly one Edge-LLM plugin")
        _validate_packaged_elf(runtime_candidates[0])
        _validate_packaged_elf(plugin)
    return runtime_candidates[0], plugin, dependency


def _resolve_runtime_payloads(
    output: Path,
    parameters: Mapping[str, Any],
    manifest_sha256: str,
    profile: ProfileSpec,
) -> tuple[Path, Path, _EdgeDependency | None]:
    explicit_runtime = _parameter_or_environment(parameters, "runtime_library", _RUNTIME_ENV)
    if explicit_runtime:
        if not _test_payload_injection_enabled(parameters):
            raise AdapterError("Edge-LLM runtime payload overrides are test-only")
        runtime = _require_payload(
            parameters, "runtime_library", _RUNTIME_ENV, "Qwen Edge-LLM runtime DSO"
        )
        built_plugin = None
        dependency = None
    else:
        runtime, built_plugin, dependency = _build_runtime_dso(
            output, parameters, manifest_sha256, profile
        )
    explicit_plugin = _parameter_or_environment(parameters, "runtime_plugin", _PLUGIN_ENV)
    if explicit_plugin:
        if not _test_payload_injection_enabled(parameters):
            raise AdapterError("Edge-LLM plugin payload overrides are test-only")
        plugin = _require_payload(
            parameters, "runtime_plugin", _PLUGIN_ENV, "Qwen Edge-LLM TensorRT plugin"
        )
    elif built_plugin is not None:
        plugin = built_plugin
    else:
        raise AdapterError(
            "On-demand fake runtime build requires an explicit runtime_plugin payload"
        )
    return runtime, plugin, dependency


def _copy_payload(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)
    shutil.copymode(source, destination, follow_symlinks=False)


def _build(
    request: Mapping[str, Any],
    output: Path,
    profile: ProfileSpec,
) -> dict[str, Any]:
    dependency_spec = _load_dependency_spec()
    parameters = _require_mapping(request["parameters"], "request.parameters")
    manifest_sha256 = _validated_runtime_compile_binding(request, profile)
    reason = _profile_parameter_reason(parameters, profile)
    if reason:
        raise AdapterError(reason)
    model = _require_mapping(request["model"], "request.model")
    revision = str(model["revision"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AdapterError(f"Capsule build output is not empty: {output}")
    artifacts = output / "artifacts"
    artifacts.mkdir()
    engine_attempt: Path | None = None
    try:
        runtime_source, plugin_source, dependency = _resolve_runtime_payloads(
            output, parameters, manifest_sha256, profile
        )
        engine_source, vocab_size, engine_attempt = _build_or_resolve_engine(
            parameters, output, dependency, plugin_source, profile, revision
        )
        engine_destination = artifacts / "engine.dir"
        shutil.copytree(engine_source, engine_destination, symlinks=False)
        _copy_payload(runtime_source, artifacts / RUNTIME_LIBRARY)
        _copy_payload(plugin_source, artifacts / EDGE_LLM_PLUGIN)
        _validate_artifact_tree(artifacts)
        edge_commit = dependency_spec.downstream_commit
        tensorrt_version = dependency_spec.tensorrt_version_text
        cuda_version = dependency_spec.cuda_version_text
        descriptor = {
            "schema_version": 1,
            # The generic host owns and validates this opaque selection token.
            # Capsule code only carries it from the build request to its output.
            "build_binding": dict(
                _require_mapping(request.get("build_binding"), "request.build_binding")
            ),
            "implementation_id": IMPLEMENTATION_ID,
            "profile_id": profile.profile_id,
            "operation": profile.operation,
            "model": {"id": profile.model_id, "revision": revision},
            "target": {
                **dict(request["target"]),
                **dict(profile.target),
                "gpu_memory_mib": profile.minimum_memory_mib,
            },
            "runtime": {
                "abi": 1,
                "library": RUNTIME_LIBRARY,
                "plugin": EDGE_LLM_PLUGIN,
            },
            "artifacts": {
                "layout": "directory-tree-v1",
                "engine_dir": "engine.dir",
                "runtime_library": RUNTIME_LIBRARY,
                "runtime_plugin": EDGE_LLM_PLUGIN,
            },
            "limits": {
                "max_input_length": profile.max_input_length,
                "max_cache_length": profile.max_cache_length,
                "max_batch_size": profile.max_batch_size,
                "vocab_size": vocab_size,
            },
            "versions": {
                "model_revision": revision,
                "edge_llm": dependency_spec.downstream_version,
                "edge_llm_commit": edge_commit,
                "tensorrt": tensorrt_version,
                "cuda": cuda_version,
            },
            "bundle_info": {
                "model_type": str(profile.engine["model"]),
                "family": "qwen",
                "trt_version": tensorrt_version,
                "gpu_name": profile.target["gpu_name"],
                "vocab_size": vocab_size,
                "hidden_size": int(profile.engine["hidden_size"]),
                "num_layers": int(profile.engine["num_hidden_layers"]),
                "num_attention_heads": int(profile.engine["num_attention_heads"]),
                "num_key_value_heads": int(profile.engine["num_key_value_heads"]),
                "max_cache_length": profile.max_cache_length,
                "runtime_strategy": "text_generation",
                "precision": profile.precision,
                "quantization": profile.quantization,
            },
            "bundle_config": {
                "model_id": profile.model_id,
                "model_revision": revision,
                "model_type": str(profile.engine["model"]),
                "family": "qwen",
                "runtime_provider": IMPLEMENTATION_ID,
                "runtime_strategy": "text_generation",
                "precision": profile.precision,
                "quantization": profile.quantization,
                "max_cache_length": profile.max_cache_length,
            },
        }
        (output / "descriptor.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        if engine_attempt is not None:
            shutil.rmtree(engine_attempt, ignore_errors=True)
        runtime_build = output / ".runtime-build"
        if runtime_build.exists():
            shutil.rmtree(runtime_build, ignore_errors=True)
        default_workspace = output / ".edge-build-workspace"
        if default_workspace.exists():
            shutil.rmtree(default_workspace, ignore_errors=True)

    return {"schema_version": 1, "descriptor": "descriptor.json", "artifacts": "artifacts"}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("probe", "build"))
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "build" and args.output is None:
        parser.error("build requires --output")
    if args.operation == "probe" and args.output is not None:
        parser.error("probe does not accept --output")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(list(sys.argv[1:] if argv is None else argv))
        request = _load_request(args.request)
        profiles = _load_profiles()
        profile, reason = _select_profile(request, profiles)
        if args.operation == "probe":
            if profile is None:
                response = {
                    "schema_version": 1,
                    "supported": False,
                    "reason": reason,
                }
            else:
                response = {
                    "schema_version": 1,
                    "supported": True,
                    "profile_id": profile.profile_id,
                    "profile_sha256": _qualified_semantic_source_sha256(profile),
                }
        else:
            assert args.output is not None
            if profile is None:
                raise AdapterError(reason)
            _select_parent_active_cuda_device()
            response = _build(
                request,
                args.output.resolve(),
                profile,
            )
        print(json.dumps(response, sort_keys=True, separators=(",", ":")))
        return 0
    except (AdapterError, OSError, RuntimeError, ValueError) as exc:
        print(f"Qwen Edge-LLM adapter error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
