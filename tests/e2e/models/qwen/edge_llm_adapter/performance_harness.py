# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned MC-versus-direct EdgeLLM performance qualification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


SCHEMA = "trtmc.qwen.edgellm.performance-runner.v1"
SUMMARY_SCHEMA = "trtmc.qwen.edgellm.performance.v1"
WARMUPS = 5
MEASURED_REQUESTS = 30
REPETITIONS = 3
MEDIAN_RATIO_MAX = 1.05
P95_RATIO_MAX = 1.10
THROUGHPUT_RATIO_MIN = 0.95
PROMPT = "Reply with one short sentence about accelerated computing."
MAX_NEW_TOKENS = 32
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEPENDENCY_LOCK_PATH = (
    _REPOSITORY_ROOT
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "qwen"
    / "edge_llm_adapter"
    / "dependency.lock"
)
_EDGE_BUILD_STAMP = ".trtmc-edge-build-stamp.json"
_EDGE_BUILD_TARGETS = ("edgellmCore", "NvInfer_edgellm_plugin", "llm_build")
_EDGE_PRODUCT_LAYOUTS = tuple(
    frozenset(
        (
            f"cpp/{configuration}libedgellmCore.a",
            f"{configuration}libNvInfer_edgellm_plugin.so.1.0",
            f"examples/llm/{configuration}llm_build",
        )
    )
    for configuration in ("", "Release/", "RelWithDebInfo/", "Debug/")
)
_INFERENCE_BINARY_LAYOUTS = tuple(
    f"examples/llm/{configuration}llm_inference"
    for configuration in ("", "Release/", "RelWithDebInfo/", "Debug/")
)


class PerformanceContractError(RuntimeError):
    """A runner, build input, or result violates the qualification contract."""


@dataclass(frozen=True)
class DependencyPins:
    edge_name: str
    edge_source: str
    edge_version: str
    edge_commit: str
    tensorrt_version: str
    tensorrt_version_parts: tuple[int, int, int, int]
    cuda_version: str
    cuda_runtime_version: int


def _required_string(table: object, key: str, context: str) -> str:
    if not isinstance(table, dict):
        raise PerformanceContractError(f"dependency lock {context} must be a table")
    value = table.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise PerformanceContractError(
            f"dependency lock {context}.{key} must be a non-empty canonical string"
        )
    return value


def load_dependency_pins(path: Path = DEPENDENCY_LOCK_PATH) -> DependencyPins:
    """Load and validate the performance cohort from the model-owned lock."""

    if path.is_symlink() or not path.is_file():
        raise PerformanceContractError(f"dependency lock must be a regular file: {path}")
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PerformanceContractError(f"unable to read dependency lock {path}: {exc}") from exc
    if value.get("schema_version") != 1:
        raise PerformanceContractError("dependency lock schema_version must be 1")

    downstream = value.get("downstream")
    edge_name = _required_string(downstream, "name", "downstream")
    edge_source = _required_string(downstream, "source", "downstream")
    edge_version = _required_string(downstream, "version", "downstream")
    edge_tag = _required_string(downstream, "tag", "downstream")
    edge_commit = _required_string(downstream, "commit", "downstream")
    edge_source_mode = _required_string(downstream, "source_mode", "downstream")
    if edge_name != "tensorrt-edge-llm" or edge_source_mode != "git":
        raise PerformanceContractError("dependency lock must identify the git EdgeLLM runtime")
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", edge_version) is None:
        raise PerformanceContractError("dependency lock downstream.version must be semantic")
    if edge_tag != f"v{edge_version}":
        raise PerformanceContractError("dependency lock downstream.tag must match its version")
    if re.fullmatch(r"[0-9a-f]{40}", edge_commit) is None:
        raise PerformanceContractError(
            "dependency lock downstream.commit must be a lowercase Git commit"
        )

    tensorrt_version = _required_string(value.get("tensorrt"), "version", "tensorrt")
    tensorrt_match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)", tensorrt_version)
    if tensorrt_match is None:
        raise PerformanceContractError(
            "dependency lock tensorrt.version must have major.minor.patch.build"
        )
    tensorrt_version_parts = tuple(int(part) for part in tensorrt_match.groups())

    cuda_version = _required_string(value.get("cuda"), "version", "cuda")
    cuda_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", cuda_version)
    if cuda_match is None:
        raise PerformanceContractError("dependency lock cuda.version must have major.minor")
    cuda_major, cuda_minor = (int(part) for part in cuda_match.groups())
    if cuda_minor >= 100:
        raise PerformanceContractError("dependency lock CUDA minor version is unsupported")

    return DependencyPins(
        edge_name=edge_name,
        edge_source=edge_source,
        edge_version=edge_version,
        edge_commit=edge_commit,
        tensorrt_version=tensorrt_version,
        tensorrt_version_parts=tensorrt_version_parts,
        cuda_version=cuda_version,
        cuda_runtime_version=cuda_major * 1000 + cuda_minor * 10,
    )


DEPENDENCY_PINS = load_dependency_pins()
EDGE_NAME = DEPENDENCY_PINS.edge_name
EDGE_SOURCE = DEPENDENCY_PINS.edge_source
EDGE_VERSION = DEPENDENCY_PINS.edge_version
EDGE_COMMIT = DEPENDENCY_PINS.edge_commit
TENSORRT_VERSION = DEPENDENCY_PINS.tensorrt_version
TENSORRT_VERSION_PARTS = DEPENDENCY_PINS.tensorrt_version_parts
CUDA_VERSION = DEPENDENCY_PINS.cuda_version
CUDA_RUNTIME_VERSION = DEPENDENCY_PINS.cuda_runtime_version


@dataclass(frozen=True)
class EdgeRuntimeIdentity:
    name: str
    version: str
    commit: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "commit": self.commit}


@dataclass(frozen=True)
class PerformanceRunners:
    direct: Path
    model_connect: Path
    environment: dict[str, str]
    runtime_identity: EdgeRuntimeIdentity


def _run(
    command: Sequence[str | Path],
    *,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    normalized = [str(item) for item in command]
    result = subprocess.run(
        normalized,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=None if environment is None else dict(environment),
    )
    if result.returncode != 0:
        raise PerformanceContractError(
            f"command failed ({result.returncode}): {' '.join(normalized)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def parse_cmake_cache(path: Path) -> dict[str, str]:
    """Read the typed assignments needed from an EdgeLLM CMake cache."""

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PerformanceContractError(f"unable to read EdgeLLM CMake cache {path}: {exc}") from exc
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, _separator, _type = key_and_type.partition(":")
        if key and value and not value.endswith("-NOTFOUND"):
            values[key] = value
    return values


def _cache_path(
    cache: Mapping[str, str],
    names: Sequence[str],
    description: str,
    *,
    directory: bool,
) -> Path:
    for name in names:
        raw = cache.get(name, "").strip()
        if not raw:
            continue
        try:
            path = Path(raw).expanduser().resolve(strict=True)
        except OSError:
            continue
        if path.is_dir() if directory else path.is_file():
            return path
    raise PerformanceContractError(
        f"EdgeLLM CMake cache does not identify {description}: {', '.join(names)}"
    )


def edge_build_root(inference_binary: Path) -> Path:
    """Find the configured EdgeLLM build owning the official inference binary."""

    binary = inference_binary.resolve(strict=True)
    for parent in binary.parents:
        if (parent / "CMakeCache.txt").is_file():
            return parent
    raise PerformanceContractError(
        f"official EdgeLLM inference binary is not below a CMake build: {binary}"
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_edge_build_stamp(build_directory: Path) -> dict[str, Any]:
    path = build_directory / _EDGE_BUILD_STAMP
    if path.is_symlink() or not path.is_file():
        raise PerformanceContractError(
            f"EdgeLLM build is missing a regular {_EDGE_BUILD_STAMP}: {build_directory}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PerformanceContractError(f"EdgeLLM build stamp is malformed: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerformanceContractError(f"EdgeLLM build stamp is malformed: {path}")
    return value


def _validate_edge_git_source(source: Path) -> str:
    top_level = _run(
        ["git", "-C", source, "rev-parse", "--show-toplevel"], timeout=30
    ).stdout.strip()
    try:
        resolved_top_level = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise PerformanceContractError(
            f"EdgeLLM git top-level is unavailable: {top_level or '<none>'}"
        ) from exc
    if resolved_top_level != source:
        raise PerformanceContractError(
            f"EdgeLLM CMake source is not its git top-level: {source} != {resolved_top_level}"
        )

    revision = _run(["git", "-C", source, "rev-parse", "HEAD"], timeout=30).stdout.strip()
    if revision != EDGE_COMMIT:
        raise PerformanceContractError(
            f"EdgeLLM source must be pinned at {EDGE_COMMIT}; got {revision or '<none>'}"
        )
    origin = _run(["git", "-C", source, "remote", "get-url", "origin"], timeout=30).stdout.strip()
    if origin != EDGE_SOURCE:
        raise PerformanceContractError(
            f"EdgeLLM source origin must be {EDGE_SOURCE}; got {origin or '<none>'}"
        )
    changes = _run(
        [
            "git",
            "-C",
            source,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=none",
        ],
        timeout=30,
    ).stdout
    if changes.strip():
        raise PerformanceContractError(
            "EdgeLLM source must have no tracked, untracked, ignored, or submodule changes; "
            f"git status reported: {changes.strip()[:4000]}"
        )

    submodules = _run(
        ["git", "-C", source, "submodule", "status", "--recursive"], timeout=30
    ).stdout
    invalid_submodules = [
        line for line in submodules.splitlines() if line and not line.startswith(" ")
    ]
    if invalid_submodules:
        raise PerformanceContractError(
            "EdgeLLM recursive submodules must be initialized at their pinned commits; "
            f"git submodule status reported: {'; '.join(invalid_submodules)[:4000]}"
        )
    submodule_changes = _run(
        [
            "git",
            "-C",
            source,
            "submodule",
            "--quiet",
            "foreach",
            "--recursive",
            "git status --porcelain=v1 --untracked-files=all --ignored=matching",
        ],
        timeout=30,
    ).stdout
    if submodule_changes.strip():
        raise PerformanceContractError(
            "EdgeLLM recursive submodules must have no tracked, untracked, or ignored changes; "
            f"git status reported: {submodule_changes.strip()[:4000]}"
        )
    return revision


def validate_edge_build_provenance(
    build_directory: Path, cache: Mapping[str, str]
) -> EdgeRuntimeIdentity:
    """Fail closed unless the direct runner build is the pinned clean EdgeLLM source."""

    build_directory = build_directory.resolve(strict=True)
    source = _cache_path(
        cache, ("CMAKE_HOME_DIRECTORY",), "the pinned EdgeLLM source", directory=True
    )
    if cache.get("CMAKE_HOME_DIRECTORY", "").strip() != str(source):
        raise PerformanceContractError(
            "EdgeLLM CMAKE_HOME_DIRECTORY must be the canonical source path"
        )

    stamp = _read_edge_build_stamp(build_directory)
    if set(stamp) != {"products", "recipe", "recipe_sha256", "schema_version"}:
        raise PerformanceContractError("EdgeLLM build stamp has an invalid schema")
    if type(stamp["schema_version"]) is not int or stamp["schema_version"] != 1:
        raise PerformanceContractError("EdgeLLM build stamp has an invalid schema version")
    recipe = stamp["recipe"]
    if not isinstance(recipe, dict) or set(recipe) != {
        "configure_definitions",
        "edge_commit",
        "edge_version",
        "schema_version",
        "source",
        "targets",
        "toolchain_sha256",
    }:
        raise PerformanceContractError("EdgeLLM build stamp recipe has an invalid schema")
    configure_definitions = recipe["configure_definitions"]
    if (
        not isinstance(configure_definitions, dict)
        or not configure_definitions
        or any(
            not isinstance(name, str) or not name or not isinstance(value, str) or not value
            for name, value in configure_definitions.items()
        )
    ):
        raise PerformanceContractError(
            "EdgeLLM build stamp recipe has invalid configure definitions"
        )
    if (
        type(recipe["schema_version"]) is not int
        or recipe["schema_version"] != 1
        or recipe["edge_version"] != EDGE_VERSION
        or recipe["edge_commit"] != EDGE_COMMIT
        or recipe["source"] != str(source)
        or recipe["targets"] != list(_EDGE_BUILD_TARGETS)
        or not isinstance(recipe["toolchain_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", recipe["toolchain_sha256"]) is None
    ):
        raise PerformanceContractError(
            "EdgeLLM build stamp recipe does not identify the pinned source and build"
        )
    expected_recipe_sha256 = _canonical_sha256(recipe)
    if stamp["recipe_sha256"] != expected_recipe_sha256:
        raise PerformanceContractError("EdgeLLM build stamp recipe digest does not match")
    if cache.get("TRTMC_EDGE_BUILD_RECIPE_SHA256", "").strip() != expected_recipe_sha256:
        raise PerformanceContractError(
            "EdgeLLM CMake cache is not bound to the stamped build recipe"
        )

    products = stamp["products"]
    if not isinstance(products, dict) or frozenset(products) not in _EDGE_PRODUCT_LAYOUTS:
        raise PerformanceContractError("EdgeLLM build stamp has an invalid product inventory")
    for relative, expected_sha256 in products.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise PerformanceContractError("EdgeLLM build stamp has invalid product metadata")
        product = build_directory / relative
        if product.is_symlink() or not product.is_file():
            raise PerformanceContractError(f"EdgeLLM stamped product is unavailable: {relative}")
        try:
            product.resolve(strict=True).relative_to(build_directory)
        except (OSError, ValueError) as exc:
            raise PerformanceContractError(
                f"EdgeLLM stamped product escapes its build directory: {relative}"
            ) from exc
        if _sha256(product) != expected_sha256:
            raise PerformanceContractError(
                f"EdgeLLM stamped product digest does not match: {relative}"
            )

    revision = _validate_edge_git_source(source)
    return EdgeRuntimeIdentity(EDGE_NAME, recipe["edge_version"], revision)


def prepare_official_inference_binary(build_directory: Path) -> Path:
    """Build EdgeLLM's official direct runner from the qualified build tree."""

    build_directory = build_directory.resolve(strict=True)
    cache_path = build_directory / "CMakeCache.txt"
    cache = parse_cmake_cache(cache_path)
    runtime_identity = validate_edge_build_provenance(build_directory, cache)
    cmake = shutil.which("cmake")
    if cmake is None:
        raise PerformanceContractError("cmake is required to build EdgeLLM llm_inference")
    _run(
        [
            cmake,
            "--build",
            build_directory,
            "--parallel",
            "4",
            "--target",
            "llm_inference",
        ],
        timeout=1_800,
    )
    binaries = []
    for relative_path in _INFERENCE_BINARY_LAYOUTS:
        candidate = build_directory / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            continue
        binaries.append(candidate.resolve(strict=True))
    if len(binaries) != 1:
        raise PerformanceContractError(
            "qualified EdgeLLM build must produce exactly one official llm_inference; "
            f"found {len(binaries)} in {build_directory}"
        )
    if (
        validate_edge_build_provenance(build_directory, parse_cmake_cache(cache_path))
        != runtime_identity
    ):
        raise PerformanceContractError(
            "EdgeLLM build provenance changed while building official llm_inference"
        )
    return binaries[0]


def build_performance_runners(
    *,
    source_directory: Path,
    build_directory: Path,
    inference_binary: Path,
    mc_core_library: Path,
    mc_include_directory: Path,
) -> PerformanceRunners:
    """Build the two private runners against the exact installed/runtime inputs."""

    cmake = shutil.which("cmake")
    if cmake is None:
        raise PerformanceContractError("cmake is required to build performance runners")
    edge_build = edge_build_root(inference_binary)
    cache = parse_cmake_cache(edge_build / "CMakeCache.txt")
    runtime_identity = validate_edge_build_provenance(edge_build, cache)
    edge_source = Path(cache["CMAKE_HOME_DIRECTORY"]).resolve(strict=True)
    trt_include = _cache_path(
        cache,
        ("TRT_INCLUDE_DIR", "TensorRT_INCLUDE_DIR"),
        "TensorRT headers",
        directory=True,
    )
    trt_library = _cache_path(
        cache,
        ("NVINFER_LIB", "TensorRT_LIBRARY"),
        "the TensorRT runtime library",
        directory=False,
    )
    cuda_include = _cache_path(
        cache,
        ("CUDA_RUNTIME_API_INCLUDE_DIR", "CUDA_INCLUDE_DIR"),
        "CUDA headers",
        directory=True,
    )
    cudart = _cache_path(cache, ("CUDART_LIB",), "the CUDA runtime library", directory=False)
    cuda_driver = _cache_path(
        cache, ("CUDA_DRIVER_LIB",), "the CUDA driver library", directory=False
    )
    cuda_compiler = _cache_path(
        cache, ("CMAKE_CUDA_COMPILER",), "the CUDA compiler", directory=False
    )

    configure = [
        cmake,
        "-S",
        str(source_directory),
        "-B",
        str(build_directory),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_COMPILER={cuda_compiler}",
        f"-DTRTMC_EDGE_LLM_SOURCE_DIR={edge_source}",
        f"-DTRTMC_EDGE_LLM_BUILD_DIR={edge_build}",
        f"-DTRTMC_EDGE_LLM_SOURCE={EDGE_SOURCE}",
        f"-DTRTMC_EDGE_LLM_VERSION={EDGE_VERSION}",
        f"-DTRTMC_EDGE_LLM_COMMIT={EDGE_COMMIT}",
        f"-DTRTMC_TENSORRT_INCLUDE_DIR={trt_include}",
        f"-DTRTMC_TENSORRT_LIBRARY={trt_library}",
        f"-DTRTMC_TENSORRT_VERSION={TENSORRT_VERSION}",
        f"-DTRTMC_TENSORRT_MAJOR={TENSORRT_VERSION_PARTS[0]}",
        f"-DTRTMC_TENSORRT_MINOR={TENSORRT_VERSION_PARTS[1]}",
        f"-DTRTMC_TENSORRT_PATCH={TENSORRT_VERSION_PARTS[2]}",
        f"-DTRTMC_TENSORRT_BUILD={TENSORRT_VERSION_PARTS[3]}",
        f"-DTRTMC_CUDA_INCLUDE_DIR={cuda_include}",
        f"-DTRTMC_CUDART_LIBRARY={cudart}",
        f"-DTRTMC_CUDA_DRIVER_LIBRARY={cuda_driver}",
        f"-DTRTMC_CUDA_VERSION={CUDA_VERSION}",
        f"-DTRTMC_CUDA_RUNTIME_VERSION={CUDA_RUNTIME_VERSION}",
        f"-DTRTMC_MC_INCLUDE_DIR={mc_include_directory.resolve(strict=True)}",
        f"-DTRTMC_MC_CORE_LIBRARY={mc_core_library.resolve(strict=True)}",
    ]
    for cache_name in ("CMAKE_CXX_COMPILER", "CMAKE_CUDA_HOST_COMPILER"):
        configured = cache.get(cache_name, "").strip()
        if configured:
            configure.append(f"-D{cache_name}={configured}")
    _run(configure, timeout=300)
    _run(
        [
            cmake,
            "--build",
            str(build_directory),
            "--parallel",
            "4",
            "--target",
            "trtmc_qwen_edgellm_direct_performance",
            "trtmc_qwen_edgellm_mc_performance",
        ],
        timeout=1_800,
    )
    if validate_edge_build_provenance(edge_build, cache) != runtime_identity:
        raise PerformanceContractError("EdgeLLM build provenance changed while building runners")

    direct = (build_directory / "trtmc_qwen_edgellm_direct_performance").resolve(strict=True)
    model_connect = (build_directory / "trtmc_qwen_edgellm_mc_performance").resolve(strict=True)
    library_directories = (
        str(mc_core_library.resolve(strict=True).parent),
        str(trt_library.parent),
        str(cudart.parent),
        *(os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)),
    )
    library_path = os.pathsep.join(dict.fromkeys(path for path in library_directories if path))
    return PerformanceRunners(
        direct,
        model_connect,
        {**os.environ, "LD_LIBRARY_PATH": library_path},
        runtime_identity,
    )


def artifact_root(repository: Path) -> Path:
    """Resolve the required external artifact directory and keep it out of source."""

    raw = os.environ.get("TRTMC_PERF_ARTIFACT_DIR", "").strip()
    if not raw:
        raise PerformanceContractError(
            "TRTMC_PERF_ARTIFACT_DIR is required for A100 performance qualification"
        )
    path = Path(raw).expanduser().resolve()
    source = repository.resolve(strict=True)
    if path == source or source in path.parents:
        raise PerformanceContractError("TRTMC_PERF_ARTIFACT_DIR must be outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_identity(repository: Path) -> dict[str, Any]:
    """Bind results to either an exact archive digest or the current Git state."""

    archive_sha256 = os.environ.get("TRTMC_TESTED_SOURCE_SHA256", "").strip()
    if archive_sha256:
        if re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None:
            raise PerformanceContractError(
                "TRTMC_TESTED_SOURCE_SHA256 must be exactly 64 lowercase hexadecimal digits"
            )
        return {"kind": "archive_sha256", "value": archive_sha256, "dirty": False}

    repository = repository.resolve(strict=True)
    if not (repository / ".git").exists():
        raise PerformanceContractError(
            "TRTMC_TESTED_SOURCE_SHA256 is required when the tested source has no .git metadata"
        )
    revision = _run(["git", "-C", repository, "rev-parse", "HEAD"], timeout=30).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise PerformanceContractError(f"git returned an invalid source revision: {revision!r}")
    status = _run(
        ["git", "-C", repository, "status", "--porcelain=v1", "--untracked-files=all"],
        timeout=30,
    ).stdout
    return {"kind": "git_revision", "value": revision, "dirty": bool(status.strip())}


def runner_request(kind: str, **runtime: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "runtime": {"kind": kind, **runtime},
        "prompt": PROMPT,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_chat_template": True,
            "enable_thinking": False,
        },
        "warmups_per_repetition": WARMUPS,
        "measured_requests_per_repetition": MEASURED_REQUESTS,
        "require_native_token_ids": True,
        "synchronize_each_request": True,
    }


_RUNNER_RESULT_KEYS = {
    "schema",
    "runtime_kind",
    "runtime_initializations",
    "decoding_cuda_graph_captured",
    "observed_tensorrt_version",
    "observed_cuda_runtime_version",
    "native_token_ids",
    "synchronized_each_request",
    "warmups_completed",
    "measured_elapsed_ms",
    "iterations",
}


def validate_runner_result(result: Mapping[str, Any], expected_kind: str) -> None:
    if set(result) != _RUNNER_RESULT_KEYS:
        raise PerformanceContractError(f"{expected_kind} runner returned an invalid result schema")
    expected = {
        "schema": SCHEMA,
        "runtime_kind": expected_kind,
        "runtime_initializations": 1,
        "decoding_cuda_graph_captured": True,
        "observed_tensorrt_version": TENSORRT_VERSION,
        "observed_cuda_runtime_version": CUDA_RUNTIME_VERSION,
        "native_token_ids": True,
        "synchronized_each_request": True,
        "warmups_completed": WARMUPS,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise PerformanceContractError(
                f"{expected_kind} runner returned {key}={result[key]!r}, expected {value!r}"
            )
    elapsed = result["measured_elapsed_ms"]
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise PerformanceContractError(f"{expected_kind} runner returned invalid elapsed time")
    iterations = result["iterations"]
    if not isinstance(iterations, list) or len(iterations) != MEASURED_REQUESTS:
        raise PerformanceContractError(
            f"{expected_kind} runner did not return {MEASURED_REQUESTS} measurements"
        )
    for iteration in iterations:
        if not isinstance(iteration, dict) or set(iteration) != {
            "latency_ms",
            "generated",
            "token_ids",
        }:
            raise PerformanceContractError(f"{expected_kind} runner returned an invalid iteration")
        latency = iteration["latency_ms"]
        token_ids = iteration["token_ids"]
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency <= 0
            or not isinstance(iteration["generated"], str)
            or not iteration["generated"].strip()
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise PerformanceContractError(
                f"{expected_kind} runner returned an invalid measured response"
            )


def _run_one(
    runner: Path,
    request: Mapping[str, Any],
    stem: str,
    directory: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], Path, Path]:
    request_path = directory / f"{stem}-request.json"
    result_path = directory / f"{stem}-result.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    _run(
        [runner, "--request", request_path, "--output", result_path],
        timeout=7_200,
        environment=environment,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_runner_result(result, str(request["runtime"]["kind"]))
    return result, request_path, result_path


def run_repetitions(
    runners: PerformanceRunners,
    *,
    bundle: Path,
    engine_directory: Path,
    plugin: Path,
    runtime_cache: Path,
    artifact_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Run both long-lived paths three times, alternating their order."""

    artifact_directory.mkdir(parents=True, exist_ok=False)
    direct_request = runner_request(
        "edgellm-direct",
        engine_dir=str(engine_directory.resolve(strict=True)),
        plugin=str(plugin.resolve(strict=True)),
    )
    mc_request = runner_request(
        "model-connect",
        bundle=str(bundle.resolve(strict=True)),
        runtime_cache=str(runtime_cache.resolve()),
    )
    direct_results: list[dict[str, Any]] = []
    mc_results: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    for repetition in range(REPETITIONS):
        direct_item = (runners.direct, direct_request, "direct", direct_results)
        mc_item = (runners.model_connect, mc_request, "mc", mc_results)
        for runner, request, label, destination in (
            (direct_item, mc_item) if repetition % 2 == 0 else (mc_item, direct_item)
        ):
            result, request_path, result_path = _run_one(
                runner,
                request,
                f"{label}-repetition-{repetition + 1}",
                artifact_directory,
                runners.environment,
            )
            destination.append(result)
            raw_paths.extend((request_path, result_path))
    return direct_results, mc_results, raw_paths


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_performance(
    *,
    profile_id: str,
    model_id: str,
    revision: str,
    source_identity: Mapping[str, Any],
    runtime_identity: EdgeRuntimeIdentity,
    direct_results: Sequence[Mapping[str, Any]],
    mc_results: Sequence[Mapping[str, Any]],
    raw_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate parity, calculate the three gates, and return the artifact payload."""

    if set(source_identity) != {"kind", "value", "dirty"} or source_identity["kind"] not in {
        "archive_sha256",
        "git_revision",
    }:
        raise PerformanceContractError("source identity has an invalid schema")
    identity_value = source_identity["value"]
    identity_dirty = source_identity["dirty"]
    expected_length = 64 if source_identity["kind"] == "archive_sha256" else 40
    if (
        not isinstance(identity_value, str)
        or re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", identity_value) is None
        or not isinstance(identity_dirty, bool)
        or (source_identity["kind"] == "archive_sha256" and identity_dirty)
    ):
        raise PerformanceContractError("source identity has invalid values")
    if runtime_identity != EdgeRuntimeIdentity(EDGE_NAME, EDGE_VERSION, EDGE_COMMIT):
        raise PerformanceContractError("runtime identity is not the pinned EdgeLLM release")
    if len(direct_results) != REPETITIONS or len(mc_results) != REPETITIONS:
        raise PerformanceContractError(f"both runtimes must provide {REPETITIONS} repetitions")
    for result in direct_results:
        validate_runner_result(result, "edgellm-direct")
    for result in mc_results:
        validate_runner_result(result, "model-connect")

    direct_iterations = [item for result in direct_results for item in result["iterations"]]
    mc_iterations = [item for result in mc_results for item in result["iterations"]]
    expected_measurements = REPETITIONS * MEASURED_REQUESTS
    if (
        len(direct_iterations) != expected_measurements
        or len(mc_iterations) != expected_measurements
    ):
        raise PerformanceContractError("runner result count changed after validation")

    reference_text = direct_iterations[0]["generated"]
    reference_tokens = direct_iterations[0]["token_ids"]
    parity_passed = all(
        direct["generated"] == reference_text
        and mc["generated"] == reference_text
        and direct["token_ids"] == reference_tokens
        and mc["token_ids"] == reference_tokens
        for direct, mc in zip(direct_iterations, mc_iterations, strict=True)
    )
    direct_latencies = [float(item["latency_ms"]) for item in direct_iterations]
    mc_latencies = [float(item["latency_ms"]) for item in mc_iterations]
    direct_median = statistics.median(direct_latencies)
    mc_median = statistics.median(mc_latencies)
    direct_p95 = _percentile(direct_latencies, 0.95)
    mc_p95 = _percentile(mc_latencies, 0.95)
    direct_elapsed = sum(float(result["measured_elapsed_ms"]) for result in direct_results)
    mc_elapsed = sum(float(result["measured_elapsed_ms"]) for result in mc_results)
    aggregate_metrics = (
        direct_median,
        mc_median,
        direct_p95,
        mc_p95,
        direct_elapsed,
        mc_elapsed,
    )
    if not all(math.isfinite(metric) for metric in aggregate_metrics):
        raise PerformanceContractError("performance aggregates contain a non-finite value")
    median_ratio = mc_median / direct_median
    p95_ratio = mc_p95 / direct_p95
    throughput_ratio = direct_elapsed / mc_elapsed
    if not all(math.isfinite(ratio) for ratio in (median_ratio, p95_ratio, throughput_ratio)):
        raise PerformanceContractError("performance ratios contain a non-finite value")

    failures = []
    if source_identity["kind"] == "git_revision" and identity_dirty:
        failures.append("source_checkout_dirty")
    if not parity_passed:
        failures.append("output_or_token_parity")
    if median_ratio > MEDIAN_RATIO_MAX:
        failures.append("median_latency_ratio")
    if p95_ratio > P95_RATIO_MAX:
        failures.append("p95_latency_ratio")
    if throughput_ratio < THROUGHPUT_RATIO_MIN:
        failures.append("throughput_ratio")
    return {
        "schema": SUMMARY_SCHEMA,
        "source_identity": dict(source_identity),
        "profile_id": profile_id,
        "model_id": model_id,
        "model_revision": revision,
        "runtime": runtime_identity.as_dict(),
        "measurement": {
            "warmups_per_repetition": WARMUPS,
            "measured_requests_per_repetition": MEASURED_REQUESTS,
            "repetitions": REPETITIONS,
            "total_measurements_per_runtime": expected_measurements,
            "cuda_synchronization": "cudaDeviceSynchronize-before-and-after",
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_chat_template": True,
            "enable_thinking": False,
        },
        "parity": {
            "passed": parity_passed,
            "generated_text": reference_text,
            "token_ids": reference_tokens,
        },
        "metrics": {
            "mc_median_ms": mc_median,
            "direct_median_ms": direct_median,
            "mc_p95_ms": mc_p95,
            "direct_p95_ms": direct_p95,
            "mc_measured_elapsed_ms": mc_elapsed,
            "direct_measured_elapsed_ms": direct_elapsed,
            "mc_to_direct_median_ratio": median_ratio,
            "mc_to_direct_p95_ratio": p95_ratio,
            "mc_to_direct_throughput_ratio": throughput_ratio,
        },
        "thresholds": {
            "median_ratio_max": MEDIAN_RATIO_MAX,
            "p95_ratio_max": P95_RATIO_MAX,
            "throughput_ratio_min": THROUGHPUT_RATIO_MIN,
        },
        "raw_artifacts": [
            {"name": path.name, "sha256": _sha256(path)} for path in sorted(raw_paths)
        ],
        "failures": failures,
        "passed": not failures,
    }


def safe_profile_name(profile_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", profile_id)
