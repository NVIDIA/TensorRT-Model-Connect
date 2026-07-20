#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run the complete Qwen EdgeLLM A100 qualification locally."""

from __future__ import annotations

import argparse
import ast
import email.parser
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import venv
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[5]
TEST_ROOT = Path(__file__).resolve().parent
BUILDER_ROOT = (
    REPOSITORY / "python" / "tensorrt_model_connect" / "families" / "qwen" / "edge_llm_adapter"
)
EDGE_SOURCE_URL = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
EDGE_TAG = "v0.9.0"
EDGE_COMMIT = "1ac0f2b99642045125e1c5ac7b109434ba3b36c7"
TENSORRT_VERSION = (11, 2, 0, 113)
TENSORRT_VERSION_TEXT = ".".join(str(component) for component in TENSORRT_VERSION)
CUDA_VERSION = "13.3"
CUDA_VERSION_ENCODED = 13030
GPU_NAME = "NVIDIA A100 80GB PCIe"
REQUIRED_TOOLS = ("cc", "c++", "cmake", "git", "ldd", "ninja", "nvidia-smi", "readelf")
COEXISTENCE_MODEL_IDS = frozenset(
    {
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instruct-2507",
    }
)


class QualificationError(RuntimeError):
    """The host or one qualification step violates the pinned profile."""


@dataclass(frozen=True)
class TensorRtInputs:
    root: Path
    include_dir: Path
    library: Path
    onnx_parser_library: Path
    python_wheel: Path


@dataclass(frozen=True)
class CudaInputs:
    root: Path
    include_dir: Path
    cudart_library: Path
    driver_library: Path
    compiler: Path


@dataclass(frozen=True)
class Profile:
    leaf: str
    model_id: str
    strict_test: Path
    runner_builder: Path
    edge_source_environment: str
    edge_build_environment: str


@dataclass(frozen=True)
class InstalledModelConnect:
    python: Path
    package: Path
    binary: Path
    core_library: Path
    sdk_include: Path


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    normalized = [str(item) for item in command]
    print("$ " + shlex.join(normalized), flush=True)
    result = subprocess.run(
        normalized,
        cwd=cwd,
        env=None if env is None else dict(env),
        check=False,
        text=True,
        capture_output=capture,
    )
    if capture and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        detail = result.stderr.strip() if capture else ""
        suffix = f"\n{detail}" if detail else ""
        raise QualificationError(
            f"command exited with {result.returncode}: {shlex.join(normalized)}{suffix}"
        )
    return result


def _output(command: Sequence[str | Path], *, cwd: Path | None = None) -> str:
    return _run(command, cwd=cwd, capture=True).stdout.strip()


def _require_file(path: Path, description: str, *, executable: bool = False) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"{description} is unavailable: {path}: {exc}") from exc
    if not resolved.is_file():
        raise QualificationError(f"{description} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise QualificationError(f"{description} is not executable: {resolved}")
    return resolved


def _require_directory(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"{description} is unavailable: {path}: {exc}") from exc
    if not resolved.is_dir():
        raise QualificationError(f"{description} is not a directory: {resolved}")
    return resolved


def _first_file(directories: Sequence[Path], patterns: Sequence[str], description: str) -> Path:
    matches: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for candidate in directory.glob(pattern):
                if candidate.is_file():
                    resolved = candidate.resolve(strict=True)
                    matches[str(resolved)] = resolved
    if not matches:
        raise QualificationError(f"unable to find {description}")
    return sorted(matches.values(), key=lambda path: (len(path.name), path.name))[0]


def _macro(path: Path, name: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)\b", text, re.MULTILINE)
    if match is None:
        raise QualificationError(f"{path} does not define {name}")
    return int(match.group(1))


def _tensorrt_header_version(path: Path) -> tuple[int, int, int, int]:
    text = path.read_text(encoding="utf-8")

    def component(public: str, enterprise: str) -> int:
        for name in (enterprise, public):
            match = re.search(
                rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)\b",
                text,
                re.MULTILINE,
            )
            if match is not None:
                return int(match.group(1))
        raise QualificationError(f"{path} does not define {public} or {enterprise}")

    return tuple(
        component(public, enterprise)
        for public, enterprise in (
            ("NV_TENSORRT_MAJOR", "TRT_MAJOR_ENTERPRISE"),
            ("NV_TENSORRT_MINOR", "TRT_MINOR_ENTERPRISE"),
            ("NV_TENSORRT_PATCH", "TRT_PATCH_ENTERPRISE"),
            ("NV_TENSORRT_BUILD", "TRT_BUILD_ENTERPRISE"),
        )
    )


def _wheel_metadata(path: Path) -> tuple[str, str]:
    wheel = _require_file(path, "TensorRT Python wheel")
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise QualificationError(
                    f"TensorRT Python wheel must contain one METADATA file: {wheel}"
                )
            metadata = email.parser.Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise QualificationError(f"unable to inspect TensorRT Python wheel {wheel}: {exc}") from exc
    name = str(metadata.get("Name", "")).strip().lower().replace("_", "-")
    version = str(metadata.get("Version", "")).strip()
    if name != "tensorrt" or version != TENSORRT_VERSION_TEXT:
        raise QualificationError(
            f"TensorRT Python wheel must provide TensorRT {TENSORRT_VERSION_TEXT}; "
            f"found {name or '<missing>'} {version or '<missing>'}"
        )
    if "x86_64" not in wheel.name or "cp312" not in wheel.name:
        raise QualificationError(
            f"TensorRT Python wheel must target CPython 3.12 on x86_64: {wheel.name}"
        )
    return name, version


def _resolve_tensorrt(root: Path, python_wheel: Path) -> TensorRtInputs:
    resolved = _require_directory(root, "TensorRT root")
    include_candidates = (resolved / "include", resolved / "include" / "x86_64-linux-gnu")
    include_dir = next(
        (candidate for candidate in include_candidates if (candidate / "NvInfer.h").is_file()),
        None,
    )
    if include_dir is None or not (include_dir / "NvOnnxParser.h").is_file():
        raise QualificationError(f"TensorRT root has incomplete headers: {resolved}")
    version_header = include_dir / "NvInferVersion.h"
    version = _tensorrt_header_version(version_header)
    if version != TENSORRT_VERSION:
        raise QualificationError(
            f"TensorRT headers are {'.'.join(map(str, version))}, not {TENSORRT_VERSION_TEXT}"
        )
    library_directories = (
        resolved / "lib",
        resolved / "lib64",
        resolved / "lib" / "x86_64-linux-gnu",
        resolved / "targets" / "x86_64-linux-gnu" / "lib",
    )
    library = _first_file(
        library_directories, ("libnvinfer.so.11", "libnvinfer.so.11.*"), "libnvinfer.so.11"
    )
    parser = _first_file(
        (library.parent,),
        ("libnvonnxparser.so.11", "libnvonnxparser.so.11.*"),
        "libnvonnxparser.so.11 beside libnvinfer",
    )
    _wheel_metadata(python_wheel)
    return TensorRtInputs(
        resolved,
        include_dir.resolve(strict=True),
        library,
        parser,
        _require_file(python_wheel, "TensorRT Python wheel"),
    )


def _resolve_cuda(root: Path) -> CudaInputs:
    resolved = _require_directory(root, "CUDA root")
    include_candidates = (resolved / "include", resolved / "targets" / "x86_64-linux" / "include")
    include_dir = next(
        (
            candidate
            for candidate in include_candidates
            if (candidate / "cuda_runtime_api.h").is_file()
        ),
        None,
    )
    if include_dir is None:
        raise QualificationError(f"CUDA root has no cuda_runtime_api.h: {resolved}")
    encoded = _macro(include_dir / "cuda.h", "CUDA_VERSION")
    if encoded != CUDA_VERSION_ENCODED:
        raise QualificationError(
            f"CUDA headers encode {encoded}, not CUDA {CUDA_VERSION} ({CUDA_VERSION_ENCODED})"
        )
    compiler = _require_file(resolved / "bin" / "nvcc", "CUDA compiler", executable=True)
    version_output = _output([compiler, "--version"])
    version_match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", version_output)
    if (
        version_match is None
        or f"{version_match.group(1)}.{version_match.group(2)}" != CUDA_VERSION
    ):
        raise QualificationError(f"CUDA compiler is not release {CUDA_VERSION}: {compiler}")
    library_directories = (
        resolved / "lib64",
        resolved / "lib",
        resolved / "targets" / "x86_64-linux" / "lib",
    )
    cudart = _first_file(
        library_directories,
        ("libcudart.so.13", "libcudart.so.13.*"),
        "CUDA 13 runtime library",
    )
    driver = _first_file(
        (
            resolved / "lib64" / "stubs",
            resolved / "lib" / "stubs",
            resolved / "targets" / "x86_64-linux" / "lib" / "stubs",
            *library_directories,
        ),
        ("libcuda.so", "libcuda.so.1"),
        "CUDA driver stub below the CUDA root",
    )
    return CudaInputs(resolved, include_dir.resolve(strict=True), cudart, driver, compiler)


def _active_gpu_name() -> str:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )
    rows: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(",", 2))
        if len(fields) == 3:
            rows.append(fields)
    if not rows:
        raise QualificationError("nvidia-smi returned no GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if not visible:
        return rows[0][2]
    if visible == "-1":
        raise QualificationError("CUDA_VISIBLE_DEVICES disables every GPU")
    for index, uuid, name in rows:
        if visible in {index, uuid}:
            return name
    raise QualificationError(f"CUDA_VISIBLE_DEVICES selects unknown device {visible!r}")


def _literal_assignment(path: Path, name: str) -> object:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            values.append(ast.literal_eval(statement.value))
    if len(values) != 1:
        raise QualificationError(f"{path} must declare exactly one literal {name}")
    return values[0]


def _discover_profiles(repository: Path = REPOSITORY) -> tuple[Profile, ...]:
    builder_root = repository / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter"
    test_root = repository / "tests/e2e/models/qwen/edge_llm_adapter"
    profiles: list[Profile] = []
    for manifest in sorted(builder_root.glob("*/IMPLEMENTATION.toml")):
        leaf = manifest.parent.name
        strict_test = test_root / leaf / "test_a100_e2e.py"
        runner_builder = test_root / leaf / "build_runners.py"
        if not strict_test.is_file() or not runner_builder.is_file():
            raise QualificationError(
                f"Qwen EdgeLLM profile {leaf} is missing its strict test or runner builder"
            )
        with manifest.open("rb") as source:
            model_id = tomllib.load(source).get("model", {}).get("id")
        if not isinstance(model_id, str) or not model_id:
            raise QualificationError(f"Qwen EdgeLLM profile has no model.id: {manifest}")
        mapping = _literal_assignment(strict_test, "_PUBLIC_EDGE_BUILD_ENVIRONMENT")
        expected_public = {"TRTMC_EDGE_LLM_SOURCE_DIR", "TRTMC_EDGE_LLM_BUILD_DIR"}
        if not isinstance(mapping, dict) or set(mapping) != expected_public:
            raise QualificationError(
                f"{strict_test} must map the two public EdgeLLM qualification inputs"
            )
        if not all(
            isinstance(value, str) and value.startswith("_TRTMC_INTERNAL_")
            for value in mapping.values()
        ):
            raise QualificationError(f"{strict_test} contains invalid internal build mappings")
        profiles.append(
            Profile(
                leaf,
                model_id,
                strict_test,
                runner_builder,
                mapping["TRTMC_EDGE_LLM_SOURCE_DIR"],
                mapping["TRTMC_EDGE_LLM_BUILD_DIR"],
            )
        )
    if not profiles:
        raise QualificationError("this checkout contains no Qwen EdgeLLM profiles")
    return tuple(profiles)


def _preflight(
    arguments: argparse.Namespace,
) -> tuple[TensorRtInputs, CudaInputs, tuple[Profile, ...]]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise QualificationError(
            f"qualification requires CPython 3.12; found {sys.implementation.name} {platform.python_version()}"
        )
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise QualificationError(
            f"qualification requires Linux x86_64; found {platform.system()} {platform.machine()}"
        )
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise QualificationError("qualification tools are unavailable: " + ", ".join(missing))
    gpu_name = _active_gpu_name()
    if gpu_name != GPU_NAME:
        raise QualificationError(f"qualification requires {GPU_NAME}; found {gpu_name}")
    _require_outside_repository(arguments.work_dir, "qualification work directory")
    if arguments.hf_cache is not None:
        _require_outside_repository(arguments.hf_cache, "Hugging Face cache")
    tensorrt = _resolve_tensorrt(arguments.tensorrt_root, arguments.tensorrt_python_wheel)
    cuda = _resolve_cuda(arguments.cuda_root)
    profiles = _discover_profiles()
    return tensorrt, cuda, profiles


def _create_venv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    python = path / "bin" / "python"
    return _require_file(python, "virtual-environment Python", executable=True)


def _require_outside_repository(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    repository = REPOSITORY.resolve(strict=True)
    try:
        resolved.relative_to(repository)
    except ValueError:
        return resolved
    raise QualificationError(f"{description} must be outside the source checkout: {resolved}")


def _prepend_path(environment: dict[str, str], name: str, paths: Sequence[Path]) -> None:
    existing = environment.get(name, "")
    entries = [*(str(path) for path in paths), *existing.split(os.pathsep)]
    environment[name] = os.pathsep.join(dict.fromkeys(entry for entry in entries if entry))


def _toolchain_environment(tensorrt: TensorRtInputs, cuda: CudaInputs) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "TENSORRT_ROOT": str(tensorrt.root),
            "TRT_ROOT": str(tensorrt.root),
            "CUDA_HOME": str(cuda.root),
            "CUDA_PATH": str(cuda.root),
            "CUDAToolkit_ROOT": str(cuda.root),
        }
    )
    _prepend_path(environment, "PATH", (cuda.compiler.parent,))
    _prepend_path(
        environment, "LD_LIBRARY_PATH", (tensorrt.library.parent, cuda.cudart_library.parent)
    )
    _prepend_path(environment, "CMAKE_PREFIX_PATH", (tensorrt.root, cuda.root))
    return environment


def _build_model_connect_wheel(
    run_root: Path,
    tensorrt: TensorRtInputs,
    cuda: CudaInputs,
) -> Path:
    build_python = _create_venv(run_root / "wheel-build-venv")
    _run([build_python, "-m", "pip", "install", "--disable-pip-version-check", "build>=1.2"])
    wheel_directory = run_root / "wheel"
    wheel_directory.mkdir(parents=True)
    environment = _toolchain_environment(tensorrt, cuda)
    environment.update(
        {
            "CONAN_PY_BUILD_PROFILE_AUTODETECT": "1",
            "TRTMC_TRT_INCLUDE_DIR": str(tensorrt.include_dir),
            "TRTMC_TRT_LIBRARY": str(tensorrt.library),
            "TRTMC_CUDA_INCLUDE_DIR": str(cuda.include_dir),
            "TRTMC_CUDART_LIBRARY": str(cuda.cudart_library),
            "WHEEL_PYVER": "py312",
            "WHEEL_ABI": "none",
            "WHEEL_ARCH": "linux_x86_64",
        }
    )
    _run(
        [
            build_python,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            wheel_directory,
            "-C",
            f"build-dir={run_root / 'wheel-build'}",
            ".",
        ],
        cwd=REPOSITORY,
        env=environment,
    )
    wheels = sorted(wheel_directory.glob("*-py312-none-linux_x86_64.whl"))
    if len(wheels) != 1:
        raise QualificationError(
            f"full same-host build produced {len(wheels)} linux_x86_64 wheels: {wheels}"
        )
    return wheels[0].resolve(strict=True)


def _install_model_connect(
    run_root: Path,
    wheel: Path,
    tensorrt: TensorRtInputs,
    cuda: CudaInputs,
) -> InstalledModelConnect:
    python = _create_venv(run_root / "installed")
    site_packages = Path(
        _output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ]
        )
    ).resolve(strict=True)
    tensorrt_libs = site_packages / "tensorrt_libs"
    tensorrt_libs.symlink_to(tensorrt.library.parent, target_is_directory=True)
    install_environment = _toolchain_environment(tensorrt, cuda)
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            tensorrt.python_wheel,
        ],
        env=install_environment,
    )
    imported = _run(
        [
            python,
            "-I",
            "-c",
            "import tensorrt; print(tensorrt.__version__)",
        ],
        env=install_environment,
        capture=True,
    )
    if imported.stdout.strip().splitlines()[-1] != TENSORRT_VERSION_TEXT:
        raise QualificationError(
            f"installed TensorRT binding is not {TENSORRT_VERSION_TEXT}: {imported.stdout!r}"
        )
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            wheel,
            "pytest>=7",
        ],
        env=install_environment,
    )
    result = _run(
        [
            python,
            "-I",
            "-c",
            (
                "import json,pathlib,tensorrt_model_connect as m; "
                "print(json.dumps({'package':str(pathlib.Path(m.__file__).resolve().parent)}))"
            ),
        ],
        capture=True,
    )
    package = Path(json.loads(result.stdout.splitlines()[-1])["package"]).resolve(strict=True)
    binary = _require_file(package / "bin" / "trtmc", "wheel-bundled trtmc", executable=True)
    cores = {
        str(path.resolve(strict=True)): path.resolve(strict=True)
        for path in (package / "bin").glob("libtrtmc_core.so*")
        if path.is_file()
    }
    if len(cores) != 1:
        raise QualificationError(
            f"wheel must contain one uniquely resolved libtrtmc_core beside {binary}: "
            f"{sorted(cores)}"
        )
    core = _require_file(next(iter(cores.values())), "wheel-bundled libtrtmc_core")
    sdk_include = _require_directory(
        package / "runtime_provider" / "_sdk" / "include",
        "wheel-bundled optimized-runtime SDK include root",
    )
    for relative in ("trtmc/pipeline.h", "runtime/providers/optimized_runtime_factory.h"):
        _require_file(sdk_include / relative, "wheel-bundled optimized-runtime SDK header")
    for native in (binary, core):
        if package not in native.parents:
            raise QualificationError(f"wheel native payload escaped its package: {native}")
    return InstalledModelConnect(python, package, binary, core, sdk_include)


def _validate_edge_source(source: Path) -> Path:
    resolved = _require_directory(source, "TensorRT Edge-LLM source")
    if _output(["git", "-C", resolved, "rev-parse", "HEAD"]) != EDGE_COMMIT:
        raise QualificationError(f"EdgeLLM source is not pinned at {EDGE_COMMIT}: {resolved}")
    if _output(["git", "-C", resolved, "remote", "get-url", "origin"]) != EDGE_SOURCE_URL:
        raise QualificationError(f"EdgeLLM source has the wrong origin: {resolved}")
    status = _output(
        [
            "git",
            "-C",
            resolved,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=none",
        ]
    )
    if status:
        raise QualificationError(f"EdgeLLM source is not clean: {resolved}\n{status[:4000]}")
    submodules = _output(["git", "-C", resolved, "submodule", "status", "--recursive"])
    invalid = [line for line in submodules.splitlines() if line and not line.startswith(" ")]
    if invalid:
        raise QualificationError("EdgeLLM submodules are not pinned: " + "; ".join(invalid))
    required = (
        resolved / "CMakeLists.txt",
        resolved / "tensorrt_edgellm/scripts/export.py",
        resolved / "3rdParty/nlohmannJson/include/nlohmann/json.hpp",
    )
    for path in required:
        _require_file(path, "pinned EdgeLLM source input")
    return resolved


def _acquire_edge_source(work_root: Path) -> Path:
    source = work_root / "edge" / f"TensorRT-Edge-LLM-{EDGE_TAG}"
    if source.exists():
        return _validate_edge_source(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--branch",
            EDGE_TAG,
            "--single-branch",
            "--no-checkout",
            EDGE_SOURCE_URL,
            source,
        ]
    )
    _run(["git", "-C", source, "checkout", "--detach", EDGE_COMMIT])
    _run(["git", "-C", source, "submodule", "update", "--init", "--recursive"])
    return _validate_edge_source(source)


def _qualification_environment(
    installed: InstalledModelConnect,
    tensorrt: TensorRtInputs,
    cuda: CudaInputs,
    edge_source: Path,
    edge_build: Path,
    profiles: Sequence[Profile],
    hf_cache: Path | None,
) -> dict[str, str]:
    environment = _toolchain_environment(tensorrt, cuda)
    environment.update(
        {
            "TRTMC_BINARY": str(installed.binary),
            "TRTMC_CORE_LIBRARY": str(installed.core_library),
            "TRTMC_INSTALLED_PYTHON": str(installed.python),
            "TRTMC_EDGE_LLM_SOURCE_DIR": str(edge_source),
            "TRTMC_EDGE_LLM_BUILD_DIR": str(edge_build),
        }
    )
    _prepend_path(environment, "PATH", (installed.python.parent, installed.binary.parent))
    for profile in profiles:
        environment[profile.edge_source_environment] = str(edge_source)
        environment[profile.edge_build_environment] = str(edge_build)
    if hf_cache is not None:
        hf_cache.mkdir(parents=True, exist_ok=True)
        environment["HF_HOME"] = str(hf_cache)
    return environment


def _seed_edge_build(
    run_root: Path,
    installed: InstalledModelConnect,
    profile: Profile,
    environment: dict[str, str],
) -> Path:
    seed = run_root / "seed"
    seed.mkdir()
    environment = dict(environment)
    environment["TRTMC_PYTHON_PROFILE_ROOT"] = str(run_root / "exporter-profiles")
    bundle = seed / f"{profile.leaf}.trtfb"
    _run(
        [
            installed.binary,
            "build",
            profile.model_id,
            "-o",
            bundle,
            "--precision",
            "fp16",
            "--max-cache-length",
            "4096",
            "--max-batch-size",
            "4",
        ],
        cwd=seed,
        env=environment,
    )
    inspected = _run([installed.binary, "inspect", bundle], cwd=seed, env=environment, capture=True)
    if "optimized_runtime.json" not in inspected.stdout:
        raise QualificationError("seed build did not produce a delegated EdgeLLM bundle")
    return bundle


def _edge_plugin(edge_build: Path) -> Path:
    candidates = sorted(edge_build.rglob("libNvInfer_edgellm_plugin.so.1.0"))
    resolved = {str(path.resolve(strict=True)): path.resolve(strict=True) for path in candidates}
    if len(resolved) != 1:
        raise QualificationError(
            f"shared EdgeLLM build has {len(resolved)} exact plugin products: {sorted(resolved)}"
        )
    if not (edge_build / ".trtmc-edge-build-stamp.json").is_file():
        raise QualificationError(f"shared EdgeLLM build has no MC build stamp: {edge_build}")
    return next(iter(resolved.values()))


def _build_runner(
    run_root: Path,
    profile: Profile,
    installed: InstalledModelConnect,
    tensorrt: TensorRtInputs,
    cuda: CudaInputs,
    edge_source: Path,
    edge_build: Path,
    plugin: Path,
    environment: dict[str, str],
) -> tuple[Path, Path]:
    build = run_root / "runners" / profile.leaf
    runner_environment = dict(environment)
    runner_environment.update(
        {
            "TRTMC_EDGE_LLM_SOURCE_DIR": str(edge_source),
            "TRTMC_EDGE_LLM_BUILD_DIR": str(edge_build),
            "TRTMC_EDGE_LLM_PLUGIN_LIBRARY": str(plugin),
            "TRTMC_TENSORRT_INCLUDE_DIR": str(tensorrt.include_dir),
            "TRTMC_TENSORRT_LIBRARY": str(tensorrt.library),
            "TRTMC_TENSORRT_VERSION": TENSORRT_VERSION_TEXT,
            "TRTMC_CUDA_INCLUDE_DIR": str(cuda.include_dir),
            "TRTMC_CUDART_LIBRARY": str(cuda.cudart_library),
            "TRTMC_CUDA_DRIVER_LIBRARY": str(cuda.driver_library),
            "TRTMC_CUDA_VERSION": CUDA_VERSION,
            "TRTMC_NLOHMANN_JSON_INCLUDE_DIR": str(edge_source / "3rdParty/nlohmannJson/include"),
            "TRTMC_MC_INCLUDE_DIR": str(installed.sdk_include),
            "TRTMC_MC_CORE_LIBRARY": str(installed.core_library),
        }
    )
    _run(
        [installed.python, profile.runner_builder, "--build-dir", build, "--parallel", "4"],
        cwd=run_root,
        env=runner_environment,
    )
    direct = _require_file(
        build / "trtmc_edgellm_direct_runner", "direct EdgeLLM runner", executable=True
    )
    mc = _require_file(build / "trtmc_edgellm_mc_runner", "Model Connect runner", executable=True)
    return direct, mc


def _run_strict_profile(
    run_root: Path,
    profile: Profile,
    installed: InstalledModelConnect,
    direct_runner: Path,
    mc_runner: Path,
    environment: dict[str, str],
) -> None:
    profile_environment = dict(environment)
    profile_environment.update(
        {
            "TRTMC_EDGELLM_DIRECT_RUNNER": str(direct_runner),
            "TRTMC_EDGELLM_MC_RUNNER": str(mc_runner),
        }
    )
    temporary = run_root / "pytest" / profile.leaf
    _run(
        [
            installed.python,
            "-m",
            "pytest",
            "-vv",
            "-s",
            "--basetemp",
            temporary,
            profile.strict_test,
        ],
        cwd=REPOSITORY,
        env=profile_environment,
    )


def _run_coexistence_if_complete(
    run_root: Path,
    profiles: Sequence[Profile],
    installed: InstalledModelConnect,
    environment: dict[str, str],
) -> None:
    coexistence = TEST_ROOT / "coexistence" / "test_a100_coexistence.py"
    available_models = {profile.model_id for profile in profiles}
    if not COEXISTENCE_MODEL_IDS.issubset(available_models) or not coexistence.is_file():
        print("coexistence=skipped (requires all three leaves and the coexistence test)")
        return
    _run(
        [
            installed.python,
            "-m",
            "pytest",
            "-vv",
            "-s",
            "--basetemp",
            run_root / "pytest" / "coexistence",
            coexistence,
        ],
        cwd=REPOSITORY,
        env=environment,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorrt-root", required=True, type=Path)
    parser.add_argument("--tensorrt-python-wheel", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--hf-cache", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parse_args(argv)
        tensorrt, cuda, profiles = _preflight(arguments)
        work_root = _require_outside_repository(arguments.work_dir, "qualification work directory")
        work_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = work_root / "runs" / f"{timestamp}-{os.getpid()}"
        run_root.mkdir(parents=True)
        print(f"qualification_work_dir={run_root}")
        print("profiles=" + ",".join(profile.leaf for profile in profiles))

        wheel = _build_model_connect_wheel(run_root, tensorrt, cuda)
        installed = _install_model_connect(run_root, wheel, tensorrt, cuda)
        edge_source = _acquire_edge_source(work_root)
        edge_build = work_root / "edge" / f"build-{EDGE_COMMIT[:12]}-trt11.2-cuda13.3-sm80"
        environment = _qualification_environment(
            installed,
            tensorrt,
            cuda,
            edge_source,
            edge_build,
            profiles,
            arguments.hf_cache.expanduser().resolve() if arguments.hf_cache else None,
        )
        seed_bundle = _seed_edge_build(run_root, installed, profiles[0], environment)
        plugin = _edge_plugin(edge_build)

        for profile in profiles:
            direct, mc = _build_runner(
                run_root,
                profile,
                installed,
                tensorrt,
                cuda,
                edge_source,
                edge_build,
                plugin,
                environment,
            )
            _run_strict_profile(run_root, profile, installed, direct, mc, environment)
        _run_coexistence_if_complete(run_root, profiles, installed, environment)

        print("qualification=passed")
        print(f"wheel={wheel}")
        print(f"installed_package={installed.package}")
        print(f"seed_bundle={seed_bundle}")
        print(f"edge_source={edge_source}")
        print(f"edge_build={edge_build}")
        print(f"run_artifacts={run_root}")
        return 0
    except (QualificationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Qwen EdgeLLM A100 qualification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
