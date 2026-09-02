# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in read-only providers for local system environment discovery."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from .commands import CommandSpec, EnvironmentPath, PathScope
from .cohorts import normalize_architecture
from .models import DevToolkitError, ToolchainObservation
from .provisioning import ContextHandle, ProvisionPolicy
from .resolution import (
    CudaSource,
    ContextLock,
    EnvironmentRequest,
    IncompatibleCombination,
    ProviderDescriptor,
    ToolchainCandidate,
)
from .runner import Runner, command_output
from .toolchain import (
    CUDA_RELEASE,
    observe_local_toolchain,
    tensorrt_header_version,
)


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


class LocalExecutionContext:
    descriptor = ProviderDescriptor("local", "trtmc-devtoolkit-local==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        *,
        repository: Path,
        runner: Runner,
    ) -> ContextLock:
        del repository, runner
        actual = normalize_architecture(platform.machine())
        if request.architecture is not None:
            requested = normalize_architecture(request.architecture)
            if requested != actual:
                raise IncompatibleCombination(
                    f"Local target architecture is {actual}; requested {requested}"
                )
        release = _os_release()
        return ContextLock(
            provider=self.descriptor,
            operating_system=platform.system().lower(),
            architecture=actual,
            identity={
                "os_id": release.get("ID", "unknown"),
                "os_version": release.get("VERSION_ID", "unknown"),
            },
            locator={
                "python": request.target.options.get("python", "python3"),
                "gpu": request.target.options.get("gpu", "0"),
            },
        )

    def provision(
        self,
        lock,
        *,
        repository: Path,
        state_dir: Path,
        policy: ProvisionPolicy,
        runner: Runner,
    ) -> ContextHandle:
        base_python = str(lock.context.locator.get("python", "python3"))
        python = base_python
        if policy is not ProvisionPolicy.ADOPT_ONLY:
            venv = state_dir / "venv"
            python_path = venv / "bin" / "python"
            if not python_path.is_file():
                command: list[str | Path] = [base_python, "-m", "venv"]
                if lock.toolchain.origin != "managed":
                    command.append("--system-site-packages")
                command.append(venv)
                runner.run(command, cwd=repository)
            python = str(python_path)
        cuda_root = str(lock.toolchain.identity.get("cuda_root", ""))
        tensorrt_library = str(lock.toolchain.identity.get("tensorrt_library", ""))
        library_paths = [str(Path(tensorrt_library).parent)]
        for directory in ("lib64", "lib"):
            candidate = Path(cuda_root) / directory
            if candidate.is_dir():
                library_paths.append(str(candidate))
        if cuda_root:
            cudart = _first_library(Path(cuda_root), "libcudart.so", lock.context.architecture)
            if cudart is not None and str(cudart.parent) not in library_paths:
                library_paths.append(str(cudart.parent))
        environment = {
            "PATH": f"{Path(python).parent}:{Path(cuda_root) / 'bin'}:{os.environ.get('PATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(lock.context.locator.get("gpu", "0")),
            "CUDA_HOME": cuda_root,
            "CUDA_PATH": cuda_root,
            "CUDAToolkit_ROOT": cuda_root,
            "TRTMC_TRT_INCLUDE_DIR": str(lock.toolchain.identity.get("tensorrt_include_dir", "")),
            "TRTMC_TRT_LIBRARY": tensorrt_library,
            "TRTMC_TRT_LIBRARY_DIR": str(Path(tensorrt_library).parent),
            "LD_LIBRARY_PATH": ":".join(library_paths),
        }
        return ContextHandle(
            provider=self.descriptor,
            identity=lock.context.identity,
            locator={"python": python, "gpu": lock.context.locator.get("gpu", "0")},
            environment=environment,
        )

    def execute(
        self,
        context: ContextHandle,
        command: CommandSpec,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
        check: bool,
        capture_output: bool,
    ):
        def render(value: str | EnvironmentPath) -> str:
            if not isinstance(value, EnvironmentPath):
                return value
            if value.scope is PathScope.REPOSITORY:
                return str(repository / Path(value.path))
            if value.scope is PathScope.STATE:
                return str(state_dir / Path(value.path))
            return str(value.path)

        return runner.run(
            [render(argument) for argument in command.arguments],
            cwd=Path(render(command.cwd)),
            env={**dict(context.environment), **dict(command.environment)},
            check=check,
            capture_output=capture_output,
        )


def _first_library(root: Path, name: str, architecture: str) -> Path | None:
    triples = {
        "aarch64": ("aarch64-linux", "aarch64-linux-gnu", "sbsa-linux"),
        "x86_64": ("x86_64-linux", "x86_64-linux-gnu"),
    }[architecture]
    directories = [root / "lib64", root / "lib"]
    directories.extend(root / "targets" / triple / "lib" for triple in triples)
    return next(
        (directory / name for directory in directories if (directory / name).is_file()), None
    )


def _native_version_script(library: Path) -> str:
    return (
        "import ctypes; "
        f"lib=ctypes.CDLL({str(library)!r}); "
        "names=('Major','Minor','Patch','Build'); "
        "fs=[getattr(lib, f'getInferLib{name}Version') for name in names]; "
        "[setattr(f, 'restype', ctypes.c_int32) for f in fs]; "
        "print('.'.join(str(f()) for f in fs))"
    )


def _docker_inspect(runner: Runner, repository: Path, container: str) -> dict[str, object]:
    output = command_output(
        runner,
        ["docker", "inspect", "--type", "container", "--format", "{{json .}}", container],
        cwd=repository,
        timeout=30,
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise DevToolkitError(f"Could not inspect Docker container {container}: {error}") from error
    if not isinstance(payload, dict):
        raise DevToolkitError(f"Docker inspect returned invalid data for {container}")
    state = payload.get("State")
    if not isinstance(state, dict) or state.get("Running") is not True:
        raise DevToolkitError(f"Docker container {container} must already be running")
    return payload


_CONTAINER_PROBE_SCRIPT = r"""
import ctypes
import ctypes.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tensorrt

nvcc = shutil.which("nvcc")
if not nvcc:
    raise RuntimeError("nvcc is not available")
nvcc_text = subprocess.check_output([nvcc, "--version"], text=True)
cuda_match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_text, re.I)
if cuda_match is None:
    raise RuntimeError("nvcc did not report a CUDA release")
cuda_root = Path(nvcc).resolve().parent.parent
cuda_libraries = []
for name in ("libcudart.so", "libcublas.so", "libcurand.so"):
    matches = [path for path in cuda_root.rglob(name) if path.is_file()]
    cuda_libraries.append(bool(matches))
cuda_complete = (cuda_root / "include" / "cuda.h").is_file() and all(cuda_libraries)

configured_library = os.environ.get("TRTMC_TRT_LIBRARY")
library_name = configured_library or ctypes.util.find_library("nvinfer") or "libnvinfer.so"
library = ctypes.CDLL(library_name)
names = ("Major", "Minor", "Patch", "Build")
functions = [getattr(library, f"getInferLib{name}Version") for name in names]
for function in functions:
    function.restype = ctypes.c_int32
native = ".".join(str(function()) for function in functions)
mapped = []
maps = Path("/proc/self/maps")
if maps.is_file():
    mapped = [
        line.rsplit(None, 1)[-1]
        for line in maps.read_text().splitlines()
        if "libnvinfer.so" in line and line.rsplit(None, 1)[-1].startswith("/")
    ]
library_path = str(Path(configured_library).resolve()) if configured_library else ""
if not library_path and mapped:
    library_path = str(Path(mapped[0]).resolve())

header_candidates = []
configured_include = os.environ.get("TRTMC_TRT_INCLUDE_DIR") or os.environ.get("TRT_INC_DIR")
if configured_include:
    header_candidates.append(Path(configured_include) / "NvInferVersion.h")
header_candidates.extend(Path("/usr/include").glob("**/NvInferVersion.h"))
header_candidates.extend(Path("/usr/local").glob("**/NvInferVersion.h"))
header = next((path for path in header_candidates if path.is_file()), None)
if header is None:
    raise RuntimeError("NvInferVersion.h is not available")
definitions = dict(
    re.findall(r"^#define\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b", header.read_text(), re.M)
)
parts = []
for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
    value = definitions.get(f"NV_TENSORRT_{name}", "")
    parts.append(definitions.get(value, value))
headers = ".".join(parts)
print(json.dumps({
    "architecture": platform.machine(),
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "python_executable": sys.executable,
    "cuda": cuda_match.group(1),
    "cuda_root": str(cuda_root),
    "cuda_complete": cuda_complete,
    "tensorrt_python": tensorrt.__version__,
    "tensorrt_native": native,
    "tensorrt_headers": headers,
    "tensorrt_include_dir": str(header.parent.resolve()),
    "tensorrt_library": library_path or library_name,
}))
""".strip()


def _container_observation(
    runner: Runner,
    repository: Path,
    container: str,
    image_id: str,
    python: str,
) -> tuple[ToolchainObservation, str, bool]:
    output = command_output(
        runner,
        ["docker", "exec", container, python, "-c", _CONTAINER_PROBE_SCRIPT],
        cwd=repository,
        timeout=30,
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise DevToolkitError(
            f"Container toolchain probe returned invalid JSON: {error}"
        ) from error
    required = (
        "python",
        "python_executable",
        "cuda",
        "cuda_root",
        "tensorrt_python",
        "tensorrt_native",
        "tensorrt_headers",
        "tensorrt_include_dir",
        "tensorrt_library",
        "architecture",
    )
    if not isinstance(payload, dict) or any(not payload.get(name) for name in required):
        raise DevToolkitError("Container toolchain probe omitted required observations")
    architecture = normalize_architecture(str(payload["architecture"]))
    observation = ToolchainObservation(
        python_version=str(payload["python"]),
        cuda_version=str(payload["cuda"]),
        tensorrt_python_version=str(payload["tensorrt_python"]),
        tensorrt_native_version=str(payload["tensorrt_native"]),
        tensorrt_header_version=str(payload["tensorrt_headers"]),
        tensorrt_include_dir=str(payload["tensorrt_include_dir"]),
        tensorrt_library=str(payload["tensorrt_library"]),
        cuda_root=str(payload["cuda_root"]),
        image_id=image_id,
        architecture=architecture,
    )
    return observation, str(payload["python_executable"]), payload.get("cuda_complete") is True


class DockerExecutionContext:
    """Adopt a running user container without imposing a Dockerfile or image version."""

    descriptor = ProviderDescriptor("docker", "trtmc-devtoolkit-docker-adoption==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        *,
        repository: Path,
        runner: Runner,
    ) -> ContextLock:
        container = request.target.options.get("container")
        if not isinstance(container, str) or not container:
            raise DevToolkitError(
                "Docker resolution currently requires a running container for adoption"
            )
        inspected = _docker_inspect(runner, repository, container)
        image_id = inspected.get("Image")
        if not isinstance(image_id, str) or not image_id:
            raise DevToolkitError(f"Docker container {container} has no image identity")
        architecture = (
            normalize_architecture(request.architecture)
            if request.architecture is not None
            else normalize_architecture(
                command_output(
                    runner,
                    ["docker", "exec", container, "uname", "-m"],
                    cwd=repository,
                    timeout=30,
                )
            )
        )
        return ContextLock(
            provider=self.descriptor,
            operating_system="linux",
            architecture=architecture,
            identity={"image_id": image_id, "runtime": "docker"},
            locator={
                "container": container,
                "workspace": request.target.options.get(
                    "workspace", "/workspace/tensorrt-model-connect"
                ),
                "target_state": request.target.options.get("state", "/tmp/trtmc-devtoolkit"),
                "python": request.target.options.get("python", "python3"),
            },
        )

    def provision(
        self,
        lock,
        *,
        repository: Path,
        state_dir: Path,
        policy: ProvisionPolicy,
        runner: Runner,
    ) -> ContextHandle:
        del state_dir
        if policy is ProvisionPolicy.CREATE:
            raise DevToolkitError("The built-in Docker provider is adoption-only")
        container = str(lock.context.locator["container"])
        inspected = _docker_inspect(runner, repository, container)
        if inspected.get("Image") != lock.context.identity.get("image_id"):
            raise DevToolkitError(f"Docker container {container} changed after resolution")
        return ContextHandle(
            provider=self.descriptor,
            identity=lock.context.identity,
            locator={
                **dict(lock.context.locator),
                "python": lock.toolchain.identity.get(
                    "python_executable", lock.context.locator.get("python", "python3")
                ),
            },
        )

    def execute(
        self,
        context: ContextHandle,
        command: CommandSpec,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
        check: bool,
        capture_output: bool,
    ):
        del state_dir

        def render(value: str | EnvironmentPath) -> str:
            if not isinstance(value, EnvironmentPath):
                return value
            if value.scope is PathScope.REPOSITORY:
                return str(PurePosixPath(str(context.locator["workspace"])) / value.path)
            if value.scope is PathScope.STATE:
                return str(PurePosixPath(str(context.locator["target_state"])) / value.path)
            return str(value.path)

        arguments = ["docker", "exec", "--workdir", render(command.cwd)]
        for name, value in sorted(
            {**dict(context.environment), **dict(command.environment)}.items()
        ):
            arguments.extend(["--env", f"{name}={value}"])
        arguments.append(str(context.locator["container"]))
        arguments.extend(render(argument) for argument in command.arguments)
        return runner.run(
            arguments,
            cwd=repository,
            check=check,
            capture_output=capture_output,
        )


class ContainerImageToolchainSource:
    descriptor = ProviderDescriptor("container-image", "trtmc-devtoolkit-container-image==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if context.provider.name != "docker":
            return ()
        container = str(context.locator["container"])
        try:
            observed, python_executable, complete_cuda = _container_observation(
                runner,
                repository,
                container,
                str(context.identity["image_id"]),
                str(context.locator["python"]),
            )
        except (DevToolkitError, OSError, subprocess.TimeoutExpired):
            return ()
        versions = {
            observed.tensorrt_python_version,
            observed.tensorrt_native_version,
            observed.tensorrt_header_version,
        }
        if not complete_cuda or len(versions) != 1:
            return ()
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="image",
                cuda_source="image",
                tensorrt=observed.tensorrt_native_version,
                cuda=observed.cuda_version,
                python=observed.python_version,
                identity={
                    "image_id": observed.image_id,
                    "python_executable": python_executable,
                    "cuda_root": observed.cuda_root,
                    "tensorrt_include_dir": observed.tensorrt_include_dir,
                    "tensorrt_library": observed.tensorrt_library,
                },
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: DockerExecutionContext,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ContextHandle:
        del lock, execution, repository, state_dir, runner
        return context

    def observe(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: DockerExecutionContext,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del execution
        observed, _, complete_cuda = _container_observation(
            runner,
            repository,
            str(context.locator["container"]),
            str(context.identity["image_id"]),
            str(context.locator["python"]),
        )
        if not complete_cuda:
            raise DevToolkitError("Container CUDA toolkit became incomplete after resolution")
        return observed


class SystemToolchainSource:
    """Discover one complete, already-installed local CUDA/TensorRT toolchain."""

    descriptor = ProviderDescriptor("system", "trtmc-devtoolkit-system==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if context.provider.name != "local" or request.target.options.get("prefix"):
            return ()
        discovered_cuda = self.discover_cuda(context, repository, runner)
        if discovered_cuda is None:
            return ()
        cuda_root, nvcc, cuda_version = discovered_cuda
        include_dir = Path(
            os.environ.get("TRTMC_TRT_INCLUDE_DIR")
            or os.environ.get("TRT_INC_DIR")
            or f"/usr/include/{context.architecture}-linux-gnu"
        )
        library_dir = Path(
            os.environ.get("TRTMC_TRT_LIBRARY_DIR")
            or os.environ.get("TRT_LIB_DIR")
            or f"/usr/lib/{context.architecture}-linux-gnu"
        )
        library = Path(os.environ.get("TRTMC_TRT_LIBRARY") or library_dir / "libnvinfer.so")
        header = include_dir / "NvInferVersion.h"
        if not header.is_file() or not library.is_file():
            return ()
        python = str(context.locator.get("python", "python3"))
        try:
            python_version = command_output(
                runner,
                [
                    python,
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                cwd=repository,
                timeout=30,
            )
            tensorrt_python = command_output(
                runner,
                [python, "-c", "import tensorrt; print(tensorrt.__version__)"],
                cwd=repository,
                timeout=30,
            )
            tensorrt_native = command_output(
                runner,
                [python, "-c", _native_version_script(library)],
                cwd=repository,
                timeout=30,
            )
            tensorrt_header = tensorrt_header_version(header)
        except (DevToolkitError, OSError, subprocess.TimeoutExpired):
            return ()
        if len({tensorrt_python, tensorrt_native, tensorrt_header}) != 1:
            return ()
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="system",
                tensorrt=tensorrt_native,
                cuda=cuda_version,
                python=python_version,
                identity={
                    "cuda_root": str(cuda_root),
                    "nvcc": str(nvcc),
                    "tensorrt_include_dir": str(include_dir),
                    "tensorrt_library_dir": str(library.parent),
                    "tensorrt_library": str(library),
                },
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: LocalExecutionContext,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ContextHandle:
        del lock, execution, repository, state_dir, runner
        return context

    def observe(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: LocalExecutionContext,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del execution
        identity = lock.toolchain.identity
        return observe_local_toolchain(
            runner,
            repository=repository,
            python=str(context.locator["python"]),
            nvcc=str(identity["nvcc"]),
            tensorrt_include_dir=Path(str(identity["tensorrt_include_dir"])),
            tensorrt_library=Path(str(identity["tensorrt_library"])),
            environment=dict(context.environment),
            cuda_root=Path(str(identity["cuda_root"])),
        )

    @staticmethod
    def _cuda_root() -> Path | None:
        configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if configured:
            return Path(configured).resolve()
        nvcc = shutil.which("nvcc")
        return Path(nvcc).resolve().parent.parent if nvcc else None

    @staticmethod
    def _complete_cuda(root: Path, architecture: str) -> bool:
        required = (
            root / "bin" / "nvcc",
            root / "include" / "cuda.h",
            _first_library(root, "libcudart.so", architecture),
            _first_library(root, "libcublas.so", architecture),
            _first_library(root, "libcurand.so", architecture),
        )
        return all(path is not None and path.is_file() for path in required)

    @classmethod
    def discover_cuda(
        cls,
        context: ContextLock,
        repository: Path,
        runner: Runner,
    ) -> tuple[Path, Path, str] | None:
        cuda_root = cls._cuda_root()
        if cuda_root is None:
            return None
        return cls.discover_cuda_at(cuda_root, context, repository, runner)

    @classmethod
    def discover_cuda_at(
        cls,
        cuda_root: Path,
        context: ContextLock,
        repository: Path,
        runner: Runner,
    ) -> tuple[Path, Path, str] | None:
        cuda_root = cuda_root.expanduser().resolve()
        if not cls._complete_cuda(cuda_root, context.architecture):
            return None
        nvcc = cuda_root / "bin" / "nvcc"
        try:
            nvcc_output = command_output(
                runner,
                [nvcc, "--version"],
                cwd=repository,
                timeout=30,
            )
        except (DevToolkitError, OSError, subprocess.TimeoutExpired):
            return None
        match = CUDA_RELEASE.search(nvcc_output)
        return (cuda_root, nvcc, match.group(1)) if match is not None else None


class PrefixToolchainSource(SystemToolchainSource):
    """Adopt a complete CUDA/TensorRT toolchain rooted at a user-owned prefix."""

    descriptor = ProviderDescriptor("prefix", "trtmc-devtoolkit-prefix==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        configured = request.target.options.get("prefix")
        if context.provider.name != "local" or not isinstance(configured, str):
            return ()
        discovered_cuda = self.discover_cuda_at(
            Path(configured), context, repository, runner
        )
        if discovered_cuda is None:
            return ()
        root, nvcc, cuda_version = discovered_cuda
        include_dir = root / "include"
        header = include_dir / "NvInferVersion.h"
        library = _first_library(root, "libnvinfer.so", context.architecture)
        if not header.is_file() or library is None:
            return ()
        python = str(context.locator.get("python", "python3"))
        cudart = _first_library(root, "libcudart.so", context.architecture)
        assert cudart is not None
        probe_environment = {
            "PATH": f"{root / 'bin'}:{os.environ.get('PATH', '')}",
            "CUDA_HOME": str(root),
            "CUDA_PATH": str(root),
            "LD_LIBRARY_PATH": (
                f"{library.parent}:{cudart.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            ),
        }
        try:
            python_version = command_output(
                runner,
                [
                    python,
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                cwd=repository,
                env=probe_environment,
                timeout=30,
            )
            tensorrt_python = command_output(
                runner,
                [python, "-c", "import tensorrt; print(tensorrt.__version__)"],
                cwd=repository,
                env=probe_environment,
                timeout=30,
            )
            tensorrt_native = command_output(
                runner,
                [python, "-c", _native_version_script(library)],
                cwd=repository,
                env=probe_environment,
                timeout=30,
            )
            tensorrt_header = tensorrt_header_version(header)
        except (DevToolkitError, OSError, subprocess.TimeoutExpired):
            return ()
        if len({tensorrt_python, tensorrt_native, tensorrt_header}) != 1:
            return ()
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="prefix",
                cuda_source="prefix",
                tensorrt=tensorrt_native,
                cuda=cuda_version,
                python=python_version,
                identity={
                    "cuda_root": str(root),
                    "nvcc": str(nvcc),
                    "tensorrt_include_dir": str(include_dir),
                    "tensorrt_library_dir": str(library.parent),
                    "tensorrt_library": str(library),
                },
            ),
        )


class ManagedArtifactToolchainSource:
    """Resolve caller-supplied, digest-pinned managed toolchain artifacts."""

    descriptor = ProviderDescriptor("managed-artifacts", "trtmc-devtoolkit-managed-artifacts==2", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if context.provider.name != "local" or not request.artifacts:
            return ()
        headers = next(
            (artifact for artifact in request.artifacts if artifact.name == "tensorrt-headers"),
            None,
        )
        if (
            headers is None
            or not urllib.parse.urlparse(headers.uri).path.endswith(".deb")
            or not any(
                urllib.parse.urlparse(artifact.uri).path.endswith(".whl")
                for artifact in request.artifacts
            )
        ):
            return ()
        policy = request.cuda
        configured_prefix = request.target.options.get("prefix")
        if isinstance(configured_prefix, str):
            system_cuda = SystemToolchainSource.discover_cuda_at(
                Path(configured_prefix), context, repository, runner
            )
            existing_cuda_source: CudaSource = "prefix"
        else:
            system_cuda = SystemToolchainSource.discover_cuda(context, repository, runner)
            existing_cuda_source = "system"
        cuda_source = "managed"
        system_cuda_root: str | None = None
        system_nvcc: str | None = None
        if policy.kind == "system-first" and system_cuda is not None:
            cuda_root, nvcc, cuda = system_cuda
            cuda_source = existing_cuda_source
            system_cuda_root = str(cuda_root)
            system_nvcc = str(nvcc)
        elif policy.kind == "system-only":
            if system_cuda is None or (
                policy.version is not None and system_cuda[2] != policy.version
            ):
                return ()
            cuda_root, nvcc, cuda = system_cuda
            cuda_source = existing_cuda_source
            system_cuda_root = str(cuda_root)
            system_nvcc = str(nvcc)
        elif (
            policy.kind == "exact" and system_cuda is not None and system_cuda[2] == policy.version
        ):
            cuda_root, nvcc, cuda = system_cuda
            cuda_source = existing_cuda_source
            system_cuda_root = str(cuda_root)
            system_nvcc = str(nvcc)
        else:
            cuda = policy.fallback if policy.kind == "system-first" else policy.version
        if cuda is None:
            return ()
        cuda_major = cuda.split(".", 1)[0]
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="managed",
                cuda_source=cuda_source,
                tensorrt=request.tensorrt,
                cuda=cuda,
                python=request.python,
                identity={
                    "layout_schema": 1,
                    "cuda_module": f"nvidia.cu{cuda_major}",
                    "tensorrt_lib_distribution": f"tensorrt_cu{cuda_major}_libs",
                    "system_cuda_root": system_cuda_root,
                    "system_nvcc": system_nvcc,
                },
                artifacts=request.artifacts,
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: LocalExecutionContext,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ContextHandle:
        del execution
        python = Path(str(context.locator["python"]))
        downloads = state_dir / "managed-artifacts"
        paths = {
            artifact.name: self._download(artifact, downloads)
            for artifact in lock.toolchain.artifacts
        }
        wheels = [
            paths[artifact.name]
            for artifact in lock.toolchain.artifacts
            if urllib.parse.urlparse(artifact.uri).path.endswith(".whl")
        ]
        if not wheels:
            raise DevToolkitError("Managed toolchain lock contains no wheel artifacts")
        runner.run(
            [python, "-m", "pip", "install", "--no-index", "--no-deps", *wheels],
            cwd=repository,
        )
        runner.run([python, "-m", "pip", "check"], cwd=repository)
        headers_archive = paths["tensorrt-headers"]
        headers_root = state_dir / "managed-toolchain" / "headers" / lock.lock_id
        marker = headers_root / ".complete"
        if not marker.is_file():
            headers_root.mkdir(parents=True, exist_ok=True)
            runner.run(
                ["dpkg-deb", "--extract", headers_archive, headers_root],
                cwd=repository,
            )
            marker.write_text("ready\n", encoding="utf-8")
        matching_headers = [
            path
            for path in headers_root.rglob("NvInferVersion.h")
            if tensorrt_header_version(path) == lock.toolchain.tensorrt
        ]
        if len(matching_headers) != 1:
            raise DevToolkitError(
                "Managed artifacts must provide exactly one matching NvInferVersion.h"
            )
        if lock.toolchain.cuda_source in {"system", "prefix"}:
            cuda_expression = (
                f"Path({str(lock.toolchain.identity['system_cuda_root'])!r}).resolve()"
            )
        else:
            cuda_expression = (
                "Path(next(iter(importlib.import_module("
                f"{str(lock.toolchain.identity['cuda_module'])!r}).__path__))).resolve()"
            )
        location_script = (
            "import importlib, importlib.metadata as m, json; from pathlib import Path; "
            f"cuda={cuda_expression}; "
            f"dist=m.distribution({str(lock.toolchain.identity['tensorrt_lib_distribution'])!r}); "
            "trt=Path(dist.locate_file('tensorrt_libs')).resolve(); "
            "libs=sorted(trt.glob('libnvinfer.so*'), key=lambda p: (p.name != 'libnvinfer.so', p.name)); "
            "print(json.dumps({'cuda_root':str(cuda),'trt_library':str(libs[0].resolve())}))"
        )
        try:
            locations = json.loads(
                command_output(
                    runner,
                    [python, "-c", location_script],
                    cwd=repository,
                    timeout=30,
                )
            )
            cuda_root = Path(locations["cuda_root"])
            tensorrt_library = Path(locations["trt_library"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DevToolkitError(f"Could not locate managed toolchain paths: {error}") from error
        nvcc = cuda_root / "bin" / "nvcc"
        if not SystemToolchainSource._complete_cuda(
            cuda_root, lock.context.architecture
        ):
            raise DevToolkitError("Managed artifacts produced an incomplete CUDA toolkit")
        if not tensorrt_library.is_file():
            raise DevToolkitError("Managed artifacts produced an incomplete TensorRT toolchain")
        cudart = _first_library(cuda_root, "libcudart.so", lock.context.architecture)
        cublas = _first_library(cuda_root, "libcublas.so", lock.context.architecture)
        assert cudart is not None and cublas is not None
        environment = {
            **dict(context.environment),
            "PATH": f"{python.parent}:{cuda_root / 'bin'}:{os.environ.get('PATH', '')}",
            "CUDA_HOME": str(cuda_root),
            "CUDA_PATH": str(cuda_root),
            "CUDAToolkit_ROOT": str(cuda_root),
            "TRTMC_CUDA_INCLUDE_DIR": str(cuda_root / "include"),
            "TRTMC_CUDART_LIBRARY": str(cudart),
            "TRTMC_CUBLAS_LIBRARY": str(cublas),
            "TRTMC_TRT_INCLUDE_DIR": str(matching_headers[0].parent),
            "TRTMC_TRT_LIBRARY": str(tensorrt_library),
            "TRTMC_TRT_LIBRARY_DIR": str(tensorrt_library.parent),
            "LD_LIBRARY_PATH": f"{tensorrt_library.parent}:{cudart.parent}",
        }
        return ContextHandle(
            provider=context.provider,
            identity=context.identity,
            locator={
                **dict(context.locator),
                "cuda_root": str(cuda_root),
                "nvcc": str(nvcc),
                "tensorrt_include_dir": str(matching_headers[0].parent),
                "tensorrt_library": str(tensorrt_library),
            },
            environment=environment,
        )

    def observe(
        self,
        lock,
        context: ContextHandle,
        *,
        execution: LocalExecutionContext,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del execution
        return observe_local_toolchain(
            runner,
            repository=repository,
            python=str(context.locator["python"]),
            nvcc=str(context.locator["nvcc"]),
            tensorrt_include_dir=Path(str(context.locator["tensorrt_include_dir"])),
            tensorrt_library=Path(str(context.locator["tensorrt_library"])),
            environment=dict(context.environment),
            cuda_root=Path(str(context.locator["cuda_root"])),
        )

    @staticmethod
    def _download(artifact, root: Path) -> Path:
        filename = Path(urllib.parse.urlparse(artifact.uri).path).name
        if not filename:
            raise DevToolkitError(f"Artifact URI has no filename: {artifact.uri}")
        destination = root / f"{artifact.sha256}-{filename}"
        if (
            destination.is_file()
            and ManagedArtifactToolchainSource._sha256(destination) == artifact.sha256
        ):
            return destination
        root.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + f".partial-{os.getpid()}")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(artifact.uri, timeout=120) as response:
                with partial.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        stream.write(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise DevToolkitError(
                    f"Artifact checksum mismatch for {artifact.name}: {digest.hexdigest()}"
                )
            partial.replace(destination)
        except (OSError, urllib.error.URLError) as error:
            raise DevToolkitError(f"Could not download {artifact.name}: {error}") from error
        finally:
            partial.unlink(missing_ok=True)
        return destination

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
