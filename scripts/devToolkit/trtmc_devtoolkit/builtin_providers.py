# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in read-only providers for local system environment discovery."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from .commands import CommandSpec, EnvironmentPath, PathScope, state_path
from .models import DevToolkitError, ToolchainObservation, ToolchainRuntime
from .platforms import normalize_architecture
from .provisioning import ContextHandle, ProvisionPolicy, ToolchainHandle
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


def _parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    return _parse_os_release(path.read_text(encoding="utf-8"))


class LocalExecutionContext:
    descriptor = ProviderDescriptor("local", "trtmc-devtoolkit-local==3", 1)

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
            execution={
                "python": request.target.options.get("python", "python3"),
                "gpu": request.target.options.get("gpu", "0"),
            },
            locator={
                "python": request.target.options.get("python", "python3"),
                "gpu": request.target.options.get("gpu", "0"),
            },
            capabilities=frozenset({"host-filesystem", "posix-process"}),
            qualification={"execution": "local"},
        )

    def provision(
        self,
        context: ContextLock,
        *,
        inherit_system_packages: bool,
        repository: Path,
        state_dir: Path,
        policy: ProvisionPolicy,
        runner: Runner,
    ) -> ContextHandle:
        base_python = str(context.locator.get("python", "python3"))
        python = base_python
        if policy is not ProvisionPolicy.ADOPT_ONLY and inherit_system_packages:
            venv = state_dir / "venv"
            python_path = venv / "bin" / "python"
            if not python_path.is_file():
                command: list[str | Path] = [base_python, "-m", "venv"]
                if inherit_system_packages:
                    command.append("--system-site-packages")
                command.append(venv)
                runner.run(command, cwd=repository)
            python = str(python_path)
        environment = {
            "PATH": f"{Path(python).parent}:{os.environ.get('PATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(context.locator.get("gpu", "0")),
        }
        handle = ContextHandle(
            provider=self.descriptor,
            identity=context.identity,
            execution_identity={
                "python": python,
                "gpu": context.locator.get("gpu", "0"),
            },
            locator={"python": python, "gpu": context.locator.get("gpu", "0")},
            environment=environment,
            capabilities=context.capabilities,
        )
        return replace(
            handle,
            _executor=lambda command, check, capture_output: self.execute(
                handle,
                command,
                repository=repository,
                state_dir=state_dir,
                runner=runner,
                check=check,
                capture_output=capture_output,
            ),
            _path_mapper=lambda value: (
                str(repository / Path(value.path))
                if isinstance(value, EnvironmentPath) and value.scope is PathScope.REPOSITORY
                else str(state_dir / Path(value.path))
                if isinstance(value, EnvironmentPath) and value.scope is PathScope.STATE
                else str(value.path)
                if isinstance(value, EnvironmentPath)
                else str(value)
            ),
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


def _toolchain_environment(
    runtime: ToolchainRuntime,
    architecture: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    base = os.environ if base_environment is None else base_environment
    cuda_root = Path(runtime.cuda_root)
    tensorrt_library = Path(runtime.tensorrt_library)
    library_paths = [str(tensorrt_library.parent)]
    for directory in ("lib64", "lib"):
        candidate = cuda_root / directory
        if candidate.is_dir() and str(candidate) not in library_paths:
            library_paths.append(str(candidate))
    cudart = _first_library(cuda_root, "libcudart.so", architecture)
    if cudart is not None and str(cudart.parent) not in library_paths:
        library_paths.append(str(cudart.parent))
    inherited_libraries = base.get("LD_LIBRARY_PATH", "")
    if inherited_libraries:
        library_paths.append(inherited_libraries)
    return {
        "PATH": (
            f"{Path(runtime.python_executable).parent}:{cuda_root / 'bin'}:{base.get('PATH', '')}"
        ),
        "CUDA_HOME": runtime.cuda_root,
        "CUDA_PATH": runtime.cuda_root,
        "CUDAToolkit_ROOT": runtime.cuda_root,
        "TRTMC_TRT_INCLUDE_DIR": runtime.tensorrt_include_dir,
        "TRTMC_TRT_LIBRARY": runtime.tensorrt_library,
        "TRTMC_TRT_LIBRARY_DIR": str(tensorrt_library.parent),
        "LD_LIBRARY_PATH": ":".join(library_paths),
    }


def _native_version_script(library: Path) -> str:
    return (
        "import ctypes; "
        f"lib=ctypes.CDLL({str(library)!r}); "
        "names=('Major','Minor','Patch','Build'); "
        "fs=[getattr(lib, f'getInferLib{name}Version') for name in names]; "
        "[setattr(f, 'restype', ctypes.c_int32) for f in fs]; "
        "print('.'.join(str(f()) for f in fs))"
    )


def _docker_command(docker_context: str, *arguments: str) -> list[str]:
    return ["docker", "--context", docker_context, *arguments]


def _docker_daemon_id(runner: Runner, repository: Path, docker_context: str) -> str:
    daemon_id = command_output(
        runner,
        _docker_command(docker_context, "info", "--format", "{{.ID}}"),
        cwd=repository,
        timeout=30,
    )
    if not daemon_id:
        raise DevToolkitError(f"Docker context {docker_context!r} has no daemon identity")
    return daemon_id


def _require_docker_client_version(
    runner: Runner,
    repository: Path,
    docker_context: str,
) -> None:
    output = command_output(
        runner,
        _docker_command(
            docker_context,
            "version",
            "--format",
            "{{.Client.Version}}",
        ),
        cwd=repository,
        timeout=30,
    )
    match = re.match(r"([0-9]+)\.([0-9]+)", output)
    if match is None or tuple(int(part) for part in match.groups()) < (20, 10):
        raise DevToolkitError(
            f"Docker CLI 20.10 or newer is required for private exec env files; found {output}"
        )


def _docker_inspect(
    runner: Runner,
    repository: Path,
    docker_context: str,
    container: str,
) -> dict[str, object]:
    output = command_output(
        runner,
        _docker_command(
            docker_context,
            "inspect",
            "--type",
            "container",
            "--format",
            "{{json .}}",
            container,
        ),
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


def _require_docker_binding(
    runner: Runner,
    repository: Path,
    *,
    docker_context: str,
    daemon_id: str,
    container_id: str,
    image_id: str,
) -> None:
    _require_docker_client_version(runner, repository, docker_context)
    observed_daemon = _docker_daemon_id(runner, repository, docker_context)
    if observed_daemon != daemon_id:
        raise DevToolkitError(
            f"Docker daemon changed after resolution: expected {daemon_id}, "
            f"observed {observed_daemon}"
        )
    inspected = _docker_inspect(runner, repository, docker_context, container_id)
    if inspected.get("Id") != container_id:
        raise DevToolkitError("Docker container identity changed after resolution")
    if inspected.get("Image") != image_id:
        raise DevToolkitError("Docker container image changed after resolution")


_DOCKER_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@contextmanager
def _docker_environment_file(
    state_dir: Path,
    environment: Mapping[str, str],
) -> Iterator[Path | None]:
    if not environment:
        yield None
        return
    for name, value in environment.items():
        if _DOCKER_ENVIRONMENT_NAME.fullmatch(name) is None:
            raise DevToolkitError(f"Invalid Docker environment name: {name!r}")
        if not isinstance(value, str) or any(character in value for character in "\r\n\0"):
            raise DevToolkitError(
                f"Docker environment value for {name!r} must be a single text line"
            )
    secret_dir = state_dir / ".secrets"
    secret_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(secret_dir, 0o700)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=secret_dir,
            prefix="docker-environment-",
            suffix=".list",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            for name, value in sorted(environment.items()):
                stream.write(f"{name}={value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_CONTAINER_PROBE_SCRIPT = r"""
import ctypes
import ctypes.util
import hashlib
import itertools
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
    cuda_libraries.append(any(path.is_file() for path in cuda_root.rglob(name)))
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

configured_include = os.environ.get("TRTMC_TRT_INCLUDE_DIR") or os.environ.get("TRT_INC_DIR")
configured_headers = (
    (Path(configured_include) / "NvInferVersion.h",) if configured_include else ()
)
header_candidates = itertools.chain(
    configured_headers,
    Path("/usr/include").rglob("NvInferVersion.h"),
    Path("/usr/local").rglob("NvInferVersion.h"),
)
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
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
evidence = {
    "nvcc": sha256(nvcc),
    "tensorrt-header": sha256(header),
}
if library_path:
    evidence["tensorrt-library"] = sha256(library_path)
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
    "evidence": evidence,
}))
""".strip()


def _container_observation(
    runner: Runner,
    repository: Path,
    docker_context: str,
    daemon_id: str,
    container_id: str,
    image_id: str,
    python: str,
) -> tuple[ToolchainObservation, str, bool]:
    _require_docker_binding(
        runner,
        repository,
        docker_context=docker_context,
        daemon_id=daemon_id,
        container_id=container_id,
        image_id=image_id,
    )
    output = command_output(
        runner,
        _docker_command(
            docker_context,
            "exec",
            container_id,
            python,
            "-c",
            _CONTAINER_PROBE_SCRIPT,
        ),
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
        "evidence",
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
        evidence={str(name): str(digest) for name, digest in dict(payload["evidence"]).items()},
    )
    return observation, str(payload["python_executable"]), payload.get("cuda_complete") is True


class DockerExecutionContext:
    """Adopt a running user container without imposing a Dockerfile or image version."""

    descriptor = ProviderDescriptor("docker", "trtmc-devtoolkit-docker-adoption==6", 1)

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
        docker_context = request.target.options.get("docker_context")
        if docker_context is None:
            docker_context = command_output(
                runner,
                ["docker", "context", "show"],
                cwd=repository,
                timeout=30,
            )
        if not isinstance(docker_context, str) or not docker_context:
            raise DevToolkitError("Docker resolution requires a non-empty context name")
        _require_docker_client_version(runner, repository, docker_context)
        daemon_id = _docker_daemon_id(runner, repository, docker_context)
        inspected = _docker_inspect(runner, repository, docker_context, container)
        container_id = inspected.get("Id")
        if not isinstance(container_id, str) or not container_id:
            raise DevToolkitError(f"Docker container {container} has no container identity")
        image_id = inspected.get("Image")
        if not isinstance(image_id, str) or not image_id:
            raise DevToolkitError(f"Docker container {container} has no image identity")
        architecture = (
            normalize_architecture(request.architecture)
            if request.architecture is not None
            else normalize_architecture(
                command_output(
                    runner,
                    _docker_command(docker_context, "exec", container_id, "uname", "-m"),
                    cwd=repository,
                    timeout=30,
                )
            )
        )
        release = _parse_os_release(
            command_output(
                runner,
                _docker_command(
                    docker_context,
                    "exec",
                    container_id,
                    "cat",
                    "/etc/os-release",
                ),
                cwd=repository,
                timeout=30,
            )
        )
        _require_docker_binding(
            runner,
            repository,
            docker_context=docker_context,
            daemon_id=daemon_id,
            container_id=container_id,
            image_id=image_id,
        )
        return ContextLock(
            provider=self.descriptor,
            operating_system="linux",
            architecture=architecture,
            identity={
                "container_id": container_id,
                "daemon_id": daemon_id,
                "image_id": image_id,
                "os_id": release.get("ID", "unknown"),
                "os_version": release.get("VERSION_ID", "unknown"),
                "runtime": "docker",
            },
            execution={
                "workspace": request.target.options.get(
                    "workspace", "/workspace/tensorrt-model-connect"
                ),
                "target_state": request.target.options.get("state", "/tmp/trtmc-devtoolkit"),
                "python": request.target.options.get("python", "python3"),
            },
            locator={
                "container": container,
                "docker_context": docker_context,
                "workspace": request.target.options.get(
                    "workspace", "/workspace/tensorrt-model-connect"
                ),
                "target_state": request.target.options.get("state", "/tmp/trtmc-devtoolkit"),
                "python": request.target.options.get("python", "python3"),
            },
            capabilities=frozenset({"container-process", "posix-process"}),
            qualification={"execution": "container"},
        )

    def provision(
        self,
        context: ContextLock,
        *,
        inherit_system_packages: bool,
        repository: Path,
        state_dir: Path,
        policy: ProvisionPolicy,
        runner: Runner,
    ) -> ContextHandle:
        del inherit_system_packages
        if policy is ProvisionPolicy.CREATE:
            raise DevToolkitError("The built-in Docker provider is adoption-only")
        _require_docker_binding(
            runner,
            repository,
            docker_context=str(context.locator["docker_context"]),
            daemon_id=str(context.identity["daemon_id"]),
            container_id=str(context.identity["container_id"]),
            image_id=str(context.identity["image_id"]),
        )
        handle = ContextHandle(
            provider=self.descriptor,
            identity=context.identity,
            execution_identity={
                "workspace": context.locator["workspace"],
                "target_state": context.locator["target_state"],
                "python": context.locator["python"],
            },
            locator={**dict(context.locator), **dict(context.execution)},
            capabilities=context.capabilities,
        )
        return replace(
            handle,
            _executor=lambda command, check, capture_output: self.execute(
                handle,
                command,
                repository=repository,
                state_dir=state_dir,
                runner=runner,
                check=check,
                capture_output=capture_output,
            ),
            _path_mapper=lambda value: (
                str(PurePosixPath(str(handle.locator["workspace"])) / value.path)
                if isinstance(value, EnvironmentPath) and value.scope is PathScope.REPOSITORY
                else str(PurePosixPath(str(handle.locator["target_state"])) / value.path)
                if isinstance(value, EnvironmentPath) and value.scope is PathScope.STATE
                else str(value.path)
                if isinstance(value, EnvironmentPath)
                else str(value)
            ),
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
                return str(PurePosixPath(str(context.locator["workspace"])) / value.path)
            if value.scope is PathScope.STATE:
                return str(PurePosixPath(str(context.locator["target_state"])) / value.path)
            return str(value.path)

        docker_context = str(context.locator["docker_context"])
        daemon_id = str(context.identity["daemon_id"])
        container_id = str(context.identity["container_id"])
        image_id = str(context.identity["image_id"])
        _require_docker_binding(
            runner,
            repository,
            docker_context=docker_context,
            daemon_id=daemon_id,
            container_id=container_id,
            image_id=image_id,
        )
        environment = {**dict(context.environment), **dict(command.environment)}
        with _docker_environment_file(state_dir, environment) as environment_file:
            arguments = _docker_command(
                docker_context,
                "exec",
                "--workdir",
                render(command.cwd),
            )
            if environment_file is not None:
                arguments.extend(["--env-file", str(environment_file)])
            arguments.append(container_id)
            arguments.extend(render(argument) for argument in command.arguments)
            return runner.run(
                arguments,
                cwd=repository,
                check=check,
                capture_output=capture_output,
            )


class ContainerImageToolchainSource:
    descriptor = ProviderDescriptor("container-image", "trtmc-devtoolkit-container-image==5", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if "container-process" not in context.capabilities:
            return ()
        try:
            observed, python_executable, complete_cuda = _container_observation(
                runner,
                repository,
                str(context.locator["docker_context"]),
                str(context.identity["daemon_id"]),
                str(context.identity["container_id"]),
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
                },
                runtime=ToolchainRuntime(
                    python_executable=python_executable,
                    cuda_root=observed.cuda_root or "",
                    nvcc=str(PurePosixPath(observed.cuda_root or "") / "bin" / "nvcc"),
                    tensorrt_include_dir=observed.tensorrt_include_dir,
                    tensorrt_library=observed.tensorrt_library,
                ),
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ToolchainHandle:
        del context, repository, state_dir, runner
        if lock.toolchain.runtime is None:
            raise DevToolkitError("Container toolchain lock has no runtime")
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=lock.toolchain.runtime,
        )

    def observe(
        self,
        lock,
        context: ContextHandle,
        toolchain: ToolchainHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del lock
        observed, _, complete_cuda = _container_observation(
            runner,
            repository,
            str(context.locator["docker_context"]),
            str(context.identity["daemon_id"]),
            str(context.identity["container_id"]),
            str(context.identity["image_id"]),
            toolchain.runtime.python_executable,
        )
        if not complete_cuda:
            raise DevToolkitError("Container CUDA toolkit became incomplete after resolution")
        return observed


class SystemToolchainSource:
    """Discover one complete, already-installed local CUDA/TensorRT toolchain."""

    descriptor = ProviderDescriptor("system", "trtmc-devtoolkit-system==3", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if "host-filesystem" not in context.capabilities or request.toolchain_options.get("prefix"):
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
                runtime=ToolchainRuntime(
                    python_executable=python,
                    cuda_root=str(cuda_root),
                    nvcc=str(nvcc),
                    tensorrt_include_dir=str(include_dir),
                    tensorrt_library=str(library),
                ),
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ToolchainHandle:
        del repository, state_dir, runner
        if lock.toolchain.runtime is None:
            raise DevToolkitError("Adopted system toolchain lock has no runtime")
        runtime = replace(
            lock.toolchain.runtime,
            python_executable=str(context.locator["python"]),
        )
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=runtime,
            environment=_toolchain_environment(runtime, lock.context.architecture),
        )

    def observe(
        self,
        lock,
        context: ContextHandle,
        toolchain: ToolchainHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del lock
        runtime = toolchain.runtime
        return observe_local_toolchain(
            runner,
            repository=repository,
            python=runtime.python_executable,
            nvcc=runtime.nvcc,
            tensorrt_include_dir=Path(runtime.tensorrt_include_dir),
            tensorrt_library=Path(runtime.tensorrt_library),
            environment=dict(context.environment),
            cuda_root=Path(runtime.cuda_root),
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


@dataclass(frozen=True)
class TargetRuntimeBaseline:
    python: str
    python_executable: str
    cuda: str
    cuda_root: str
    nvcc: str
    cuda_source: CudaSource


@dataclass(frozen=True)
class TargetPythonBaseline:
    python: str
    python_executable: str


_TARGET_PYTHON_SCRIPT = (
    "import json, sys; "
    "print(json.dumps({'python': "
    "f'{sys.version_info.major}.{sys.version_info.minor}', "
    "'python_executable': sys.executable}))"
)


_CONTAINER_BASELINE_SCRIPT = r"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

nvcc = shutil.which("nvcc")
if not nvcc:
    raise RuntimeError("nvcc is not available")
text = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release\s+([0-9]+\.[0-9]+)", text, re.I)
if match is None:
    raise RuntimeError("nvcc did not report a CUDA release")
root = Path(nvcc).resolve().parent.parent
required = (
    root / "include" / "cuda.h",
    next(root.rglob("libcudart.so"), None),
    next(root.rglob("libcublas.so"), None),
    next(root.rglob("libcurand.so"), None),
)
if not all(path is not None and path.is_file() for path in required):
    raise RuntimeError("CUDA toolkit is incomplete")
print(json.dumps({
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "python_executable": sys.executable,
    "cuda": match.group(1),
    "cuda_root": str(root),
    "nvcc": str(Path(nvcc).resolve()),
}))
""".strip()


def target_python_baseline(
    context: ContextLock,
    repository: Path,
    runner: Runner,
) -> TargetPythonBaseline | None:
    """Observe target Python without requiring CUDA or TensorRT."""

    python = str(context.locator.get("python", "python3"))
    try:
        if "host-filesystem" in context.capabilities:
            output = command_output(
                runner,
                [python, "-c", _TARGET_PYTHON_SCRIPT],
                cwd=repository,
                timeout=30,
            )
        elif "container-process" in context.capabilities:
            _require_docker_binding(
                runner,
                repository,
                docker_context=str(context.locator["docker_context"]),
                daemon_id=str(context.identity["daemon_id"]),
                container_id=str(context.identity["container_id"]),
                image_id=str(context.identity["image_id"]),
            )
            output = command_output(
                runner,
                _docker_command(
                    str(context.locator["docker_context"]),
                    "exec",
                    str(context.identity["container_id"]),
                    python,
                    "-c",
                    _TARGET_PYTHON_SCRIPT,
                ),
                cwd=repository,
                timeout=30,
            )
        else:
            return None
        payload = json.loads(output)
        return TargetPythonBaseline(
            python=str(payload["python"]),
            python_executable=str(payload["python_executable"]),
        )
    except (
        DevToolkitError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return None


def target_runtime_baseline(
    context: ContextLock,
    repository: Path,
    runner: Runner,
) -> TargetRuntimeBaseline | None:
    """Observe Python and a complete CUDA toolkit without requiring TensorRT."""

    if "host-filesystem" in context.capabilities:
        discovered = SystemToolchainSource.discover_cuda(context, repository, runner)
        if discovered is None:
            return None
        cuda_root, nvcc, cuda = discovered
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
        except (DevToolkitError, OSError, subprocess.TimeoutExpired):
            return None
        return TargetRuntimeBaseline(
            python_version,
            python,
            cuda,
            str(cuda_root),
            str(nvcc),
            "system",
        )
    if "container-process" not in context.capabilities:
        return None
    try:
        _require_docker_binding(
            runner,
            repository,
            docker_context=str(context.locator["docker_context"]),
            daemon_id=str(context.identity["daemon_id"]),
            container_id=str(context.identity["container_id"]),
            image_id=str(context.identity["image_id"]),
        )
        output = command_output(
            runner,
            _docker_command(
                str(context.locator["docker_context"]),
                "exec",
                str(context.identity["container_id"]),
                str(context.locator.get("python", "python3")),
                "-c",
                _CONTAINER_BASELINE_SCRIPT,
            ),
            cwd=repository,
            timeout=30,
        )
        payload = json.loads(output)
        return TargetRuntimeBaseline(
            str(payload["python"]),
            str(payload["python_executable"]),
            str(payload["cuda"]),
            str(payload["cuda_root"]),
            str(payload["nvcc"]),
            "image",
        )
    except (
        DevToolkitError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return None


class PrefixToolchainSource(SystemToolchainSource):
    """Adopt a complete CUDA/TensorRT toolchain rooted at a user-owned prefix."""

    descriptor = ProviderDescriptor("prefix", "trtmc-devtoolkit-prefix==3", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        configured = request.toolchain_options.get("prefix")
        if "host-filesystem" not in context.capabilities or not isinstance(configured, str):
            return ()
        discovered_cuda = self.discover_cuda_at(Path(configured), context, repository, runner)
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
                runtime=ToolchainRuntime(
                    python_executable=python,
                    cuda_root=str(root),
                    nvcc=str(nvcc),
                    tensorrt_include_dir=str(include_dir),
                    tensorrt_library=str(library),
                ),
            ),
        )


class ManagedArtifactToolchainSource:
    """Resolve caller-supplied, digest-pinned managed toolchain artifacts."""

    descriptor = ProviderDescriptor("managed-artifacts", "trtmc-devtoolkit-managed-artifacts==6", 1)

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        if not ({"host-filesystem", "container-process"} & context.capabilities):
            return ()
        if not request.artifacts:
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
        configured_prefix = request.toolchain_options.get("cuda_prefix")
        if isinstance(configured_prefix, str) and "host-filesystem" in context.capabilities:
            system_cuda = SystemToolchainSource.discover_cuda_at(
                Path(configured_prefix), context, repository, runner
            )
            existing_cuda_source: CudaSource = "prefix"
        else:
            baseline = target_runtime_baseline(context, repository, runner)
            system_cuda = (
                (Path(baseline.cuda_root), Path(baseline.nvcc), baseline.cuda)
                if baseline is not None
                else None
            )
            existing_cuda_source = baseline.cuda_source if baseline is not None else "system"
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
        cuda_artifacts: tuple[str, ...] = ()
        if cuda_source == "managed":
            configured_cuda_artifacts = request.toolchain_options.get("cuda_artifacts")
            artifact_names = {artifact.name for artifact in request.artifacts}
            if (
                not isinstance(configured_cuda_artifacts, tuple)
                or not configured_cuda_artifacts
                or any(
                    not isinstance(name, str) or name not in artifact_names
                    for name in configured_cuda_artifacts
                )
            ):
                return ()
            cuda_artifacts = configured_cuda_artifacts
        configured_python_bootstrap = request.toolchain_options.get(
            "python_bootstrap_artifacts", ()
        )
        artifact_names = {artifact.name for artifact in request.artifacts}
        if (
            not isinstance(configured_python_bootstrap, tuple)
            or any(
                not isinstance(name, str) or not name or name not in artifact_names
                for name in configured_python_bootstrap
            )
        ):
            return ()
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="managed",
                cuda_source=cuda_source,
                tensorrt=request.tensorrt,
                cuda=cuda,
                python=request.python,
                identity={
                    "layout_schema": 4,
                    "cuda_module": f"nvidia.cu{cuda_major}",
                    "tensorrt_lib_distribution": f"tensorrt_cu{cuda_major}_libs",
                    "system_cuda_root": system_cuda_root,
                    "system_nvcc": system_nvcc,
                    "cuda_artifacts": cuda_artifacts,
                    "python_bootstrap_artifacts": configured_python_bootstrap,
                    "cuda_release": request.toolchain_options.get("cuda_release"),
                },
                artifacts=request.artifacts,
            ),
        )

    def provision(
        self,
        lock,
        context: ContextHandle,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ToolchainHandle:
        if context.supports_target_operations:
            return self._provision_target(lock, context, repository=repository, runner=runner)
        python = Path(str(context.locator["python"]))
        downloads = state_dir / "managed-artifacts"
        paths = {
            artifact.name: self._download(artifact, downloads)
            for artifact in lock.toolchain.artifacts
        }
        packages = [
            paths[artifact.name]
            for artifact in lock.toolchain.artifacts
            if urllib.parse.urlparse(artifact.uri).path.endswith((".whl", ".tar.gz"))
        ]
        if not packages:
            raise DevToolkitError("Managed toolchain lock contains no Python packages")
        runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                *packages,
            ],
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
        if lock.toolchain.cuda_source in {"system", "prefix", "image"}:
            cuda_expression = (
                f"Path({str(lock.toolchain.identity['system_cuda_root'])!r}).resolve()"
            )
        else:
            cuda_root = self._materialize_local_cuda(lock, paths, state_dir, runner, repository)
            cuda_expression = f"Path({str(cuda_root)!r}).resolve()"
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
        if not SystemToolchainSource._complete_cuda(cuda_root, lock.context.architecture):
            raise DevToolkitError("Managed artifacts produced an incomplete CUDA toolkit")
        if not tensorrt_library.is_file():
            raise DevToolkitError("Managed artifacts produced an incomplete TensorRT toolchain")
        cudart = _first_library(cuda_root, "libcudart.so", lock.context.architecture)
        cublas = _first_library(cuda_root, "libcublas.so", lock.context.architecture)
        assert cudart is not None and cublas is not None
        runtime = ToolchainRuntime(
            python_executable=str(python),
            cuda_root=str(cuda_root),
            nvcc=str(nvcc),
            tensorrt_include_dir=str(matching_headers[0].parent),
            tensorrt_library=str(tensorrt_library),
        )
        environment = {
            **_toolchain_environment(runtime, lock.context.architecture),
            "TRTMC_CUDA_INCLUDE_DIR": str(cuda_root / "include"),
            "TRTMC_CUDART_LIBRARY": str(cudart),
            "TRTMC_CUBLAS_LIBRARY": str(cublas),
        }
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=runtime,
            environment=environment,
        )

    def _provision_target(
        self,
        lock,
        context: ContextHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> ToolchainHandle:
        del repository, runner
        root_path = state_path(f"managed-toolchain/{lock.lock_id}")
        root = context.map_path(root_path)
        base_python = str(context.execution_identity["python"])
        venv = str(PurePosixPath(root) / "venv")
        venv_python = str(PurePosixPath(venv) / "bin" / "python")
        ready_marker = str(PurePosixPath(root) / ".complete")
        headers_root = str(PurePosixPath(root) / "headers")
        cuda_root = (
            str(PurePosixPath(root) / "cuda")
            if lock.toolchain.cuda_source == "managed"
            else None
        )
        context.execute(CommandSpec(("mkdir", "-p", root)))
        complete = context.execute(
            CommandSpec(("test", "-f", ready_marker)),
            check=False,
            capture_output=True,
        )
        newly_materialized = complete.returncode != 0
        if newly_materialized:
            downloads = context.map_path(state_path("artifact-cache"))
            context.execute(CommandSpec(("mkdir", "-p", downloads)))
            paths: dict[str, str] = {}
            for artifact in lock.toolchain.artifacts:
                filename = Path(urllib.parse.urlsplit(artifact.uri).path).name
                if not filename:
                    raise DevToolkitError(f"Artifact URI has no filename for {artifact.name}")
                destination = str(PurePosixPath(downloads) / artifact.sha256 / filename)
                context.execute(
                    CommandSpec(
                        (base_python, "-c", _TARGET_DOWNLOAD_SCRIPT, destination),
                        environment={
                            "TRTMC_ARTIFACT_URI": artifact.uri,
                            "TRTMC_ARTIFACT_SHA256": artifact.sha256,
                        },
                    )
                )
                paths[artifact.name] = destination

            exists = context.execute(
                CommandSpec(("test", "-x", venv_python)),
                check=False,
                capture_output=True,
            )
            if exists.returncode != 0:
                created = context.execute(
                    CommandSpec((base_python, "-m", "venv", venv)),
                    check=False,
                    capture_output=True,
                )
                if created.returncode != 0:
                    context.execute(
                        CommandSpec((base_python, "-c", _TARGET_REMOVE_PATH_SCRIPT, venv))
                    )
                    context.execute(
                        CommandSpec((base_python, "-m", "venv", "--without-pip", venv))
                    )

            bootstrap_names = self._python_bootstrap_artifact_names(lock)
            if bootstrap_names:
                bootstrap_paths = tuple(paths[name] for name in bootstrap_names)
                context.execute(
                    CommandSpec(
                        (
                            venv_python,
                            "-m",
                            "pip",
                            "install",
                            "--no-cache-dir",
                            "--no-index",
                            "--no-deps",
                            *bootstrap_paths,
                        ),
                        environment={"PYTHONPATH": ":".join(bootstrap_paths)},
                    )
                )
            else:
                needs_wheel_builder = any(
                    urllib.parse.urlsplit(artifact.uri).path.endswith(".tar.gz")
                    for artifact in lock.toolchain.artifacts
                )
                python_build_ready = context.execute(
                    CommandSpec(
                        (
                            venv_python,
                            "-c",
                            (
                                "import pip, setuptools, wheel"
                                if needs_wheel_builder
                                else "import pip"
                            ),
                        )
                    ),
                    check=False,
                    capture_output=True,
                )
                if python_build_ready.returncode != 0:
                    raise DevToolkitError(
                        "Target Python cannot build wheels; the immutable lock must include "
                        "python_bootstrap_artifacts for pip, setuptools, and wheel"
                    )

            if cuda_root is not None:
                context.execute(CommandSpec(("mkdir", "-p", cuda_root)))
                for name in self._cuda_artifact_names(lock):
                    archive = paths.get(name)
                    if archive is None:
                        raise DevToolkitError(f"Managed CUDA artifact {name!r} is missing")
                    context.execute(
                        CommandSpec(
                            (
                                "tar",
                                "--extract",
                                "--xz",
                                "--file",
                                archive,
                                "--directory",
                                cuda_root,
                                "--strip-components=1",
                                "--no-same-owner",
                            )
                        )
                    )
                context.execute(
                    CommandSpec(
                        (base_python, "-c", _TARGET_NORMALIZE_CUDA_LAYOUT_SCRIPT, cuda_root)
                    )
                )

            bootstrap_set = set(bootstrap_names)
            wheels = [
                paths[artifact.name]
                for artifact in lock.toolchain.artifacts
                if artifact.name not in bootstrap_set
                and urllib.parse.urlsplit(artifact.uri).path.endswith(".whl")
            ]
            source_packages = [
                paths[artifact.name]
                for artifact in lock.toolchain.artifacts
                if urllib.parse.urlsplit(artifact.uri).path.endswith(".tar.gz")
            ]
            if source_packages:
                built_wheels = str(PurePosixPath(root) / "built-wheels")
                context.execute(CommandSpec(("mkdir", "-p", built_wheels)))
                for source_package in source_packages:
                    context.execute(
                        CommandSpec(
                            (
                                venv_python,
                                "-m",
                                "pip",
                                "wheel",
                                "--no-cache-dir",
                                "--no-deps",
                                "--no-build-isolation",
                                "--wheel-dir",
                                built_wheels,
                                source_package,
                            )
                        )
                    )
                wheel_result = context.execute(
                    CommandSpec((venv_python, "-c", _TARGET_WHEELS_SCRIPT, built_wheels)),
                    capture_output=True,
                )
                wheels.extend(line for line in wheel_result.stdout.splitlines() if line)
            if not wheels:
                raise DevToolkitError("Managed toolchain lock contains no Python packages")
            context.execute(
                CommandSpec(
                    (
                        venv_python,
                        "-m",
                        "pip",
                        "install",
                        "--no-cache-dir",
                        "--no-index",
                        "--no-deps",
                        "--no-build-isolation",
                        *wheels,
                    )
                )
            )
            context.execute(CommandSpec((venv_python, "-m", "pip", "check")))

            context.execute(CommandSpec(("mkdir", "-p", headers_root)))
            headers_archives = [
                paths[artifact.name]
                for artifact in lock.toolchain.artifacts
                if urllib.parse.urlsplit(artifact.uri).path.endswith(".deb")
            ]
            if not headers_archives:
                raise DevToolkitError("Managed toolchain lock contains no TensorRT headers")
            for archive in headers_archives:
                context.execute(CommandSpec(("dpkg-deb", "--extract", archive, headers_root)))

        header_result = context.execute(
            CommandSpec(
                (
                    venv_python,
                    "-c",
                    _TARGET_HEADER_SCRIPT,
                    headers_root,
                    lock.toolchain.tensorrt,
                )
            ),
            capture_output=True,
        )
        include_dir = header_result.stdout.strip()
        if not include_dir:
            raise DevToolkitError("Managed artifacts produced no matching TensorRT headers")

        locations_result = context.execute(
            CommandSpec(
                (
                    venv_python,
                    "-c",
                    _TARGET_LOCATIONS_SCRIPT,
                    str(lock.toolchain.identity["tensorrt_lib_distribution"]),
                    cuda_root or str(lock.toolchain.identity["system_cuda_root"]),
                )
            ),
            capture_output=True,
        )
        try:
            locations = json.loads(locations_result.stdout)
            cuda_root = str(locations["cuda_root"])
            tensorrt_library = str(locations["tensorrt_library"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DevToolkitError(f"Could not locate managed toolchain paths: {error}") from error
        runtime = ToolchainRuntime(
            python_executable=venv_python,
            cuda_root=cuda_root,
            nvcc=str(PurePosixPath(cuda_root) / "bin" / "nvcc"),
            tensorrt_include_dir=include_dir,
            tensorrt_library=tensorrt_library,
        )
        base_environment_result = context.execute(
            CommandSpec((base_python, "-c", _TARGET_ENVIRONMENT_SCRIPT)),
            capture_output=True,
        )
        try:
            raw_environment = json.loads(base_environment_result.stdout)
            if not isinstance(raw_environment, dict) or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in raw_environment.items()
            ):
                raise TypeError
            base_environment = raw_environment
        except (json.JSONDecodeError, TypeError) as error:
            raise DevToolkitError(
                f"Could not observe target process environment: {error}"
            ) from error
        environment = _toolchain_environment(
            runtime,
            lock.context.architecture,
            base_environment=base_environment,
        )
        if newly_materialized:
            context.execute(CommandSpec(("touch", ready_marker)))
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=runtime,
            environment=environment,
        )

    @staticmethod
    def _cuda_artifact_names(lock) -> tuple[str, ...]:
        raw = lock.toolchain.identity.get("cuda_artifacts", ())
        if (
            not isinstance(raw, tuple)
            or not raw
            or any(not isinstance(name, str) or not name for name in raw)
        ):
            raise DevToolkitError("Managed CUDA requires a non-empty digest-pinned component set")
        return raw

    @staticmethod
    def _python_bootstrap_artifact_names(lock) -> tuple[str, ...]:
        raw = lock.toolchain.identity.get("python_bootstrap_artifacts", ())
        artifact_names = {artifact.name for artifact in lock.toolchain.artifacts}
        if (
            not isinstance(raw, tuple)
            or any(
                not isinstance(name, str) or not name or name not in artifact_names
                for name in raw
            )
        ):
            raise DevToolkitError("Managed Python bootstrap artifact set is invalid")
        return raw

    @classmethod
    def _materialize_local_cuda(
        cls,
        lock,
        paths: dict[str, Path],
        state_dir: Path,
        runner: Runner,
        repository: Path,
    ) -> Path:
        cuda_root = state_dir / "managed-toolchain" / lock.lock_id / "cuda"
        marker = cuda_root / ".complete"
        if not marker.is_file():
            cuda_root.mkdir(parents=True, exist_ok=True)
            for name in cls._cuda_artifact_names(lock):
                archive = paths.get(name)
                if archive is None:
                    raise DevToolkitError(f"Managed CUDA artifact {name!r} is missing")
                runner.run(
                    [
                        "tar",
                        "--extract",
                        "--xz",
                        "--file",
                        archive,
                        "--directory",
                        cuda_root,
                        "--strip-components=1",
                        "--no-same-owner",
                    ],
                    cwd=repository,
                )
            marker.touch()
        lib = cuda_root / "lib"
        lib64 = cuda_root / "lib64"
        if lib.is_dir() and not lib64.exists() and not lib64.is_symlink():
            lib64.symlink_to("lib", target_is_directory=True)
        if not all((lib64 / name).is_file() for name in ("libcudart_static.a", "libcudadevrt.a")):
            raise DevToolkitError("Managed CUDA lacks static compiler runtime libraries")
        return cuda_root

    def observe(
        self,
        lock,
        context: ContextHandle,
        toolchain: ToolchainHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation:
        del lock
        runtime = toolchain.runtime
        if "container-process" in context.capabilities:
            result = context.execute(
                CommandSpec(
                    (
                        runtime.python_executable,
                        "-c",
                        _TARGET_TOOLCHAIN_PROBE_SCRIPT,
                        runtime.nvcc,
                        runtime.tensorrt_include_dir,
                        runtime.tensorrt_library,
                        runtime.cuda_root,
                    ),
                    environment=dict(toolchain.environment),
                ),
                capture_output=True,
            )
            try:
                payload = json.loads(result.stdout)
                return ToolchainObservation(
                    python_version=str(payload["python"]),
                    cuda_version=str(payload["cuda"]),
                    tensorrt_python_version=str(payload["tensorrt_python"]),
                    tensorrt_native_version=str(payload["tensorrt_native"]),
                    tensorrt_header_version=str(payload["tensorrt_headers"]),
                    tensorrt_include_dir=str(payload["tensorrt_include_dir"]),
                    tensorrt_library=str(payload["tensorrt_library"]),
                    cuda_root=str(payload["cuda_root"]),
                    image_id=str(context.identity.get("image_id", "")) or None,
                    architecture=normalize_architecture(str(payload["architecture"])),
                    evidence={
                        str(name): str(digest) for name, digest in dict(payload["evidence"]).items()
                    },
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DevToolkitError(
                    f"Managed target toolchain probe returned invalid data: {error}"
                ) from error
        return observe_local_toolchain(
            runner,
            repository=repository,
            python=runtime.python_executable,
            nvcc=runtime.nvcc,
            tensorrt_include_dir=Path(runtime.tensorrt_include_dir),
            tensorrt_library=Path(runtime.tensorrt_library),
            environment=dict(context.environment),
            cuda_root=Path(runtime.cuda_root),
        )

    @staticmethod
    def _download(artifact, root: Path) -> Path:
        filename = Path(urllib.parse.urlparse(artifact.uri).path).name
        if not filename:
            raise DevToolkitError(f"Artifact URI has no filename: {artifact.uri}")
        destination = root / artifact.sha256 / filename
        if (
            destination.is_file()
            and ManagedArtifactToolchainSource._sha256(destination) == artifact.sha256
        ):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        partial: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{filename}.",
                suffix=".partial",
                delete=False,
            ) as stream:
                partial = Path(stream.name)
                with urllib.request.urlopen(artifact.uri, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        stream.write(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise DevToolkitError(
                    f"Artifact checksum mismatch for {artifact.name}: {digest.hexdigest()}"
                )
            os.replace(partial, destination)
        except (OSError, urllib.error.URLError) as error:
            raise DevToolkitError(f"Could not download {artifact.name}: {error}") from error
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        return destination

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


_TARGET_WHEELS_SCRIPT = r"""
import sys
from pathlib import Path

wheels = sorted(Path(sys.argv[1]).glob("*.whl"))
if not wheels:
    raise SystemExit("source package produced no wheel")
print("\n".join(str(wheel.resolve()) for wheel in wheels))
""".strip()


_TARGET_REMOVE_PATH_SCRIPT = r"""
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists() or path.is_symlink():
    shutil.rmtree(path)
""".strip()


_TARGET_ENVIRONMENT_SCRIPT = r"""
import json
import os

print(json.dumps({
    "PATH": os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}))
""".strip()


_TARGET_NORMALIZE_CUDA_LAYOUT_SCRIPT = r"""
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
library = root / "lib"
library64 = root / "lib64"
if library.is_dir() and not os.path.lexists(library64):
    library64.symlink_to("lib", target_is_directory=True)
required = (library64 / "libcudart_static.a", library64 / "libcudadevrt.a")
if not all(path.is_file() for path in required):
    raise SystemExit("managed CUDA lacks static compiler runtime libraries")
""".strip()


_TARGET_DOWNLOAD_SCRIPT = r"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

destination = Path(sys.argv[1])
expected = os.environ["TRTMC_ARTIFACT_SHA256"]
destination.parent.mkdir(parents=True, exist_ok=True)
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
if destination.is_file():
    if sha256(destination) == expected:
        raise SystemExit(0)
digest = hashlib.sha256()
partial = None
try:
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        partial = Path(stream.name)
        with urllib.request.urlopen(os.environ["TRTMC_ARTIFACT_URI"], timeout=120) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
except Exception as error:
    if partial is not None:
        partial.unlink(missing_ok=True)
    raise SystemExit(f"artifact download failed: {type(error).__name__}")
if digest.hexdigest() != expected:
    if partial is not None:
        partial.unlink(missing_ok=True)
    raise SystemExit("artifact checksum mismatch")
os.replace(partial, destination)
""".strip()


_TARGET_HEADER_SCRIPT = r"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = sys.argv[2]
matches = []
for header in root.rglob("NvInferVersion.h"):
    definitions = dict(re.findall(
        r"^#define\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b",
        header.read_text(),
        re.M,
    ))
    parts = []
    for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        value = definitions.get(f"NV_TENSORRT_{name}", "")
        parts.append(definitions.get(value, value))
    if ".".join(parts) == expected:
        matches.append(header.parent.resolve())
if len(matches) != 1:
    raise SystemExit(f"expected one matching TensorRT header tree, found {len(matches)}")
print(matches[0])
""".strip()


_TARGET_LOCATIONS_SCRIPT = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

distribution = metadata.distribution(sys.argv[1])
libraries = Path(distribution.locate_file("tensorrt_libs")).resolve()
candidates = sorted(
    libraries.glob("libnvinfer.so*"),
    key=lambda path: (path.name != "libnvinfer.so", path.name),
)
if not candidates:
    raise SystemExit("TensorRT library was not installed")
cuda_root = Path(sys.argv[2]).resolve()
required = (
    cuda_root / "bin" / "nvcc",
    cuda_root / "include" / "cuda.h",
    next(cuda_root.rglob("libcudart.so"), None),
    next(cuda_root.rglob("libcublas.so"), None),
    next(cuda_root.rglob("libcurand.so"), None),
)
if not all(path is not None and path.is_file() for path in required):
    raise SystemExit("CUDA toolkit is incomplete")
print(json.dumps({
    "cuda_root": str(cuda_root),
    "tensorrt_library": str(candidates[0].resolve()),
}))
""".strip()


_TARGET_TOOLCHAIN_PROBE_SCRIPT = r"""
import ctypes
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import tensorrt

nvcc, include_dir, library, cuda_root = map(Path, sys.argv[1:5])
text = subprocess.check_output([nvcc, "--version"], text=True)
cuda = re.search(r"release\s+([0-9]+\.[0-9]+)", text, re.I)
if cuda is None:
    raise SystemExit("nvcc did not report a CUDA release")
header = include_dir / "NvInferVersion.h"
definitions = dict(re.findall(
    r"^#define\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b",
    header.read_text(),
    re.M,
))
parts = []
for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
    value = definitions.get(f"NV_TENSORRT_{name}", "")
    parts.append(definitions.get(value, value))
native_library = ctypes.CDLL(str(library))
functions = [
    getattr(native_library, f"getInferLib{name}Version")
    for name in ("Major", "Minor", "Patch", "Build")
]
for function in functions:
    function.restype = ctypes.c_int32
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
print(json.dumps({
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "cuda": cuda.group(1),
    "tensorrt_python": tensorrt.__version__,
    "tensorrt_native": ".".join(str(function()) for function in functions),
    "tensorrt_headers": ".".join(parts),
    "tensorrt_include_dir": str(include_dir),
    "tensorrt_library": str(library),
    "cuda_root": str(cuda_root),
    "architecture": platform.machine(),
    "evidence": {
        "nvcc": sha256(nvcc),
        "tensorrt-header": sha256(header),
        "tensorrt-library": sha256(library),
    },
}))
""".strip()
