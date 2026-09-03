# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public capability API tests for the repository-local TRTMC DevToolkit."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLKIT_ROOT = REPO_ROOT / "scripts" / "devToolkit"
sys.path.insert(0, str(DEVTOOLKIT_ROOT))

from trtmc_devtoolkit import (  # noqa: E402
    ArtifactPin,
    ArtifactUnavailable,
    AttestationFailed,
    CommandSpec,
    CudaPolicy,
    DevToolkitError,
    DevToolkit,
    EnvironmentRequest,
    ExecutionTarget,
    IncompatibleCombination,
    JsonQualificationSource,
    JsonToolchainCatalog,
    NvidiaPackageIndexCatalog,
    ProvisionPolicy,
    ToolchainRuntime,
    TrtmcBuildRecipe,
    ToolchainObservation,
    repository_path,
)
from trtmc_devtoolkit.spi import (  # noqa: E402
    ContextHandle,
    ContextLock,
    ExecutionContext,
    ProviderDescriptor,
    ProviderRegistry,
    QualificationRegistry,
    ToolchainCandidate,
    ToolchainHandle,
    ToolchainSource,
)
from trtmc_devtoolkit import receipt as receipt_module  # noqa: E402
from trtmc_devtoolkit import catalogs as catalogs_module  # noqa: E402
from trtmc_devtoolkit.building import BuildContext  # noqa: E402
from trtmc_devtoolkit.builtin_providers import (  # noqa: E402
    ManagedArtifactToolchainSource,
)


def test_extension_protocols_are_isolated_in_spi() -> None:
    assert ExecutionContext.__name__ == "ExecutionContext"
    assert ToolchainSource.__name__ == "ToolchainSource"


def test_nvidia_catalog_resolves_a_digest_pinned_development_toolchain() -> None:
    version = "11.2.0.113"
    meta = {
        "urls": [
            {
                "filename": f"tensorrt_cu13-{version}.tar.gz",
                "url": f"https://files.example/tensorrt_cu13-{version}.tar.gz",
                "digests": {"sha256": "1" * 64},
            }
        ]
    }
    bindings = {
        "urls": [
            {
                "filename": (
                    f"tensorrt_cu13_bindings-{version}-cp312-none-manylinux_2_28_x86_64.whl"
                ),
                "url": f"https://files.example/bindings-{version}.whl",
                "digests": {"sha256": "2" * 64},
            }
        ]
    }
    simple = (
        f'<a href="tensorrt_cu13_libs-{version}-py3-none-manylinux_2_28_x86_64.whl'
        f'#sha256={"3" * 64}">runtime</a>'
    ).encode()
    packages = gzip.compress(
        "\n".join(
            (
                "Package: libnvinfer-headers-dev",
                f"Version: {version}-1+cuda13.3",
                "Architecture: amd64",
                f"Filename: ./headers-{version}.deb",
                f"SHA256: {'4' * 64}",
                "",
            )
        ).encode()
    )

    def fetch(url: str) -> bytes | None:
        if url.endswith(f"/tensorrt-cu13/{version}/json"):
            return json.dumps(meta).encode()
        if url.endswith(f"/tensorrt-cu13-bindings/{version}/json"):
            return json.dumps(bindings).encode()
        if url.endswith("/tensorrt-cu13-libs/"):
            return simple
        if url.endswith("/ubuntu2404/x86_64/Packages.gz"):
            return packages
        raise AssertionError(url)

    catalog = NvidiaPackageIndexCatalog(fetch=fetch)
    artifacts, headers_version = catalog._distribution(
        version,
        "13.0",
        "3.12",
        "x86_64",
        "ubuntu",
        "24.04",
    )

    assert [artifact.name for artifact in artifacts] == [
        "tensorrt-python",
        "tensorrt-bindings",
        "tensorrt-libs",
        "tensorrt-headers",
    ]
    assert [artifact.sha256 for artifact in artifacts] == [
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
    ]
    assert headers_version == f"{version}-1+cuda13.3"


def test_nvidia_catalog_resolves_managed_cuda_when_target_has_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "11.2.0.113"
    cuda_components = (
        "cuda_crt",
        "cuda_cudart",
        "cuda_culibos",
        "cuda_nvcc",
        "libcublas",
        "libcurand",
        "libnvvm",
        "cccl",
    )
    redistrib = {
        "release_label": "13.3.0",
        **{
            component: {
                "linux-x86_64": {
                    "relative_path": f"{component}/{component}.tar.xz",
                    "sha256": str(index) * 64,
                }
            }
            for index, component in enumerate(cuda_components, start=1)
        },
    }
    meta = {
        "urls": [
            {
                "filename": f"tensorrt_cu13-{version}.tar.gz",
                "url": f"https://files.example/tensorrt_cu13-{version}.tar.gz",
                "digests": {"sha256": "a" * 64},
            }
        ]
    }
    bindings = {
        "urls": [
            {
                "filename": (
                    f"tensorrt_cu13_bindings-{version}-cp312-none-manylinux_2_28_x86_64.whl"
                ),
                "url": f"https://files.example/bindings-{version}.whl",
                "digests": {"sha256": "b" * 64},
            }
        ]
    }
    simple = (
        f'<a href="tensorrt_cu13_libs-{version}-py3-none-manylinux_2_28_x86_64.whl'
        f'#sha256={"c" * 64}">runtime</a>'
    ).encode()
    packages = gzip.compress(
        "\n".join(
            (
                "Package: libnvinfer-headers-dev",
                f"Version: {version}-1+cuda13.3",
                "Architecture: amd64",
                f"Filename: ./headers-{version}.deb",
                f"SHA256: {'d' * 64}",
                "",
            )
        ).encode()
    )
    bootstrap = {
        "pip": ("24.0", "pip-24.0-py3-none-any.whl", "e" * 64),
        "setuptools": (
            "68.1.2",
            "setuptools-68.1.2-py3-none-any.whl",
            "f" * 64,
        ),
        "wheel": ("0.42.0", "wheel-0.42.0-py3-none-any.whl", "0" * 64),
    }

    def fetch(url: str) -> bytes | None:
        if url.endswith("/redistrib_13.3.0.json"):
            return json.dumps(redistrib).encode()
        if url.endswith(f"/tensorrt-cu13/{version}/json"):
            return json.dumps(meta).encode()
        if url.endswith(f"/tensorrt-cu13-bindings/{version}/json"):
            return json.dumps(bindings).encode()
        if url.endswith("/tensorrt-cu13-libs/"):
            return simple
        if url.endswith("/ubuntu2404/x86_64/Packages.gz"):
            return packages
        for package, (package_version, filename, digest) in bootstrap.items():
            if url.endswith(f"/{package}/{package_version}/json"):
                return json.dumps(
                    {
                        "urls": [
                            {
                                "filename": filename,
                                "url": f"https://files.example/{filename}",
                                "digests": {"sha256": digest},
                            }
                        ]
                    }
                ).encode()
        raise AssertionError(url)

    monkeypatch.setattr(catalogs_module, "target_runtime_baseline", lambda *args: None)
    monkeypatch.setattr(
        catalogs_module,
        "target_python_baseline",
        lambda *args: type(
            "PythonBaseline",
            (),
            {"python": "3.12", "python_executable": "/usr/bin/python3"},
        )(),
    )
    context = ContextLock(
        provider=ProviderDescriptor("test", "tests==1", 1),
        operating_system="linux",
        architecture="x86_64",
        identity={"os_id": "ubuntu", "os_version": "24.04"},
        capabilities=frozenset({"container-process"}),
    )
    candidate = NvidiaPackageIndexCatalog(fetch=fetch).resolve(
        EnvironmentRequest(
            tensorrt=version,
            target=ExecutionTarget("test"),
        ),
        context,
        repository=tmp_path,
        runner=CommandRecordingRunner(),
    )[0]

    assert candidate.cuda == "13.3"
    assert candidate.cuda_source == "managed"
    assert candidate.identity["cuda_release"] == "13.3.0"
    assert candidate.identity["cuda_artifacts"] == tuple(
        f"cuda-component-{component}" for component in cuda_components
    )
    assert candidate.identity["python_bootstrap_artifacts"] == (
        "python-bootstrap-pip",
        "python-bootstrap-setuptools",
        "python-bootstrap-wheel",
    )
    assert [artifact.name for artifact in candidate.artifacts[-3:]] == list(
        candidate.identity["python_bootstrap_artifacts"]
    )
    assert len(candidate.artifacts) == 15


def test_nvidia_catalog_ignores_a_malformed_cuda_manifest() -> None:
    catalog = NvidiaPackageIndexCatalog(fetch=lambda _url: json.dumps([]).encode())

    artifacts, release = catalog._cuda_distribution("13.3", "x86_64")

    assert artifacts == ()
    assert release == ""


def test_json_catalog_supplies_a_private_exact_version_from_target_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "11.2.0.113"
    manifest = tmp_path / "private-toolchains.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "toolchains": [
                    {
                        "id": "gb300-trt-1120113",
                        "tensorrt": version,
                        "python": "3.12",
                        "architecture": "x86_64",
                        "cuda": {"source": "target", "major": "13"},
                        "artifacts": [
                            {
                                "name": "tensorrt-bindings",
                                "uri": "artifacts/bindings.whl",
                                "sha256": "1" * 64,
                            },
                            {
                                "name": "tensorrt-libs",
                                "uri": "https://artifacts.example/libs.whl",
                                "sha256": "2" * 64,
                            },
                            {
                                "name": "tensorrt-headers",
                                "uri": "https://artifacts.example/headers.deb",
                                "sha256": "3" * 64,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        catalogs_module,
        "target_runtime_baseline",
        lambda *args: type(
            "RuntimeBaseline",
            (),
            {
                "python": "3.12",
                "python_executable": "/usr/bin/python3",
                "cuda": "13.0",
                "cuda_root": "/usr/local/cuda-13.0",
                "nvcc": "/usr/local/cuda-13.0/bin/nvcc",
                "cuda_source": "image",
            },
        )(),
    )
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    registry.register_toolchain(ManagedArtifactToolchainSource())
    registry.register_catalog(JsonToolchainCatalog((manifest,)))
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt=version,
            target=ExecutionTarget("test-local"),
            architecture="x86_64",
            toolchain_options={"catalog": "json-toolchain-catalog"},
        )
    )

    assert lock.tensorrt == version
    assert lock.cuda == "13.0"
    assert lock.toolchain.cuda_source == "image"
    assert lock.toolchain.identity["catalog"]["record_id"] == "gb300-trt-1120113"
    assert lock.toolchain.artifacts[0].uri == (tmp_path / "artifacts/bindings.whl").as_uri()


def test_public_api_exposes_capabilities_without_plan_or_apply(tmp_path: Path) -> None:
    toolkit = DevToolkit.from_checkout(tmp_path)

    assert not hasattr(toolkit, "plan")
    assert not hasattr(toolkit, "apply")


def test_generic_qualification_source_is_explicit(tmp_path: Path) -> None:
    root = tmp_path / "qualifications"
    root.mkdir()
    (root / "qualified.json").write_text(
        json.dumps(
            {
                "id": "qualified",
                "status": "supported",
                "requirements": {"tensorrt": "11.2.0.113", "execution": ["local"]},
            }
        ),
        encoding="utf-8",
    )
    records = QualificationRegistry((JsonQualificationSource((root,)),)).load_all()

    assert records
    assert all(record.reference.digest for record in records)


class BuiltinProbeRunner:
    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, env, check, capture_output, timeout
        arguments = [str(item) for item in command]
        if arguments[-1] == "--version":
            output = "Cuda compilation tools, release 12.8, V12.8.0\n"
        elif "sys.version_info" in arguments[-1]:
            output = "3.12\n"
        elif "tensorrt.__version__" in arguments[-1]:
            output = "11.2.0.113\n"
        elif "getInferLib" in arguments[-1]:
            output = "11.2.0.113\n"
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, 0, output, "")


class DockerAdoptionRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []
        self.environment_files: list[tuple[Path, str, int]] = []
        self.client_version = "28.0.0"
        self.daemon_id = "daemon-789"
        self.container_id = "container-123"
        self.tensorrt_version = "11.0.2.2"

    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, check, capture_output, timeout
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        self.environments.append(dict(env) if env is not None else None)
        if arguments == ["docker", "context", "show"]:
            output = "test-context\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "version"]:
            output = self.client_version + "\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "info"]:
            output = self.daemon_id + "\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "inspect"]:
            output = json.dumps(
                {
                    "Id": self.container_id,
                    "Image": "sha256:image-456",
                    "State": {"Running": True},
                    "Config": {"Image": "campaign:latest"},
                }
            )
        elif arguments[:4] == ["docker", "--context", "test-context", "exec"]:
            if "--env-file" in arguments:
                environment_file = Path(arguments[arguments.index("--env-file") + 1])
                self.environment_files.append(
                    (
                        environment_file,
                        environment_file.read_text(encoding="utf-8"),
                        environment_file.stat().st_mode & 0o777,
                    )
                )
            if arguments[-2:] == ["cat", "/etc/os-release"]:
                output = 'ID="ubuntu"\nVERSION_ID="24.04"\n'
            elif "uname" in arguments:
                output = "aarch64\n"
            else:
                output = json.dumps(
                    {
                        "python": "3.12",
                        "python_executable": "/usr/bin/python3",
                        "cuda": "12.8",
                        "cuda_root": "/usr/local/cuda-12.8",
                        "tensorrt_python": self.tensorrt_version,
                        "tensorrt_native": self.tensorrt_version,
                        "tensorrt_headers": self.tensorrt_version,
                        "tensorrt_include_dir": "/usr/include/aarch64-linux-gnu",
                        "tensorrt_library": ("/usr/lib/aarch64-linux-gnu/libnvinfer.so.11"),
                        "cuda_complete": True,
                        "architecture": "aarch64",
                        "evidence": {
                            "nvcc": "1" * 64,
                            "tensorrt-header": "2" * 64,
                            "tensorrt-library": "3" * 64,
                        },
                    }
                )
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, 0, output, "")


class ManagedProvisionRunner:
    def __init__(self, root: Path) -> None:
        self.header_include: Path | None = None
        self.cuda = root / "managed-cuda"
        self.trt = root / "managed-trt"
        for directory in (self.cuda / "bin", self.cuda / "include", self.cuda / "lib"):
            directory.mkdir(parents=True)
        for relative in (
            "bin/nvcc",
            "include/cuda.h",
            "lib/libcudart.so",
            "lib/libcublas.so",
            "lib/libcurand.so",
        ):
            (self.cuda / relative).touch()
        self.trt.mkdir()
        (self.trt / "libnvinfer.so").touch()

    def run(self, command, **kwargs):
        check = kwargs.get("check", True)
        arguments = [str(item) for item in command]
        output = ""
        returncode = 0
        script = arguments[2] if len(arguments) > 2 and arguments[1] == "-c" else ""
        if arguments[0] == "test":
            path = Path(arguments[-1])
            returncode = 0 if path.is_file() else 1
        elif arguments[:2] == ["mkdir", "-p"]:
            Path(arguments[-1]).mkdir(parents=True, exist_ok=True)
        elif arguments[0] == "touch":
            Path(arguments[-1]).touch()
        elif arguments[:2] == ["dpkg-deb", "--extract"]:
            include = Path(arguments[3]) / "usr" / "include" / "x86_64-linux-gnu"
            self.header_include = include
            include.mkdir(parents=True)
            (include / "NvInferVersion.h").write_text(
                "\n".join(
                    (
                        "#define NV_TENSORRT_MAJOR 11",
                        "#define NV_TENSORRT_MINOR 2",
                        "#define NV_TENSORRT_PATCH 0",
                        "#define NV_TENSORRT_BUILD 113",
                    )
                ),
                encoding="utf-8",
            )
        elif "expected one matching TensorRT header tree" in script:
            assert self.header_include is not None
            output = str(self.header_include) + "\n"
        elif "metadata.distribution" in script or "m.distribution" in arguments[-1]:
            if not (self.cuda / "lib" / "libcurand.so").is_file():
                raise DevToolkitError("Managed artifacts produced an incomplete CUDA toolkit")
            output = json.dumps(
                {
                    "cuda_root": str(self.cuda),
                    "tensorrt_library": str(self.trt / "libnvinfer.so"),
                }
            )
        elif "os.environ.get" in script and '"LD_LIBRARY_PATH"' in script:
            output = json.dumps(
                {
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
                    "LD_LIBRARY_PATH": "",
                }
            )
        elif arguments[-1] == "--version":
            output = "Cuda compilation tools, release 13.3, V13.3.0\n"
        elif "sys.version_info" in arguments[-1]:
            output = "3.12\n"
        elif "tensorrt.__version__" in arguments[-1]:
            output = "11.2.0.113\n"
        elif "getInferLib" in arguments[-1]:
            output = "11.2.0.113\n"
        result = subprocess.CompletedProcess(arguments, returncode, output, "")
        if check and returncode != 0:
            raise DevToolkitError(f"simulated target command failure: {arguments}")
        return result


class CommandRecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.artifact_digest = "c" * 64

    def run(self, command, **kwargs):
        del kwargs
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        output = "trtmc 0.1\n"
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            output = "b" * 40 + "\n"
        elif arguments[-3:-1] == ["diff", "--binary"]:
            output = ""
        elif arguments[-3:-1] == ["ls-files", "--others"]:
            output = ""
        elif arguments[0] == "sha256sum":
            output = f"{self.artifact_digest}  {arguments[1]}\n"
        elif "CUDA runtime headers are missing" in arguments[-2]:
            output = json.dumps(
                {
                    "include": "/cuda/include",
                    "cudart": "/cuda/lib64/libcudart.so",
                    "cublas": "/cuda/lib64/libcublas.so",
                }
            )
        elif "shutil.which" in arguments[-1]:
            output = "Ninja\n"
        return subprocess.CompletedProcess(arguments, 0, output, "")


class BlockingBuildRunner(CommandRecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.first_configure_entered = threading.Event()
        self.release_first_configure = threading.Event()
        self.second_configure_entered = threading.Event()
        self._configure_calls = 0
        self._guard = threading.Lock()

    def run(self, command, **kwargs):
        arguments = [str(item) for item in command]
        if arguments[:2] == ["cmake", "-S"]:
            with self._guard:
                self._configure_calls += 1
                call = self._configure_calls
            if call == 1:
                self.first_configure_entered.set()
                assert self.release_first_configure.wait(timeout=5)
            else:
                self.second_configure_entered.set()
        return super().run(command, **kwargs)


class NonzeroCommandRunner:
    def __init__(self) -> None:
        self.checks: list[bool] = []

    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, env, capture_output, timeout
        arguments = [str(item) for item in command]
        self.checks.append(check)
        return subprocess.CompletedProcess(arguments, 7, "", "expected failure")


class SecretEchoFailureRunner:
    def run(self, command, *, env=None, **kwargs):
        del command, kwargs
        raise RuntimeError(f"runner leaked environment: {env}")


class StaticLocalContext:
    descriptor = ProviderDescriptor("test-local", "tests==1", 1)

    def resolve(self, request, *, repository, runner):
        del repository, runner
        return ContextLock(
            provider=self.descriptor,
            operating_system="linux",
            architecture=request.architecture or "aarch64",
            identity={"kind": "local"},
            execution={"gpu": "0", "python": "python3"},
            locator={"gpu": "0"},
            capabilities=frozenset({"host-filesystem", "posix-process"}),
            qualification={"execution": "local"},
        )

    def provision(
        self,
        context,
        *,
        inherit_system_packages,
        repository,
        state_dir,
        policy,
        runner,
    ):
        del repository, runner
        del inherit_system_packages
        assert policy is ProvisionPolicy.ADOPT_OR_CREATE
        state_dir.mkdir(parents=True, exist_ok=True)
        return ContextHandle(
            provider=self.descriptor,
            identity=context.identity,
            execution_identity={"root": str(state_dir), "gpu": "0"},
            locator={"root": str(state_dir)},
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )

    def execute(
        self,
        context,
        command,
        *,
        repository,
        state_dir,
        runner,
        check,
        capture_output,
    ):
        del context
        arguments = [
            str(repository / argument.path)
            if hasattr(argument, "scope") and argument.scope.value == "repository"
            else str(state_dir / argument.path)
            if hasattr(argument, "scope") and argument.scope.value == "state"
            else str(argument)
            for argument in command.arguments
        ]
        cwd = repository / command.cwd.path
        return runner.run(
            arguments,
            cwd=cwd,
            env=command.environment,
            check=check,
            capture_output=capture_output,
        )


class BlockingLocalContext(StaticLocalContext):
    def __init__(self) -> None:
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.second_entered = threading.Event()
        self._guard = threading.Lock()
        self._calls = 0

    def provision(
        self,
        context,
        *,
        inherit_system_packages,
        repository,
        state_dir,
        policy,
        runner,
    ):
        with self._guard:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.first_entered.set()
            assert self.release_first.wait(timeout=5)
        else:
            self.second_entered.set()
        return super().provision(
            context,
            inherit_system_packages=inherit_system_packages,
            repository=repository,
            state_dir=state_dir,
            policy=policy,
            runner=runner,
        )


class ExistingToolchainSource:
    descriptor = ProviderDescriptor("test-system", "tests==1", 1)

    def __init__(self) -> None:
        self.requested_versions: list[str] = []

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        self.requested_versions.append(request.tensorrt)
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="system",
                tensorrt=request.tensorrt,
                cuda="12.8",
                python=request.python,
                identity={"prefix": "/opt/nvidia"},
                runtime=ToolchainRuntime(
                    python_executable="python3",
                    cuda_root="/opt/nvidia/cuda",
                    nvcc="/opt/nvidia/cuda/bin/nvcc",
                    tensorrt_include_dir="/opt/nvidia/include",
                    tensorrt_library="/opt/nvidia/lib/libnvinfer.so",
                ),
            ),
        )

    def provision(self, lock, context, *, repository, state_dir, runner):
        del repository, state_dir, runner
        assert lock.toolchain.runtime is not None
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=replace(
                lock.toolchain.runtime,
                python_executable=str(context.locator.get("python", "python3")),
            ),
        )

    def observe(self, lock, context, toolchain, *, repository, runner):
        del context, repository, runner
        return ToolchainObservation(
            python_version=lock.toolchain.python,
            cuda_version=lock.toolchain.cuda,
            tensorrt_python_version=lock.toolchain.tensorrt,
            tensorrt_native_version=lock.toolchain.tensorrt,
            tensorrt_header_version=lock.toolchain.tensorrt,
            tensorrt_include_dir=toolchain.runtime.tensorrt_include_dir,
            tensorrt_library=toolchain.runtime.tensorrt_library,
            cuda_root=toolchain.runtime.cuda_root,
            evidence={"toolchain": "a" * 64},
        )


class ReplacementToolchainSource(ExistingToolchainSource):
    descriptor = ProviderDescriptor("test-system", "tests==2", 1)


class AlternateExistingToolchainSource(ExistingToolchainSource):
    descriptor = ProviderDescriptor("alternate-system", "tests==1", 1)


class FailingReattestationSource(ExistingToolchainSource):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0

    def observe(self, lock, context, toolchain, *, repository, runner):
        self.observations += 1
        if self.observations > 1:
            raise RuntimeError("preflight contained super-secret")
        return super().observe(
            lock,
            context,
            toolchain,
            repository=repository,
            runner=runner,
        )


class MutableEvidenceSource(ExistingToolchainSource):
    def __init__(self) -> None:
        super().__init__()
        self.digest = "a" * 64

    def observe(self, lock, context, toolchain, *, repository, runner):
        observed = super().observe(
            lock,
            context,
            toolchain,
            repository=repository,
            runner=runner,
        )
        return replace(observed, evidence={"toolchain": self.digest})


class EmptySystemToolchainSource:
    descriptor = ProviderDescriptor("empty-system", "tests==1", 1)

    def resolve(self, request, context, *, repository, runner):
        del request, context, repository, runner
        return ()


class RecordingToolchainCatalog:
    descriptor = ProviderDescriptor("test-catalog", "tests==1", 1)

    def __init__(self, materializer: ProviderDescriptor) -> None:
        self.materializer = materializer
        self.requests: list[str] = []

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        self.requests.append(request.tensorrt)
        return (
            ToolchainCandidate(
                provider=self.materializer,
                origin="managed",
                cuda_source="managed",
                tensorrt=request.tensorrt,
                cuda="13.3",
                python=request.python,
                identity={"catalog": self.descriptor.name},
                artifacts=(
                    ArtifactPin(
                        "toolchain",
                        "https://example.invalid/toolchain.tar.xz",
                        "f" * 64,
                    ),
                ),
            ),
        )


class ManagedContainerCatalog:
    descriptor = ProviderDescriptor("container-catalog", "tests==1", 1)

    def __init__(self, materializer: ProviderDescriptor) -> None:
        self.materializer = materializer

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        suffixes = ("module.tar.gz", "bindings.whl", "libs.whl", "headers.deb")
        names = (
            "tensorrt-python",
            "tensorrt-bindings",
            "tensorrt-libs",
            "tensorrt-headers",
        )
        return (
            ToolchainCandidate(
                provider=self.materializer,
                origin="managed",
                cuda_source="image",
                tensorrt=request.tensorrt,
                cuda="13.0",
                python=request.python,
                identity={
                    "layout_schema": 2,
                    "tensorrt_lib_distribution": "tensorrt_cu13_libs",
                    "system_cuda_root": "/usr/local/cuda",
                    "system_nvcc": "/usr/local/cuda/bin/nvcc",
                },
                artifacts=tuple(
                    ArtifactPin(name, f"https://example.invalid/{suffix}", str(index) * 64)
                    for index, (name, suffix) in enumerate(zip(names, suffixes), start=1)
                ),
            ),
        )


class BootstrapManagedContainerCatalog(ManagedContainerCatalog):
    descriptor = ProviderDescriptor("bootstrap-container-catalog", "tests==1", 1)

    def resolve(self, request, context, *, repository, runner):
        candidate = super().resolve(
            request,
            context,
            repository=repository,
            runner=runner,
        )[0]
        names = (
            "python-bootstrap-pip",
            "python-bootstrap-setuptools",
            "python-bootstrap-wheel",
        )
        artifacts = tuple(
            ArtifactPin(name, f"https://example.invalid/{name}.whl", str(index) * 64)
            for index, name in enumerate(names, start=5)
        )
        return (
            replace(
                candidate,
                identity={
                    **dict(candidate.identity),
                    "python_bootstrap_artifacts": names,
                },
                artifacts=(*candidate.artifacts, *artifacts),
            ),
        )


class ManagedCudaContainerCatalog(ManagedContainerCatalog):
    descriptor = ProviderDescriptor("managed-cuda-container-catalog", "tests==1", 1)

    def resolve(self, request, context, *, repository, runner):
        candidate = super().resolve(
            request,
            context,
            repository=repository,
            runner=runner,
        )[0]
        cuda_artifact = ArtifactPin(
            "cuda-component-cuda-nvcc",
            "https://example.invalid/cuda-nvcc.tar.xz",
            "5" * 64,
        )
        return (
            replace(
                candidate,
                cuda="13.3",
                cuda_source="managed",
                identity={
                    **dict(candidate.identity),
                    "system_cuda_root": None,
                    "system_nvcc": None,
                    "cuda_artifacts": (cuda_artifact.name,),
                    "cuda_release": "13.3.0",
                },
                artifacts=(*candidate.artifacts, cuda_artifact),
            ),
        )


class ManagedContainerContext:
    descriptor = ProviderDescriptor("test-container", "tests==1", 1)

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.command_environments: list[dict[str, str]] = []
        self.cuda_version = "13.0"
        self.venv_requires_bootstrap = False
        self.target_files: set[str] = set()

    def resolve(self, request, *, repository, runner):
        del request, repository, runner
        return ContextLock(
            provider=self.descriptor,
            operating_system="linux",
            architecture="x86_64",
            identity={"image_id": "sha256:test-image"},
            execution={"python": "/usr/bin/python3", "target_state": "/target/state"},
            locator={"python": "/usr/bin/python3", "target_state": "/target/state"},
            capabilities=frozenset({"container-process", "posix-process"}),
            qualification={"execution": "container"},
        )

    def provision(self, context, **kwargs):
        del kwargs

        def execute(command, check, capture_output):
            del capture_output
            arguments = [str(item) for item in command.arguments]
            self.commands.append(arguments)
            self.command_environments.append(dict(command.environment))
            returncode = 0
            if arguments[0] == "test":
                returncode = 0 if arguments[-1] in self.target_files else 1
            elif (
                self.venv_requires_bootstrap
                and arguments[:3] == ["/usr/bin/python3", "-m", "venv"]
                and "--without-pip" not in arguments
            ):
                returncode = 1
            elif arguments[0] == "touch":
                self.target_files.add(arguments[-1])
            output = ""
            script = arguments[2] if len(arguments) > 2 and arguments[1] == "-c" else ""
            if "os.environ.get" in script and '"LD_LIBRARY_PATH"' in script:
                output = json.dumps(
                    {
                        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
                        "LD_LIBRARY_PATH": "/image/lib",
                    }
                )
            elif "expected one matching TensorRT header tree" in script:
                output = (
                    "/target/state/managed-toolchain/lock/headers/usr/include/x86_64-linux-gnu\n"
                )
            elif "metadata.distribution" in script:
                output = json.dumps(
                    {
                        "cuda_root": arguments[-1],
                        "tensorrt_library": "/target/state/venv/tensorrt_libs/libnvinfer.so",
                    }
                )
            elif "source package produced no wheel" in script:
                output = "/target/state/managed-toolchain/lock/built-wheels/tensorrt.whl\n"
            elif '"tensorrt_python"' in script:
                output = json.dumps(
                    {
                        "python": "3.12",
                        "cuda": self.cuda_version,
                        "tensorrt_python": "11.2.0.113",
                        "tensorrt_native": "11.2.0.113",
                        "tensorrt_headers": "11.2.0.113",
                        "tensorrt_include_dir": (
                            "/target/state/managed-toolchain/lock/headers/usr/include/"
                            "x86_64-linux-gnu"
                        ),
                        "tensorrt_library": ("/target/state/venv/tensorrt_libs/libnvinfer.so"),
                        "cuda_root": arguments[-1],
                        "architecture": "x86_64",
                        "evidence": {
                            "nvcc": "a" * 64,
                            "tensorrt-header": "b" * 64,
                            "tensorrt-library": "c" * 64,
                        },
                    }
                )
            result = subprocess.CompletedProcess(arguments, returncode, output, "")
            if check and returncode != 0:
                raise DevToolkitError(f"simulated target command failure: {arguments}")
            return result

        return ContextHandle(
            provider=self.descriptor,
            identity=context.identity,
            execution_identity={"python": "/usr/bin/python3"},
            locator=context.locator,
            capabilities=context.capabilities,
            _executor=execute,
            _path_mapper=lambda path: f"/target/state/{path.path}",
        )

    def execute(self, context, command, **kwargs):
        del kwargs
        return context.execute(command)


class ManagedCudaToolchainSource:
    descriptor = ProviderDescriptor("test-managed", "tests==1", 1)

    def __init__(self) -> None:
        self.provisioned = False

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="managed",
                tensorrt=request.tensorrt,
                cuda="13.3",
                python=request.python,
                identity={"distribution": "test"},
                artifacts=(
                    ArtifactPin(
                        name="cuda-toolkit",
                        uri="https://example.invalid/cuda-13.3.tar.xz",
                        sha256="a" * 64,
                    ),
                ),
            ),
        )

    def provision(self, lock, context, *, repository, state_dir, runner):
        del context, repository, state_dir, runner
        self.provisioned = True
        return ToolchainHandle(
            provider=self.descriptor,
            identity=lock.toolchain.identity,
            runtime=ToolchainRuntime(
                python_executable="python3",
                cuda_root="/managed/cuda",
                nvcc="/managed/cuda/bin/nvcc",
                tensorrt_include_dir="/managed/include",
                tensorrt_library="/managed/lib/libnvinfer.so",
            ),
        )

    def observe(self, lock, context, toolchain, *, repository, runner):
        del context, repository, runner
        return ToolchainObservation(
            python_version=lock.toolchain.python,
            cuda_version=lock.toolchain.cuda,
            tensorrt_python_version=lock.toolchain.tensorrt,
            tensorrt_native_version=lock.toolchain.tensorrt,
            tensorrt_header_version=lock.toolchain.tensorrt,
            tensorrt_include_dir=toolchain.runtime.tensorrt_include_dir,
            tensorrt_library=toolchain.runtime.tensorrt_library,
            cuda_root=toolchain.runtime.cuda_root,
            evidence={"toolchain": "b" * 64},
        )


class CatalogMaterializer(ManagedCudaToolchainSource):
    descriptor = ProviderDescriptor("test-catalog-materializer", "tests==1", 1)

    def resolve(self, request, context, *, repository, runner):
        del request, context, repository, runner
        return ()


def test_arbitrary_tensorrt_reaches_provider_without_a_cohort(tmp_path: Path) -> None:
    source = ExistingToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(source)
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            cuda=CudaPolicy.system_first(fallback="13.3"),
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    assert source.requested_versions == ["11.2.0.113"]
    assert lock.tensorrt == "11.2.0.113"
    assert lock.cuda == "12.8"
    assert lock.cuda_origin == "system"
    assert len(lock.lock_id) == 64
    assert not state_root.exists()


def test_missing_version_falls_through_to_a_toolchain_catalog(tmp_path: Path) -> None:
    materializer = CatalogMaterializer()
    catalog = RecordingToolchainCatalog(materializer.descriptor)
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    registry.register_toolchain(materializer)
    registry.register_catalog(catalog)
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
        )
    )

    assert catalog.requests == ["11.2.0.113"]
    assert lock.toolchain.provider == materializer.descriptor
    assert lock.toolchain.origin == "managed"
    assert lock.toolchain.artifacts[0].sha256 == "f" * 64


def test_exact_installed_toolchain_does_not_query_catalog(tmp_path: Path) -> None:
    installed = ExistingToolchainSource()
    catalog = RecordingToolchainCatalog(installed.descriptor)
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(installed)
    registry.register_catalog(catalog)
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
        )
    )

    assert lock.toolchain.origin == "system"
    assert catalog.requests == []


def test_ambiguous_installed_toolchains_do_not_query_a_catalog(tmp_path: Path) -> None:
    installed = ExistingToolchainSource()
    alternate = AlternateExistingToolchainSource()
    catalog = RecordingToolchainCatalog(installed.descriptor)
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(installed)
    registry.register_toolchain(alternate)
    registry.register_catalog(catalog)
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    with pytest.raises(IncompatibleCombination, match="ambiguous exact candidates"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                target=ExecutionTarget("test-local"),
            )
        )

    assert catalog.requests == []


def test_managed_catalog_installs_into_an_isolated_container_prefix(tmp_path: Path) -> None:
    context = ManagedContainerContext()
    materializer = ManagedArtifactToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(context)
    registry.register_toolchain(materializer)
    registry.register_catalog(ManagedContainerCatalog(materializer.descriptor))
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-container"),
        )
    )
    environment = toolkit.provision(lock)

    assert environment.observation.tensorrt_native_version == "11.2.0.113"
    pip_install = next(
        command for command in context.commands if command[1:4] == ["-m", "pip", "install"]
    )
    assert pip_install[0].startswith(f"/target/state/managed-toolchain/{lock.lock_id}/venv/")
    assert pip_install[0] != "/usr/bin/python3"
    assert "--no-index" in pip_install
    assert environment.context.environment["PATH"].endswith(
        ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    )
    assert environment.context.environment["LD_LIBRARY_PATH"].endswith(":/image/lib")


def test_managed_target_bootstraps_python_without_ensurepip(tmp_path: Path) -> None:
    context = ManagedContainerContext()
    context.venv_requires_bootstrap = True
    materializer = ManagedArtifactToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(context)
    registry.register_toolchain(materializer)
    registry.register_catalog(BootstrapManagedContainerCatalog(materializer.descriptor))
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-container"),
        )
    )
    toolkit.provision(lock)

    venv_commands = [command for command in context.commands if command[1:3] == ["-m", "venv"]]
    assert any("--without-pip" in command for command in venv_commands)
    bootstrap_install_index = next(
        index
        for index, command in enumerate(context.commands)
        if command[1:4] == ["-m", "pip", "install"]
        and "python-bootstrap-pip.whl" in " ".join(command)
    )
    bootstrap_environment = context.command_environments[bootstrap_install_index]
    assert "python-bootstrap-pip.whl" in bootstrap_environment["PYTHONPATH"]
    wheel_build = next(
        command for command in context.commands if command[1:3] == ["-m", "pip"] and "wheel" in command
    )
    assert wheel_build[0].endswith("/venv/bin/python")


def test_managed_target_reuses_a_completed_toolchain_prefix(tmp_path: Path) -> None:
    context = ManagedContainerContext()
    materializer = ManagedArtifactToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(context)
    registry.register_toolchain(materializer)
    registry.register_catalog(ManagedContainerCatalog(materializer.descriptor))
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-container"),
        )
    )

    first = toolkit.provision(lock)
    first_wheel_builds = sum(
        command[1:3] == ["-m", "pip"] and "wheel" in command for command in context.commands
    )
    first_downloads = sum(
        "artifact download failed" in command[2]
        for command in context.commands
        if len(command) > 2 and command[1] == "-c"
    )
    second = toolkit.provision(lock)

    assert second.environment_id == first.environment_id
    assert sum(
        command[1:3] == ["-m", "pip"] and "wheel" in command for command in context.commands
    ) == first_wheel_builds
    assert sum(
        "artifact download failed" in command[2]
        for command in context.commands
        if len(command) > 2 and command[1] == "-c"
    ) == first_downloads


def test_managed_cuda_components_are_extracted_into_the_target_prefix(
    tmp_path: Path,
) -> None:
    context = ManagedContainerContext()
    context.cuda_version = "13.3"
    materializer = ManagedArtifactToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(context)
    registry.register_toolchain(materializer)
    registry.register_catalog(ManagedCudaContainerCatalog(materializer.descriptor))
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-container"),
        )
    )
    environment = toolkit.provision(lock)

    extraction = next(command for command in context.commands if command[0] == "tar")
    assert "--strip-components=1" in extraction
    assert environment.toolchain.runtime.cuda_root.endswith(f"/{lock.lock_id}/cuda")
    assert environment.observation.cuda_version == "13.3"


def test_system_first_falls_back_to_managed_cuda_13_3(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    registry.register_toolchain(ManagedCudaToolchainSource())
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.0.114",
            target=ExecutionTarget("test-local"),
        )
    )

    assert lock.cuda == "13.3"
    assert lock.cuda_origin == "managed-default"
    assert lock.toolchain.artifacts[0].sha256 == "a" * 64


def test_adopt_only_never_materializes_a_managed_toolchain(tmp_path: Path) -> None:
    source = ManagedCudaToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(source)
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.0.114",
            target=ExecutionTarget("test-local"),
        )
    )

    with pytest.raises(AttestationFailed, match="adopt-only"):
        toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert source.provisioned is False


@pytest.mark.parametrize("version", ["latest", "11.2", ">=11.1", "11.2.*"])
def test_environment_request_requires_an_exact_four_part_tensorrt_version(
    version: str,
) -> None:
    with pytest.raises(DevToolkitError, match="exact four-part"):
        EnvironmentRequest(tensorrt=version, target=ExecutionTarget.local())


def test_cuda_policy_rejects_an_unknown_runtime_kind() -> None:
    with pytest.raises(DevToolkitError, match="CUDA policy kind"):
        CudaPolicy("unknown")  # type: ignore[arg-type]


def test_environment_lock_rejects_a_tampered_identity(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=registry.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    with pytest.raises(DevToolkitError, match="lock ID"):
        replace(lock, lock_id="../outside")


def test_builtin_local_provider_discovers_complete_system_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include").mkdir()
    (cuda / "lib64").mkdir()
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib64/libcudart.so",
        "lib64/libcublas.so",
        "lib64/libcurand.so",
    ):
        (cuda / relative).touch()
    trt_include = tmp_path / "trt" / "include"
    trt_library_dir = tmp_path / "trt" / "lib"
    trt_include.mkdir(parents=True)
    trt_library_dir.mkdir(parents=True)
    (trt_library_dir / "libnvinfer.so").touch()
    major, minor, patch, build = "11.2.0.113".split(".")
    (trt_include / "NvInferVersion.h").write_text(
        "\n".join(
            (
                f"#define NV_TENSORRT_MAJOR {major}",
                f"#define NV_TENSORRT_MINOR {minor}",
                f"#define NV_TENSORRT_PATCH {patch}",
                f"#define NV_TENSORRT_BUILD {build}",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("TRTMC_TRT_INCLUDE_DIR", str(trt_include))
    monkeypatch.setenv("TRTMC_TRT_LIBRARY_DIR", str(trt_library_dir))
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(python="python3.12"),
        )
    )

    assert lock.toolchain.provider.name == "system"
    assert lock.toolchain.identity["cuda_root"] == str(cuda)
    assert lock.toolchain.identity["tensorrt_include_dir"] == str(trt_include)

    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert environment.observation.cuda_version == "12.8"
    assert environment.observation.tensorrt_header_version == "11.2.0.113"


def test_resolution_distinguishes_unavailable_from_incompatible(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    unavailable = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    with pytest.raises(ArtifactUnavailable):
        unavailable.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                target=ExecutionTarget("test-local"),
            )
        )

    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    incompatible = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    with pytest.raises(IncompatibleCombination):
        incompatible.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                cuda=CudaPolicy.exact("13.3"),
                target=ExecutionTarget("test-local"),
            )
        )


def test_provision_attests_and_writes_a_v3_receipt(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    environment = toolkit.provision(lock)

    assert environment.environment_id != lock.lock_id
    assert environment.observation.tensorrt_header_version == "11.2.0.113"
    receipt = json.loads(environment.receipt.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 3
    assert receipt["lock_id"] == lock.lock_id
    assert receipt["cuda_origin"] == "system"
    assert receipt["observed"]["tensorrt_native_version"] == "11.2.0.113"


def test_provision_rejects_a_different_toolchain_provider_implementation(
    tmp_path: Path,
) -> None:
    original = ProviderRegistry()
    original.register_context(StaticLocalContext())
    original.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=original.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    replacement = ProviderRegistry()
    replacement.register_context(StaticLocalContext())
    replacement.register_toolchain(ReplacementToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=replacement.freeze(),
    )

    with pytest.raises(AttestationFailed, match="toolchain provider"):
        toolkit.provision(lock)


def test_provision_records_failure_when_locked_provider_is_not_registered(
    tmp_path: Path,
) -> None:
    original = ProviderRegistry()
    original.register_context(StaticLocalContext())
    original.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=original.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    missing_toolchain = ProviderRegistry()
    missing_toolchain.register_context(StaticLocalContext())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=missing_toolchain.freeze(),
    )

    with pytest.raises(DevToolkitError, match="Unknown toolchain source provider"):
        toolkit.provision(lock)

    state_dir = state_root / "environments" / lock.lock_id
    failure = json.loads((state_dir / "provision-failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "DevToolkitError"
    assert not (state_dir / "provision-receipt.json").exists()


def test_provision_serializes_mutation_for_the_same_environment(tmp_path: Path) -> None:
    context = BlockingLocalContext()
    registry = ProviderRegistry()
    registry.register_context(context)
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    failures: list[BaseException] = []

    def provision() -> None:
        try:
            toolkit.provision(lock)
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=provision)
    second = threading.Thread(target=provision)
    first.start()
    assert context.first_entered.wait(timeout=5)
    second.start()
    assert not context.second_entered.wait(timeout=0.2)
    context.release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert context.second_entered.is_set()
    assert failures == []


def test_json_receipt_replace_failure_preserves_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt_module.write_json(receipt, {"status": "previous"})

    def fail_replace(source, destination) -> None:
        del source, destination
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(receipt_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interrupted replace"):
        receipt_module.write_json(receipt, {"status": "new"})

    assert json.loads(receipt.read_text(encoding="utf-8")) == {"status": "previous"}
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []


def test_builtin_docker_provider_adopts_a_probed_campaign_container(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert lock.toolchain.provider.name == "container-image"
    assert lock.context.identity["daemon_id"] == "daemon-789"
    assert lock.context.identity["container_id"] == "container-123"
    assert lock.context.identity["image_id"] == "sha256:image-456"
    assert environment.context.locator["container"] == "jedha-campaign"
    assert environment.observation.tensorrt_native_version == "11.0.2.2"
    assert not any("run" in command[:5] for command in runner.commands)


def test_docker_provider_rejects_cli_without_private_exec_env_files(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    runner.client_version = "19.03.15"
    toolkit = DevToolkit.from_checkout(tmp_path, runner=runner)

    with pytest.raises(DevToolkitError, match="Docker CLI 20.10 or newer"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt="11.0.2.2",
                target=ExecutionTarget.docker(container="jedha-campaign"),
                architecture="aarch64",
            )
        )


def test_docker_environment_identity_distinguishes_container_instances(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, runner=runner)
    request = EnvironmentRequest(
        tensorrt="11.0.2.2",
        target=ExecutionTarget.docker(container="campaign"),
        architecture="aarch64",
    )

    first = toolkit.resolve(request)
    runner.container_id = "container-456"
    second = toolkit.resolve(request)

    assert first.context.identity["image_id"] == second.context.identity["image_id"]
    assert first.context.identity["container_id"] != second.context.identity["container_id"]
    assert first.lock_id != second.lock_id


def test_docker_environment_identity_includes_effective_path_mapping(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    first_toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "first-state",
        runner=runner,
    )
    second_toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "second-state",
        runner=runner,
    )
    first_lock = first_toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(
                container="jedha-campaign",
                workspace="/workspace/first",
            ),
            architecture="aarch64",
        )
    )
    second_lock = second_toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(
                container="jedha-campaign",
                workspace="/workspace/second",
            ),
            architecture="aarch64",
        )
    )

    first = first_toolkit.provision(first_lock, policy=ProvisionPolicy.ADOPT_ONLY)
    second = second_toolkit.provision(second_lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert first_lock.lock_id != second_lock.lock_id
    assert first.environment_id != second.environment_id


def test_changed_toolchain_evidence_blocks_execution(tmp_path: Path) -> None:
    source = MutableEvidenceSource()
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(source)
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)
    source.digest = "b" * 64

    with pytest.raises(AttestationFailed, match="evidence changed"):
        toolkit.run(environment, CommandSpec(("trtmc", "version")))


def test_docker_command_forwards_environment_without_values_in_argv(tmp_path: Path) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    toolkit.run(
        environment,
        CommandSpec(("trtmc", "version"), environment={"TOKEN": "super-secret"}),
    )

    assert "super-secret" not in " ".join(runner.commands[-1])
    assert "--env-file" in runner.commands[-1]
    assert "container-123" in runner.commands[-1]
    assert "jedha-campaign" not in runner.commands[-1]
    environment_path, content, mode = runner.environment_files[-1]
    assert content == "TOKEN=super-secret\n"
    assert mode == 0o600
    assert not environment_path.exists()
    assert runner.environments[-1] is None


def test_docker_command_rejects_a_changed_daemon_after_attestation(tmp_path: Path) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)
    command_count = len(runner.commands)
    runner.daemon_id = "different-daemon"

    with pytest.raises(DevToolkitError, match="Docker daemon changed"):
        toolkit.run(environment, CommandSpec(("trtmc", "version")))

    new_commands = runner.commands[command_count:]
    assert len(new_commands) == 2
    assert new_commands[1][:4] == ["docker", "--context", "test-context", "info"]


def test_docker_command_rejects_a_replaced_container_after_attestation(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)
    command_count = len(runner.commands)
    runner.container_id = "replacement-container"

    with pytest.raises(DevToolkitError, match="container identity changed"):
        toolkit.run(environment, CommandSpec(("trtmc", "version")))

    new_commands = runner.commands[command_count:]
    assert len(new_commands) == 3
    assert not any("trtmc" in command for command in new_commands)


def test_docker_command_reattests_mutable_toolchain_before_execution(tmp_path: Path) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)
    command_count = len(runner.commands)
    runner.tensorrt_version = "11.0.0.114"

    with pytest.raises(AttestationFailed, match="expected 11.0.2.2"):
        toolkit.run(environment, CommandSpec(("trtmc", "version")))

    new_commands = runner.commands[command_count:]
    assert not any("trtmc" in command for command in new_commands)


def test_generic_command_is_routed_by_the_execution_context(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = CommandRecordingRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    result = toolkit.run(
        environment,
        CommandSpec(
            ("trtmc", "inspect", repository_path("example.bundle")),
            cwd=repository_path("."),
        ),
        capture_output=True,
    )

    assert runner.commands[-1] == ["trtmc", "inspect", str(tmp_path / "example.bundle")]
    assert result.stdout == "trtmc 0.1\n"
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 3
    assert receipt["environment_id"] == environment.environment_id
    assert receipt["occurrence_id"] != receipt["invocation_digest"]


def test_run_trtmc_forwards_check_policy(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = NonzeroCommandRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    result = toolkit.run_trtmc(environment, ("version",), check=False)

    assert result.returncode == 7
    assert runner.checks == [False]


def test_command_failure_receipt_does_not_serialize_environment_values(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
        runner=SecretEchoFailureRunner(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    with pytest.raises(RuntimeError, match="super-secret"):
        toolkit.run(
            environment,
            CommandSpec(("false",), environment={"TOKEN": "super-secret"}),
        )

    receipt = next((environment.state_dir / "commands").glob("*.json"))
    assert "super-secret" not in receipt.read_text(encoding="utf-8")


def test_native_build_identity_includes_source_sm_options_and_outputs(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = CommandRecordingRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    build = toolkit.build(
        environment,
        TrtmcBuildRecipe(
            targets=("trtmc",),
            cmake_defines={"TRTMC_BUILD_TESTS": False},
            cuda_architectures=("100",),
        ),
    )

    assert len(build.build_request_id) == 64
    assert len(build.build_id) == 64
    assert build.build_id != build.build_request_id
    assert build.artifacts[0].sha256 == "c" * 64
    configure = next(command for command in runner.commands if command[:2] == ["cmake", "-S"])
    assert "-DCMAKE_CUDA_ARCHITECTURES=100-real" in configure
    assert "-DTRTMC_BUILD_TESTS=OFF" in configure
    assert "-DTRTMC_CUDA_INCLUDE_DIR=/cuda/include" in configure
    assert "-DTRTMC_CUDART_LIBRARY=/cuda/lib64/libcudart.so" in configure
    assert "-DTRTMC_CUBLAS_LIBRARY=/cuda/lib64/libcublas.so" in configure
    assert not any(command[1:4] == ["-m", "pip", "install"] for command in runner.commands)
    receipt = json.loads(build.receipt.read_text(encoding="utf-8"))
    assert receipt["environment_id"] == environment.environment_id
    assert receipt["source"]["revision"] == "b" * 40
    assert receipt["artifacts"][0]["sha256"] == "c" * 64
    source_git = next(command for command in runner.commands if "rev-parse" in command)
    assert source_git[:3] == [
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
    ]

    command = toolkit.run_trtmc(environment, ("version",), build=build)
    command_receipt = json.loads(command.receipt.read_text(encoding="utf-8"))
    assert command_receipt["provenance"] == {
        "build_id": build.build_id,
        "artifact:trtmc": "c" * 64,
    }
    assert runner.commands[-1][0].endswith(f"builds/{build.build_request_id}/build/trtmc")

    runner.artifact_digest = "d" * 64
    with pytest.raises(DevToolkitError, match="changed after build"):
        toolkit.run_trtmc(environment, ("version",), build=build)
    assert runner.commands[-1][0] == "sha256sum"


def test_native_recipe_resolves_thor_architecture_through_cuda_driver() -> None:
    runtime = ToolchainRuntime(
        python_executable="/managed/venv/bin/python",
        cuda_root="/managed/cuda",
        nvcc="/managed/cuda/bin/nvcc",
        tensorrt_include_dir="/managed/include",
        tensorrt_library="/managed/lib/libnvinfer.so",
    )

    def probe(command: CommandSpec) -> str:
        arguments = [str(argument) for argument in command.arguments]
        script = arguments[2] if len(arguments) > 2 and arguments[1] == "-c" else ""
        if "cuDeviceGetAttribute" in script:
            return "110\n"
        if "CUDA runtime headers are missing" in script:
            return json.dumps(
                {
                    "include": "/managed/cuda/include",
                    "cudart": "/managed/cuda/lib/libcudart.so",
                    "cublas": "/managed/cuda/lib/libcublas.so",
                }
            )
        if "shutil.which" in script:
            return "Unix Makefiles\n"
        if arguments[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise AssertionError(arguments)

    inputs = TrtmcBuildRecipe().inputs(
        BuildContext(runtime=runtime, architecture="aarch64", _probe=probe)
    )

    assert inputs["cuda_architectures"] == ("110",)
    assert inputs["generator"] == "Unix Makefiles"


def test_native_recipe_auto_generator_prefers_ninja_when_available() -> None:
    runtime = ToolchainRuntime(
        python_executable="python3",
        cuda_root="/cuda",
        nvcc="/cuda/bin/nvcc",
        tensorrt_include_dir="/trt/include",
        tensorrt_library="/trt/lib/libnvinfer.so",
    )

    generator_probes: list[list[str]] = []

    def probe(command: CommandSpec) -> str:
        script = str(command.arguments[2])
        if "CUDA runtime headers are missing" in script:
            return json.dumps(
                {
                    "include": "/cuda/include",
                    "cudart": "/cuda/lib/libcudart.so",
                    "cublas": "/cuda/lib/libcublas.so",
                }
            )
        if "shutil.which" in script:
            generator_probes.append([str(argument) for argument in command.arguments])
            return "Ninja\n"
        raise AssertionError(command.arguments)

    inputs = TrtmcBuildRecipe(cuda_architectures=("110",)).inputs(
        BuildContext(runtime=runtime, architecture="aarch64", _probe=probe)
    )

    assert inputs["generator"] == "Ninja"
    assert len(generator_probes) == 1


def test_native_build_records_attestation_preflight_failure(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(FailingReattestationSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    with pytest.raises(RuntimeError, match="super-secret"):
        toolkit.build(environment, TrtmcBuildRecipe(cuda_architectures=("100",)))

    receipts = list((environment.state_dir / "builds" / "preflight").glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["stage"] == "attestation"
    assert payload["environment_id"] == environment.environment_id
    assert payload["error_type"] == "RuntimeError"
    assert "super-secret" not in receipts[0].read_text(encoding="utf-8")


def test_identical_builds_are_serialized(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = BlockingBuildRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)
    recipe = TrtmcBuildRecipe(cuda_architectures=("100",))
    results = []

    first = threading.Thread(target=lambda: results.append(toolkit.build(environment, recipe)))
    second = threading.Thread(target=lambda: results.append(toolkit.build(environment, recipe)))
    first.start()
    assert runner.first_configure_entered.wait(timeout=5)
    second.start()
    assert not runner.second_configure_entered.wait(timeout=0.2)
    runner.release_first_configure.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not runner.second_configure_entered.is_set()
    assert len(results) == 2
    assert results[0].build_request_id == results[1].build_request_id


def test_completed_build_receipt_skips_reexecution(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = CommandRecordingRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)
    recipe = TrtmcBuildRecipe(cuda_architectures=("100",), generator="Ninja")

    first = toolkit.build(environment, recipe)
    first_configures = sum(command[:2] == ["cmake", "-S"] for command in runner.commands)
    first_builds = sum(command[:2] == ["cmake", "--build"] for command in runner.commands)
    second = toolkit.build(environment, recipe)

    assert second.build_id == first.build_id
    assert sum(command[:2] == ["cmake", "-S"] for command in runner.commands) == first_configures
    assert sum(command[:2] == ["cmake", "--build"] for command in runner.commands) == first_builds


def test_qualification_is_optional_provenance_not_an_allowlist(
    tmp_path: Path,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "qualified.json").write_text(
        json.dumps(
            {
                "id": "qualified-trt",
                "status": "supported",
                "requirements": {
                    "execution": ["local"],
                    "tensorrt": "11.2.0.113",
                    "cuda": "12.8",
                    "python": ["3.12"],
                    "architecture": ["aarch64"],
                },
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualifications=(JsonQualificationSource((presets,)),),
    )

    qualified = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
            preset="qualified-trt",
            require_qualification=True,
        )
    )
    unrestricted = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    assert qualified.qualifications[0].name == "qualified-trt"
    assert len(qualified.qualifications[0].digest) == 64
    assert qualified.lock_id == unrestricted.lock_id


def test_required_qualification_fails_without_restricting_default_resolution(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())
    request = EnvironmentRequest(
        tensorrt="11.2.0.113",
        target=ExecutionTarget("test-local"),
        architecture="aarch64",
    )

    assert toolkit.resolve(request).qualifications == ()
    with pytest.raises(IncompatibleCombination, match="qualification"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt=request.tensorrt,
                target=request.target,
                architecture=request.architecture,
                require_qualification=True,
            )
        )


def test_invalid_optional_qualification_metadata_does_not_gate_resolution(
    tmp_path: Path,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "broken.json").write_text("{not-json", encoding="utf-8")
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualifications=(JsonQualificationSource((presets,)),),
    )
    request = EnvironmentRequest(
        tensorrt="11.2.0.113",
        target=ExecutionTarget("test-local"),
        architecture="aarch64",
    )

    assert toolkit.resolve(request).qualifications == ()
    with pytest.raises(DevToolkitError, match="Invalid qualification"):
        toolkit.resolve(replace(request, require_qualification=True))


@pytest.mark.parametrize("invalid", ([], {"python": 3.12}, {"python": [3.12]}))
def test_qualification_rejects_invalid_requirement_shapes(
    tmp_path: Path,
    invalid: object,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    payload = {
        "id": "malformed",
        "status": "supported",
        "requirements": invalid,
    }
    (presets / "malformed.json").write_text(json.dumps(payload), encoding="utf-8")
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualifications=(JsonQualificationSource((presets,)),),
    )

    with pytest.raises(DevToolkitError, match="Invalid qualification"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                target=ExecutionTarget("test-local"),
                architecture="aarch64",
                require_qualification=True,
            )
        )


def test_builtin_managed_source_accepts_arbitrary_trt_with_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    artifacts = (
        ArtifactPin(
            name="tensorrt-headers",
            uri="https://example.invalid/libnvinfer-headers.deb",
            sha256="d" * 64,
        ),
        ArtifactPin(
            name="tensorrt-wheel",
            uri="https://example.invalid/tensorrt-11.2.0.113.whl?token=super-secret",
            sha256="e" * 64,
        ),
        ArtifactPin(
            name="cuda-component-nvcc",
            uri="https://example.invalid/cuda-nvcc.tar.xz",
            sha256="f" * 64,
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
            toolchain_options={"cuda_artifacts": ["cuda-component-nvcc"]},
        )
    )

    assert lock.toolchain.provider.name == "managed-artifacts"
    assert lock.toolchain.cuda == "13.3"
    assert lock.cuda_origin == "managed-default"
    assert lock.toolchain.artifacts == artifacts
    assert "super-secret" not in json.dumps(lock.as_dict())


def test_builtin_managed_source_materializes_and_attests_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    headers = tmp_path / "libnvinfer-headers.deb"
    wheel = tmp_path / "tensorrt-11.2.0.113-py3-none-any.whl"
    cuda = tmp_path / "cuda-nvcc.tar.xz"
    headers.write_bytes(b"pinned headers")
    wheel.write_bytes(b"pinned wheel")
    cuda.write_bytes(b"pinned CUDA")

    def pin(name: str, path: Path) -> ArtifactPin:
        return ArtifactPin(
            name,
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=ManagedProvisionRunner(tmp_path),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=(
                pin("tensorrt-headers", headers),
                pin("tensorrt-wheel", wheel),
                pin("cuda-component-nvcc", cuda),
            ),
            toolchain_options={"cuda_artifacts": ["cuda-component-nvcc"]},
        )
    )

    environment = toolkit.provision(lock)

    assert environment.observation.cuda_version == "13.3"
    assert environment.observation.tensorrt_native_version == "11.2.0.113"
    assert environment.context.environment["CUDA_HOME"] == str(tmp_path / "managed-cuda")
    receipt = json.loads(environment.receipt.read_text(encoding="utf-8"))
    assert receipt["observed"]["tensorrt_header_version"] == "11.2.0.113"


def test_builtin_managed_source_rejects_an_incomplete_cuda_toolkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    headers = tmp_path / "headers.deb"
    wheel = tmp_path / "tensorrt.whl"
    cuda = tmp_path / "cuda-nvcc.tar.xz"
    headers.write_bytes(b"headers")
    wheel.write_bytes(b"wheel")
    cuda.write_bytes(b"CUDA")
    artifacts = tuple(
        ArtifactPin(
            name,
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for name, path in (
            ("tensorrt-headers", headers),
            ("tensorrt-wheel", wheel),
            ("cuda-component-nvcc", cuda),
        )
    )
    runner = ManagedProvisionRunner(tmp_path)
    (runner.cuda / "lib" / "libcurand.so").unlink()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
            toolchain_options={"cuda_artifacts": ["cuda-component-nvcc"]},
        )
    )

    with pytest.raises(DevToolkitError, match="incomplete CUDA"):
        toolkit.provision(lock)


def test_managed_tensorrt_uses_complete_system_cuda_before_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = tmp_path / "cuda"
    for directory in (cuda / "bin", cuda / "include", cuda / "lib64"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib64/libcudart.so",
        "lib64/libcublas.so",
        "lib64/libcurand.so",
    ):
        (cuda / relative).touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("TRTMC_TRT_INCLUDE_DIR", str(tmp_path / "missing-trt"))
    artifacts = (
        ArtifactPin(
            "tensorrt-headers",
            "https://example.invalid/headers.deb",
            "1" * 64,
        ),
        ArtifactPin(
            "tensorrt-wheel",
            "https://example.invalid/tensorrt.whl",
            "2" * 64,
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
        )
    )

    assert lock.toolchain.origin == "managed"
    assert lock.toolchain.cuda_source == "system"
    assert lock.cuda == "12.8"
    assert lock.cuda_origin == "system"


def test_environment_identity_depends_on_resolution_not_request_spelling(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    providers = registry.freeze()
    toolkit_a = DevToolkit.from_checkout(tmp_path, providers=providers)
    toolkit_b = DevToolkit.from_checkout(tmp_path, providers=providers)
    common = {
        "tensorrt": "11.2.0.113",
        "target": ExecutionTarget("test-local"),
        "architecture": "aarch64",
    }

    default_lock = toolkit_a.resolve(EnvironmentRequest(**common))
    explicit_lock = toolkit_b.resolve(EnvironmentRequest(**common, cuda=CudaPolicy.exact("12.8")))

    assert default_lock.lock_id == explicit_lock.lock_id
    assert default_lock.cuda_origin == "system"
    assert explicit_lock.cuda_origin == "explicit"


def test_builtin_prefix_provider_uses_user_owned_toolchain_root(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "toolchain"
    for directory in (prefix / "bin", prefix / "include", prefix / "lib"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib/libcudart.so",
        "lib/libcublas.so",
        "lib/libcurand.so",
        "lib/libnvinfer.so",
    ):
        (prefix / relative).touch()
    major, minor, patch, build = "11.2.0.113".split(".")
    (prefix / "include" / "NvInferVersion.h").write_text(
        "\n".join(
            (
                f"#define NV_TENSORRT_MAJOR {major}",
                f"#define NV_TENSORRT_MINOR {minor}",
                f"#define NV_TENSORRT_PATCH {patch}",
                f"#define NV_TENSORRT_BUILD {build}",
            )
        ),
        encoding="utf-8",
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            toolchain="prefix",
            toolchain_options={"prefix": str(prefix)},
        )
    )

    assert lock.toolchain.provider.name == "prefix"
    assert lock.toolchain.origin == "prefix"
    assert lock.toolchain.identity["cuda_root"] == str(prefix)


def test_managed_tensorrt_follows_cuda_from_explicit_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "cuda"
    for directory in (prefix / "bin", prefix / "include", prefix / "lib"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib/libcudart.so",
        "lib/libcublas.so",
        "lib/libcurand.so",
    ):
        (prefix / relative).touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    artifacts = (
        ArtifactPin(
            "tensorrt-headers",
            "https://example.invalid/headers.deb",
            "3" * 64,
        ),
        ArtifactPin(
            "tensorrt-wheel",
            "https://example.invalid/tensorrt.whl",
            "4" * 64,
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            toolchain_options={"cuda_prefix": str(prefix)},
            artifacts=artifacts,
        )
    )

    assert lock.toolchain.provider.name == "managed-artifacts"
    assert lock.toolchain.cuda_source == "prefix"
    assert lock.toolchain.cuda == "12.8"
    assert lock.cuda_origin == "system"
