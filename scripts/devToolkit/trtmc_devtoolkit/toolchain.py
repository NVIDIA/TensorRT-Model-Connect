# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-owned CUDA/TensorRT toolchain provisioning for LocalTarget."""

from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import DevToolkitError, DownloadArtifact, PreparationPlan
from .runner import Runner, command_output


CUDA_RELEASE = re.compile(r"release\s+([0-9]+\.[0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ManagedToolchain:
    cuda_root: Path
    tensorrt_include_dir: Path
    tensorrt_library_dir: Path
    tensorrt_library: Path
    cudart_library: Path
    cublas_library: Path

    def environment(self, venv: Path, *, gpu: str) -> dict[str, str]:
        library_path = ":".join(
            (
                str(self.tensorrt_library_dir),
                str(self.cuda_root / "lib"),
            )
        )
        return {
            "PATH": f"{venv / 'bin'}:{self.cuda_root / 'bin'}:{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(venv),
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_HOME": str(self.cuda_root),
            "CUDA_PATH": str(self.cuda_root),
            "CUDAToolkit_ROOT": str(self.cuda_root),
            "TRTMC_CUDA_INCLUDE_DIR": str(self.cuda_root / "include"),
            "TRTMC_CUDART_LIBRARY": str(self.cudart_library),
            "TRTMC_CUBLAS_LIBRARY": str(self.cublas_library),
            "TRTMC_TRT_INCLUDE_DIR": str(self.tensorrt_include_dir),
            "TRTMC_TRT_LIBRARY": str(self.tensorrt_library),
            "TRTMC_TRT_LIBRARY_DIR": str(self.tensorrt_library_dir),
            "TRT_INC_DIR": str(self.tensorrt_include_dir),
            "TRT_LIB_DIR": str(self.tensorrt_library_dir),
            "LD_LIBRARY_PATH": library_path,
        }


class ManagedLocalProvisioner:
    def __init__(self, repository: Path, runner: Runner):
        self.repository = repository.resolve()
        self.runner = runner

    def prepare(self, plan: PreparationPlan, venv: Path) -> ManagedToolchain:
        python = venv / "bin" / "python"
        requirements = [
            f"{package.name}=={package.version}"
            for package in plan.cohort.managed_local.python_packages
        ]
        self.runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *requirements,
            ],
            cwd=self.repository,
        )
        self.runner.run(
            [python, "-m", "pip", "check"],
            cwd=self.repository,
        )

        contract = plan.cohort.architectures[plan.architecture]
        include_dir = self._prepare_headers(plan, contract.tensorrt_headers)
        cuda_root = Path(
            command_output(
                self.runner,
                [
                    python,
                    "-c",
                    (
                        "from pathlib import Path; import nvidia.cu13; "
                        "print(Path(next(iter(nvidia.cu13.__path__))).resolve())"
                    ),
                ],
                cwd=self.repository,
            )
        )
        tensorrt_library_dir = Path(
            command_output(
                self.runner,
                [
                    python,
                    "-c",
                    (
                        "import importlib.metadata as m; from pathlib import Path; "
                        "print(Path(m.distribution('tensorrt_cu13_libs')"
                        ".locate_file('tensorrt_libs')).resolve())"
                    ),
                ],
                cwd=self.repository,
            )
        )
        self._require_directory(cuda_root, "managed CUDA root")
        self._require_directory(tensorrt_library_dir, "managed TensorRT library directory")

        cudart = self._ensure_linker_name(cuda_root / "lib", "libcudart.so", "13")
        cublas = self._ensure_linker_name(cuda_root / "lib", "libcublas.so", "13")
        for linker_name, major in (
            ("libcublasLt.so", "13"),
            ("libcurand.so", "10"),
        ):
            self._ensure_linker_name(cuda_root / "lib", linker_name, major)
        tensorrt = self._ensure_linker_name(
            tensorrt_library_dir,
            "libnvinfer.so",
            plan.cohort.tensorrt_version.split(".", 1)[0],
        )
        for linker_name in (
            "libnvinfer_plugin.so",
            "libnvonnxparser.so",
        ):
            self._ensure_linker_name(
                tensorrt_library_dir,
                linker_name,
                plan.cohort.tensorrt_version.split(".", 1)[0],
            )

        toolchain = ManagedToolchain(
            cuda_root=cuda_root,
            tensorrt_include_dir=include_dir,
            tensorrt_library_dir=tensorrt_library_dir,
            tensorrt_library=tensorrt,
            cudart_library=cudart,
            cublas_library=cublas,
        )
        self._verify(plan, python, toolchain)
        return toolchain

    def _prepare_headers(
        self,
        plan: PreparationPlan,
        artifact: DownloadArtifact,
    ) -> Path:
        cache = plan.state_dir.parent / "downloads"
        filename = Path(urllib.parse.urlparse(artifact.url).path).name
        if not filename:
            raise DevToolkitError(f"TensorRT header URL has no filename: {artifact.url}")
        archive = cache / f"{artifact.sha256[:16]}-{filename}"
        self._download(artifact, archive)
        root = plan.state_dir / "managed-toolchain" / "tensorrt-headers"
        marker = root / ".artifact-sha256"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != artifact.sha256:
            root.mkdir(parents=True, exist_ok=True)
            self.runner.run(
                ["dpkg-deb", "--extract", archive, root],
                cwd=self.repository,
            )
            marker.write_text(artifact.sha256 + "\n", encoding="utf-8")
        matches = sorted(root.glob("usr/include/*-linux-gnu/NvInferVersion.h"))
        if len(matches) != 1:
            raise DevToolkitError(
                f"Expected one NvInferVersion.h in managed header artifact, found {matches}"
            )
        return matches[0].parent

    def _download(self, artifact: DownloadArtifact, destination: Path) -> None:
        if destination.is_file() and self._sha256(destination) == artifact.sha256:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + f".partial-{os.getpid()}")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(artifact.url, timeout=120) as response:
                with partial.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        stream.write(chunk)
            actual = digest.hexdigest()
            if actual != artifact.sha256:
                raise DevToolkitError(
                    f"TensorRT header artifact checksum mismatch: expected "
                    f"{artifact.sha256}, got {actual}"
                )
            partial.replace(destination)
        except (OSError, urllib.error.URLError) as error:
            raise DevToolkitError(
                f"Could not download TensorRT header artifact {artifact.url}: {error}"
            ) from error
        finally:
            if partial.exists():
                partial.unlink()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _require_directory(path: Path, name: str) -> None:
        if not path.is_dir():
            raise DevToolkitError(f"Could not locate {name}: {path}")

    @staticmethod
    def _ensure_linker_name(directory: Path, linker_name: str, major: str) -> Path:
        linker = directory / linker_name
        versioned = directory / f"{linker_name}.{major}"
        if not versioned.is_file():
            raise DevToolkitError(f"Managed toolchain is missing {versioned}")
        if linker.is_symlink() and linker.resolve() == versioned.resolve():
            return linker
        if linker.exists() or linker.is_symlink():
            raise DevToolkitError(
                f"Managed toolchain linker name does not select {versioned}: {linker}"
            )
        linker.symlink_to(versioned.name)
        return linker

    def _verify(
        self,
        plan: PreparationPlan,
        python: Path,
        toolchain: ManagedToolchain,
    ) -> None:
        environment = toolchain.environment(
            python.parent.parent,
            gpu=plan.request.target.gpu,
        )
        nvcc_output = command_output(
            self.runner,
            [toolchain.cuda_root / "bin" / "nvcc", "--version"],
            cwd=self.repository,
            env=environment,
        )
        match = CUDA_RELEASE.search(nvcc_output)
        if match is None or match.group(1) != plan.cohort.cuda_version:
            raise DevToolkitError(
                f"Managed nvcc must report CUDA {plan.cohort.cuda_version}; got {nvcc_output}"
            )
        python_version = command_output(
            self.runner,
            [python, "-c", "import tensorrt; print(tensorrt.__version__)"],
            cwd=self.repository,
            env=environment,
        )
        if python_version != plan.cohort.tensorrt_version:
            raise DevToolkitError(
                f"Managed TensorRT Python must be {plan.cohort.tensorrt_version}; "
                f"got {python_version}"
            )
        native_version = command_output(
            self.runner,
            [
                python,
                "-c",
                (
                    "import ctypes; "
                    f"lib=ctypes.CDLL({str(toolchain.tensorrt_library)!r}); "
                    "names=('Major','Minor','Patch','Build'); "
                    "fs=[getattr(lib, f'getInferLib{name}Version') for name in names]; "
                    "[setattr(f, 'restype', ctypes.c_int32) for f in fs]; "
                    "print('.'.join(str(f()) for f in fs))"
                ),
            ],
            cwd=self.repository,
            env=environment,
        )
        if native_version != plan.cohort.tensorrt_version:
            raise DevToolkitError(
                f"Managed TensorRT native library must be {plan.cohort.tensorrt_version}; "
                f"got {native_version}"
            )
        header_version = self._header_version(toolchain.tensorrt_include_dir / "NvInferVersion.h")
        if header_version != plan.cohort.tensorrt_version:
            raise DevToolkitError(
                f"Managed TensorRT headers must be {plan.cohort.tensorrt_version}; "
                f"got {header_version}"
            )

    @staticmethod
    def _header_version(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        definitions = dict(
            re.findall(r"^#define\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b", text, re.MULTILINE)
        )
        parts: list[str] = []
        for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
            value = definitions.get(f"NV_TENSORRT_{name}", "")
            value = definitions.get(value, value)
            if not value.isdigit():
                raise DevToolkitError(f"Could not resolve TensorRT {name.lower()} from {path}")
            parts.append(value)
        return ".".join(parts)
