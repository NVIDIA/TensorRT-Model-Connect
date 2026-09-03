# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional build recipes composed with the generic DevToolkit builder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType

from .building import BuildContext, BuildPlan
from .commands import CommandSpec, EnvironmentPath, repository_path
from .models import DevToolkitError


def _define_value(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


_CUDA_BUILD_INPUTS_SCRIPT = r"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
include = root / "include"
if not (include / "cuda_runtime_api.h").is_file():
    raise SystemExit("CUDA runtime headers are missing")
def library(name):
    preferred = (root / "lib64" / name, root / "lib" / name)
    matches = [path for path in preferred if path.is_file()]
    if not matches:
        matches = [path for path in root.rglob(name) if path.is_file()]
    if not matches:
        raise SystemExit(f"CUDA library {name} is missing")
    return str(matches[0].resolve())
print(json.dumps({
    "include": str(include),
    "cudart": library("libcudart.so"),
    "cublas": library("libcublas.so"),
}))
""".strip()


@dataclass(frozen=True)
class TrtmcBuildRecipe:
    """Sample recipe for the repository's native TRTMC targets."""

    descriptor: str = field(default="trtmc-native==2", init=False)
    targets: tuple[str, ...] = ("trtmc", "trtmc_backend_trt")
    cmake_defines: Mapping[str, str | int | bool] = field(default_factory=dict)
    cuda_architectures: tuple[str, ...] | None = None
    build_type: str = "Release"
    generator: str = "Ninja"
    jobs: int | None = None
    outputs: Mapping[str, str] = field(default_factory=lambda: {"trtmc": "trtmc"})

    def __post_init__(self) -> None:
        if not self.targets or any(not target for target in self.targets):
            raise DevToolkitError("A TRTMC build requires non-empty targets")
        if self.cuda_architectures is not None and not self.cuda_architectures:
            raise DevToolkitError("cuda_architectures cannot be empty")
        if self.jobs is not None and self.jobs < 1:
            raise DevToolkitError("Build jobs must be positive")
        for name, relative in self.outputs.items():
            path = PurePosixPath(relative)
            if not name or path.is_absolute() or ".." in path.parts:
                raise DevToolkitError("Build output paths must be named, safe relative paths")
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.cuda_architectures is not None:
            object.__setattr__(self, "cuda_architectures", tuple(self.cuda_architectures))
        object.__setattr__(self, "cmake_defines", MappingProxyType(dict(self.cmake_defines)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def inputs(self, context: BuildContext) -> Mapping[str, object]:
        architectures = self.cuda_architectures
        if architectures is None:
            output = context.probe(
                CommandSpec(
                    (
                        "nvidia-smi",
                        "--query-gpu=compute_cap",
                        "--format=csv,noheader,nounits",
                    )
                )
            )
            architectures = tuple(
                line.strip().replace(".", "") for line in output.splitlines() if line.strip()
            )
        if not architectures:
            raise DevToolkitError("Could not resolve a CUDA architecture for the TRTMC recipe")
        try:
            cuda = json.loads(
                context.probe(
                    CommandSpec(
                        (
                            context.runtime.python_executable,
                            "-c",
                            _CUDA_BUILD_INPUTS_SCRIPT,
                            context.runtime.cuda_root,
                        )
                    )
                )
            )
            cuda_inputs = {name: str(cuda[name]) for name in ("include", "cudart", "cublas")}
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DevToolkitError(f"Could not resolve locked CUDA build inputs: {error}") from error
        return {
            "targets": self.targets,
            "cmake_defines": dict(self.cmake_defines),
            "cuda_architectures": architectures,
            "build_type": self.build_type,
            "generator": self.generator,
            "jobs": self.jobs,
            "outputs": dict(self.outputs),
            "cuda": cuda_inputs,
        }

    def plan(
        self,
        context: BuildContext,
        inputs: Mapping[str, object],
        build_dir: EnvironmentPath,
    ) -> BuildPlan:
        architectures = tuple(str(value) for value in inputs["cuda_architectures"])
        commands: list[CommandSpec] = []
        defines: dict[str, str | int | bool] = {
            "TRTMC_BUILD_BACKEND_TRT": True,
            "TRTMC_BUILD_BACKEND_RTX": False,
            **dict(self.cmake_defines),
            "CMAKE_CUDA_ARCHITECTURES": ";".join(
                item if not item.isdigit() else f"{item}-real" for item in architectures
            ),
            "TRTMC_TRT_INCLUDE_DIR": context.runtime.tensorrt_include_dir,
            "TRTMC_TRT_LIBRARY": context.runtime.tensorrt_library,
            "TRTMC_CUDA_INCLUDE_DIR": str(inputs["cuda"]["include"]),
            "TRTMC_CUDART_LIBRARY": str(inputs["cuda"]["cudart"]),
            "TRTMC_CUBLAS_LIBRARY": str(inputs["cuda"]["cublas"]),
        }
        configure: list[str | EnvironmentPath] = [
            "cmake",
            "-S",
            repository_path("."),
            "-B",
            build_dir,
            "-G",
            self.generator,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
        ]
        configure.extend(
            f"-D{name}={_define_value(value)}" for name, value in sorted(defines.items())
        )
        commands.append(CommandSpec(configure))
        build_command: list[str | EnvironmentPath] = [
            "cmake",
            "--build",
            build_dir,
            "--parallel",
        ]
        if self.jobs is not None:
            build_command.append(str(self.jobs))
        build_command.extend(("--target", *self.targets))
        commands.append(CommandSpec(build_command))
        return BuildPlan(
            commands=tuple(commands),
            outputs={
                name: EnvironmentPath(build_dir.scope, build_dir.path / relative)
                for name, relative in self.outputs.items()
            },
        )
