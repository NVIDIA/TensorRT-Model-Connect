# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed host and local-toolchain environment probes."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .models import DevToolkitError, EnvironmentCohort, PrepareRequest, ProbeResult
from .runner import Runner, command_output


CUDA_RELEASE = re.compile(r"release\s+([0-9]+\.[0-9]+)", re.IGNORECASE)


def _native_tensorrt_version_script(library: Path) -> str:
    return (
        "import ctypes; "
        f"lib=ctypes.CDLL({str(library)!r}); "
        "names=('Major','Minor','Patch','Build'); "
        "funcs=[getattr(lib, f'getInferLib{name}Version') for name in names]; "
        "[setattr(func, 'restype', ctypes.c_int32) for func in funcs]; "
        "print('.'.join(str(func()) for func in funcs))"
    )


def _probe_command(
    runner: Runner,
    command: list[str],
    *,
    cwd: Path,
    name: str,
) -> tuple[ProbeResult, str]:
    try:
        output = command_output(runner, command, cwd=cwd, timeout=30)
    except (DevToolkitError, OSError) as error:
        return ProbeResult(name, "fail", str(error)), ""
    return ProbeResult(name, "pass", output or "available"), output


class EnvironmentDoctor:
    def __init__(self, repository: Path, runner: Runner):
        self.repository = repository.resolve()
        self.runner = runner

    def inspect(
        self,
        request: PrepareRequest,
        cohort: EnvironmentCohort,
        architecture: str,
    ) -> tuple[tuple[ProbeResult, ...], str]:
        results: list[ProbeResult] = []
        selected = cohort.architectures[architecture]
        if shutil.disk_usage(self.repository).free < 20 * 1024**3:
            results.append(
                ProbeResult("disk-space", "warning", "less than 20 GiB is free")
            )
        else:
            results.append(ProbeResult("disk-space", "pass", "at least 20 GiB is free"))

        gpu_result, gpu_output = _probe_command(
            self.runner,
            [
                "nvidia-smi",
                "-i",
                request.target.gpu,
                "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd=self.repository,
            name="gpu",
        )
        results.append(gpu_result)
        sm = ""
        if gpu_output:
            parts = [part.strip() for part in gpu_output.splitlines()[0].split(",")]
            if len(parts) >= 4:
                sm = parts[3].replace(".", "")
        if request.target.kind == "docker":
            docker_result, _ = _probe_command(
                self.runner,
                ["docker", "version", "--format", "{{.Server.Version}}"],
                cwd=self.repository,
                name="docker",
            )
            results.append(docker_result)
        else:
            for name, command in (
                ("cmake", ["cmake", "--version"]),
                ("ninja", ["ninja", "--version"]),
            ):
                result, _ = _probe_command(
                    self.runner, command, cwd=self.repository, name=name
                )
                results.append(result)
            python_result, python_output = _probe_command(
                self.runner,
                [
                    request.target.python,
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                cwd=self.repository,
                name="python",
            )
            if (
                python_result.status == "pass"
                and python_output != request.python_version
            ):
                python_result = ProbeResult(
                    "python",
                    "fail",
                    f"requested {request.python_version}; found {python_output}",
                )
            results.append(python_result)
            nvcc_result, nvcc_output = _probe_command(
                self.runner,
                ["nvcc", "--version"],
                cwd=self.repository,
                name="cuda-toolkit",
            )
            if nvcc_result.status == "pass":
                match = CUDA_RELEASE.search(nvcc_output)
                if match is None or match.group(1) != cohort.cuda_version:
                    nvcc_result = ProbeResult(
                        "cuda-toolkit",
                        "fail",
                        f"requested CUDA {cohort.cuda_version}; nvcc reported {nvcc_output}",
                    )
            results.append(nvcc_result)
            trt_result, trt_output = _probe_command(
                self.runner,
                [
                    request.target.python,
                    "-c",
                    "import tensorrt; print(tensorrt.__version__)",
                ],
                cwd=self.repository,
                name="tensorrt-python",
            )
            if trt_result.status == "pass" and trt_output != cohort.tensorrt_version:
                trt_result = ProbeResult(
                    "tensorrt-python",
                    "fail",
                    f"requested {cohort.tensorrt_version}; found {trt_output}",
                )
            results.append(trt_result)
            library = Path(selected.tensorrt_library_dir) / "libnvinfer.so"
            native_result, native_output = _probe_command(
                self.runner,
                [
                    request.target.python,
                    "-c",
                    _native_tensorrt_version_script(library),
                ],
                cwd=self.repository,
                name="tensorrt-native",
            )
            if (
                native_result.status == "pass"
                and native_output != cohort.tensorrt_version
            ):
                native_result = ProbeResult(
                    "tensorrt-native",
                    "fail",
                    f"requested {cohort.tensorrt_version}; found {native_output}",
                )
            results.append(native_result)
            include_dir = Path(
                os.environ.get("TRTMC_TRT_INCLUDE_DIR")
                or os.environ.get("TRT_INC_DIR")
                or selected.tensorrt_include_dir
            )
            header = include_dir / "NvInferVersion.h"
            results.append(
                ProbeResult(
                    "tensorrt-headers",
                    "pass" if header.is_file() else "fail",
                    str(header),
                )
            )
        failures = [result for result in results if result.status == "fail"]
        if failures:
            detail = "; ".join(f"{result.name}: {result.detail}" for result in failures)
            raise DevToolkitError(f"Environment doctor failed: {detail}")
        if not sm:
            raise DevToolkitError("Environment doctor could not resolve the selected GPU SM")
        return tuple(results), sm
