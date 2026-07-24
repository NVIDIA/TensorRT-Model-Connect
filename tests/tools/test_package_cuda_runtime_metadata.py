# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools.ci.package import WheelArchiveValidator
from tools.ci.process import CiError


PINNED_REQUIREMENTS = (
    "tensorrt==11.2.0.113",
    "apache-tvm-ffi==0.1.12",
    "cuda-python==13.3.1",
    "nvidia-cuda-runtime==13.3.29",
    "nvidia-cuda-nvrtc==13.3.33",
    "nvidia-cudnn-cu13==9.20.0.48",
)


class _AuditContext:
    def output(self, _command: list[object]) -> str:
        return "platform tag: manylinux_2_34_aarch64"


def _write_test_wheel(tmp_path: Path, requirements: tuple[str, ...]) -> Path:
    platform = "manylinux_2_34_aarch64"
    wheel = tmp_path / f"tensorrt_model_connect-0.1.0-py3-none-{platform}.whl"
    files = {
        "tensorrt_model_connect/bin/trtmc": b"\x7fELF",
        "tensorrt_model_connect/bin/trtmc_benchmark_worker": b"\x7fELF",
        "tensorrt_model_connect/bin/libtrtmc_core.so": b"\x7fELF",
        "tensorrt_model_connect/bin/libtrtmc_backend_trt.so": b"\x7fELF",
        "tensorrt_model_connect/bin/libtrtmc_trt_plugins.so": b"\x7fELF",
        "tensorrt_model_connect/benchmark/_catalog/example/MODEL.toml": b"",
        "tensorrt_model_connect/benchmark/_catalog/example/manifests/example.json": b"{}",
        "tensorrt_model_connect/benchmark/_catalog/example/data/Recording.wav": b"",
        "tensorrt_model_connect/benchmark/_catalog/example/data/flux2-fp8-scales.json": b"{}",
        "tensorrt_model_connect/benchmark/_catalog/example/data/test_img.jpeg": b"",
        "tensorrt_model_connect-0.1.0.data/scripts/trtmc": b"\x7fELF",
        "tensorrt_model_connect-0.1.0.data/scripts/trtmc-bench": b"",
        "tensorrt_model_connect-0.1.0.data/scripts/libtrtmc_core.so": b"\x7fELF",
        "tensorrt_model_connect-0.1.0.dist-info/METADATA": (
            "\n".join(f"Requires-Dist: {requirement}" for requirement in requirements)
        ).encode(),
        "tensorrt_model_connect-0.1.0.dist-info/WHEEL": (
            f"Tag: py3-none-{platform}\n"
        ).encode(),
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return wheel


@pytest.mark.dynamic_memory
def test_wheel_archive_accepts_exact_cuda_13_3_runtime_metadata(
    tmp_path: Path,
) -> None:
    wheel = _write_test_wheel(tmp_path, PINNED_REQUIREMENTS)
    WheelArchiveValidator(_AuditContext(), "manylinux_2_34_aarch64").validate(
        [wheel]
    )


@pytest.mark.dynamic_memory
@pytest.mark.parametrize(
    ("missing", "message"),
    (
        ("cuda-python==13.3.1", "pinned CUDA Python 13.3.1"),
        ("nvidia-cuda-runtime==13.3.29", "pinned CUDA runtime 13.3.29"),
        ("nvidia-cuda-nvrtc==13.3.33", "pinned CUDA NVRTC 13.3.33"),
    ),
)
def test_wheel_archive_rejects_missing_exact_cuda_13_3_runtime_metadata(
    tmp_path: Path, missing: str, message: str
) -> None:
    requirements = tuple(
        requirement for requirement in PINNED_REQUIREMENTS if requirement != missing
    )
    wheel = _write_test_wheel(tmp_path, requirements)
    with pytest.raises(CiError, match=message):
        WheelArchiveValidator(
            _AuditContext(), "manylinux_2_34_aarch64"
        ).validate([wheel])
