# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci.package import (
    _probe_backend_identity,
    _required_tensorrt_version,
    _validate_backend_files,
    _validate_backend_identity,
)
from tools.ci.process import CiError


TENSORRT_VERSION = "11.1.0.106"
TENSORRT_METADATA = """\
Metadata-Version: 2.4
Name: tensorrt-model-connect
Version: 0.1.0
Requires-Dist: tensorrt==11.1.0.106; platform_machine == "aarch64"
Requires-Dist: tensorrt==11.1.0.106; platform_machine == "x86_64"
"""


def _backend_files(*names: str, distinct: bool = False) -> dict[str, bytes]:
    return {
        name: (f"payload-{index}".encode() if distinct else b"same-backend")
        for index, name in enumerate(names)
    }


def test_tensorrt_requirement_resolves_one_exact_version() -> None:
    assert _required_tensorrt_version(TENSORRT_METADATA) == TENSORRT_VERSION


def test_tensorrt_requirement_rejects_mixed_versions() -> None:
    metadata = TENSORRT_METADATA.replace(
        '11.1.0.106; platform_machine == "x86_64"',
        '11.2.0.113; platform_machine == "x86_64"',
    )

    with pytest.raises(CiError, match="one exact TensorRT dependency version"):
        _required_tensorrt_version(metadata)


@pytest.mark.parametrize(
    "requirement",
    ("tensorrt>=11.2", "tensorrt @ https://example.invalid/tensorrt.whl"),
)
def test_tensorrt_requirement_rejects_non_exact_pin(requirement: str) -> None:
    metadata = TENSORRT_METADATA + f"Requires-Dist: {requirement}\n"

    with pytest.raises(CiError, match="only exact TensorRT dependency pins"):
        _required_tensorrt_version(metadata)


def test_backend_files_match_wheel_tensorrt_abi() -> None:
    _validate_backend_files(
        "wheel",
        TENSORRT_VERSION,
        _backend_files("libtrtmc_backend_trt.so", "libtrtmc_backend_trt_11_1.so"),
    )


@pytest.mark.parametrize(
    "names",
    (
        ("libtrtmc_backend_trt.so", "libtrtmc_backend_trt_11_2.so"),
        (
            "libtrtmc_backend_trt.so",
            "libtrtmc_backend_trt_11_1.so",
            "libtrtmc_backend_trt_11_2.so",
        ),
    ),
)
def test_backend_files_reject_wrong_or_extra_abi(names: tuple[str, ...]) -> None:
    with pytest.raises(CiError, match="expected TensorRT backend files"):
        _validate_backend_files("wheel", TENSORRT_VERSION, _backend_files(*names))


def test_backend_files_reject_different_generic_payload() -> None:
    with pytest.raises(CiError, match="backend DSOs differ"):
        _validate_backend_files(
            "wheel",
            TENSORRT_VERSION,
            _backend_files(
                "libtrtmc_backend_trt.so",
                "libtrtmc_backend_trt_11_1.so",
                distinct=True,
            ),
        )


def test_backend_identity_matches_wheel_tensorrt_version() -> None:
    _validate_backend_identity("wheel", TENSORRT_VERSION, "11_1", TENSORRT_VERSION)


def test_backend_contract_derives_future_tensorrt_abi() -> None:
    version = "11.2.0.113"
    metadata = TENSORRT_METADATA.replace(TENSORRT_VERSION, version)

    assert _required_tensorrt_version(metadata) == version
    _validate_backend_files(
        "wheel",
        version,
        _backend_files("libtrtmc_backend_trt.so", "libtrtmc_backend_trt_11_2.so"),
    )
    _validate_backend_identity("wheel", version, "11_2", version)


def test_backend_identity_rejects_wrong_abi() -> None:
    with pytest.raises(CiError, match="reports ABI 11_2"):
        _validate_backend_identity("wheel", TENSORRT_VERSION, "11_2", TENSORRT_VERSION)


def test_backend_identity_rejects_wrong_runtime() -> None:
    with pytest.raises(CiError, match="reports runtime 11.2.0.113"):
        _validate_backend_identity("wheel", TENSORRT_VERSION, "11_1", "11.2.0.113")


def test_backend_probe_reads_exported_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Export:
        argtypes = None
        restype = None

        def __init__(self, value: bytes):
            self.value = value

        def __call__(self) -> bytes:
            return self.value

    class _Library:
        trtmc_backend_abi = _Export(b"11_1")
        trtmc_backend_runtime_version = _Export(b"11.1.0.106")

    monkeypatch.setattr("tools.ci.package.ctypes.CDLL", lambda _path: _Library())

    assert _probe_backend_identity(Path("libtrtmc_backend_trt.so")) == (
        "11_1",
        "11.1.0.106",
    )
    assert _Library.trtmc_backend_abi.argtypes == []
    assert _Library.trtmc_backend_runtime_version.argtypes == []
