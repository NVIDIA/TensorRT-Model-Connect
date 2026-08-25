# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

import _pyproject_backend as backend
from tools.ci.package import (
    _package_variant_version,
    _probe_backend_identity,
    _required_tensorrt_version,
    _target_tensorrt_version,
    _validate_archive_backend_identity,
    _validate_backend_files,
    _validate_backend_identity,
    _validate_package_variant,
)
from tools.ci.process import CiError


TENSORRT_VERSION = "11.1.0.106"
REPO_ROOT = Path(__file__).resolve().parents[2]
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


@pytest.mark.parametrize(
    ("tensorrt_version", "package_version"),
    (("11.1.0.106", "0.1.0+trt111"), ("11.2.0.113", "0.1.0+trt112")),
)
def test_package_profile_resolves_unique_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tensorrt_version: str,
    package_version: str,
) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", tensorrt_version)

    project = backend._resolved_project_metadata(REPO_ROOT)

    assert project["version"] == package_version
    assert "dynamic" not in project
    assert [
        dependency for dependency in project["dependencies"] if dependency.startswith("tensorrt==")
    ] == [
        f'tensorrt=={tensorrt_version}; platform_machine == "aarch64"',
        f'tensorrt=={tensorrt_version}; platform_machine == "x86_64"',
    ]
    assert _package_variant_version(REPO_ROOT, tensorrt_version) == package_version


def test_package_ci_abi_validation_does_not_require_source_package_import() -> None:
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path.cwd()))
        from tools.ci.package import (
            _package_variant_version,
            _validate_backend_files,
            _validate_backend_identity,
        )

        version = "11.1.0.106"
        package_version = _package_variant_version(Path.cwd(), version)
        _validate_backend_files(
            "wheel",
            version,
            {
                "libtrtmc_backend_trt.so": b"backend",
                "libtrtmc_backend_trt_11_1.so": b"backend",
            },
        )
        _validate_backend_identity("wheel", version, "11_1", version)
        print(package_version)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("+trt111")


def test_package_profile_default_preserves_development_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_PACKAGE_TENSORRT_VERSION", raising=False)

    project = backend._resolved_project_metadata(REPO_ROOT)

    assert project["version"] == "0.1.0"
    assert _required_tensorrt_version(backend._metadata_text(project)) == TENSORRT_VERSION


def test_python_only_wheel_uses_selected_profile_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", "11.2.0.113")
    monkeypatch.chdir(REPO_ROOT)

    filename = backend.build_editable(str(tmp_path), {"py-only": "true"})

    assert filename == "tensorrt_model_connect-0.1.0+trt112-py3-none-any.whl"
    with zipfile.ZipFile(tmp_path / filename) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
    assert "Version: 0.1.0+trt112" in metadata
    assert _required_tensorrt_version(metadata) == "11.2.0.113"


def test_package_target_is_explicit_and_exact() -> None:
    with pytest.raises(CiError, match="must select an exact wheel target"):
        _target_tensorrt_version({}, required=True)
    with pytest.raises(CiError, match="exact four-part version"):
        _target_tensorrt_version(
            {"TRTMC_PACKAGE_TENSORRT_VERSION": "11.2"},
            required=True,
        )


def test_native_backend_profile_is_scoped_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ConanBackend:
        @staticmethod
        def _get_project_metadata(_project_dir: Path) -> dict[str, object]:
            return {"name": "unresolved"}

    fake = _ConanBackend()
    original_reader = fake._get_project_metadata
    monkeypatch.setattr(backend, "_conan_build_backend", lambda: fake)
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", "11.2.0.113")
    monkeypatch.setenv("TRTMC_PACKAGE_VERSION", "untrusted")
    monkeypatch.chdir(REPO_ROOT)

    with backend._variant_conan_build_backend() as selected:
        assert selected._get_project_metadata(REPO_ROOT)["version"] == "0.1.0+trt112"
        assert backend.os.environ["TRTMC_PACKAGE_VERSION"] == "0.1.0+trt112"

    assert fake._get_project_metadata == original_reader
    assert backend.os.environ["TRTMC_PACKAGE_VERSION"] == "untrusted"


def test_prepared_metadata_must_match_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", TENSORRT_VERSION)
    expected = backend._resolved_project_metadata(REPO_ROOT)
    dist_info = tmp_path / "tensorrt_model_connect-0.1.0+trt111.dist-info"
    dist_info.mkdir()
    stale = backend._metadata_text(expected).replace("0.1.0+trt111", "0.1.0+trt112")
    (dist_info / "METADATA").write_text(stale, encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the selected package variant"):
        backend._validate_prepared_metadata(tmp_path, expected)


def test_prepared_metadata_rejects_wrong_directory_or_extra_tensorrt_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", TENSORRT_VERSION)
    expected = backend._resolved_project_metadata(REPO_ROOT)
    dist_info = tmp_path / "tensorrt_model_connect-0.1.0+trt112.dist-info"
    dist_info.mkdir()
    metadata = backend._metadata_text(expected)
    (dist_info / "METADATA").write_text(metadata, encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata directory"):
        backend._validate_prepared_metadata(tmp_path, expected)

    correct = tmp_path / "tensorrt_model_connect-0.1.0+trt111.dist-info"
    dist_info.rename(correct)
    (correct / "METADATA").write_text(
        metadata + "Requires-Dist: tensorrt>=11\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exact per-architecture pins"):
        backend._validate_prepared_metadata(tmp_path, expected)


def test_variant_sdist_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_TENSORRT_VERSION", TENSORRT_VERSION)

    with pytest.raises(RuntimeError, match="apply to wheels"):
        backend.build_sdist(str(tmp_path))


def test_wheel_variant_must_match_explicit_target() -> None:
    metadata = TENSORRT_METADATA.replace("Version: 0.1.0", "Version: 0.1.0+trt111")
    assert _validate_package_variant(
        "wheel",
        metadata,
        REPO_ROOT,
        TENSORRT_VERSION,
        "tensorrt_model_connect-0.1.0+trt111-py312-none-linux_aarch64.whl",
    ) == (TENSORRT_VERSION, "0.1.0+trt111")

    with pytest.raises(CiError, match=r"expected 0.1.0\+trt111"):
        _validate_package_variant(
            "wheel",
            TENSORRT_METADATA,
            REPO_ROOT,
            TENSORRT_VERSION,
            "tensorrt_model_connect-0.1.0-py312-none-linux_aarch64.whl",
        )


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


@pytest.mark.parametrize(
    ("version", "abi"),
    (("11.2.0.113", "11_2"), ("100.42.0.113", "100_42")),
)
def test_backend_contract_derives_future_tensorrt_abi(version: str, abi: str) -> None:
    metadata = TENSORRT_METADATA.replace(TENSORRT_VERSION, version)

    assert _required_tensorrt_version(metadata) == version
    _validate_backend_files(
        "wheel",
        version,
        _backend_files("libtrtmc_backend_trt.so", f"libtrtmc_backend_trt_{abi}.so"),
    )
    _validate_backend_identity("wheel", version, abi, version)


def test_backend_identity_rejects_wrong_abi() -> None:
    with pytest.raises(CiError, match="reports ABI 11_2"):
        _validate_backend_identity("wheel", TENSORRT_VERSION, "11_2", TENSORRT_VERSION)


def test_backend_identity_rejects_wrong_runtime() -> None:
    with pytest.raises(CiError, match="reports runtime 11.2.0.113"):
        _validate_backend_identity("wheel", TENSORRT_VERSION, "11_1", "11.2.0.113")


def test_archive_backend_identity_rejects_renamed_wrong_dso(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.ci.package._probe_backend_identity",
        lambda _path: ("11_2", "11.2.0.113"),
    )

    with pytest.raises(CiError, match="reports ABI 11_2"):
        _validate_archive_backend_identity(
            "wheel",
            TENSORRT_VERSION,
            {
                "libtrtmc_core.so": b"core",
                "libtrtmc_backend_trt.so": b"backend",
                "libtrtmc_backend_trt_11_1.so": b"backend",
            },
        )


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
