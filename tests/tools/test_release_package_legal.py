# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

import _pyproject_backend as backend
from tools import model_ci
from tools.ci.package import RELEASE_LEGAL_FILES, WheelArchiveValidator
from tools.ci.process import CiError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _Context:
    repository = REPOSITORY_ROOT


def _validate_legal_payload(wheel: Path) -> None:
    validator = WheelArchiveValidator(_Context(), "manylinux_2_39_aarch64")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        validator._validate_legal_payload(
            wheel,
            archive,
            names,
            metadata_name,
            archive.read(metadata_name).decode(),
        )


def _synthetic_wheel(
    path: Path,
    *,
    metadata: str,
    omitted: str | None = None,
    stale: str | None = None,
) -> Path:
    dist_info = "tensorrt_model_connect-0.1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        for relative in RELEASE_LEGAL_FILES:
            if relative == omitted:
                continue
            content = b"stale\n" if relative == stale else (REPOSITORY_ROOT / relative).read_bytes()
            archive.writestr(f"{dist_info}/licenses/{relative}", content)
    return path


def test_pyproject_declares_pep639_release_legal_files() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]

    assert project["license"] == "Apache-2.0"
    assert tuple(project["license-files"]) == RELEASE_LEGAL_FILES
    assert "ASSET_LICENSES.md" in model_ci.PLATFORM_PROJECTION_EXACT
    assert "ASSET_LICENSES.md" in model_ci.LEGAL_OR_DOC_EXACT


def test_python_only_editable_wheel_contains_release_legal_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    filename = backend.build_editable(str(tmp_path), {"py-only": "true"})
    wheel = tmp_path / filename

    _validate_legal_payload(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        assert metadata["Metadata-Version"] == "2.4"
        assert metadata["License-Expression"] == "Apache-2.0"
        assert sorted(metadata.get_all("License-File")) == sorted(RELEASE_LEGAL_FILES)


def test_wheel_legal_validator_rejects_missing_file(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata())
    wheel = _synthetic_wheel(
        tmp_path / "missing.whl",
        metadata=metadata,
        omitted="NOTICE",
    )

    with pytest.raises(CiError, match="packaged legal file is missing"):
        _validate_legal_payload(wheel)


def test_wheel_legal_validator_rejects_stale_file(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata())
    wheel = _synthetic_wheel(
        tmp_path / "stale.whl",
        metadata=metadata,
        stale="ASSET_LICENSES.md",
    )

    with pytest.raises(CiError, match="packaged legal file is stale"):
        _validate_legal_payload(wheel)


def test_wheel_legal_validator_rejects_missing_metadata_declaration(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata()).replace(
        "License-File: NOTICE\n", ""
    )
    wheel = _synthetic_wheel(tmp_path / "metadata.whl", metadata=metadata)

    with pytest.raises(CiError, match="package metadata must declare legal files"):
        _validate_legal_payload(wheel)
