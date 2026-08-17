# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static publication gates for reviewed SAM2 qualification artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
SAM2_RUNTIME = ROOT / "src/runtime/models/sam2"
MANIFEST = SAM2_RUNTIME / "MODEL.toml"
PINS = SAM2_RUNTIME / "sam2_production_qualification_pins.cpp"
AUTHORITY_ID = "sam2-l4-trt11.1-contract5-0001"
RECORD_NAME = f"{AUTHORITY_ID}.qualification-record.json"
AUDIT_NAME = f"{AUTHORITY_ID}.qualification-audit.json"
EXPECTED_DATA_FILES = (RECORD_NAME, AUDIT_NAME)
PIN_COUNT_RE = re.compile(r"std::array\s*<\s*NativeQualificationStaticPin\s*,\s*(\d+)\s*>")
LOWERCASE_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _pin_count(pin_source: Path) -> int:
    match = PIN_COUNT_RE.search(pin_source.read_text(encoding="utf-8"))
    assert match is not None, "production SAM2 pin array extent must remain statically auditable"
    return int(match.group(1))


def _published_pair_is_complete(runtime_dir: Path, pin_count: int) -> bool:
    paths = tuple(runtime_dir / name for name in EXPECTED_DATA_FILES)
    assert not any(path.is_symlink() for path in paths), (
        "SAM2 qualification record and audit manifest must be regular files, not symlinks"
    )
    present = tuple(path.is_file() for path in paths)
    assert present in {(False, False), (True, True)}, (
        "SAM2 qualification record and audit manifest must be published as a pair"
    )
    if pin_count != 0:
        assert present == (True, True), (
            "active SAM2 production pins require the reviewed record and audit manifest"
        )
    return present == (True, True)


def _validate_published_pair(record_path: Path, audit_path: Path) -> None:
    record_bytes = record_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    record = json.loads(record_bytes)
    audit = json.loads(audit_bytes)
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()

    assert record_bytes == (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert audit_bytes == (json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    assert record["schema_version"] == 2
    assert record["artifact_type"] == "sam2_native_qualification_record"
    assert record["authority_id"] == AUTHORITY_ID
    assert record["self_authorizing"] is False
    assert audit["schema_version"] == 1
    assert audit["artifact_type"] == "sam2_native_qualification_audit_manifest"
    assert audit["self_authorizing"] is False
    assert audit["pin_mutation_supported"] is False
    assert audit["record"]["sha256"] == record_sha256
    assert audit["record"]["size_bytes"] == len(record_bytes)

    if _pin_count(PINS) != 0:
        assert LOWERCASE_SHA256_RE.fullmatch(record_sha256)
        assert record_sha256 in PINS.read_text(encoding="utf-8")


def test_sam2_manifest_predeclares_only_the_fixed_public_qualification_pair() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = tuple(manifest["runtime_optional_data_files"])

    assert declared == EXPECTED_DATA_FILES
    for value in declared:
        relative = PurePosixPath(value)
        assert not relative.is_absolute()
        assert relative.as_posix() == value
        assert all(part not in {"", ".", ".."} for part in relative.parts)


def test_sam2_qualification_pair_is_optional_until_a_production_pin_is_active() -> None:
    record_path = SAM2_RUNTIME / RECORD_NAME
    audit_path = SAM2_RUNTIME / AUDIT_NAME
    if _published_pair_is_complete(SAM2_RUNTIME, _pin_count(PINS)):
        _validate_published_pair(record_path, audit_path)


def test_nonempty_pin_extent_fails_closed_without_the_publication_pair(tmp_path: Path) -> None:
    assert _published_pair_is_complete(tmp_path, 0) is False
    with pytest.raises(
        AssertionError,
        match="active SAM2 production pins require the reviewed record and audit manifest",
    ):
        _published_pair_is_complete(tmp_path, 1)


def test_publication_pair_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "external-record.json"
    target.write_text("{}\n", encoding="utf-8")
    (tmp_path / RECORD_NAME).symlink_to(target)

    with pytest.raises(AssertionError, match="regular files, not symlinks"):
        _published_pair_is_complete(tmp_path, 0)


def test_sam2_qualification_data_has_exact_optional_packaging_and_install_routes() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    conan = (ROOT / "conanfile.py").read_text(encoding="utf-8")

    assert "TRTMC_MODEL_${_trtmc_model_var}_OPTIONAL_DATA_FILES" in cmake
    assert '"${CMAKE_INSTALL_DATADIR}/trtmc/model_data/${_trtmc_model}"' in cmake
    assert "COMPONENT trtmc_model_data" in cmake
    assert "OPTIONAL" in cmake
    assert "_runtime_optional_data_files(self.source_folder)" in conan
    assert 'package_module / "model_data" / model' in conan
