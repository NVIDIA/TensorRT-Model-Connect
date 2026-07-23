# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixed-noise, stagewise OpenPI reference artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

import pytest

from tests.e2e.models.openpi import qualification


def _asset(uri: str, digit: str) -> dict[str, str]:
    return {"uri": uri, "sha256": digit * 64}


def _capture_spec(tmp_path: Path, profile: str = "pi05_droid") -> Path:
    contract = qualification.load_contract(profile)
    tensors = {}
    sources = tmp_path / "capture"
    sources.mkdir()
    for index, (name, expected) in enumerate(
        qualification._expected_tensor_contract(contract).items(), start=1
    ):
        byte_length = qualification._tensor_byte_length(expected["dtype"], expected["shape"])
        source = sources / f"{name}.bin"
        with source.open("wb") as handle:
            handle.seek(byte_length - 1)
            handle.write(bytes([index % 251 + 1]))
        tensors[name] = {"source": str(source), **expected}

    spec = {
        "schema_version": 1,
        "artifact_type": "openpi_reference_capture_spec",
        "profile_name": profile,
        "upstream": {
            "repository": qualification.UPSTREAM_REPOSITORY,
            "commit": qualification.UPSTREAM_COMMIT,
            "checkpoint": _asset(contract["upstream"]["checkpoint_uri"], "a"),
            "tokenizer": _asset("openpi://paligemma_tokenizer.model", "b"),
            "normalization": _asset(f"openpi://{profile}/norm_stats.json", "c"),
        },
        "case": {"id": "synthetic-qualification-case", "prompt": "pick up the block"},
        "exporter": {"name": "openpi-jax-capture", "version": "pinned-test"},
        "tensors": tensors,
    }
    path = tmp_path / "capture-spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_reference_paths_accept_only_standard_hf_snapshot_symlinks(tmp_path: Path) -> None:
    repo_cache = tmp_path / "models--NVIDIA--openpi"
    blobs = repo_cache / "blobs"
    snapshot = repo_cache / "snapshots" / ("a" * 40)
    logical_root = snapshot / "trtmc_openpi" / "reference"
    blobs.mkdir(parents=True)
    logical_root.mkdir(parents=True)
    blob = blobs / ("b" * 64)
    blob.write_bytes(b"reference")
    logical = logical_root / "reference.bin"
    logical.symlink_to(os.path.relpath(blob, logical.parent))

    assert (
        qualification._resolve_relative_file(
            logical_root,
            PurePosixPath("reference.bin"),
            label="reference",
        )
        == blob.resolve()
    )

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    escaped = logical_root / "escaped.bin"
    escaped.symlink_to(outside)
    with pytest.raises(qualification.OpenPIQualificationError, match="escapes"):
        qualification._resolve_relative_file(
            logical_root,
            PurePosixPath("escaped.bin"),
            label="reference",
        )


def test_reference_generator_materializes_and_hashes_every_stage(tmp_path: Path) -> None:
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), tmp_path / "artifact"
    )
    artifact = qualification.validate_reference_artifact(artifact_path)

    assert artifact["fixed_external_noise"] == {
        "tensor": "initial_noise",
        "provided_by_caller": True,
        "rng_backend": "external",
    }
    assert {f"velocity_{step:02d}" for step in range(10)} <= set(artifact["tensors"])
    assert {f"flow_state_{step:02d}" for step in range(11)} <= set(artifact["tensors"])
    assert artifact["tensors"]["normalized_actions"]["shape"] == [1, 15, 32]
    assert artifact["tensors"]["physical_actions"]["shape"] == [1, 15, 8]


def test_reference_validator_rejects_generated_noise_claim(tmp_path: Path) -> None:
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), tmp_path / "artifact"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["fixed_external_noise"]["provided_by_caller"] = False
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(qualification.OpenPIQualificationError, match="caller-supplied"):
        qualification.validate_reference_artifact(artifact_path)


def test_reference_validator_rejects_missing_intermediate_flow_step(tmp_path: Path) -> None:
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), tmp_path / "artifact"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    del artifact["tensors"]["velocity_06"]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(qualification.OpenPIQualificationError, match="missing stagewise tensors"):
        qualification.validate_reference_artifact(artifact_path, verify_payloads=False)


def test_reference_validator_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), tmp_path / "artifact"
    )
    noise_path = artifact_path.parent / "tensors" / "initial_noise.bin"
    with noise_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"X")

    with pytest.raises(qualification.OpenPIQualificationError, match="SHA-256 mismatch"):
        qualification.validate_reference_artifact(artifact_path)


def test_reference_generator_rejects_incomplete_capture_before_publishing(tmp_path: Path) -> None:
    spec_path = _capture_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    del spec["tensors"]["initial_noise"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(qualification.OpenPIQualificationError, match="missing stagewise tensors"):
        qualification.materialize_reference_artifact(spec_path, tmp_path / "artifact")


def test_reference_set_binds_independent_case_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), corpus / "case-0000"
    )
    index_path = qualification.materialize_reference_set(
        [artifact_path], corpus / "reference-set.json"
    )

    reference_set = qualification.validate_reference_set(index_path)
    assert reference_set["profile_name"] == "pi05_droid"
    assert reference_set["case_count"] == 1
    assert reference_set["cases"][0]["id"] == "synthetic-qualification-case"
    with pytest.raises(qualification.OpenPIQualificationError, match="requires 512"):
        qualification.validate_reference_set(index_path, minimum_cases=512)


def test_reference_set_rejects_duplicate_case_identity(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    artifact_path = qualification.materialize_reference_artifact(
        _capture_spec(tmp_path), corpus / "case-0000"
    )

    with pytest.raises(qualification.OpenPIQualificationError, match="Duplicate reference case id"):
        qualification.materialize_reference_set(
            [artifact_path, artifact_path], corpus / "reference-set.json"
        )
