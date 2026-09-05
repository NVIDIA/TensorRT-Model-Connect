# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.ci.context import CiContext
from tools.ci.model_artifact_cache import (
    ModelArtifactCache,
    ModelArtifactCacheWarmer,
    ModelArtifactContract,
    ModelArtifactFile,
    parse_model_artifact_contract,
)
from tools.ci.process import CiError


def _owner(url: str = "https://api.ngc.nvidia.com/v2/models/example") -> dict:
    content = b"pinned model"
    return {
        "model_artifact_cache": {
            "relative_path": "example/ngc-1",
            "environment_variable": "TRTMC_EXAMPLE_MODEL_DIR",
            "files": [
                {
                    "path": "model.onnx",
                    "url": url,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            ],
        }
    }


def test_parse_model_artifact_contract():
    contract = parse_model_artifact_contract(_owner(), "example", Path("MODEL.toml"), "premerge")
    assert contract is not None
    assert contract.relative_path == "example/ngc-1"
    assert contract.files[0].path == "model.onnx"


@pytest.mark.parametrize(
    "url",
    [
        "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/model.pt",
        "https://raw.githubusercontent.com/example/project/revision/query.json",
    ],
)
def test_parse_accepts_digest_pinned_public_model_origins(url: str):
    assert (
        parse_model_artifact_contract(_owner(url), "example", Path("MODEL.toml"), "premerge")
        is not None
    )


def test_parse_accepts_large_digest_pinned_checkpoint() -> None:
    owner = _owner("https://openfold3-data.s3.amazonaws.com/openfold3-parameters/model.pt")
    owner["model_artifact_cache"]["files"][0]["size"] = 2_287_872_989
    assert (
        parse_model_artifact_contract(owner, "example", Path("MODEL.toml"), "premerge") is not None
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://api.ngc.nvidia.com/model",
        "https://example.com/model",
        "https://user@api.ngc.nvidia.com/model",
        "https://user:secret@api.ngc.nvidia.com/model",
        "https://api.ngc.nvidia.com/model#fragment",
    ],
)
def test_parse_rejects_unapproved_origins_or_unsafe_urls(url: str):
    with pytest.raises(CiError, match="allowed HTTPS URL"):
        parse_model_artifact_contract(_owner(url), "example", Path("MODEL.toml"), "premerge")


def test_prepare_copies_only_verified_cached_files(tmp_path: Path):
    content = b"pinned model"
    item = ModelArtifactFile(
        "model.onnx",
        "https://api.ngc.nvidia.com/v2/models/example",
        hashlib.sha256(content).hexdigest(),
        len(content),
    )
    contract = ModelArtifactContract("example", "example/ngc-1", "TRTMC_EXAMPLE_MODEL_DIR", (item,))
    cache = tmp_path / "cache"
    cached_file = cache / "model-artifacts/example/ngc-1/model.onnx"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_bytes(content)
    repository = tmp_path / "repo"
    repository.mkdir()
    context = CiContext(repository, {"TRTMC_MODEL_ARTIFACT_CACHE_ROOT": str(cache)})
    assert ModelArtifactCacheWarmer(context).warm_contract(contract) == cached_file.parent

    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()
    ModelArtifactCache(context, "example").prepare(contract.as_payload(), work, artifacts)
    assert (work / "model-artifacts/example/ngc-1/model.onnx").read_bytes() == content
    evidence = json.loads((artifacts / "model-artifact-cache.json").read_text())
    assert evidence["isolation"] == "selected-digest-private"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "relative_path": "example/ngc-1",
            "environment_variable": "TRTMC_EXAMPLE_MODEL_DIR",
            "files": [None],
        },
        {
            "relative_path": "example/ngc-1",
            "environment_variable": "TRTMC_EXAMPLE_MODEL_DIR",
            "files": [{"path": "model.onnx"}],
        },
    ],
)
def test_prepare_rejects_malformed_payload_as_ci_error(tmp_path: Path, payload: dict):
    repository = tmp_path / "repo"
    repository.mkdir()
    context = CiContext(repository, {})

    with pytest.raises(CiError):
        ModelArtifactCache(context, "example").prepare(
            payload, tmp_path / "work", tmp_path / "artifacts"
        )


def test_cached_digest_mismatch_fails_closed(tmp_path: Path):
    contract = parse_model_artifact_contract(_owner(), "example", Path("MODEL.toml"), "premerge")
    assert contract is not None
    cache = tmp_path / "cache"
    cached_file = cache / "model-artifacts/example/ngc-1/model.onnx"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_bytes(b"broken model")
    repository = tmp_path / "repo"
    repository.mkdir()
    context = CiContext(repository, {"TRTMC_MODEL_ARTIFACT_CACHE_ROOT": str(cache)})
    with pytest.raises(CiError, match="pinned size/digest"):
        ModelArtifactCacheWarmer(context).warm_contract(contract)


def test_cache_root_must_not_be_inside_repository(tmp_path: Path):
    contract = parse_model_artifact_contract(_owner(), "example", Path("MODEL.toml"), "premerge")
    assert contract is not None
    repository = tmp_path / "repo"
    repository.mkdir()
    cache = repository / ".model-artifact-cache"
    context = CiContext(repository, {"TRTMC_MODEL_ARTIFACT_CACHE_ROOT": str(cache)})

    with pytest.raises(CiError, match="cache root is invalid"):
        ModelArtifactCacheWarmer(context).warm_contract(contract)
    assert not cache.exists()
