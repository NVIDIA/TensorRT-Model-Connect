# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require the manifest's checkpoint and build settings before qualification."""

import json

import pytest

from families.molmo2.tests import test_e2e
from tensorrt_model_connect import BuildRequest, provenance


@pytest.fixture
def inspection(tmp_path, monkeypatch):
    manifest = test_e2e.MANIFEST
    checkpoint = (
        tmp_path
        / ("models--" + manifest["hf_id"].replace("/", "--"))
        / "snapshots"
        / manifest["hf_revision"]
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    request = BuildRequest(
        checkpoint,
        tmp_path / "prebuilt.bundle",
        manifest["family"],
        manifest["task"],
        manifest["precision"],
        max_sequence_length=manifest["max_sequence_length"],
        tensor_parallel_size=manifest["tensor_parallel_size"],
    )
    monkeypatch.setattr(provenance, "_producer", lambda: {})
    monkeypatch.setattr(provenance, "_toolchain", lambda: {})
    return {
        "family": manifest["family"],
        "task": manifest["task"],
        "backend": "trt",
        "provenance": provenance.collect(request),
    }


def test_builder_recorded_manifest_is_accepted(inspection):
    test_e2e._validate_bundle(inspection)


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", "fp32"),
        ("max_sequence_length", 1024),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("max_batch_size", 2),
        ("image_height", 378),
        ("image_width", 378),
        ("video_num_frames", 2),
        ("quantization", "fp8"),
        ("fp32_layers", [0]),
        ("dynamic_kv_cache", True),
        ("mtp_seq_len", 2),
        ("graph_transform", True),
        ("max_batch_size", True),
    ],
)
def test_mismatched_build_settings_are_rejected(inspection, field, value):
    inspection["provenance"]["build"][field] = value
    with pytest.raises(AssertionError, match=f"build.{field}"):
        test_e2e._validate_bundle(inspection)


@pytest.mark.parametrize("field", ["hf_id", "revision"])
def test_mismatched_checkpoint_identity_is_rejected(inspection, field):
    inspection["provenance"]["checkpoint"][field] = "different-checkpoint"
    with pytest.raises(AssertionError, match="checkpoint identity"):
        test_e2e._validate_bundle(inspection)


@pytest.mark.parametrize("section", ["checkpoint", "build", "version"])
def test_incomplete_provenance_is_rejected(inspection, section):
    del inspection["provenance"][section]
    with pytest.raises(AssertionError):
        test_e2e._validate_bundle(inspection)


def test_missing_optional_setting_is_not_assumed_to_be_default(inspection):
    del inspection["provenance"]["build"]["quantization"]
    with pytest.raises(AssertionError, match="build.quantization"):
        test_e2e._validate_bundle(inspection)


def test_unknown_checkpoint_identity_is_rejected(inspection):
    inspection["provenance"]["checkpoint"] = None
    with pytest.raises(AssertionError, match="checkpoint identity"):
        test_e2e._validate_bundle(inspection)


def test_mismatched_prebuilt_bundle_fails_before_reference_or_generation(
    inspection, tmp_path, tmp_path_factory, monkeypatch
):
    checkpoint = next(tmp_path.glob("models--*/snapshots/*"))
    bundle = tmp_path / "prebuilt.bundle"
    bundle.touch()
    binary = tmp_path / "trtmc"
    binary.touch()
    monkeypatch.setenv("TRTMC_MOLMO2_MODEL_DIR", str(checkpoint))
    monkeypatch.setenv("TRTMC_MOLMO2_BUNDLE", str(bundle))
    monkeypatch.setenv("TRTMC_BINARY", str(binary))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    inspection["provenance"]["build"]["max_sequence_length"] = 1024
    monkeypatch.setattr(
        test_e2e.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps(inspection)
    )
    monkeypatch.setattr(
        test_e2e, "build", lambda *_args: pytest.fail("a supplied bundle must not be rebuilt")
    )
    with pytest.raises(AssertionError, match="build.max_sequence_length"):
        test_e2e.artifacts.__wrapped__(tmp_path_factory)()
