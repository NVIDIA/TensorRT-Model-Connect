# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebuilt profiling must validate the actual builder-recorded checkpoint."""
import json
from pathlib import Path
import struct

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from ..bundle_provenance import checkpoint_identity, _read_provenance
from .test_e2e import _validation_bundle, _validation_revision


def _inputs(tmp_path, monkeypatch):
    model_dir = tmp_path / "models--example--model" / "snapshots" / ("a" * 40)
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type":"gpt2"}')
    (model_dir / "model.safetensors").write_bytes(b"checkpoint weights")
    manifest = {"name": "pilot", "task": "text_generation", "precision": "fp16",
                "max_sequence_length": 256, "tensor_parallel_size": 1, "hf_id": "example/model"}
    bundle = tmp_path / "served.bundle"
    monkeypatch.setenv("TRTMC_E2E_BUNDLE", str(bundle))
    return model_dir, manifest, bundle


def _write_bundle(bundle, manifest, checkpoint):
    writer = BundleWriter(bundle)
    writer.set_header(family="gpt2", task="text_generation", backend="trt")
    writer.add_bytes("engine.plan", b"fixture engine")
    writer.add_json("checkpoint_provenance.json", {
        "version": 1, "checkpoint": checkpoint,
        "build": {"precision": manifest["precision"],
                  "max_sequence_length": manifest["max_sequence_length"],
                  "tensor_parallel_size": manifest["tensor_parallel_size"],
                  "quantization": "none", "fp32_layers": []},
    })
    writer.finish()


def test_prebuilt_bundle_requires_matching_profile_and_exact_revision(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _write_bundle(bundle, manifest, checkpoint_identity(model_dir))
    with pytest.raises(AssertionError, match="matching manifest"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    with pytest.raises(AssertionError, match="exact checkpoint"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    assert _validation_revision(manifest) == "a" * 40
    assert _validation_bundle(manifest, model_dir, tmp_path / "unused") == bundle
    with pytest.raises(AssertionError, match="differs"):
        _validation_revision({**manifest, "hf_revision": "b" * 40})
    bundle.unlink()
    with pytest.raises(AssertionError):
        _validation_bundle(manifest, model_dir, tmp_path / "unused")


@pytest.mark.parametrize("checkpoint", [None, {"hf_id": "example/other", "revision": "a" * 40},
                                       {"hf_id": "example/model", "revision": "b" * 40}])
def test_prebuilt_bundle_rejects_another_checkpoint(tmp_path, monkeypatch, checkpoint):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _write_bundle(bundle, manifest, checkpoint)
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    with pytest.raises(AssertionError, match="checkpoint provenance differs"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused")


@pytest.mark.parametrize("field,value", [("precision", "bf16"), ("max_sequence_length", 128),
                                        ("tensor_parallel_size", 2), ("quantization", "fp8")])
def test_prebuilt_bundle_rejects_another_build_profile(tmp_path, monkeypatch, field, value):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _write_bundle(bundle, manifest, checkpoint_identity(model_dir))
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    with pytest.raises(AssertionError, match="build profile differs"):
        _validation_bundle({**manifest, field: value}, model_dir, tmp_path / "unused")


def test_prebuilt_bundle_requires_builder_provenance(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    writer = BundleWriter(bundle)
    writer.set_header(family="gpt2", task="text_generation", backend="trt")
    writer.add_bytes("engine.plan", b"fixture engine")
    writer.finish()
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    with pytest.raises(AssertionError, match="lacks build provenance"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused")


def test_prebuilt_validation_rejects_an_unpinned_local_directory(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _write_bundle(bundle, manifest, checkpoint_identity(model_dir))
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    with pytest.raises(AssertionError, match="selected immutable HF snapshot"):
        _validation_bundle(manifest, tmp_path, tmp_path / "unused")


def test_builder_records_snapshot_revision(tmp_path):
    snapshot = tmp_path / "models--example--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    assert checkpoint_identity(snapshot) == {"hf_id": "example/model", "revision": "a" * 40}
    assert checkpoint_identity(tmp_path) is None


@pytest.mark.parametrize("header", [
    None, [], {},
    {"family": "gpt2", "task": "text_generation"},
    {"family": "gpt2", "task": "text_generation", "sections": []},
    {"family": "gpt2", "task": "text_generation", "sections": {"checkpoint_provenance.json": []}},
    *[{"family": "gpt2", "task": "text_generation", "sections": {
        "checkpoint_provenance.json": section}}
      for section in ({}, {"offset": "0", "length": 2}, {"offset": True, "length": 2},
                      {"offset": 0, "length": "2"}, {"offset": 0, "length": False},
                      {"offset": -1, "length": 2}, {"offset": 0, "length": 0},
                      {"offset": 0, "length": 3}, {"offset": 3, "length": 2})],
])
def test_prebuilt_reader_rejects_malformed_header_shapes(tmp_path, header):
    bundle = tmp_path / "malformed.bundle"
    encoded = json.dumps(header).encode()
    bundle.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(encoded)) + encoded + b"{}")
    with pytest.raises(AssertionError):
        _read_provenance(bundle)


@pytest.mark.parametrize("payload", [b"[]", b"null", b"false", b"0", b'"text"', b"{", b"\xff"])
def test_prebuilt_reader_rejects_malformed_provenance(tmp_path, payload):
    bundle = tmp_path / "malformed.bundle"
    writer = BundleWriter(bundle)
    writer.set_header(family="gpt2", task="text_generation", backend="trt")
    writer.add_bytes("checkpoint_provenance.json", payload)
    writer.finish()
    with pytest.raises(AssertionError, match="invalid prebuilt provenance"):
        _read_provenance(bundle)


def test_prebuilt_reader_rejects_a_truncated_header(tmp_path):
    bundle = tmp_path / "malformed.bundle"
    bundle.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", 100) + b"{}")
    with pytest.raises(AssertionError, match="truncated prebuilt bundle header"):
        _read_provenance(bundle)


@pytest.mark.parametrize("optimized", [False, True], ids=["normal", "optimized"])
@pytest.mark.parametrize("invalid", ["version", "checkpoint", "build", "signature"])
def test_prebuilt_guards_survive_optimized_python(tmp_path, monkeypatch, optimized, invalid):
    source = Path(__file__).parents[1] / "bundle_provenance.py"
    namespace = {"__name__": "optimized_provenance"}
    exec(
        compile(source.read_text(encoding="utf-8"), str(source), "exec", optimize=int(optimized)),
        namespace,
    )
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    selected = checkpoint_identity(model_dir)
    _write_bundle(bundle, manifest, selected)
    validate = namespace["validate_prebuilt_bundle"]
    assert validate(bundle, model_dir, manifest, "a" * 40)["checkpoint"] == selected

    if invalid == "signature":
        bundle.write_bytes(b"INVALID!" + bundle.read_bytes()[8:])
    else:
        provenance = _read_provenance(bundle)
        provenance[invalid] = None
        writer = BundleWriter(bundle)
        writer.set_header(family="gpt2", task="text_generation", backend="trt")
        writer.add_json("checkpoint_provenance.json", provenance)
        writer.finish()
    with pytest.raises(AssertionError):
        validate(bundle, model_dir, manifest, "a" * 40)
