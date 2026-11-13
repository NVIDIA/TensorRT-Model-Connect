# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebuilt profiling must validate the checkpoint recorded by the core build provenance."""
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.provenance import collect
from . import test_e2e
from .test_e2e import _validation_bundle, _validation_revision

_FAMILY = "qwen"
_BINARY = Path("/opt/trtmc/bin/trtmc")


def _inputs(tmp_path, monkeypatch):
    model_dir = tmp_path / "models--example--model" / "snapshots" / ("a" * 40)
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(f'{{"model_type":"{_FAMILY}"}}')
    manifest = {"name": "pilot", "task": "text_generation", "precision": "fp16",
                "max_sequence_length": 256, "tensor_parallel_size": 1, "hf_id": "example/model"}
    bundle = tmp_path / "served.bundle"
    bundle.write_bytes(b"fixture bundle")
    monkeypatch.setenv("TRTMC_E2E_BUNDLE", str(bundle))
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    return model_dir, manifest, bundle


def _recorded(manifest, model_dir, **changes):
    """The provenance that `trtmc build` records for the manifest's profile."""
    profile = {"precision": manifest["precision"],
               "max_sequence_length": manifest["max_sequence_length"],
               "tensor_parallel_size": manifest["tensor_parallel_size"], **changes}
    return collect(BuildRequest(model_dir=model_dir, output_path=model_dir.parent / "built.bundle",
                                family=_FAMILY, task="text_generation", **profile))


def _inspect_returns(monkeypatch, bundle, provenance, family=_FAMILY, task="text_generation"):
    """Stand in for `trtmc inspect`; the native reader and its provenance output have their own tests."""
    def inspect(binary, path):
        assert (binary, path) == (_BINARY, bundle)
        return {"format": 1, "family": family, "task": task, "backend": "trt",
                "sections": {}, "provenance": provenance}
    monkeypatch.setattr(test_e2e, "_inspect", inspect)


def test_prebuilt_bundle_requires_matching_profile_and_exact_revision(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, _recorded(manifest, model_dir))
    monkeypatch.delenv("TRTMC_E2E_PROFILE")
    with pytest.raises(AssertionError, match="matching manifest"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "")
    with pytest.raises(AssertionError, match="exact checkpoint"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    assert _validation_revision(manifest) == "a" * 40
    assert _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY) == bundle
    with pytest.raises(AssertionError, match="differs"):
        _validation_revision({**manifest, "hf_revision": "b" * 40})
    bundle.unlink()
    with pytest.raises(AssertionError):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


@pytest.mark.parametrize("quantization", [None, "none"])
def test_prebuilt_bundle_accepts_either_spelling_of_no_quantization(tmp_path, monkeypatch, quantization):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    recorded = _recorded(manifest, model_dir, quantization=quantization)
    _inspect_returns(monkeypatch, bundle, recorded)
    for profile in (manifest, {**manifest, "quantization": "none"}):
        assert _validation_bundle(profile, model_dir, tmp_path / "unused", _BINARY) == bundle


@pytest.mark.parametrize("checkpoint", [None, {"hf_id": "example/other", "revision": "a" * 40},
                                       {"hf_id": "example/model", "revision": "b" * 40}])
def test_prebuilt_bundle_rejects_another_checkpoint(tmp_path, monkeypatch, checkpoint):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, {**_recorded(manifest, model_dir), "checkpoint": checkpoint})
    with pytest.raises(AssertionError, match="checkpoint provenance differs"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


@pytest.mark.parametrize("field,value", [("precision", "bf16"), ("max_sequence_length", 128),
                                        ("tensor_parallel_size", 2), ("quantization", "fp8"),
                                        ("fp32_layers", [0])])
def test_prebuilt_bundle_rejects_another_build_profile(tmp_path, monkeypatch, field, value):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, _recorded(manifest, model_dir))
    with pytest.raises(AssertionError, match="build profile differs"):
        _validation_bundle({**manifest, field: value}, model_dir, tmp_path / "unused", _BINARY)


def test_prebuilt_bundle_rejects_a_build_without_the_manifest_sequence_length(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, _recorded(manifest, model_dir, max_sequence_length=None))
    with pytest.raises(AssertionError, match="build profile differs"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


@pytest.mark.parametrize("provenance", [None, [], "provenance"])
def test_prebuilt_bundle_requires_builder_provenance(tmp_path, monkeypatch, provenance):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, provenance)
    with pytest.raises(AssertionError, match="lacks build provenance"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


@pytest.mark.parametrize("change", [{"version": 2}, {"build": None}])
def test_prebuilt_bundle_rejects_unknown_provenance(tmp_path, monkeypatch, change):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, {**_recorded(manifest, model_dir), **change})
    with pytest.raises(AssertionError, match="unsupported prebuilt provenance|build profile differs"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


@pytest.mark.parametrize("family,task", [("other", "text_generation"), (_FAMILY, "embedding")])
def test_prebuilt_bundle_rejects_another_family_or_task(tmp_path, monkeypatch, family, task):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, _recorded(manifest, model_dir), family, task)
    with pytest.raises(AssertionError, match="wrong family or task"):
        _validation_bundle(manifest, model_dir, tmp_path / "unused", _BINARY)


def test_prebuilt_validation_rejects_an_unpinned_local_directory(tmp_path, monkeypatch):
    model_dir, manifest, bundle = _inputs(tmp_path, monkeypatch)
    _inspect_returns(monkeypatch, bundle, _recorded(manifest, model_dir))
    with pytest.raises(AssertionError, match="selected immutable HF snapshot"):
        _validation_bundle(manifest, tmp_path, tmp_path / "unused", _BINARY)
