# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebuilt profiling validation retains the owning family's reference checks."""
import pytest
from .test_e2e import _validation_bundle, _validation_revision


def test_prebuilt_bundle_requires_matching_profile_and_exact_revision(tmp_path, monkeypatch):
    bundle = tmp_path / "served.bundle"
    bundle.write_bytes(b"test")
    manifest = {"name": "pilot", "task": "text_generation"}
    monkeypatch.setenv("TRTMC_E2E_BUNDLE", str(bundle))
    with pytest.raises(AssertionError, match="matching manifest"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_PROFILE", "pilot")
    with pytest.raises(AssertionError, match="exact checkpoint"):
        _validation_revision(manifest)
    monkeypatch.setenv("TRTMC_E2E_CHECKPOINT_REVISION", "a" * 40)
    assert _validation_revision(manifest) == "a" * 40
    assert _validation_bundle(manifest, tmp_path, tmp_path / "unused") == bundle
    with pytest.raises(AssertionError, match="differs"):
        _validation_revision({**manifest, "hf_revision": "b" * 40})
    bundle.unlink()
    with pytest.raises(AssertionError):
        _validation_bundle(manifest, tmp_path, tmp_path / "unused")
