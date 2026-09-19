# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject missing or altered dependency terms and preserve bundle notices."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from families.hstu import native_attention_build as build
from tensorrt_model_connect.build import content_cache_key


FAMILY = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((FAMILY / "native_attention_source.json").read_text())


@pytest.fixture
def licensed_source(tmp_path, monkeypatch):
    """Small source fixture; actual legal texts remain the checked-in copies."""
    family = tmp_path / "family"
    family.mkdir()
    shutil.copytree(FAMILY / "third_party", family / "third_party")
    root = tmp_path / "source"
    source = root / "fbgemm_gpu/experimental/hstu/src/hstu_ampere"
    source.mkdir(parents=True)
    manifest = copy.deepcopy(MANIFEST)
    for name in manifest["files"]:
        data = ("// test-only header " + name + "\n").encode()
        (source / name).write_bytes(data)
        manifest["files"][name] = content_cache_key(
            manifest["content_key_domain"], name.encode(), data
        )
    for relative, entry in manifest["licenses"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((family / "third_party" / entry["notice_file"]).read_bytes())
    cutlass = root / "external/cutlass"
    (cutlass / "include/cute").mkdir(parents=True)
    (cutlass / "include/cute/tensor.hpp").write_text("// test-only CUTLASS header\n")
    manifest["cutlass"]["include_content_key"] = build.cutlass_content_key(cutlass)
    (family / "native_attention_source.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(build, "HERE", family)
    return root, source, manifest


def test_pinned_notice_copies_preserve_full_upstream_terms():
    assert MANIFEST["revision"] == "43791a0ade113a0ad5530c2a4948870dd0f7e417"
    notices = build.native_attention_notices(MANIFEST)
    for entry in [*MANIFEST["licenses"].values(), MANIFEST["cccl_notice"]]:
        data = (FAMILY / "third_party" / entry["notice_file"]).read_bytes()
        assert content_cache_key("hstu-native-license-v1", data) == entry["content_key"]
        assert data in notices
    assert b"Copyright (c) 2023, Tri Dao." in notices
    assert b"Duane Merrill" in notices
    assert b"LLVM Exceptions" in notices
    assert b"LicenseRef-NvidiaProprietary" not in notices


def test_source_with_matching_licenses_is_accepted(licensed_source):
    _, source, manifest = licensed_source
    assert build.verify_source(source) == manifest


@pytest.mark.parametrize("path", list(MANIFEST["licenses"]))
@pytest.mark.parametrize("mutation", ["missing", "truncated", "conflicting_terms"])
def test_source_license_changes_fail_before_compilation(licensed_source, path, mutation):
    root, source, _ = licensed_source
    license_file = root / path
    if mutation == "missing":
        license_file.unlink()
    elif mutation == "truncated":
        license_file.write_bytes(license_file.read_bytes().splitlines()[0])
    else:
        license_file.write_bytes(license_file.read_bytes() + b"\nLicenseRef-NvidiaProprietary\n")
    with pytest.raises(ValueError, match="license differs"):
        build.verify_source(source)


@pytest.mark.parametrize("entry", [*MANIFEST["licenses"].values(), MANIFEST["cccl_notice"]])
def test_redistributed_license_copy_must_match_source(licensed_source, entry):
    _, _, manifest = licensed_source
    (build.HERE / "third_party" / entry["notice_file"]).write_text("license omitted\n")
    with pytest.raises(ValueError, match="license differs"):
        build.native_attention_notices(manifest)
