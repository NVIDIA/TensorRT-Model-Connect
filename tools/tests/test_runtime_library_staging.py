# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared native dependencies survive isolation without family-prefix scans."""

import json
from types import SimpleNamespace

import pytest

from tools.ci.e2e import E2ERunner
from tools.ci.package import native_runtime_libraries
from tools.ci.process import CiError


_REQUIRED = (
    "libtrtmc_core.so",
    "libtrtmc_runtime.so",
    "libtrtmc_c.so",
    "libtrtmc_c.so.1",
    "libtrtmc_backend_trt.so",
    "libtrtmc_model_alpha.so",
)


def setup_tree(tmp_path, libraries=None):
    artifact = tmp_path / "actual"
    staged = tmp_path / "outer"
    artifact.mkdir()
    staged.mkdir()
    for name in _REQUIRED:
        (artifact / name).write_bytes(name.encode())
        (staged / name).symlink_to(artifact / name)
    declaration = artifact / "families/alpha/runtime-libraries.json"
    if libraries is not None:
        declaration.parent.mkdir(parents=True)
        declaration.write_text(json.dumps({"schema_version": 1, "libraries": libraries}))
    runner = E2ERunner(SimpleNamespace(repository=tmp_path))
    return artifact, staged, declaration, runner


def test_outer_omission_is_recovered_only_from_selected_artifact(tmp_path):
    names = ["libtrtmc_alpha_sampler.so", "libtrtmc_alpha_w2.so", "libtrtmc_byok_tvm_ffi.so"]
    artifact, staged, declaration, runner = setup_tree(tmp_path, names)
    for name in names + ["libtrtmc_alpha_beta_sampler.so", "libtrtmc_model_beta.so"]:
        (artifact / name).write_bytes(name.encode())
    assert not (staged / "families").exists()
    with runner._isolated_runtime_root(staged, "alpha") as isolated:
        for name in names:
            assert (isolated / name).is_symlink()
            assert (isolated / name).resolve() == artifact / name
        assert (isolated / "families/alpha/runtime-libraries.json").resolve() == declaration
        assert not (isolated / "libtrtmc_alpha_beta_sampler.so").exists()
        assert not (isolated / "libtrtmc_model_beta.so").exists()


@pytest.mark.parametrize("libraries", [None, []])
def test_legacy_and_empty_declarations_preserve_existing_files(tmp_path, libraries):
    artifact, staged, declaration, runner = setup_tree(tmp_path, libraries)
    (artifact / "libtrtmc_alpha_optional.so").write_bytes(b"unlisted")
    with runner._isolated_runtime_root(staged, "alpha") as isolated:
        assert all((isolated / name).is_file() for name in _REQUIRED)
        assert not (isolated / "libtrtmc_alpha_optional.so").exists()
        assert (isolated / "families/alpha/runtime-libraries.json").exists() == (libraries == [])


@pytest.mark.parametrize(
    "entry",
    [
        "../libescape.so",
        "/tmp/libescape.so",
        "sub/libescape.so",
        "libtrtmc_model_beta.so",
        "libtrtmc_cli_beta.so",
        "libtrtmc_model_alpha.so",
        "libtrtmc_cli_alpha.so",
        "",
        None,
        1,
    ],
)
def test_invalid_or_plugin_entries_are_rejected(tmp_path, entry):
    _, staged, _, runner = setup_tree(tmp_path, [entry])
    with pytest.raises(CiError, match="runtime library declaration"):
        with runner._isolated_runtime_root(staged, "alpha"):
            pytest.fail("invalid closure was staged")


@pytest.mark.parametrize(
    "data",
    [
        {"schema_version": 2, "libraries": []},
        {"schema_version": True, "libraries": []},
        {"schema_version": 1, "libraries": "libexample.so"},
        {"schema_version": 1, "libraries": [], "unknown": 1},
        [],
    ],
)
def test_schema_is_strict(tmp_path, data):
    _, staged, declaration, _ = setup_tree(tmp_path, [])
    declaration.write_text(json.dumps(data))
    with pytest.raises(CiError, match="runtime library declaration"):
        native_runtime_libraries(staged / "libtrtmc_model_alpha.so", "alpha")


def test_duplicate_and_missing_libraries_fail_without_fallback(tmp_path):
    artifact, staged, declaration, _ = setup_tree(tmp_path, ["libhelper.so", "libhelper.so"])
    (artifact / "libhelper.so").write_bytes(b"helper")
    with pytest.raises(CiError, match="duplicate"):
        native_runtime_libraries(staged / "libtrtmc_model_alpha.so", "alpha")
    declaration.write_text(json.dumps({"schema_version": 1, "libraries": ["libmissing.so"]}))
    (staged / "libmissing.so").write_bytes(b"not the selected artifact")
    with pytest.raises(CiError, match="missing"):
        native_runtime_libraries(staged / "libtrtmc_model_alpha.so", "alpha")


@pytest.mark.parametrize("name", ["libtrtmc_core.so", "libtrtmc_byok_tvm_ffi.so"])
def test_conflicting_existing_core_or_bridge_is_rejected(tmp_path, name):
    artifact, staged, _, runner = setup_tree(tmp_path, [name])
    (artifact / name).write_bytes(b"selected")
    if (staged / name).is_symlink():
        (staged / name).unlink()
    (staged / name).write_bytes(b"other build")
    with pytest.raises(CiError, match="conflicting staged runtime library"):
        with runner._isolated_runtime_root(staged, "alpha"):
            pytest.fail("mixed builds were accepted")


def test_identical_existing_bridge_is_retained(tmp_path):
    name = "libtrtmc_byok_tvm_ffi.so"
    artifact, staged, _, runner = setup_tree(tmp_path, [name])
    (artifact / name).write_bytes(b"bridge")
    (staged / name).symlink_to(artifact / name)
    with runner._isolated_runtime_root(staged, "alpha") as isolated:
        assert (isolated / name).resolve() == artifact / name


def test_declared_library_cannot_redirect_to_other_artifact(tmp_path):
    artifact, staged, _, _ = setup_tree(tmp_path, ["libhelper.so"])
    other = tmp_path / "libhelper.so"
    other.write_bytes(b"other")
    (artifact / "libhelper.so").symlink_to(other)
    with pytest.raises(CiError, match="outside"):
        native_runtime_libraries(staged / "libtrtmc_model_alpha.so", "alpha")


def test_declaration_cannot_redirect_to_other_family(tmp_path):
    artifact, staged, declaration, _ = setup_tree(tmp_path, [])
    other = artifact / "families/beta/runtime-libraries.json"
    other.parent.mkdir()
    other.write_text(declaration.read_text())
    declaration.unlink()
    declaration.symlink_to(other)
    with pytest.raises(CiError, match="owning artifact directory"):
        native_runtime_libraries(staged / "libtrtmc_model_alpha.so", "alpha")


def test_declared_soname_is_preserved_for_versioned_library(tmp_path):
    artifact, staged, _, runner = setup_tree(tmp_path, ["libhelper.so"])
    versioned = artifact / "libhelper.so.1"
    versioned.write_bytes(b"versioned helper")
    (artifact / "libhelper.so").symlink_to(versioned)
    with runner._isolated_runtime_root(staged, "alpha") as isolated:
        assert (isolated / "libhelper.so").resolve() == versioned
        assert not (isolated / "libhelper.so.1").exists()
