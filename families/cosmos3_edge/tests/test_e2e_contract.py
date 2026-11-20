# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the existing CI runtime-layout adapter."""

import os
from families.cosmos3_edge.tests.test_e2e import _runner


def _driver(build):
    suffix = ".exe" if os.name == "nt" else ""
    path = build / "families/cosmos3_edge" / ("cosmos3_edge_inference_test" + suffix)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test executable")
    return path


def test_uses_declared_native_build_root(tmp_path, monkeypatch):
    build = tmp_path / "build"
    expected = _driver(build)
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(build))
    assert _runner(tmp_path / "runtime") == expected


def test_resolves_family_runtime_symlink(tmp_path, monkeypatch):
    build, runtime = tmp_path / "build", tmp_path / "runtime"
    expected = _driver(build)
    runtime.mkdir()
    library = build / "libtrtmc_model_cosmos3_edge.so"
    library.write_bytes(b"family library")
    (runtime / library.name).symlink_to(library)
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    assert _runner(runtime) == expected
