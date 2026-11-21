# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from families.modernbert.tests import test_e2e


@pytest.fixture
def sdk_layout(tmp_path, monkeypatch):
    binary = tmp_path / "wheel/bin/trtmc"
    binary.parent.mkdir(parents=True)
    binary.touch()
    native_build = tmp_path / "native-build"
    native_build.mkdir()
    consumers = [native_build / f"test_modernbert_sdk_{language}" for language in ("c", "cpp")]
    for consumer in consumers:
        consumer.touch()
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(native_build))
    monkeypatch.setattr(test_e2e, "record_evidence", lambda *args: None)
    return binary, native_build, consumers


def invoke(binary, tmp_path):
    return test_e2e._native(binary, tmp_path, tmp_path / "bundle", tmp_path,
                            {"task": "text_to_pooled_features"}, {"prompt": "hello"}, tmp_path)


def test_sdk_consumers_run_from_native_build_with_installed_cli(sdk_layout, tmp_path, monkeypatch):
    binary, _, consumers = sdk_layout
    calls = []
    result = {"task": "text_to_pooled_features", "pooling": "cls",
              "normalization": "none", "values": [0.25, -0.5]}

    def run(path, *args):
        calls.append(Path(path))
        return result

    monkeypatch.setattr(test_e2e, "_run_json", run)
    assert invoke(binary, tmp_path) == result
    assert calls == [binary, *consumers]


def test_sdk_consumers_require_native_build_directory(sdk_layout, tmp_path, monkeypatch):
    binary, _, _ = sdk_layout
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR")
    monkeypatch.setattr(test_e2e, "_run_json", lambda *args: {})
    with pytest.raises(AssertionError, match="TRTMC_NATIVE_BUILD_DIR"):
        invoke(binary, tmp_path)


@pytest.mark.parametrize("language", ("c", "cpp"))
def test_sdk_consumers_require_built_files(sdk_layout, tmp_path, monkeypatch, language):
    binary, native_build, _ = sdk_layout
    missing = native_build / f"test_modernbert_sdk_{language}"
    missing.unlink()
    result = {"task": "text_to_pooled_features", "pooling": "cls",
              "normalization": "none", "values": [0.25, -0.5]}
    monkeypatch.setattr(test_e2e, "_run_json", lambda *args: result)
    with pytest.raises(AssertionError, match=f"test_modernbert_sdk_{language}"):
        invoke(binary, tmp_path)
