# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise CI's pinned cache and separate native-build/runtime directories."""

import json
from pathlib import Path

from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest

from families.molmo2.tests.inputs import checkpoint_directory, native_tool, tokenizer_file

MANIFEST = json.loads((Path(__file__).parent / "manifests/molmo2-8b.json").read_text())


@pytest.fixture
def cached_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTMC_MOLMO2_MODEL_DIR", raising=False)
    monkeypatch.delenv("TRTMC_MOLMO2_TOKENIZER_DIR", raising=False)
    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    repository = cache / ("models--" + MANIFEST["hf_id"].replace("/", "--"))
    directory = repository / "snapshots" / MANIFEST["hf_revision"]
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer.json").write_text("{}")
    (repository / "refs").mkdir()
    (repository / "refs/main").write_text("0" * 40)
    other = repository / "snapshots" / ("0" * 40)
    other.mkdir()
    (other / "config.json").write_text("{}")
    (other / "tokenizer.json").write_text("{}")
    return directory


def test_pinned_cache_without_model_override(cached_checkpoint):
    assert checkpoint_directory(MANIFEST["hf_id"], MANIFEST["hf_revision"]) == cached_checkpoint
    assert tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"]) == (
        cached_checkpoint / "tokenizer.json"
    )


def test_missing_pinned_cache_does_not_use_main(cached_checkpoint):
    cached_checkpoint.rename(cached_checkpoint.with_name("1" * 40))
    with pytest.raises(LocalEntryNotFoundError):
        checkpoint_directory(MANIFEST["hf_id"], MANIFEST["hf_revision"])


@pytest.mark.parametrize("name", ["TRTMC_MOLMO2_MODEL_DIR", "TRTMC_MOLMO2_TOKENIZER_DIR"])
@pytest.mark.parametrize("kind", ["empty", "missing", "file"])
def test_invalid_explicit_directory_does_not_use_cache(cached_checkpoint, monkeypatch, name, kind):
    path = cached_checkpoint / kind
    if kind == "file":
        path.write_text("not a directory")
    monkeypatch.setenv(name, "" if kind == "empty" else str(path))
    with pytest.raises(AssertionError, match=name):
        tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"])


def test_explicit_checkpoint_directory(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.delenv("TRTMC_MOLMO2_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("TRTMC_MOLMO2_MODEL_DIR", str(tmp_path))
    assert checkpoint_directory(MANIFEST["hf_id"], MANIFEST["hf_revision"]) == tmp_path
    assert tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"]) == tmp_path / "tokenizer.json"


def test_explicit_tokenizer_directory_needs_no_weights_or_model(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setenv("TRTMC_MOLMO2_TOKENIZER_DIR", str(tmp_path))
    monkeypatch.setenv("TRTMC_MOLMO2_MODEL_DIR", str(tmp_path / "missing-model"))
    assert tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"]) == tmp_path / "tokenizer.json"


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json"])
def test_missing_checkpoint_input_fails(cached_checkpoint, filename):
    (cached_checkpoint / filename).unlink()
    with pytest.raises(AssertionError, match=filename):
        tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"])


@pytest.mark.parametrize(
    "name", ["test_molmo2_preprocess", "test_molmo2_tokenizer", "molmo2_qualification"]
)
def test_native_tools_use_build_separate_from_packaged_runtime(tmp_path, monkeypatch, name):
    native = tmp_path / "native"
    runtime = tmp_path / "wheel/bin"
    runtime.mkdir(parents=True)
    tool = native / "families/molmo2" / name
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(native))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(runtime))
    assert native_tool(name) == tool


def test_native_tool_can_use_combined_local_build(tmp_path, monkeypatch):
    tool = tmp_path / "families/molmo2/test_molmo2_tokenizer"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    assert native_tool(tool.name) == tool
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path / "missing-native"))
    with pytest.raises(AssertionError, match="not executable"):
        native_tool(tool.name)


def test_native_tool_must_be_executable(tmp_path, monkeypatch):
    tool = tmp_path / "families/molmo2/test_molmo2_tokenizer"
    tool.parent.mkdir(parents=True)
    tool.write_text("not executable")
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="not executable"):
        native_tool(tool.name)
