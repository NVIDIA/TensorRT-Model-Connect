# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError

from families.musicgen.tests.inputs import MANIFEST, checkpoint_dir, tokenizer_binary


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    monkeypatch.delenv("TRTMC_MUSICGEN_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(root))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root))
    return root / "models--facebook--musicgen-medium"


def test_offline_cache_uses_the_manifest_revision_instead_of_main(cache):
    pinned = cache / "snapshots" / MANIFEST["hf_revision"]
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}")
    other = cache / "snapshots" / ("0" * 40)
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    (cache / "refs").mkdir()
    (cache / "refs/main").write_text(other.name)
    assert checkpoint_dir() == pinned
    assert checkpoint_dir(MANIFEST["hf_id"], other.name) == other


def test_missing_pinned_cache_fails_offline(cache):
    with pytest.raises(LocalEntryNotFoundError):
        checkpoint_dir()


def test_explicit_checkpoint_does_not_require_a_hub_cache(cache, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    monkeypatch.setenv("TRTMC_MUSICGEN_MODEL_DIR", str(root))
    assert checkpoint_dir() == root


@pytest.mark.parametrize("kind", ["empty", "missing", "file", "incomplete"])
def test_invalid_explicit_checkpoint_is_rejected(cache, tmp_path, monkeypatch, kind):
    root = tmp_path / "invalid"
    if kind == "file":
        root.write_text("not a checkpoint")
    elif kind == "incomplete":
        root.mkdir()
    monkeypatch.setenv("TRTMC_MUSICGEN_MODEL_DIR", "" if kind == "empty" else str(root))
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoint_dir()


def test_tokenizer_probe_uses_the_native_build_directory(tmp_path, monkeypatch):
    binary = tmp_path / "test_musicgen_tokenizer"
    binary.touch()
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path / "packaged-libraries"))
    assert tokenizer_binary() == binary


def test_tokenizer_probe_supports_a_direct_runtime_build(tmp_path, monkeypatch):
    binary = tmp_path / "test_musicgen_tokenizer"
    binary.touch()
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    assert tokenizer_binary() == binary


def test_missing_tokenizer_probe_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="test_musicgen_tokenizer"):
        tokenizer_binary()
