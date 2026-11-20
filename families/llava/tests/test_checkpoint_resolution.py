# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the declared checkpoint in offline CI without an explicit override."""

import json
from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest

from families.llava.tests.test_native_inputs import _checkpoint_dir


def test_cached_checkpoint_without_model_override(monkeypatch, tmp_path):
    monkeypatch.delenv("TRTMC_LLAVA_MODEL_DIR", raising=False)
    manifest = json.loads((Path(__file__).parent / "manifests/llava-1.5-7b.json").read_text())
    calls = []

    def cached_snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_snapshot)
    assert _checkpoint_dir() == tmp_path
    assert calls == [{"repo_id": manifest["hf_id"], "revision": manifest["hf_revision"],
                      "local_files_only": True}]


def test_explicit_checkpoint_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTMC_LLAVA_MODEL_DIR", str(tmp_path))

    def unexpected_lookup(**kwargs):
        raise AssertionError("An explicit checkpoint must not query the Hub cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_lookup)
    assert _checkpoint_dir() == tmp_path


def test_invalid_explicit_checkpoint_fails(monkeypatch, tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_text("checkpoint")
    monkeypatch.setenv("TRTMC_LLAVA_MODEL_DIR", str(path))
    with pytest.raises(AssertionError, match="must be a checkpoint directory"):
        _checkpoint_dir()


def test_missing_cache_skips_source_only_environment(monkeypatch):
    monkeypatch.delenv("TRTMC_LLAVA_MODEL_DIR", raising=False)

    def missing_snapshot(**kwargs):
        raise LocalEntryNotFoundError("No cached snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing_snapshot)
    with pytest.raises(pytest.skip.Exception, match="pinned checkpoint tokenizer"):
        _checkpoint_dir()


def test_cache_errors_are_not_skipped(monkeypatch):
    monkeypatch.delenv("TRTMC_LLAVA_MODEL_DIR", raising=False)

    def unreadable_snapshot(**kwargs):
        raise PermissionError("Unreadable checkpoint cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unreadable_snapshot)
    with pytest.raises(PermissionError, match="Unreadable checkpoint cache"):
        _checkpoint_dir()
