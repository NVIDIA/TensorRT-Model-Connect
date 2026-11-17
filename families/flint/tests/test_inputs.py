# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the same cache and helper paths provided by protected CI."""

import pytest

from families.flint.tests.inputs import MANIFEST, checkpoint_dir, native_binary


def test_checkpoint_uses_pinned_offline_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.delenv("TRTMC_FLINT_CHECKPOINT", raising=False)
    cache = tmp_path / "hub"
    repository = cache / ("models--" + MANIFEST["hf_id"].replace("/", "--"))
    pinned = repository / "snapshots" / MANIFEST["hf_revision"]
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}")
    other = repository / "snapshots" / ("0" * 40)
    other.mkdir()
    (other / "config.json").write_text("{}")
    (repository / "refs").mkdir()
    (repository / "refs/main").write_text(other.name)
    download = huggingface_hub.snapshot_download
    calls = []

    def cached_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return download(repo_id, cache_dir=cache, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_download)

    assert checkpoint_dir() == pinned
    assert calls == [(MANIFEST["hf_id"], {
        "revision": MANIFEST["hf_revision"], "local_files_only": True
    })]


@pytest.mark.parametrize("value", ["", "missing", "incomplete"])
def test_invalid_checkpoint_override_fails(monkeypatch, tmp_path, value):
    if value == "incomplete":
        (tmp_path / value).mkdir()
    monkeypatch.setenv("TRTMC_FLINT_CHECKPOINT", str(tmp_path / value) if value else "")
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoint_dir()


def test_valid_checkpoint_override(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("TRTMC_FLINT_CHECKPOINT", str(tmp_path))
    assert checkpoint_dir() == tmp_path


@pytest.mark.parametrize("name", ["flint_tokenizer_probe", "flint_logits_trace"])
def test_helpers_use_native_build_separate_from_wheel(monkeypatch, tmp_path, name):
    native = tmp_path / "native"
    native.mkdir()
    (native / name).touch()
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(native))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path / "wheel"))
    assert native_binary(name) == native / name


def test_missing_native_helper_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="flint_logits_trace"):
        native_binary("flint_logits_trace")


def test_missing_native_build_fails(monkeypatch):
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    monkeypatch.delenv("TRTMC_RUNTIME_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="TRTMC_NATIVE_BUILD_DIR"):
        native_binary("flint_tokenizer_probe")
