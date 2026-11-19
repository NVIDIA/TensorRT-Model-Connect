# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the staged image and checkpoint without E2E network access."""

import socket
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from . import test_e2e as e2e


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("TRTMC_BLIP_MODEL_DIR", "TRTMC_BLIP_REFERENCE_IMAGE", "TRTMC_BLIP_IMAGE"):
        monkeypatch.delenv(name, raising=False)

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("BLIP E2E asset resolution must not access the network")

    monkeypatch.setattr(socket, "create_connection", unexpected_download)
    monkeypatch.setattr(socket, "getaddrinfo", unexpected_download)
    monkeypatch.setattr(urllib.request, "urlretrieve", unexpected_download)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected_download)
    )


def stage_image(directory):
    directory.mkdir(exist_ok=True)
    image = directory / e2e.MANIFEST["external_files"][0]["path"]
    image.write_bytes(b"the same pinned image")
    return image


def stage_checkpoint(directory):
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text("{}")
    return directory


def cached_checkpoint(monkeypatch, directory):
    checkpoint = stage_checkpoint(directory)

    def snapshot(repo_id, *, revision, local_files_only):
        assert repo_id == e2e.MANIFEST["hf_id"]
        assert revision == e2e.MANIFEST["hf_revision"]
        assert local_files_only is True
        return str(checkpoint)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    return checkpoint


def test_ci_image_directory_and_cached_checkpoint_are_separate(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "external")
    checkpoint = cached_checkpoint(monkeypatch, tmp_path / "hub-snapshot")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(image.parent))

    assert e2e.input_image() == image
    assert e2e.checkpoint_dir() == checkpoint


def test_local_checkpoint_and_explicit_reference_need_no_hub(tmp_path, monkeypatch):
    checkpoint = stage_checkpoint(tmp_path / "checkpoint")
    image = stage_image(tmp_path / "reference")
    copy = tmp_path / "input.jpg"
    copy.write_bytes(image.read_bytes())
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(checkpoint))
    monkeypatch.setenv("TRTMC_BLIP_REFERENCE_IMAGE", str(image))
    monkeypatch.setenv("TRTMC_BLIP_IMAGE", str(copy))

    assert e2e.input_image() == copy
    assert e2e.checkpoint_dir() == checkpoint


def test_reference_override_works_with_cached_checkpoint(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "reference")
    checkpoint = cached_checkpoint(monkeypatch, tmp_path / "hub-snapshot")
    monkeypatch.setenv("TRTMC_BLIP_REFERENCE_IMAGE", str(image))

    assert e2e.input_image() == image
    assert e2e.checkpoint_dir() == checkpoint


def test_combined_local_assets_do_not_require_cached_checkpoint(tmp_path, monkeypatch):
    checkpoint = stage_checkpoint(tmp_path / "checkpoint")
    image = stage_image(checkpoint)
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(checkpoint))

    assert e2e.input_image() == image
    assert e2e.checkpoint_dir() == checkpoint


def test_missing_image_fails_before_loading_torch_or_model():
    with pytest.raises(FileNotFoundError, match="TRTMC_BLIP_REFERENCE_IMAGE"):
        e2e.artifacts.__wrapped__(None, None)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_missing_staged_image_never_downloads(tmp_path, monkeypatch, kind):
    if kind == "directory":
        (tmp_path / e2e.MANIFEST["external_files"][0]["path"]).mkdir()
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="reference image was not staged"):
        e2e.input_image()


def test_explicit_missing_reference_does_not_fall_back_to_staged_image(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "external")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(image.parent))
    monkeypatch.setenv("TRTMC_BLIP_REFERENCE_IMAGE", str(tmp_path / "missing.jpg"))

    with pytest.raises(FileNotFoundError, match="reference image was not staged"):
        e2e.input_image()


def test_different_input_bytes_cannot_replace_the_reference(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "external")
    other = tmp_path / "different.jpg"
    other.write_bytes(b"different image")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(image.parent))
    monkeypatch.setenv("TRTMC_BLIP_IMAGE", str(other))

    with pytest.raises(AssertionError, match="must match the pinned image"):
        e2e.input_image()


def test_missing_input_copy_is_not_replaced_with_the_reference(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "external")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(image.parent))
    monkeypatch.setenv("TRTMC_BLIP_IMAGE", str(tmp_path / "missing.jpg"))

    with pytest.raises(FileNotFoundError):
        e2e.input_image()


def test_invalid_explicit_model_directory_does_not_use_cached_checkpoint(tmp_path, monkeypatch):
    cached_checkpoint(monkeypatch, tmp_path / "hub-snapshot")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="neither config.json nor the staged image"):
        e2e.checkpoint_dir()


def test_missing_cached_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    image = stage_image(tmp_path / "external")
    monkeypatch.setenv("TRTMC_BLIP_MODEL_DIR", str(image.parent))

    def missing_snapshot(_repo_id, *, revision, local_files_only):
        assert revision == e2e.MANIFEST["hf_revision"]
        assert local_files_only is True
        raise FileNotFoundError("pinned snapshot was not staged")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=missing_snapshot)
    )
    with pytest.raises(FileNotFoundError, match="pinned snapshot was not staged"):
        e2e.checkpoint_dir()
