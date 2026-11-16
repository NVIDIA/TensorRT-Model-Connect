# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the same offline asset paths used by protected family tests."""

import httpx
import pytest
from huggingface_hub import constants
from PIL import Image

from . import test_e2e as e2e


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.delenv("TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))

    def unexpected_network(*args, **kwargs):
        pytest.fail("checkpoint resolution attempted a network request")

    monkeypatch.setattr(httpx.Client, "request", unexpected_network)
    return cache


def test_manifest_revision_resolves_from_offline_cache(offline_cache):
    root = offline_cache / ("models--" + e2e.MANIFEST["hf_id"].replace("/", "--"))
    pinned = root / "snapshots" / e2e.MANIFEST["hf_revision"]
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}")
    other = root / "snapshots" / ("f" * 40)
    other.mkdir()
    (root / "refs").mkdir()
    (root / "refs/main").write_text(other.name)

    assert e2e._model_dir() == pinned


def test_missing_pinned_checkpoint_fails_without_network(offline_cache):
    with pytest.raises(AssertionError, match="exact cached checkpoint"):
        e2e._model_dir()


def test_explicit_model_directory_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR", str(tmp_path))
    assert e2e._model_dir() == tmp_path


@pytest.mark.parametrize("kind", ["missing", "file", "empty"])
def test_invalid_explicit_model_does_not_fall_back(offline_cache, tmp_path, monkeypatch, kind):
    cached = (
        offline_cache
        / ("models--" + e2e.MANIFEST["hf_id"].replace("/", "--"))
        / "snapshots"
        / e2e.MANIFEST["hf_revision"]
    )
    cached.mkdir(parents=True)
    path = tmp_path / "model"
    if kind == "file":
        path.write_text("not a directory")
    monkeypatch.setenv("TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR", "" if kind == "empty" else str(path))
    with pytest.raises(AssertionError):
        e2e._model_dir()


def test_provisioned_document_needs_no_environment_override(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE", raising=False)
    path = tmp_path / "layout_demo.jpg"
    Image.new("RGB", (4, 4)).save(path)
    assert e2e._image_path(tmp_path) == path


def test_missing_document_explains_offline_provisioning(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE", raising=False)
    with pytest.raises(AssertionError, match="provision the official"):
        e2e._image_path(tmp_path)


def test_explicit_image_path_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / "document.png"
    Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setenv("TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE", str(path))
    assert e2e._image_path(tmp_path) == path


@pytest.mark.parametrize("kind", ["missing", "directory", "empty"])
def test_invalid_explicit_image_does_not_fall_back(tmp_path, monkeypatch, kind):
    path = tmp_path / "document.png"
    if kind == "directory":
        path.mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "layout_demo.jpg")
    monkeypatch.setenv("TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE", "" if kind == "empty" else str(path))
    with pytest.raises(AssertionError):
        e2e._image_path(tmp_path)
