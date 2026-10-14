# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline E2E consumes the declared input without changing either model's pixels."""

from types import SimpleNamespace
from io import BytesIO
import urllib.request

import numpy as np
from PIL import Image
import pytest

from . import test_e2e


@pytest.fixture
def processor_stub():
    # Rendering has separate original-reference fixtures. This transport control
    # only supplies the context count required to construct the input cases.
    return SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "rendered prompt",
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: "rendered history",
            encode=lambda *args, **kwargs: range(32760),
        ),
    )


@pytest.mark.parametrize("offline", [False, True])
def test_checkpoint_resolution_preserves_parsed_offline_policy(monkeypatch, tmp_path, offline):
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.delenv("TRTMC_LFM25_VL_CHECKPOINT", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    calls = []

    def snapshot(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert test_e2e._checkpoint() == tmp_path
    assert calls == [
        (
            (test_e2e.CHECKPOINT,),
            {"revision": test_e2e.REVISION, "local_files_only": offline},
        )
    ]


def test_local_checkpoint_override_avoids_hub_resolution(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setenv("TRTMC_LFM25_VL_CHECKPOINT", str(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail("local checkpoint override attempted Hub resolution")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    assert test_e2e._checkpoint() == tmp_path


def test_manifest_declares_original_model_card_photo():
    assert test_e2e.MANIFEST["external_files"] == [
        {
            "path": "model-card.jpg",
            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        }
    ]


def test_staged_input_resolver_requires_directory(monkeypatch):
    monkeypatch.delenv("TRTMC_LFM2_VL_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="TRTMC_LFM2_VL_MODEL_DIR"):
        test_e2e._model_card_image()


def test_missing_staged_photo_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTMC_LFM2_VL_MODEL_DIR", str(tmp_path))

    def deny_network(*args, **kwargs):
        pytest.fail("missing explicitly staged input attempted a network fallback")

    monkeypatch.setattr(urllib.request, "urlopen", deny_network)
    with pytest.raises(FileNotFoundError, match="staged model-card image is missing"):
        test_e2e._fixtures(tmp_path, None)


@pytest.mark.parametrize("hub_offline", ["0", "1"])
def test_unstaged_input_downloads_independently_of_hub_mode(
    monkeypatch, tmp_path, processor_stub, hub_offline
):
    monkeypatch.delenv("TRTMC_LFM2_VL_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", hub_offline)
    encoded = BytesIO()
    Image.new("RGB", (11, 7), "red").save(encoded, format="PNG")
    data = encoded.getvalue()
    calls, evidence = [], []

    def download(url, *, timeout):
        calls.append((url, timeout))
        return BytesIO(data)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(test_e2e, "record_evidence", lambda *entry: evidence.append(entry))
    test_e2e._fixtures(tmp_path, processor_stub)
    assert calls == [(test_e2e.CARD_IMAGE, 60)]
    assert (tmp_path / "model-card.jpg").read_bytes() == data
    assert evidence[0][1]["model_card_image"] == {
        "declared_source_url": test_e2e.CARD_IMAGE,
        "acquisition": "downloaded",
        "file": tmp_path / "model-card.jpg",
    }


def test_offline_input_retains_bytes_evidence_and_shared_pixels(
    monkeypatch, tmp_path, processor_stub
):
    source = tmp_path / "staged"
    source.mkdir()
    pixels = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    Image.fromarray(pixels).save(source / "model-card.jpg", format="PNG")
    monkeypatch.setenv("TRTMC_LFM2_VL_MODEL_DIR", str(source))

    def deny_network(*args, **kwargs):
        pytest.fail("offline E2E attempted a network download")

    monkeypatch.setattr(urllib.request, "urlopen", deny_network)
    evidence = []
    monkeypatch.setattr(test_e2e, "record_evidence", lambda *entry: evidence.append(entry))
    output = tmp_path / "inputs"
    output.mkdir()
    images, cases = test_e2e._fixtures(output, processor_stub)
    photo = output / "model-card.jpg"
    assert photo.read_bytes() == (source / "model-card.jpg").read_bytes()
    assert evidence == [
        (
            "inputs",
            {
                "model_card_image": {
                    "declared_source_url": test_e2e.CARD_IMAGE,
                    "acquisition": "staged",
                    "file": photo,
                }
            },
        )
    ]
    for case in cases:
        assert len(case["images"]) == len(case["media_names"])
        for media, name in zip(case["images"], case["media_names"]):
            native = np.fromfile(media["path"], dtype=np.float32).reshape(
                media["height"], media["width"], 3
            )
            assert np.array_equal(native, np.asarray(images[name], dtype=np.float32))
