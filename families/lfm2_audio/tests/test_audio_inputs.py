# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline E2E consumes declared inputs without replacing their PCM content."""

import shutil
import socket

import numpy as np
import pytest

from families.lfm2_audio.tests import test_e2e


@pytest.fixture
def recordings(tmp_path):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "staged"
    root.mkdir()
    expected = {}
    for index, name in enumerate(("asr", "question")):
        samples = np.array([0.0, -0.25, 0.5, 0.125], np.float32) * (index + 1)
        rate = (44100, 16000)[index]
        sf.write(root / (name + ".wav"), samples, rate, subtype="FLOAT")
        expected[name] = (samples, rate)
    return root, expected


def unavailable_network(*_args, **_kwargs):
    raise AssertionError("staged audio preparation attempted network access")


def test_staged_inputs_preserve_pcm_without_network(recordings, tmp_path, monkeypatch):
    root, expected = recordings
    evidence = []
    monkeypatch.setenv("TRTMC_LFM2_AUDIO_MODEL_DIR", str(root))
    monkeypatch.setattr(test_e2e.urllib.request, "urlretrieve", unavailable_network)
    monkeypatch.setattr(
        test_e2e, "record_evidence", lambda kind, item: evidence.append((kind, item))
    )
    sources = test_e2e.audio_sources(tmp_path)
    assert sources.keys() == expected.keys()
    for name, (path, rate) in sources.items():
        np.testing.assert_array_equal(np.fromfile(path, np.float32), expected[name][0])
        assert rate == expected[name][1]
    assert evidence == [
        (
            "inputs",
            {
                name: {
                    "declared_source_url": asset["url"],
                    "acquisition": "staged",
                    "file": root / asset["path"],
                }
            },
        )
        for name, asset in zip(expected, test_e2e.MANIFEST["external_files"], strict=True)
    ]


def test_missing_staged_input_fails_without_download(recordings, tmp_path, monkeypatch):
    root, _ = recordings
    (root / "question.wav").unlink()
    monkeypatch.setenv("TRTMC_LFM2_AUDIO_MODEL_DIR", str(root))
    monkeypatch.setattr(test_e2e.urllib.request, "urlretrieve", unavailable_network)
    with pytest.raises(FileNotFoundError, match="question.wav"):
        test_e2e.audio_sources(tmp_path)


@pytest.mark.parametrize("hub_offline", [False, True])
def test_download_uses_declared_inputs_independent_of_hub_mode(
    recordings, tmp_path, monkeypatch, hub_offline
):
    from huggingface_hub import constants

    root, expected = recordings
    monkeypatch.delenv("TRTMC_LFM2_AUDIO_MODEL_DIR", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", hub_offline)
    urls = []
    evidence = []

    def download(url, path):
        urls.append(url)
        shutil.copyfile(root / path.name, path)

    monkeypatch.setattr(test_e2e.urllib.request, "urlretrieve", download)
    monkeypatch.setattr(
        test_e2e, "record_evidence", lambda kind, item: evidence.append((kind, item))
    )
    sources = test_e2e.audio_sources(tmp_path)
    prefix = (
        test_e2e.MANIFEST["reference_repository"].replace("github.com", "raw.githubusercontent.com")
        + "/"
        + test_e2e.MANIFEST["reference_revision"]
        + "/assets/"
    )
    assert urls == [prefix + name + ".wav" for name in expected]
    assert evidence == [
        (
            "inputs",
            {
                name: {
                    "declared_source_url": prefix + name + ".wav",
                    "acquisition": "downloaded",
                    "file": tmp_path / (name + ".wav"),
                }
            },
        )
        for name in expected
    ]
    for name, (path, rate) in sources.items():
        np.testing.assert_array_equal(np.fromfile(path, np.float32), expected[name][0])
        assert rate == expected[name][1]


@pytest.mark.parametrize("offline,raw_flag", [(False, "0"), (True, "yes")])
def test_checkpoint_resolver_honors_parsed_offline_mode(tmp_path, monkeypatch, offline, raw_flag):
    import huggingface_hub
    from huggingface_hub import constants

    calls = []

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setenv("HF_HUB_OFFLINE", raw_flag)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert test_e2e.checkpoint_directory() == tmp_path
    assert calls == [
        (
            test_e2e.MANIFEST["hf_id"],
            {"revision": test_e2e.MANIFEST["hf_revision"], "local_files_only": offline},
        )
    ]


def staged_checkpoint(tmp_path, monkeypatch):
    from huggingface_hub import HfApi, constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(HfApi, "repo_info", unavailable_network)
    monkeypatch.setattr(HfApi, "list_repo_tree", unavailable_network)
    monkeypatch.setattr(socket.socket, "connect", unavailable_network)
    return (
        tmp_path
        / ("models--" + test_e2e.MANIFEST["hf_id"].replace("/", "--"))
        / "snapshots"
        / test_e2e.MANIFEST["hf_revision"]
    )


def test_actual_hub_uses_staged_files_without_tree_metadata(tmp_path, monkeypatch):
    snapshot = staged_checkpoint(tmp_path, monkeypatch)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert test_e2e.checkpoint_directory() == snapshot


def test_missing_cached_checkpoint_fails_without_network(tmp_path, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    staged_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(LocalEntryNotFoundError):
        test_e2e.checkpoint_directory()
