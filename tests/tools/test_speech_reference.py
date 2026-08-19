# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_model_connect.models.canary.tools import reference as canary_reference


def _install_fake_nemo_asr(
    monkeypatch,
    *,
    model_class: type,
) -> None:
    nemo = ModuleType("nemo")
    nemo.__path__ = []
    collections = ModuleType("nemo.collections")
    collections.__path__ = []
    asr = ModuleType("nemo.collections.asr")
    asr.models = SimpleNamespace(ASRModel=model_class)
    nemo.collections = collections
    collections.asr = asr
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)


def _recording_model(
    from_pretrained_calls: list[tuple[object, ...]],
    restore_calls: list[dict[str, object]],
) -> type:
    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            from_pretrained_calls.append((*args, kwargs))
            return cls()

        @classmethod
        def restore_from(cls, **kwargs):
            restore_calls.append(kwargs)
            return cls()

        def eval(self):
            return self

    return FakeModel


@pytest.mark.parametrize(
    "revision",
    [
        "87bc52657add533cd0156b3fc1aef027280754bf",
        "",
    ],
)
def test_canary_offline_reference_restores_cached_archive(
    tmp_path: Path,
    monkeypatch,
    revision: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    archive = snapshot / "canary-1b-v2.nemo"
    archive.write_bytes(b"checkpoint")
    download_calls: list[dict[str, object]] = []
    from_pretrained_calls: list[tuple[object, ...]] = []
    restore_calls: list[dict[str, object]] = []

    def fake_snapshot_download(*args, **kwargs):
        assert not args
        download_calls.append(kwargs)
        return str(snapshot)

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    _install_fake_nemo_asr(
        monkeypatch,
        model_class=_recording_model(from_pretrained_calls, restore_calls),
    )

    responses = canary_reference.run(
        SimpleNamespace(
            model="nvidia/canary-1b-v2",
            model_revision=revision,
            local_files_only=True,
            device="cpu",
            predictions=tmp_path / "hf_predictions.json",
        ),
        {"generation": {"sample_rate": 16000}},
        [],
    )

    assert responses == []
    expected_download = {
        "repo_id": "nvidia/canary-1b-v2",
        "allow_patterns": ["*.nemo"],
        "local_files_only": True,
    }
    if revision:
        expected_download["revision"] = revision
    assert download_calls == [expected_download]
    assert from_pretrained_calls == []
    assert restore_calls == [
        {
            "restore_path": str(archive),
            "map_location": "cpu",
        }
    ]


def test_canary_online_reference_keeps_nemo_from_pretrained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from_pretrained_calls: list[tuple[object, ...]] = []
    restore_calls: list[dict[str, object]] = []

    _install_fake_nemo_asr(
        monkeypatch,
        model_class=_recording_model(from_pretrained_calls, restore_calls),
    )

    responses = canary_reference.run(
        SimpleNamespace(
            model="nvidia/canary-1b-v2",
            model_revision="ignored-by-existing-online-path",
            local_files_only=False,
            device="cpu",
            predictions=tmp_path / "hf_predictions.json",
        ),
        {"generation": {"sample_rate": 16000}},
        [],
    )

    assert responses == []
    assert from_pretrained_calls == [
        (
            "nvidia/canary-1b-v2",
            {"map_location": "cpu"},
        )
    ]
    assert restore_calls == []
