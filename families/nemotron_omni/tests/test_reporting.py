# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch result IDs select their own prompt and media, including text-only inputs."""

import copy
import json
from types import SimpleNamespace

import numpy as np

from families.nemotron_omni.tests import reporting, test_e2e as e2e
from tools import e2e_evidence


def _requests(tmp_path):
    image, frame, audio = (tmp_path / name for name in ("image.rgb", "frame-7.rgb", "audio.f32"))
    np.arange(12, dtype=np.uint8).tofile(image)
    np.arange(12, dtype=np.uint8)[::-1].tofile(frame)
    np.asarray([0.0, 0.5, -0.5], dtype="<f4").tofile(audio)
    media = {
        "image": {"path": str(image), "height": 2, "width": 2},
        "video": {"frames": [{"path": str(frame), "height": 2, "width": 2}], "timestamps": [0.7]},
        "audio": {"path": str(audio), "sample_rate": 16000, "channels": 1},
    }
    combinations = {"text": [], "image": ["image"], "audio": ["audio"], "video": ["video"],
                    "image_video": ["image", "video"], "audio_video": ["audio", "video"]}
    return [{"id": name, "prompt": name + " prompt", **{key: media[key] for key in keys}}
            for name, keys in combinations.items()]


def test_original_batch_entrypoint_binds_by_id_without_borrowing_media(tmp_path, monkeypatch):
    requests = _requests(tmp_path)
    original = copy.deepcopy(requests)
    expected = [{"id": row["id"], "text": row["id"] + " answer", "token_ids": [1], "status": "passed"}
                for row in requests]
    case = {"name": "report", "modalities": [row["id"] for row in requests]}
    manifest = {"name": "report", "family": "nemotron_omni", "hf_id": "example/model",
                "task": "vision_language_generation", "precision": "fp32", "quantization": None,
                "max_sequence_length": 128, "tensor_parallel_size": 1, "bundle": "model.bundle"}
    monkeypatch.setattr(e2e, "CASES", {"report": (manifest, case)})
    monkeypatch.setattr(e2e, "ROOT", tmp_path)
    monkeypatch.setattr(e2e, "REFERENCE", {"reference_backend": "recorded fixture", "revision": "b" * 40})
    monkeypatch.setattr(e2e, "_selected", lambda *args: None)
    monkeypatch.setattr(e2e, "_runner", lambda *args: tmp_path / "runner")
    monkeypatch.setattr(e2e, "_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_progress", lambda *args: None)
    monkeypatch.setattr(e2e, "build", lambda *args: None)
    # Return a different request-list order so association cannot use zip position.
    monkeypatch.setattr(e2e, "make_inputs", lambda *args: (list(reversed(requests)), expected))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    (tmp_path / "thresholds").mkdir()
    (tmp_path / "thresholds/report.json").write_text(json.dumps({"threshold_overrides": {"normalized_text_edit_distance": 0.0}}))
    native_calls = []

    def native(command, **kwargs):
        native_calls.append(command)
        (tmp_path / "output.json").write_text(json.dumps(expected))
        return SimpleNamespace(returncode=0, stdout="batch completed", stderr="")

    monkeypatch.setattr(e2e.subprocess, "run", native)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="nemotron_omni", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    snapshots = []

    def bind(batch, request_id):
        reporting.record_request_by_id(batch, request_id)
        snapshots.append(copy.deepcopy(recorder.data["inputs"]))

    monkeypatch.setattr(e2e, "record_request_by_id", bind)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e.test_e2e("report", SimpleNamespace(config=None), tmp_path, None)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert len(native_calls) == 1 and requests == original
    assert [row["request_id"] for row in snapshots] == [row["id"] for row in expected]
    expected_sources = [set(), {"image.rgb"}, {"audio.f32"}, {"frame-7.rgb"},
                        {"image.rgb", "frame-7.rgb"}, {"audio.f32", "frame-7.rgb"}]
    for row, sources in zip(snapshots, expected_sources, strict=True):
        assert row["prompt"] == row["request_id"] + " prompt"
        assert {media["source"] for media in row["media"]} == sources
    assert recorder.data["inputs"]["request_id"] == "audio_video"
    assert recorder.data["native"]["id"] == recorder.data["reference"]["id"] == "audio_video"
    assert recorder.data["request_batch"]["requests"]


def test_unknown_request_id_clears_the_previous_default(tmp_path):
    requests = _requests(tmp_path)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="nemotron_omni", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        reporting.record_request_by_id(requests, "image")
        e2e_evidence.record_evidence("native", {"text": "old output"})
        reporting.record_request_by_id(requests, "missing")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["native"] is recorder.data["reference"] is None
    assert recorder.data["views"] == [] and recorder.data["inputs"]["media"] == []
    assert recorder.data["inputs"]["request_id"] == "missing"
    assert "preview unavailable" in recorder.data["inputs"]["text"]
