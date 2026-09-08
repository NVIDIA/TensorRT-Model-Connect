# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Listening views retain the real rate and survive the family's failed gate."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from families.nemotron_voicechat.tests import test_e2e as e2e
from families.nemotron_voicechat.tests.reporting import record_audio_views
from tools import e2e_evidence


def test_listening_views_keep_real_rates_before_comparison_failure(monkeypatch, tmp_path: Path):
    name = next(iter(e2e.CASES))
    native = tmp_path / "native.wav"
    samples = np.linspace(-0.25, 0.25, 400, dtype=np.float32)
    sf.write(native, samples, 24000, subtype="FLOAT")
    expected = {"samples": samples, "sample_rate": 22050}

    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda manifest: (tmp_path / "trtmc", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_native", lambda *args: {"audio": str(native)})
    monkeypatch.setattr(e2e, "_official_reference", lambda *args: expected)
    monkeypatch.setattr(e2e, "_thresholds", lambda case: {})

    def fail(*args):
        raise AssertionError("original audio gate failed")

    monkeypatch.setattr(e2e, "_assert_parity", fail)
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="nemotron_voicechat",
        case=name,
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="original audio gate failed"):
            e2e.test_official_checkpoint_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    views = {view["title"]: view for view in recorder.data["views"]}
    assert set(views) == {"Native audio", "Reference audio"}
    assert "24000 Hz" in views["Native audio"]["caption"]
    assert "22050 Hz" in views["Reference audio"]["caption"]
    reference = recorder.directory / views["Reference audio"]["audio"]["artifact"]
    restored, rate = sf.read(reference, dtype="float32")
    assert rate == 22050
    np.testing.assert_array_equal(restored, samples)
    assert recorder.data["failure_stage"] == "compare"
    assert recorder.data["reference"]["sample_rate"] == expected["sample_rate"]


def test_listening_views_are_opt_in_and_never_invent_reference_audio(tmp_path: Path):
    directory = tmp_path / "views"
    record_audio_views({}, {"speech_tokens": np.zeros((2, 8), dtype=np.int32)}, directory)
    assert not directory.exists()
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="nemotron_voicechat",
        case="tokens",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_audio_views({}, {"speech_tokens": np.zeros((2, 8), dtype=np.int32)}, directory)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert len(recorder.data["views"]) == 2
    assert all("audio" not in view for view in recorder.data["views"])
    assert "no waveform" in recorder.data["views"][1]["caption"]
    assert not directory.exists()


def test_bad_native_waveform_does_not_discard_available_reference(tmp_path: Path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"invalid wav")
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="nemotron_voicechat",
        case="bad-wav",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_audio_views(
            {"audio": str(broken)},
            {"samples": np.zeros(200), "sample_rate": 16000},
            tmp_path / "views",
        )
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "unavailable" in recorder.data["views"][0]["caption"]
    assert "audio" in recorder.data["views"][1]
