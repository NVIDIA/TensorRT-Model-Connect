# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the existing exact transcript check, without introducing a distance gate."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from families.parakeet_tdt.tests import test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import _assessment, render_case


def _recorder(tmp_path):
    return e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="parakeet_tdt",
        case="transcript",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )


def test_exact_normalized_transcript_check_is_readable(tmp_path):
    actual, expected = {"text": " Hello\n WORLD "}, {"text": "hello world"}
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e_evidence.record_evidence("native", actual)
        e2e_evidence.record_evidence("reference", expected)
        e2e._assert_parity(actual, expected)
        recorder.finish("passed")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    (check,) = recorder.data["reference_comparison"]["checks"]
    assert check == {
        "name": "normalized_transcript",
        "scope": "independent_reference",
        "actual": "hello world",
        "operator": "==",
        "expected": "hello world",
        "passed": True,
    }
    assert _assessment(recorder.data)["kind"] == "reference"
    assert "Normalized transcript equality" in render_case(recorder.data, recorder.directory)
    assert actual == {"text": " Hello\n WORLD "}


@pytest.mark.parametrize("actual,expected", [("wrong", "hello"), ("", ""), ("Straße", "STRASSE")])
def test_original_rejections_remain_failures_without_successful_comparisons(
    tmp_path, actual, expected
):
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e_evidence.record_evidence("native", {"text": actual})
        e2e_evidence.record_evidence("reference", {"text": expected})
        with pytest.raises(AssertionError):
            e2e._assert_parity({"text": actual}, {"text": expected})
        recorder.finish("failed", failure="original exact transcript assertion rejected the result")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "reference_comparison" not in recorder.data
    assert recorder.data["native"]["text"] == actual
    assert recorder.data["reference"]["text"] == expected
    assert _assessment(recorder.data)["kind"] == "failed"


def test_report_io_failure_does_not_change_a_matching_transcript(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)

    def unavailable(*args):
        raise OSError("comparison snapshot unavailable")

    monkeypatch.setattr(Path, "write_bytes", unavailable)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e._assert_parity({"text": "Hello"}, {"text": "hello"})
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "reference_comparison" not in recorder.data
    assert any("comparison snapshot unavailable" in issue for issue in recorder.data["issues"])


def test_audio_evidence_is_the_actual_pcm_fixture(tmp_path, monkeypatch):
    pcm = np.array([0, 100, -100, 32767], dtype="<i2")
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
    monkeypatch.setattr(e2e, "TEST_ROOT", tmp_path)
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        audio, rate, channels = e2e._audio(
            {"test_input_audio": "input.wav"}, tmp_path / "generated"
        )
    finally:
        e2e_evidence._ACTIVE.reset(token)
    artifact = recorder.directory / recorder.data["inputs"]["audio"]["artifact"]
    assert artifact.read_bytes() == source.read_bytes()
    assert (rate, channels) == (16000, 1)
    np.testing.assert_array_equal(audio, pcm.astype(np.float32) / 32768.0)


def test_reference_failure_preserves_the_native_transcript(tmp_path, monkeypatch):
    name = next(iter(e2e.CASES))
    for library in ("libtrtmc_model_parakeet_tdt.so", "libtrtmc_backend_trt.so"):
        (tmp_path / library).touch()
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    monkeypatch.setattr(e2e, "_sdk_consumer_binary", lambda: tmp_path / "consumer")
    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "build", lambda request: None)
    monkeypatch.setattr(e2e, "_audio", lambda *args: (np.zeros(4, dtype=np.float32), 16000, 1))
    monkeypatch.setattr(
        e2e.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"text": "kept native transcript"}),
            stderr="",
            args=["mock-consumer"],
        ),
    )

    def unavailable(*args):
        raise RuntimeError("reference unavailable")

    monkeypatch.setattr(e2e, "_reference", unavailable)
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(RuntimeError, match="reference unavailable"):
            e2e.test_official_checkpoint_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["native"]["text"] == "kept native transcript"
    assert recorder.data["failure_stage"] == "reference"
