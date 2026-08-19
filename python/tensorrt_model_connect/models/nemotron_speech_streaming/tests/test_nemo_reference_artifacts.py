# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for writable artifacts produced by the Nemotron NeMo reference."""

from __future__ import annotations

from tensorrt_model_connect.models.nemotron_speech_streaming.tests.e2e_plugins.references import (
    hf_transformers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def test_nemo_reference_writes_derived_audio_under_artifacts(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="nemotron-case",
        hf_id="nvidia/nemotron-speech-streaming-en-0.6b",
        family="nemotron_speech_streaming",
        runtime_strategy="nemotron_speech_streaming_speech_to_text_rnnt",
        task_strategy="speech_to_text",
        inputs={"audio": "/read-only-source/Recording.wav"},
    )
    ctx = RunContext(case=case, artifacts_dir=str(tmp_path))
    captured: dict[str, object] = {}

    def fake_reference_subprocess(**kwargs):
        captured.update(kwargs)
        return StageOutput(stage_name=kwargs["stage_name"], text="transcript")

    monkeypatch.setattr(
        hf_transformers,
        "run_reference_subprocess",
        fake_reference_subprocess,
    )

    output = hf_transformers.HfTransformersReference().run_stage(
        case, StageSpec("full_generation"), ctx
    )

    script = captured["command"][2]
    compile(script, "<nemotron-reference>", "exec")
    artifact_dir = tmp_path / case.name
    assert f"mono_path = {str(artifact_dir / 'nemo_reference_audio.wav')!r}" in script
    assert (
        f"manifest_path = {str(artifact_dir / 'nemo_reference_audio.manifest.jsonl')!r}"
    ) in script
    assert 'audio_path + ".mono.wav"' not in script
    assert output.text == "transcript"
