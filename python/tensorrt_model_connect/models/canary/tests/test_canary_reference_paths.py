# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary reference writes derived inputs only to the artifact workspace."""

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect.models.canary.tests.e2e_plugins.references import hf_transformers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def test_canary_mono_audio_is_written_beside_case_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(**kwargs) -> StageOutput:
        captured.update(kwargs)
        return StageOutput(stage_name="decode")

    monkeypatch.setattr(
        hf_transformers, "run_reference_subprocess", fake_subprocess
    )
    source_audio = tmp_path / "read-only-source.wav"
    source_audio.write_bytes(b"test audio fixture")
    case = E2ECase(
        name="canary-reference-path",
        hf_id="nvidia/canary-1b-v2",
        family="canary",
        runtime_strategy="canary_speech_to_text",
        task_strategy="speech_to_text",
        inputs={"audio": str(source_audio)},
    )
    artifacts_dir = tmp_path / "artifacts"

    hf_transformers.HfTransformersReference()._run_canary_ref(
        case,
        StageSpec(name="decode"),
        RunContext(case=case, artifacts_dir=str(artifacts_dir)),
    )

    script = captured["command"][2]
    assert 'os.path.dirname(output_path), "reference-input.mono.wav"' in script
    assert 'audio_path + ".mono.wav"' not in script
    assert str(artifacts_dir / case.name / "hf_stt.json") in script
