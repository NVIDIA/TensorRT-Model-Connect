# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the live Qwen3-Omni official-HF reference."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

from tests.e2e.models.qwen3_omni.e2e_plugins.references import torch_reference
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import get_case_by_name


MODEL_DIR = Path(__file__).resolve().parent
OFFICIAL_PROMPT = "Please say hello from Qwen3 Omni in one short sentence."


def _case(*, prompt: str = OFFICIAL_PROMPT) -> E2ECase:
    return E2ECase(
        name="qwen3-omni-reference-test",
        hf_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        family="qwen3_omni",
        runtime_strategy="qwen3_omni_multimodal",
        task_strategy="omni_multimodal",
        reference_backend="torch_reference",
        oracle_level="L1_external_reference",
        inputs={"prompt": prompt, "max_new_tokens": 16, "seed": 42},
        determinism={"seed": 42, "reruns": 0},
        metadata={
            "reference_speaker": "Ethan",
            "reference_talker_max_new_tokens": 32,
        },
    )


def _write_pcm16(path: Path, *, frames: int = 2400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * frames)


def test_manifest_uses_live_official_hf_reference() -> None:
    case = get_case_by_name("qwen3-omni-30b-a3b-instruct", MODEL_DIR)

    assert case is not None
    assert case.reference_backend == "torch_reference"
    assert case.oracle_level == "L1_external_reference"
    assert case.inputs["prompt"] == OFFICIAL_PROMPT
    assert case.inputs["seed"] == 42
    assert case.metadata["reference_speaker"] == "Ethan"
    assert case.metadata["reference_talker_max_new_tokens"] == 32
    assert "golden_snapshot_path" not in case.metadata
    assert not any(
        requirement.kind == "asset_exists"
        and "qwen3_omni_hf_reference" in str(requirement.args.get("path", ""))
        for requirement in case.preflight
    )


def test_reference_runs_direct_official_hf_command_and_materializes_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[str] = []

    def run(command, **_kwargs):
        captured.extend(command)
        audio_path = Path(command[command.index("--audio-output") + 1])
        metadata_path = Path(command[command.index("--metadata-output") + 1])
        _write_pcm16(audio_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "resolved_revision": "a" * 40,
                    "decoded_text": "Hello from Qwen-Omni!",
                    "sample_rate": 24_000,
                    "num_samples": 2400,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(torch_reference.subprocess, "run", run)
    case = _case()
    output = torch_reference.TorchReference().run_stage(
        case,
        StageSpec(name="talker_decode"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            reference_python="/profiles/qwen/bin/python",
        ),
    )

    assert captured[0] == "/profiles/qwen/bin/python"
    assert captured[1].endswith("/qwen3_omni/official_hf_audio.py")
    assert captured[captured.index("--prompt") + 1] == OFFICIAL_PROMPT
    assert captured[captured.index("--speaker") + 1] == "Ethan"
    assert captured[captured.index("--thinker-max-new-tokens") + 1] == "16"
    assert captured[captured.index("--talker-max-new-tokens") + 1] == "32"
    assert output.data["_invariant_only"] is True
    assert output.data["sample_rate"] == 24_000
    assert output.data["num_samples"] == 2400
    assert output.data["decoded_text"] == "Hello from Qwen-Omni!"
    assert Path(output.data["wav_path"]).is_file()
    assert output.text == "Hello from Qwen-Omni!"
    assert output.metadata["source"] == "official_hf_live_reference"
    assert output.metadata["command"] == captured


def test_official_runner_uses_transformers_generation_api() -> None:
    source = (MODEL_DIR / "official_hf_audio.py").read_text(encoding="utf-8")

    assert "Qwen3OmniMoeForConditionalGeneration" in source
    assert "Qwen3OmniMoeProcessor" in source
    assert "thinker_do_sample=False" in source
    assert "talker_do_sample=False" in source
    assert "speaker=arguments.speaker" in source


def test_reference_declines_non_talker_stage(tmp_path: Path) -> None:
    case = _case()
    output = torch_reference.TorchReference().run_stage(
        case,
        StageSpec(name="thinker_decode"),
        RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
    )

    assert "only supports omni_multimodal/talker_decode" in output.data["error"]
