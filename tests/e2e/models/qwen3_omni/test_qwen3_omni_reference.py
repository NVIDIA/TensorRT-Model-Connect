# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Qwen3-Omni pinned official-HF audio evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.e2e.models.qwen3_omni.e2e_plugins.references import torch_reference
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import get_case_by_name


MODEL_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = MODEL_DIR / "data" / "qwen3_omni_hf_reference.json"
OFFICIAL_PROMPT = "Please say hello from Qwen3 Omni in one short sentence."
EXPECTED_WAV_SHA256 = "2648af9d3de015de2e7c73f829e374b95f287a9c5e1c548569a33068e8aa99ef"


def _case(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    prompt: str = OFFICIAL_PROMPT,
) -> E2ECase:
    return E2ECase(
        name="qwen3-omni-reference-test",
        hf_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        family="qwen3_omni",
        runtime_strategy="qwen3_omni_multimodal",
        task_strategy="omni_multimodal",
        reference_backend="torch_reference",
        oracle_level="L3_snapshot_regression",
        inputs={"prompt": prompt, "max_new_tokens": 16, "seed": 42},
        determinism={"seed": 42, "reruns": 0},
        metadata={
            "reference_speaker": "Ethan",
            "reference_talker_max_new_tokens": 32,
            "golden_snapshot_path": str(snapshot_path),
        },
    )


def test_manifest_uses_pinned_hf_waveform_as_l4_oracle() -> None:
    case = get_case_by_name("qwen3-omni-30b-a3b-instruct", MODEL_DIR)

    assert case is not None
    assert case.reference_backend == "torch_reference"
    assert case.oracle_level == "L3_snapshot_regression"
    assert case.inputs["prompt"] == OFFICIAL_PROMPT
    assert case.inputs["seed"] == 42
    assert case.metadata["reference_speaker"] == "Ethan"
    assert case.metadata["reference_talker_max_new_tokens"] == 32
    assert Path(case.metadata["golden_snapshot_path"]) == SNAPSHOT_PATH
    assert case.threshold_overrides == {
        "audio_artifact_bytes_min": 44.0,
        "audio_duration_s_min": 0.5,
        "audio_reference_duration_ratio_min": 0.5,
        "audio_rms_min": 0.005,
        "audio_peak_min": 0.02,
        "audio_reference_waveform_cosine_min": 0.25,
    }
    assert any(
        requirement.kind == "asset_exists"
        and Path(requirement.args["path"]) == SNAPSHOT_PATH
        and requirement.gating
        for requirement in case.preflight
    )
    assert not any(
        requirement.kind == "python_module_available"
        and requirement.args.get("phase") == "reference"
        for requirement in case.preflight
    )


def test_pinned_reference_materializes_case_local_playable_hf_audio(
    tmp_path: Path,
) -> None:
    case = _case()
    output = torch_reference.TorchReference().run_stage(
        case,
        StageSpec(name="talker_decode"),
        RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
    )

    expected_wav = tmp_path / "artifacts" / case.name / "hf_reference.wav"
    assert output.data["_invariant_only"] is True
    assert output.data["reference_role"] == "automated_waveform_oracle"
    assert output.data["wav_path"] == str(expected_wav)
    assert output.data["sample_rate"] == 24_000
    assert output.data["num_samples"] == 37_845
    assert output.data["duration_s"] == pytest.approx(1.576875)
    assert output.data["decoded_text"] == "Hello from Qwen-Omni!"
    assert output.data["resolved_revision"] == ("26291f793822fb6be9555850f06dfe95f2d7e695")
    assert output.data["raw_sha256"] == EXPECTED_WAV_SHA256
    assert output.text == "Hello from Qwen-Omni!"
    assert output.metadata["source"] == "official_hf_pinned_waveform_oracle"
    assert output.metadata["comparison_mode"] == "waveform_cosine_and_invariants"
    assert hashlib.sha256(expected_wav.read_bytes()).hexdigest() == EXPECTED_WAV_SHA256
    assert output.timing_s >= 0.0


def test_pinned_reference_has_no_model_load_or_subprocess_dependency() -> None:
    source = Path(torch_reference.__file__).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import transformers" not in source
    assert "import subprocess" not in source
    assert "from_pretrained" not in source


def test_snapshot_records_complete_generation_provenance() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    assert snapshot["source"] == "official_hugging_face_qwen3_omni"
    assert snapshot["model_id"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert snapshot["prompt"] == OFFICIAL_PROMPT
    assert snapshot["system_prompt"] == torch_reference.QWEN_AUDIO_SYSTEM_PROMPT
    assert snapshot["speaker"] == "Ethan"
    assert snapshot["seed"] == 42
    assert snapshot["thinker_max_new_tokens"] == 16
    assert snapshot["talker_max_new_tokens"] == 32
    assert snapshot["thinker_do_sample"] is False
    assert snapshot["talker_do_sample"] is False
    assert snapshot["audio"]["raw_sha256"] == EXPECTED_WAV_SHA256


def test_reference_rejects_prompt_drift(tmp_path: Path) -> None:
    case = _case(prompt="A different prompt")

    with pytest.raises(
        RuntimeError, match=(r"provenance mismatch for prompt: expected 'A different prompt', got ")
    ):
        torch_reference.TorchReference().run_stage(
            case,
            StageSpec(name="talker_decode"),
            RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
        )


def test_reference_rejects_corrupt_snapshot_hash(tmp_path: Path) -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot["audio"]["gzip_sha256"] = "0" * 64
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text(json.dumps(snapshot), encoding="utf-8")
    case = _case(snapshot_path=corrupt_path)

    with pytest.raises(RuntimeError, match=r"compressed audio SHA-256"):
        torch_reference.TorchReference().run_stage(
            case,
            StageSpec(name="talker_decode"),
            RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
        )


def test_reference_rejects_missing_snapshot(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    case = _case(snapshot_path=missing)

    with pytest.raises(RuntimeError, match=r"could not be read"):
        torch_reference.TorchReference().run_stage(
            case,
            StageSpec(name="talker_decode"),
            RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
        )


def test_reference_declines_non_talker_stage(tmp_path: Path) -> None:
    case = _case()
    output = torch_reference.TorchReference().run_stage(
        case,
        StageSpec(name="thinker_decode"),
        RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts")),
    )

    assert "only supports omni_multimodal/talker_decode" in output.data["error"]
