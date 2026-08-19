# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper decoder metadata and bounded long-audio CI contract tests."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.models.whisper.config import ModelConfig
from tensorrt_model_connect.models.whisper.prompt_metadata import (
    whisper_decoder_prompt_metadata,
)

from tensorrt_model_connect.models.whisper.tests.e2e_plugins.references import hf_transformers


WHISPER_DIR = Path(__file__).resolve().parent
LONG_AUDIO_SHA256 = (
    "166d138dc95c706e4eedbebb48f4ac4c8cb1b77ea796c0bc650da518308657e2"
)


def _model_config(model_dir: Path, **raw) -> ModelConfig:
    return ModelConfig(
        model_type="whisper",
        raw={"_model_dir": str(model_dir), **raw},
    )


def test_tiny_decoder_prompt_uses_released_checkpoint_ids(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "decoder_start_token_id": 50258,
                "forced_decoder_ids": [[1, None], [2, 50359]],
                "lang_to_id": {"<|en|>": 50259},
                "task_to_id": {"transcribe": 50359},
                "no_timestamps_token_id": 50363,
            }
        ),
        encoding="utf-8",
    )
    config = _model_config(
        tmp_path,
        decoder_start_token_id=50258,
        forced_decoder_ids=[[1, 50259], [2, 50359], [3, 50363]],
    )

    assert whisper_decoder_prompt_metadata(config) == {
        "decoder_start_token_ids": [50258, 50259, 50359, 50363]
    }


def test_large_v3_turbo_decoder_prompt_does_not_use_tiny_ids(
    tmp_path: Path,
) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "decoder_start_token_id": 50258,
                "forced_decoder_ids": [[1, None], [2, 50360]],
                "lang_to_id": {"<|en|>": 50259},
                "task_to_id": {"transcribe": 50360},
                "no_timestamps_token_id": 50364,
            }
        ),
        encoding="utf-8",
    )
    config = _model_config(tmp_path, decoder_start_token_id=50258)

    assert whisper_decoder_prompt_metadata(config) == {
        "decoder_start_token_ids": [50258, 50259, 50360, 50364]
    }


def test_hf_reference_uses_the_manifest_generation_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    marker = object()

    def _capture_reference(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(
        hf_transformers, "run_reference_subprocess", _capture_reference
    )
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"test audio fixture")
    case = SimpleNamespace(
        metadata={"reference_precision": "fp32"},
        inputs={"audio": str(audio_path), "max_new_tokens": 120},
        hf_id="openai/whisper-tiny",
        name="whisper-tiny-fp16",
    )
    stage = SimpleNamespace(name="full_inference")
    ctx = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        ld_library_path="",
        reference_python_path=lambda: None,
    )

    result = hf_transformers.HfTransformersReference()._run_speech_to_text_ref(
        case, stage, ctx
    )

    assert result is marker
    script = captured["command"][2]
    assert script.index("max_new_tokens = 120") < script.index(
        "**inputs, max_new_tokens=max_new_tokens"
    )
    assert "skip_special_tokens=True)[0].strip()" in script
    assert "torch_dtype=torch.float32" in script


def test_long_audio_sentinel_has_pinned_qa_provenance() -> None:
    audio_path = (
        WHISPER_DIR
        / "data"
        / "librispeech-test-clean-6930-75918-0003.wav"
    )
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == LONG_AUDIO_SHA256
    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getparams()[:4] == (1, 2, 16000, 373040)


def test_manifests_cover_long_audio_budget_with_one_strict_sentinel() -> None:
    manifest_paths = sorted((WHISPER_DIR / "manifests").glob("whisper-*.json"))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in manifest_paths
    ]

    assert manifests
    for manifest in manifests:
        case_budget = max(
            testcase.get("max_new_tokens", 0)
            for testcase in manifest["testcases"]
        )
        assert manifest["max_cache_length"] >= 4 + case_budget

    sentinels = [
        testcase
        for manifest in manifests
        for testcase in manifest["testcases"]
        if testcase.get("test_input_audio", "").endswith(
            "librispeech-test-clean-6930-75918-0003.wav"
        )
    ]
    assert sentinels == [
        {
            "name": "whisper-tiny-fp16",
            "trace_id": "IT-E2E-WHSPT-FP16-01",
            "reference_family": "asr_whisper",
            "user_contract": "exact_transcript",
            "reference_precision": "fp32",
            "test_type": "transcription",
            "prompt": "test",
            "max_new_tokens": 120,
            "test_input_audio": (
                "data/librispeech-test-clean-6930-75918-0003.wav"
            ),
        }
    ]

    thresholds = json.loads(
        (
            WHISPER_DIR / "thresholds" / "whisper-tiny-fp16.json"
        ).read_text(encoding="utf-8")
    )["threshold_overrides"]
    assert thresholds["contract_ned_threshold"] == 0.0
    assert thresholds["contract_wer_threshold"] == 0.0
