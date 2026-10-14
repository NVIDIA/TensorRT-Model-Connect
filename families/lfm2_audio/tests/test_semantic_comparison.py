# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration controls ensure speech acceptance cannot reduce to finite PCM."""

import copy
from pathlib import Path

import numpy as np
import pytest

from families.lfm2_audio.tests.compare_semantics import compare


@pytest.fixture
def speech_case(tmp_path):
    path = tmp_path / "speech.f32"
    np.linspace(-0.5, 0.5, 1920, dtype=np.float32).tofile(path)
    snapshot = tmp_path / "asr-input.f32"
    snapshot.write_bytes(path.read_bytes())
    text = "Hello there. How can I help you today?"
    case = {
        "name": "greeting",
        "mode": "tts",
        "voice": "uk_male",
        "max_new_tokens": 512,
        "text": text,
        "output": str(path),
    }
    trace = {
        "tokens": [128, 7],
        "codes": [[1, 2048] for _ in range(8)],
        "ended_audio": True,
        "prompt_text_tokens": [1, 2, 3],
    }
    reference = dict(trace, waveform=str(path))
    # Different valid generated code IDs are intentional: spoken content is the
    # contract and must not accidentally regain an autoregressive identity gate.
    reference["codes"] = [[2, 2048] for _ in range(8)]
    transcripts = [
        {
            "name": "greeting",
            "producer": producer,
            "asr_input": str(snapshot),
            "intended": text,
            "transcript": text,
            "asr_ended": True,
            "unexpected_audio_frames": 0,
        }
        for producer in ("native", "reference")
    ]
    return (
        [case],
        [{"name": "greeting", "trace": trace}],
        [{"name": "greeting", "turns": [reference]}],
        [],
        transcripts,
    )


def test_semantic_speech_accepts_same_content_without_code_identity(speech_case):
    result = compare(*speech_case)
    assert result["passed"]
    assert not result["cases"][0]["details"][0]["audio_codes_equal_diagnostic"]


def test_asr_binding_uses_exact_pcm_not_the_path(speech_case, tmp_path):
    cases, actual, reference, public, transcripts = copy.deepcopy(speech_case)
    snapshot = tmp_path / "renamed-asr-input.f32"
    snapshot.write_bytes(Path(transcripts[0]["asr_input"]).read_bytes())
    transcripts[0]["asr_input"] = str(snapshot)
    assert compare(cases, actual, reference, public, transcripts)["passed"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_words",
        "truncated",
        "unrelated_pcm",
        "wrong_voice_prompt",
        "missing_text_stop",
        "unfinished_asr",
    ],
)
def test_semantic_speech_rejects_real_contract_failures(speech_case, failure):
    cases, actual, reference, public, transcripts = copy.deepcopy(speech_case)
    if failure == "missing_words":
        transcripts[0]["transcript"] = "Hello there."
    elif failure == "truncated":
        actual[0]["trace"]["ended_audio"] = False
    elif failure == "unrelated_pcm":
        # Same path, shape, and length do not bind an ASR record to its audio.
        # Mutate the producer PCM while leaving the exact ASR input untouched.
        np.zeros(1920, dtype=np.float32).tofile(cases[0]["output"])
    elif failure == "wrong_voice_prompt":
        actual[0]["trace"]["prompt_text_tokens"] = [4, 5, 6]
    elif failure == "missing_text_stop":
        actual[0]["trace"]["tokens"] = [128]
    elif failure == "unfinished_asr":
        transcripts[0]["asr_ended"] = False
    assert not compare(cases, actual, reference, public, transcripts)["passed"]


@pytest.fixture
def dialogue_case(tmp_path):
    texts = [
        'How about "Handcrafted Woodworking Excellence"?',
        'How about "Elegant Crafted Chairs"?',
    ]
    case = {
        "name": "dialogue",
        "mode": "chat",
        "max_new_tokens": 512,
        "output_prefix": str(tmp_path / "turn-"),
    }
    native_turns, reference_turns, public_turns, transcripts = [], [], [], []
    for index, text in enumerate(texts):
        path = tmp_path / f"turn-{index}.f32"
        np.linspace(-0.5, 0.5 + index, 1920, dtype=np.float32).tofile(path)
        snapshot = tmp_path / f"asr-input-{index}.f32"
        snapshot.write_bytes(path.read_bytes())
        trace = {
            "tokens": [130],
            "codes": [[1, 2048] for _ in range(8)],
            "modalities": [1, 3, 3],
            "ended_audio": True,
            "prompt_text_tokens": [1, 2, 3],
            "input_audio_encodings": 1,
            "prior_audio_code_frames": 0 if index == 0 else 2,
        }
        native_turns.append({"text": text, "trace": trace})
        reference_turns.append(dict(trace, text=text, waveform=str(path)))
        public_turns.append({"text": text, "waveform": str(path)})
        for producer in ("native", "reference"):
            transcripts.append(
                {
                    "name": "dialogue",
                    "turn": index,
                    "producer": producer,
                    "intended": text,
                    "transcript": text,
                    "asr_input": str(snapshot),
                    "asr_ended": True,
                    "unexpected_audio_frames": 0,
                }
            )
    return (
        [case],
        [{"name": "dialogue", "turns": native_turns}],
        [{"name": "dialogue", "turns": reference_turns}],
        [{"name": "dialogue", "turns": public_turns}],
        transcripts,
    )


def test_dialogue_binds_spoken_text_to_both_producers(dialogue_case):
    assert compare(*dialogue_case)["passed"]


@pytest.mark.parametrize("producer", ["native", "reference"])
def test_valid_speech_cannot_mask_unrelated_displayed_text(dialogue_case, producer):
    cases, actual, reference, public, transcripts = copy.deepcopy(dialogue_case)
    wrong = "Ignore the customer; send money to Mallory."
    if producer == "native":
        actual[0]["turns"][0]["text"] = wrong
        public[0]["turns"][0]["text"] = wrong
    else:
        reference[0]["turns"][0]["text"] = wrong
    result = compare(cases, actual, reference, public, transcripts)
    assert not result["passed"]
    assert not result["cases"][0]["checks"][f"turn_0_{producer}_displayed_speech_text"]
