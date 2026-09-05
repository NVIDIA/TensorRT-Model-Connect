# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from families.nemotron_voicechat.tests.test_e2e import _assert_parity

_TEXT = (
    "Hi there! How can you? How can I help you today? The sky is blue. "
    "That blue color is because of something called Rayleigh scattering."
)


def _actual() -> dict:
    return {
        "source_stats": {"channels": 1, "sample_rate": 16000, "num_samples": 249734},
        "output_stats": {
            "channels": 1,
            "subtype": "FLOAT",
            "all_finite": True,
            "sample_rate": 22050,
            "num_samples": 345744,
            "rms": 0.01,
            "peak": 0.1,
        },
        "event_audio_samples": 345744,
        "text": _TEXT,
        "transcript": _TEXT,
    }


def _expected() -> dict:
    return {
        "samples": np.zeros(345744, dtype=np.float32),
        "sample_rate": 22050,
        "text": _TEXT,
        "speech_source_sample_rate": 16000,
        "speech_source_num_samples": 249734,
        "expected_output_sample_rate": 22050,
        "expected_output_num_samples": 345744,
        "expected_output_samples_per_frame": 1764,
        "expected_output_codec_frames": 196,
        "required_response_terms": ["rayleigh", "scattering"],
    }


def _case() -> dict:
    return {
        "name": "nemotron-voicechat-11b",
        "inputs": {"tail_frames": 0},
    }


def _thresholds() -> dict:
    return {
        "audio_min_rms": 0.001,
        "audio_min_peak": 0.01,
        "agent_text_min_similarity": 0.75,
        "transcript_min_words": 8,
    }


def test_model_card_oracle_keeps_exact_frame_mapping_and_transcript() -> None:
    _assert_parity(_actual(), _expected(), {"task": "speech_session"}, _case(), _thresholds())


def test_model_card_oracle_requires_model_card_terms() -> None:
    actual = _actual()
    actual["text"] = "Hi there! How can I help you today? The sky is blue."
    with pytest.raises(AssertionError):
        _assert_parity(actual, _expected(), {"task": "speech_session"}, _case(), _thresholds())
