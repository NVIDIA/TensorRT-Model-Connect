"""Bark-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import family_hf_warm_dependencies


def test_bark_reference_dependencies_are_family_owned() -> None:
    deps = dict(family_hf_warm_dependencies("bark"))

    assert deps["tts-asr-verifier"] == "openai/whisper-large-v3-turbo"
