# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron streaming ASR-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_nemotron_streaming_reference_uses_nemo_asr_environment() -> None:
    profiles = default_execution_profiles(
        family="nemotron_speech_streaming",
        runtime_strategy="nemotron_speech_streaming_speech_to_text",
        reference_backend="torch_reference",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "nemotron_speech_streaming_reference",
    }


def test_nemotron_streaming_profile_pins_nemo_asr_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/nemotron_speech_streaming/"
        "python_profile_requirements/nemotron_speech_streaming_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert {
        "absl-py==2.5.0",
        "cffi==2.1.1",
        "datasets==3.6.0",
        "hydra-core==1.3.2",
        "lightning==2.4.0",
        "librosa==0.11.0",
        "lhotse==1.33.0",
        "matplotlib==3.11.1",
        "nemo_toolkit[asr]==2.7.3",
        "numexpr==2.10.0",
        "parso==0.8.7",
        "pandas==2.2.3",
        "protobuf==5.29.5",
        "pyarrow==20.0.0",
        "pycparser==3.0",
        "pydantic==2.10.6",
        "pytz==2026.3.post1",
        "ruamel.yaml==0.18.10",
        "sentencepiece==0.2.2",
        "tzdata==2026.3",
        "tensorboard==2.20.0",
        "numpy==1.26.4",
        "soundfile==0.14.0",
        "transformers==4.57.6",
        "webdataset==1.0.2",
    } <= set(requirements.splitlines())


def test_nemotron_streaming_profile_verifies_the_runtime_entrypoint() -> None:
    repository = Path(__file__).resolve().parents[4]
    verify = (
        repository
        / "python/tensorrt_model_connect/families/nemotron_speech_streaming/"
        "python_profile_verify.py"
    ).read_text(encoding="utf-8")

    assert "EncDecHybridRNNTCTCBPEModelWithPrompt" in verify
    assert "EncDecRNNTBPEModelWithPrompt" not in verify
