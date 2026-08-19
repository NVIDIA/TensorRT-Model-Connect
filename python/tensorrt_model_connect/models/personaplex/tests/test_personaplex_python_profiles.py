# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


def test_personaplex_reference_keeps_the_default_profile() -> None:
    profiles = default_execution_profiles(
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        reference_backend="torch_reference",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "base",
    }


def test_personaplex_full_duplex_profile_pins_evaluator_dependencies() -> None:
    repository = Path(__file__).resolve().parents[5]
    requirements = (
        repository
        / "python/tensorrt_model_connect/models/personaplex/"
        "python_profile_requirements/full_duplex_evaluator.lock.txt"
    ).read_text(encoding="utf-8")

    assert "huggingface-hub==1.22.0" in requirements
    assert "nemo_toolkit[asr]==2.7.3" in requirements
    assert "numpy==1.26.4" in requirements
    assert "scipy==1.15.3" in requirements
    assert "silero-vad==6.2.1" in requirements
    profile = load_python_profile_registry()["profiles"][
        "personaplex_full_duplex_evaluator"
    ]
    assert profile["prebuild"] is False
