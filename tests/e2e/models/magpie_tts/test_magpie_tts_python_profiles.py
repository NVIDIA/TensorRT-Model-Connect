# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie TTS-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_magpie_reference_uses_tts_environment() -> None:
    profiles = default_execution_profiles(
        family="magpie_tts",
        runtime_strategy="magpie_tts_text_to_speech",
        reference_backend="torch_reference",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "magpie_tts_reference",
    }


def test_magpie_reference_profile_pins_import_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/magpie_tts/"
        "python_profile_requirements/magpie_tts_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert {
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "fonttools==4.63.0",
        "kiwisolver==1.5.0",
        "matplotlib==3.11.1",
        "nemo_toolkit==2.7.3",
        "numpy==1.26.4",
        "pydub==0.25.1",
        "pyloudnorm==0.2.0",
        "pyparsing==3.3.2",
        "sox==1.5.0",
    } <= set(requirements.splitlines())
