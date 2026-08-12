# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_canary_reference_uses_nemo_asr_environment() -> None:
    profiles = default_execution_profiles(
        family="canary",
        runtime_strategy="canary_speech_to_text",
        reference_backend="torch_reference",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "canary_reference",
    }


def test_canary_reference_profile_pins_nemo_asr_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/canary/"
        "python_profile_requirements/canary_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert {
        "hydra-core==1.3.2",
        "lightning==2.4.0",
        "lhotse==1.33.0",
        "matplotlib==3.11.1",
        "nemo_toolkit[asr]==2.7.3",
        "parso==0.8.7",
        "numpy==1.26.4",
        "soundfile==0.14.0",
        "transformers==4.57.6",
        "webdataset==1.0.2",
    } <= set(requirements.splitlines())
