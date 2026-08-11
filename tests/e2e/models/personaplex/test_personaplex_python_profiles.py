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
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/personaplex/"
        "python_profile_requirements/full_duplex_evaluator.lock.txt"
    ).read_text(encoding="utf-8")

    assert "huggingface-hub==0.36.2" in requirements
    assert "nemo_toolkit[asr]==2.7.3" in requirements
    assert "numpy==1.26.4" in requirements
    assert "scipy==1.15.3" in requirements
    assert "silero-vad==6.2.1" in requirements
    assert "tokenizers==0.22.2" in requirements
    assert "transformers==4.57.6" in requirements
    assert {
        "hydra-core==1.3.2",
        "lightning==2.4.0",
        "omegaconf==2.3.0",
        "wrapt==2.3.0",
        "wget==3.2",
        "fiddle==0.3.0",
        "cloudpickle==3.1.2",
        "lhotse==1.33.0",
        "einops==0.8.2",
        "kaldialign==0.9.1",
        "pyannote.core==5.0.0",
        "pyannote.metrics==3.2.1",
        "webdataset==1.0.2",
        "text-unidecode==1.3",
        "ipython==9.16.1",
        "matplotlib==3.11.1",
    } <= set(requirements.splitlines())
    profile = load_python_profile_registry()["profiles"][
        "personaplex_full_duplex_evaluator"
    ]
    assert profile["prebuild"] is False
