# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie TTS-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


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
        "absl-py==2.5.0",
        "cffi==2.1.1",
        "datasets==3.6.0",
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "fonttools==4.63.0",
        "jieba==0.42.1",
        "kiwisolver==1.5.0",
        "librosa==0.11.0",
        "matplotlib==3.11.1",
        "nemo_toolkit==2.7.3",
        "numexpr==2.10.0",
        "pandas==2.2.3",
        "protobuf==5.29.5",
        "pyarrow==20.0.0",
        "parso==0.8.7",
        "numpy==1.26.4",
        "pydub==0.25.1",
        "pyloudnorm==0.2.0",
        "pyparsing==3.3.2",
        "pycparser==3.0",
        "pydantic==2.10.6",
        "pyopenjtalk==0.4.1",
        "pypinyin==0.55.0",
        "pypinyin-dict==0.9.0",
        "pytz==2026.3.post1",
        "ruamel.yaml==0.18.10",
        "sentencepiece==0.2.2",
        "tzdata==2026.3",
        "tensorboard==2.20.0",
        "tqdm==4.70.0",
        "wandb==0.23.0",
        "sox==1.5.0",
    } <= set(requirements.splitlines())

    profile = load_python_profile_registry()["profiles"]["magpie_tts_reference"]
    bootstrap = (
        repository
        / "python/tensorrt_model_connect"
        / profile["bootstrap_requirements"]
    ).read_text(encoding="utf-8")
    assert {
        "Cython==3.1.3",
        "numpy==1.26.4",
        "setuptools-scm==9.2.2",
    } <= set(bootstrap.splitlines())
