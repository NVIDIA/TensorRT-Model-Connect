# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


def test_lora_reference_uses_a_family_owned_peft_profile() -> None:
    profiles = default_execution_profiles(
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        reference_backend="hf_transformers",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "nemotron_labs_diffusion_reference",
    }

    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/nemotron_labs_diffusion/"
        "python_profile_requirements/reference.lock.txt"
    ).read_text(encoding="utf-8")
    assert "peft==0.20.0" in requirements.splitlines()

    profile = load_python_profile_registry()["profiles"][
        "nemotron_labs_diffusion_reference"
    ]
    assert profile["prebuild"] is True
