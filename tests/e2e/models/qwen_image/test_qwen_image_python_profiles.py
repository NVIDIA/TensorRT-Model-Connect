# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_qwen_image_reference_uses_family_owned_environment() -> None:
    assert default_execution_profiles(family="qwen_image") == {
        "build": "base",
        "runtime": "base",
        "reference": "qwen_image_reference",
    }


def test_qwen_image_reference_profile_pins_runtime_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/qwen_image/"
        "python_profile_requirements/qwen_image_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert set(requirements.splitlines()) == {
        "accelerate==1.14.0",
        "diffusers==0.39.0",
        "ftfy==6.3.1",
        "wcwidth==0.8.2",
    }
