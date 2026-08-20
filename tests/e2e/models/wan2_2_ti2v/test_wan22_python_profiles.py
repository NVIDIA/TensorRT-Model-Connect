# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_wan22_reference_uses_family_owned_environment() -> None:
    assert default_execution_profiles(family="wan2_2_ti2v") == {
        "build": "base",
        "runtime": "base",
        "reference": "wan2_2_ti2v_reference",
    }


def test_wan22_reference_profile_pins_selective_import_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/wan2_2_ti2v/"
        "python_profile_requirements/wan2_2_ti2v_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert set(requirements.splitlines()) == {
        "einops==0.8.1",
        "ftfy==6.3.1",
        "wcwidth==0.8.2",
    }
