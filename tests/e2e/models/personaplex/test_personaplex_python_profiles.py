# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned Python profile contracts."""

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


def test_personaplex_reference_uses_an_isolated_python_profile() -> None:
    profiles = default_execution_profiles(
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        reference_backend="torch_reference",
    )

    assert profiles == {
        "build": "base",
        "runtime": "base",
        "reference": "personaplex_reference",
    }


def test_personaplex_reference_profile_pins_sphn() -> None:
    spec = load_python_profile_registry()["profiles"]["personaplex_reference"]

    assert spec["requirements"].endswith("personaplex_reference.lock.txt")
    assert spec["system_site_packages"] is True
