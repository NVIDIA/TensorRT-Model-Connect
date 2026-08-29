# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2-owned Python profile contracts."""

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
    prebuilt_python_profile_names,
)


def test_wan22_default_reference_profile_is_prepared_for_offline_proofs() -> None:
    profiles = default_execution_profiles(family="wan2_2_ti2v")
    registry = load_python_profile_registry()

    assert profiles["reference"] == "wan2_2_ti2v_reference"
    assert profiles["reference"] in prebuilt_python_profile_names(registry)
