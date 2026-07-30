# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned Python profile contracts."""

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
)


def test_personaplex_reference_uses_the_common_reference_profile() -> None:
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
