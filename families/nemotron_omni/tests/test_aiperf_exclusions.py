# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's AIPerf profiles that the qualification excludes, and why."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

AIPERF = Path(__file__).resolve().parent / "aiperf"


@pytest.mark.parametrize(
    "profile", ("nemotron-omni-nvfp4-native", "nemotron-omni-nvfp4-text-image")
)
def test_profiles_whose_only_native_reference_is_vllm_are_excluded(profile: str) -> None:
    # The family's independent native reference is a pinned vLLM snapshot, and AIPerf has no vLLM native backend.
    # Drop this test with the exclusion once AIPerf can serve that reference.
    reason = yaml.safe_load((AIPERF / f"{profile}.yaml").read_text())["exclude"]
    assert "vLLM" in reason
