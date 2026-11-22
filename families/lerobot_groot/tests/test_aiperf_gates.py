# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The action-parity gates of the family's AIPerf profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

AIPERF = Path(__file__).resolve().parent / "aiperf"
GRIPPER = {
    "gripper_component": -1,
    "outliers": {"max_fraction": 0.02, "gate": {"max_gripper_sign_changes": 1}},
}


@pytest.mark.parametrize(
    ("profile", "arm"),
    [("groot-n17-libero-spatial-bf16", 0.045), ("groot-n17-libero-spatial-fp8", 0.068)],
)
def test_the_gripper_is_compared_by_sign_with_one_changed_observation_allowed(
    profile: str, arm: float
) -> None:
    gate = yaml.safe_load((AIPERF / f"{profile}.yaml").read_text())["benchmarks"][
        "action_chunk_parity"
    ]["gate"]
    assert gate == {"max_abs_error": arm, **GRIPPER}
