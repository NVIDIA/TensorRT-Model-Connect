# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the immutable OpenPI model-owned contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.models.openpi.qualification import (
    UPSTREAM_COMMIT,
    load_contract,
    load_thresholds,
)
from tests.e2e_harness.manifest_loader import load_model_manifest


ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("profile", "horizon", "output_dim", "discrete_state"),
    [
        ("pi05_droid", 15, 8, True),
    ],
)
def test_openpi_profile_contract_matches_pinned_upstream(
    profile: str,
    horizon: int,
    output_dim: int,
    discrete_state: bool,
) -> None:
    contract = load_contract(profile)

    assert contract["upstream"]["commit"] == UPSTREAM_COMMIT
    assert contract["images"]["order"] == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    ]
    assert contract["images"]["validity"] == [True, True, False]
    assert contract["prompt"]["discrete_state_input"] is discrete_state
    assert contract["actions"] == {
        "horizon": horizon,
        "external_dim": output_dim,
        "internal_dim": 32,
        "normalization": "quantile_q01_q99",
    }
    assert contract["prefix"]["max_physical_tokens"] == 968
    assert contract["prefix"]["kv_heads"] == 1
    assert contract["prefix"]["head_dim"] == 256
    assert contract["flow"]["steps"] == 10
    assert contract["flow"]["external_noise_required_for_parity"] is True
    assert contract["runtime"]["onnx_allowed"] is False
    assert contract["runtime"]["python_allowed"] is False
    assert contract["runtime"]["additional_frameworks_allowed"] is False


@pytest.mark.parametrize(
    ("manifest_name", "profile", "horizon", "output_dim"),
    [
        ("pi05-droid.json", "pi05_droid", 15, 8),
    ],
)
def test_openpi_e2e_manifest_preserves_action_contract(
    manifest_name: str,
    profile: str,
    horizon: int,
    output_dim: int,
) -> None:
    model = load_model_manifest(ROOT / "manifests" / manifest_name)
    assert len(model.testcases) == 1
    case = model.testcases[0]

    assert case.family == "openpi"
    assert case.runtime_strategy == "openpi_vla"
    assert case.task_strategy == "robot_action_generation"
    assert case.user_contract == "robot_action_chunk"
    assert case.reference_backend == "upstream_replay"
    assert case.oracle_level == "L1_external_reference"
    assert case.inputs["profile"] == profile
    assert case.inputs["action_horizon"] == horizon
    assert case.inputs["output_action_dim"] == output_dim
    assert case.inputs["internal_action_dim"] == 32
    assert case.inputs["fixed_external_noise"] is True
    assert case.determinism == {"seed": 42, "reruns": 100}
    assert [stage.name for stage in case.stages] == [
        "preprocess",
        "vision",
        "prefix",
        "flow",
        "actions",
    ]
    assert all(stage.required for stage in case.stages)


@pytest.mark.parametrize("profile", ["pi05_droid"])
def test_openpi_thresholds_cannot_weaken_high_accuracy_gates(profile: str) -> None:
    thresholds = load_thresholds(profile)

    assert thresholds["normalized_action_cosine_min"] >= 0.9995
    assert thresholds["normalized_action_mae_max"] <= 0.003
    assert thresholds["normalized_action_p99_abs_max"] <= 0.01
    assert thresholds["normalized_action_max_abs_max"] <= 0.02
    assert thresholds["native_latency_p50_ms_max"] <= 50.0
    assert thresholds["native_latency_p95_ms_max"] <= 60.0
    assert thresholds["torch_eager_speedup_p50_min"] >= 5.0
    assert thresholds["torch_eager_speedup_p95_min"] >= 5.0
