# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for generic batch sample acceptance."""

from __future__ import annotations

from tools.validation import catalog as validation_catalog
from tools.validation.gate_policy import evaluate_sample_acceptance


def test_batch_acceptance_allows_the_declared_failure_floor() -> None:
    policy = {"min_pass_rate": 0.98, "min_allowed_failures": 1}

    accepted = evaluate_sample_acceptance(
        policy=policy,
        sample_count=20,
        passed_count=19,
        expected_count=20,
    )
    rejected = evaluate_sample_acceptance(
        policy=policy,
        sample_count=20,
        passed_count=18,
        expected_count=20,
    )
    incomplete = evaluate_sample_acceptance(
        policy=policy,
        sample_count=19,
        passed_count=19,
        expected_count=20,
    )

    assert accepted["verdict"] == "pass"
    assert accepted["allowed_failures"] == 1
    assert rejected["verdict"] == "fail"
    assert incomplete["verdict"] == "invalid"
    assert incomplete["issues"] == [
        {"code": "incomplete_samples", "expected": 20, "actual": 19}
    ]


def test_synthetic_profiles_merge_into_generic_sample_acceptance() -> None:
    suite = {
        "id": "synthetic_quality",
        "generation": {},
        "gates": {"max_quality_drop": 0.01},
        "sample_acceptance": {
            "min_pass_rate": 0.98,
            "min_allowed_failures": 1,
        },
        "family_profiles": {
            "synthetic_family": {
                "sample_acceptance": {"min_pass_rate": 0.95},
            }
        },
        "model_profiles": {
            "synthetic_model": {
                "sample_acceptance": {"min_allowed_failures": 0},
            }
        },
    }

    resolved = validation_catalog.resolve_suite_for_model(
        suite,
        {"family": "synthetic_family", "name": "synthetic_model"},
    )

    assert resolved["sample_acceptance"] == {
        "min_pass_rate": 0.95,
        "min_allowed_failures": 0,
    }
    assert resolved["gates"] == {"max_quality_drop": 0.01}
    assert resolved["gate_policy"] == "blocking"
    assert suite["sample_acceptance"] == {
        "min_pass_rate": 0.98,
        "min_allowed_failures": 1,
    }


def test_generic_suites_declare_batch_acceptance_without_model_policy() -> None:
    raw = validation_catalog.load_structured_file(
        validation_catalog.DEFAULT_SUITES
    )
    suites = {suite["id"]: suite for suite in raw["suites"]}
    policies = sorted(
        (
            policy["min_pass_rate"],
            policy["min_allowed_failures"],
        )
        for suite in suites.values()
        if isinstance((policy := suite.get("sample_acceptance")), dict)
    )
    assert policies == [
        (0.9, 0),
        (0.9, 1),
        (0.95, 0),
        (0.95, 0),
        (0.95, 0),
        (0.95, 0),
        (0.98, 1),
    ]

    for suite in raw["suites"]:
        assert not {
            "default_model_names",
            "family_profiles",
            "model_profiles",
            "model_overrides",
            "qualification_models",
        } & set(suite)
