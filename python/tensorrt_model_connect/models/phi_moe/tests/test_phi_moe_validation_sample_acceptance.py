# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE-owned validation sample-acceptance contract."""

from tools.validation import catalog as validation_catalog


def test_phi_moe_mmlu_merges_owner_sample_acceptance() -> None:
    suites = validation_catalog.load_suites(
        _owners={"phi_moe"},
        _require_all_suites=False,
    )
    suite = validation_catalog.suite_by_id(suites, "mmlu_five_shot_mcq")
    model = next(
        model
        for model in validation_catalog.load_manifest_records(
            validation_catalog.DEFAULT_MODELS_DIR / "phi_moe"
        )
        if model["name"] == "phi-moe"
    )

    resolved = validation_catalog.resolve_suite_for_model(suite, model)

    assert suite["sample_acceptance"] == {
        "min_pass_rate": 0.98,
        "min_allowed_failures": 1,
    }
    assert resolved["sample_acceptance"] == {
        "min_pass_rate": 0.95,
        "min_allowed_failures": 1,
    }
    assert resolved["gates"] == {"max_accuracy_drop_from_hf": 0.01}
