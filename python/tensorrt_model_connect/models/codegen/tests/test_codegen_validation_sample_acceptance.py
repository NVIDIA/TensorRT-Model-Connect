# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CodeGen-owned validation sample-acceptance contract."""

from tools.validation import catalog as validation_catalog


def test_codegen_humaneval_owns_zero_failure_acceptance() -> None:
    suites = validation_catalog.load_suites(
        _owners={"codegen"},
        _require_all_suites=False,
    )
    suite = validation_catalog.suite_by_id(
        suites,
        "humaneval_code_continuation_parity",
    )
    model = next(
        model
        for model in validation_catalog.load_manifest_records(
            validation_catalog.DEFAULT_MODELS_DIR / "codegen"
        )
        if model["name"] == "codegen-350m"
    )

    resolved = validation_catalog.resolve_suite_for_model(suite, model)

    assert "sample_acceptance" not in suite
    assert suite["gate_policy"] == "observation_only"
    assert resolved["sample_acceptance"] == {
        "min_pass_rate": 1.0,
        "min_allowed_failures": 0,
    }
    assert resolved["gate_policy"] == "blocking"
    assert resolved["model_overrides"]["by_model"]["codegen-350m"] == {
        "build_generation_headroom": True,
    }
