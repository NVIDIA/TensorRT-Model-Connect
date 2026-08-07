# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only task-eval contract tests for the GPT2 family."""

from __future__ import annotations

from typing import Any

from tools.validation import engine as validation_engine


_SUITE_ID = "wikitext103_distilgpt2_continuation_parity"
_R4_RECEIPT: dict[str, Any] = {
    "run": (
        "trtmc-validate-gb300-3-20260729T084401Z-"
        "5dc37b75-all105-offline-fresh-r4"
    ),
    "sample_count": 20,
    "exact_match_rate": 0.95,
    "tie_adjusted_exact_match_rate": 0.95,
    "token_prefix_agreement": 0.9671875,
    "divergent_count": 1,
    "divergent_sample_id": "wikitext103_000011",
    "first_divergence": 22,
    "hf_token": "Cambridge",
    "trtmc_token": "Oxford",
}
_C829_RECEIPT: dict[str, Any] = {
    "run": "trtmc-validate-gb300-2-20260726-c8291be0-all105",
    "sample_count": 20,
    "exact_match_rate": 0.95,
    "tie_adjusted_exact_match_rate": 0.95,
    "token_prefix_agreement": 0.96484375,
    "divergent_count": 1,
}


def _suite() -> dict[str, Any]:
    return validation_engine.suite_by_id(
        validation_engine.load_suites(), _SUITE_ID
    )


def _apply_suite_gates(receipt: dict[str, Any]) -> dict[str, Any]:
    result = dict(receipt)
    validation_engine.apply_metric_gates(result, _suite()["gates"])
    return result


def test_distilgpt2_suite_owns_sample_level_continuation_gate() -> None:
    suite = _suite()

    assert suite["selectors"]["model_names"] == ["distilgpt2"]
    assert suite["scoring"]["scorer"] == "continuation"
    assert suite["gates"] == {"min_tie_adjusted_exact_match_rate": 0.9}
    assert suite["ci"]["eligible"] is False
    assert suite["ci"]["lane"] == "local_only"


def test_distilgpt2_deterministic_gb300_receipts_pass_gate() -> None:
    for receipt in (_C829_RECEIPT, _R4_RECEIPT):
        result = _apply_suite_gates(receipt)

        assert result["sample_count"] == 20
        assert result["exact_match_rate"] == 19 / 20
        assert result["divergent_count"] == 1
        assert result["status"] == "passed"
        assert result["gate_failures"] == []

    assert _R4_RECEIPT["divergent_sample_id"] == "wikitext103_000011"
    assert _R4_RECEIPT["first_divergence"] == 22
    assert (
        _R4_RECEIPT["hf_token"],
        _R4_RECEIPT["trtmc_token"],
    ) == ("Cambridge", "Oxford")


def test_distilgpt2_regressed_receipt_fails_closed() -> None:
    result = _apply_suite_gates(
        {
            "sample_count": 20,
            "exact_match_rate": 0.85,
            "tie_adjusted_exact_match_rate": 0.85,
            "token_prefix_agreement": 0.90,
            "divergent_count": 3,
        }
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "BenchmarkGateError"
    assert result["gate_failures"] == [
        {
            "gate": "min_tie_adjusted_exact_match_rate",
            "metric": "tie_adjusted_exact_match_rate",
            "actual": 0.85,
            "required": 0.9,
        }
    ]
