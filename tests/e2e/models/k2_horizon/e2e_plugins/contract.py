# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon-owned strict greedy-continuation contract."""

from __future__ import annotations

from .comparators.text import TextComparator
from .contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _metric(passed: bool, note: str) -> MetricResult:
    return MetricResult(
        value=1.0 if passed else 0.0,
        threshold=1.0,
        operator="==",
        passed=passed,
        note=note,
    )


def _tokens(output: StageOutput) -> list[int]:
    return [int(token) for token in (output.data or {}).get("token_ids", [])]


def _contract_config(case: E2ECase) -> dict:
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


class K2HorizonGreedyContinuationPlugin:
    reference_families = ["k2_horizon_greedy_continuation"]
    user_contract = "continuation_parity"

    def configure_reference(self, case: E2ECase) -> dict:
        return _contract_config(case)

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        numeric = TextComparator().compare(
            trt=trt_output,
            ref=ref_output,
            threshold=threshold,
            stage=StageSpec(name=trt_output.stage_name),
        )
        trt_tokens = _tokens(trt_output)
        ref_tokens = _tokens(ref_output)
        revision = str((ref_output.data or {}).get("model_revision", ""))
        expected = case.metadata.get("expected_continuation_token_ids")
        expected_tokens = [int(token) for token in expected] if isinstance(expected, list) else []
        checks = {
            "nonempty_continuation": (
                bool(trt_tokens) and bool(ref_tokens),
                f"TRT={trt_tokens}, HF={ref_tokens}",
            ),
            "exact_token_parity": (
                bool(trt_tokens) and trt_tokens == ref_tokens,
                f"TRT={trt_tokens}, HF={ref_tokens}",
            ),
            "pinned_reference_revision": (
                bool(case.hf_revision) and revision == case.hf_revision,
                f"expected={case.hf_revision}, actual={revision}",
            ),
            "expected_golden_continuation": (
                bool(expected_tokens)
                and ref_tokens == expected_tokens
                and trt_tokens == expected_tokens,
                f"expected={expected_tokens}, TRT={trt_tokens}, HF={ref_tokens}",
            ),
        }
        contract_metrics = {name: _metric(passed, note) for name, (passed, note) in checks.items()}
        metrics = {**numeric.metrics, **contract_metrics}
        contract_passed = all(passed for passed, _note in checks.values())
        passed = numeric.passed and contract_passed
        failed = [name for name, (ok, _note) in checks.items() if not ok]
        if numeric.status == StageStatus.ERROR.value:
            status = StageStatus.ERROR.value
        else:
            status = StageStatus.PASSED.value if passed else StageStatus.FAILED.value
        message_parts = [numeric.message]
        if failed:
            message_parts.append("contract failed: " + ", ".join(failed))
        return CompareResult(
            stage_name=trt_output.stage_name,
            status=status,
            metrics=metrics,
            composite_rule=f"({numeric.composite_rule}) AND " + " AND ".join(checks),
            message="; ".join(part for part in message_parts if part),
        )


plugin = K2HorizonGreedyContinuationPlugin()
