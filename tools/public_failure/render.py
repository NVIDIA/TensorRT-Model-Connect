# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render validated public-failure-v1 data as deterministic plain text."""

from __future__ import annotations

from typing import Mapping

from .contract import validate_public_failure


REASON_SUMMARIES = {
    "artifact_write_failed": "The test harness could not write its result artifact.",
    "build_failed": "The tested source failed to build.",
    "canonical_document_mismatch": "A canonical repository document does not match policy.",
    "complexity_limit_exceeded": "Source complexity exceeds the configured limit.",
    "determinism_check_failed": "The determinism check failed.",
    "github_automation_permission_denied": "GitHub Actions is not allowed to create the required pull request.",
    "gpu_capacity_unavailable": "The protected GPU did not have enough allocatable capacity.",
    "infrastructure_failed": "Protected CI infrastructure failed.",
    "metric_threshold_exceeded": "A correctness metric exceeded its allowed threshold.",
    "model_contract_failed": "A model-specific regression contract failed.",
    "model_cache_warm_failed": "A required public model cache could not be warmed.",
    "model_output_mismatch": "Model output did not match the required reference.",
    "out_of_memory": "The test ran out of memory.",
    "python_dependency_missing": "A required Python dependency is unavailable.",
    "python_package_import_failed": "The built Python package could not be imported.",
    "reference_failed": "The reference implementation failed.",
    "runtime_catalog_miss": "No qualified runtime exists for this exact Source revision.",
    "runtime_failed": "The tested runtime execution failed.",
    "runtime_image_pull_timeout": "The qualified runtime image could not be fetched in time.",
    "source_formatting_failed": "Source formatting validation failed.",
    "source_revision_mismatch": "The selected runtime does not match the tested Source inputs.",
    "spdx_preamble_invalid": "An SPDX directive is outside the approved file preamble.",
    "test_failed": "A named test failed.",
    "timed_out": "The test exceeded its time limit.",
    "unknown": "No structured failure detail was safe to disclose.",
}


def _format_number(value: object) -> str:
    return format(float(value), ".8g")


def _failure_lines(failure: Mapping[str, object], index: int) -> list[str]:
    lines = [
        f"Failure {index}",
        f"  Class: {failure['failure_class']}",
        f"  Reason: {failure['reason_code']}",
        f"  Cause: {REASON_SUMMARIES[str(failure['reason_code'])]}",
        f"  Stage: {failure['public_stage']}",
        f"  Model: {failure['model']}",
        f"  Backend: {failure['backend']}",
        "  GPU: protected-gpu",
        f"  Test: {failure['test_id']}",
    ]
    if "subject" in failure:
        lines.append(f"  Subject: {failure['subject']}")
    metric = failure.get("metric")
    if isinstance(metric, Mapping):
        lines.extend(
            [
                f"  Metric: {metric['name']}",
                f"  Observed: {_format_number(metric['observed'])}",
                (
                    f"  Requirement: {metric['operator']} "
                    f"{_format_number(metric['threshold'])}"
                ),
            ]
        )
    elif failure["disclosure"] == "withheld":
        lines.append("  Evidence: details withheld")
    return lines


def render_failure_report(report: Mapping[str, object]) -> bytes:
    """Validate and render one deterministic UTF-8 ``public-failure.log``."""
    validate_public_failure(report)
    status = "FAILED" if report["result"] == "failure" else "ERROR"
    lines = [
        "TRTMC Protected CI failure",
        "==========================",
        "",
        "This log contains only approved structured failure fields.",
        "",
        f"Status: {status}",
        f"Repository: {report['repository']}",
        f"Pull request: #{report['pr_number']}",
        f"Head commit: {report['head_sha']}",
        (
            f"Tested revision: {report['tested_revision']} "
            f"({report['tested_revision_kind']})"
        ),
        f"Run attempt: {report['run_attempt']}",
        f"Generated at: {report['generated_at']}",
        f"Disclosure policy: {report['policy_version']}",
        "",
        "Failure summary",
        "---------------",
    ]
    failures = report["failures"]
    if failures:
        for index, failure in enumerate(failures, start=1):
            if index > 1:
                lines.append("")
            lines.extend(_failure_lines(failure, index))
    else:
        lines.append("No structured failure details were safe to disclose.")
    omitted = int(report["omitted_failure_count"])
    if omitted:
        lines.extend(["", f"{omitted} additional failure(s) were omitted."])
    lines.extend(["", f"Report ID: {report['report_id']}"])
    return ("\n".join(lines) + "\n").encode("utf-8")
