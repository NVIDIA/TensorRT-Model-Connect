# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.public_failure.contract import (
    PublicFailureValidationError,
    serialize_public_failure,
    validate_public_failure,
)
from tools.public_failure.export import ExportContext, export_failure
from tools.public_failure.render import render_failure_report
from tools.public_failure.safety import PublicFailureSafetyError, assert_public_payload_safe
from tools.public_failure import build_failure_artifacts
from tools.public_failure.__main__ import main as public_failure_main


HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40


def _context(*, result: str = "failure") -> ExportContext:
    return ExportContext(
        repository="NVIDIA/TensorRT-Model-Connect",
        pr_number=123,
        head_sha=HEAD_SHA,
        base_sha=BASE_SHA,
        tested_revision=HEAD_SHA,
        run_attempt=1,
        result=result,
        generated_at="2026-08-26T00:00:00Z",
    )


def _comparison_failure() -> dict[str, object]:
    return {
        "failure_type": "compare_fail",
        "stage": "model-proof",
        "model": "patchtsmixer",
        "backend": "native",
        "gpu_type": "H100",
        "test_id": "tests/e2e/models/patchtsmixer/test_e2e.py::test_forecast",
        "reason_code": "metric_threshold_exceeded",
        "metric": {
            "name": "max_relative_l2",
            "observed": 0.021,
            "operator": "<=",
            "threshold": 0.010,
        },
    }


def _comparison_report() -> dict[str, object]:
    return export_failure({"failures": [_comparison_failure()]}, _context())


def _build_failure(*, test_id: str) -> dict[str, object]:
    return {
        "failure_type": "build_fail",
        "stage": "build",
        "model": "llama",
        "backend": "native",
        "gpu_type": "H100",
        "test_id": test_id,
        "reason_code": "build_failed",
    }


def test_export_failure_rebuilds_a_public_result_from_approved_fields() -> None:
    failure = _comparison_failure()
    failure["raw_log"] = "Bearer must-not-escape"
    failure["command"] = "private-runner --internal-only"
    report = export_failure(
        {"failures": [failure], "jenkins_url": "http://jenkins.internal/job/123"},
        _context(),
    )

    assert report == {
        "schema_version": 1,
        "policy_version": "2026-08-26",
        "report_id": f"trtmc-pr123-{HEAD_SHA[:7]}-attempt1",
        "repository": "NVIDIA/TensorRT-Model-Connect",
        "pr_number": 123,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "tested_revision": HEAD_SHA,
        "tested_revision_kind": "head",
        "run_attempt": 1,
        "result": "failure",
        "failures": [
            {
                "public_stage": "model-proof",
                "model": "patchtsmixer",
                "backend": "native",
                "gpu_type": "H100",
                "test_id": "tests/e2e/models/patchtsmixer/test_e2e.py::test_forecast",
                "failure_class": "accuracy_regression",
                "reason_code": "metric_threshold_exceeded",
                "metric": {
                    "name": "max_relative_l2",
                    "observed": 0.021,
                    "operator": "<=",
                    "threshold": 0.010,
                },
                "disclosure": "full",
            }
        ],
        "omitted_failure_count": 0,
        "generated_at": "2026-08-26T00:00:00Z",
    }


def test_unknown_sensitive_internal_fields_do_not_change_public_bytes() -> None:
    internal = {"failures": [_comparison_failure()]}
    poisoned = deepcopy(internal)
    poisoned["access_token"] = "ghp_FAKE_SECRET"
    poisoned["internal_url"] = "http://jenkins.internal/job/123"
    poisoned["employee_email"] = "someone@nvidia.com"
    poisoned["failures"][0]["raw_log"] = "Authorization: Bearer JWT"
    poisoned["failures"][0]["workspace"] = "PRIVATE_WORKSPACE_VALUE"
    poisoned["failures"][0]["metric"]["debug_payload"] = "long-base64-secret"
    context = _context()

    clean_bytes = serialize_public_failure(export_failure(internal, context))
    poisoned_bytes = serialize_public_failure(export_failure(poisoned, context))

    assert poisoned_bytes == clean_bytes


def test_exported_report_satisfies_the_public_contract() -> None:
    report = export_failure({"failures": []}, _context(result="error"))

    validate_public_failure(report)


@pytest.mark.parametrize("level", ["report", "failure", "metric"])
def test_public_contract_rejects_unknown_fields_at_every_level(level: str) -> None:
    report = _comparison_report()
    target = {
        "report": report,
        "failure": report["failures"][0],
        "metric": report["failures"][0]["metric"],
    }[level]
    target["unexpected"] = "must be rejected"

    with pytest.raises(PublicFailureValidationError, match="unknown fields"):
        validate_public_failure(report)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("result",), "success"),
        (("head_sha",), "A" * 40),
        (("pr_number",), True),
        (("failures", 0, "failure_class"), "private_exception_name"),
        (("failures", 0, "test_id"), "../../internal/test.py::test_secret"),
        (("failures", 0, "metric", "observed"), float("nan")),
    ],
)
def test_public_contract_rejects_values_outside_the_closed_schema(
    path: tuple[object, ...], invalid_value: object
) -> None:
    report = _comparison_report()
    target = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = invalid_value

    with pytest.raises(PublicFailureValidationError):
        validate_public_failure(report)


def test_unknown_internal_values_become_fixed_placeholders_without_leaking() -> None:
    private_failure_type = "InternalSchedulerSecretException"
    private_test_id = "../../private/test.py::test_token"
    report = export_failure(
        {
            "failures": [
                {
                    "failure_type": private_failure_type,
                    "stage": "slurm-gb300-secret-pool",
                    "model": "unannounced-model",
                    "backend": "private-backend",
                    "gpu_type": "engineering-sample-serial-123",
                    "test_id": private_test_id,
                    "reason_code": "private_reason_with_hostname",
                    "metric": {
                        "name": "secret_internal_metric",
                        "observed": "host.internal",
                        "operator": "approximately",
                        "threshold": "token",
                    },
                }
            ]
        },
        _context(result="error"),
    )

    validate_public_failure(report)
    assert report["failures"] == [
        {
            "public_stage": "protected-ci",
            "model": "other-model",
            "backend": "other-backend",
            "gpu_type": "protected-gpu",
            "test_id": "withheld",
            "failure_class": "unknown",
            "reason_code": "unknown",
            "disclosure": "withheld",
        }
    ]
    public_bytes = serialize_public_failure(report)
    assert private_failure_type.encode() not in public_bytes
    assert private_test_id.encode() not in public_bytes


def test_json_schema_accepts_the_exported_report_and_is_itself_valid() -> None:
    schema_path = (
        Path(__file__).parents[2] / "tools/public_failure/assets/public-failure-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = export_failure({"failures": []}, _context(result="error"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_renderer_produces_deterministic_self_contained_html() -> None:
    report = _comparison_report()

    first = render_failure_report(report)
    second = render_failure_report(report)

    assert first == second
    document = first.decode("utf-8")
    assert "TRTMC Protected CI failure report" in document
    assert "patchtsmixer" in document
    assert "max_relative_l2" in document
    assert "0.021" in document
    assert "&lt;=" in document
    assert "default-src 'none'" in document
    assert '<span class="status">FAILED</span>' in document
    assert '<table class="run-meta">' in document
    assert '<table class="failure-table">' in document
    lowered = document.lower()
    assert 'class="banner"' not in lowered
    assert 'class="card"' not in lowered
    assert 'class="eyebrow"' not in lowered
    assert "<script" not in lowered
    assert "<link" not in lowered
    assert "<img" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_poison_scan_rejects_sensitive_text_even_when_schema_allows_the_characters() -> None:
    report = export_failure(
        {
            "failures": [
                _build_failure(test_id="tests/e2e/test_ghp_FAKESECRET123456789.py::test_build")
            ]
        },
        _context(),
    )
    document = render_failure_report(report)

    with pytest.raises(PublicFailureSafetyError, match="credential-like token"):
        assert_public_payload_safe(report, document)


def test_build_failure_artifacts_runs_the_complete_local_pipeline() -> None:
    artifacts = build_failure_artifacts(
        {
            "failures": [
                _build_failure(test_id="tests/e2e/models/llama/test_llama_e2e.py::test_model_e2e")
            ]
        },
        _context(),
    )

    assert json.loads(artifacts.json_bytes) == artifacts.report
    assert b"TRTMC Protected CI failure report" in artifacts.html_bytes
    assert b"Raw logs and internal diagnostics are not included" in artifacts.html_bytes


def test_local_cli_writes_preview_files_without_publishing(tmp_path: Path) -> None:
    input_path = tmp_path / "internal.json"
    context_path = tmp_path / "context.json"
    output_dir = tmp_path / "preview"
    input_path.write_text(
        json.dumps(
            {
                "failures": [
                    _build_failure(
                        test_id="tests/e2e/models/llama/test_llama_e2e.py::test_model_e2e"
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    context_path.write_text(json.dumps(asdict(_context())), encoding="utf-8")

    exit_code = public_failure_main(
        [
            "--input",
            str(input_path),
            "--context",
            str(context_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert json.loads((output_dir / "public-failure.json").read_text())["result"] == "failure"
    assert "TRTMC Protected CI failure report" in (output_dir / "report.html").read_text()
