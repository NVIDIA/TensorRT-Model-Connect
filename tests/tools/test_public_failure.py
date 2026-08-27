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


def test_fast_foundation_stereo_failure_keeps_the_public_model_name() -> None:
    report = export_failure(
        {
            "failures": [
                {
                    "failure_type": "unit_fail",
                    "stage": "unit",
                    "model": "fast_foundation_stereo",
                    "backend": "other-backend",
                    "gpu_type": "protected-gpu",
                    "test_id": "ctest::test_fast_foundation_stereo_native_plugins",
                    "reason_code": "test_failed",
                }
            ]
        },
        _context(),
    )

    assert report["failures"][0]["model"] == "fast_foundation_stereo"


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
        "policy_version": "2026-08-27",
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


def test_metric_failure_without_valid_metric_is_withheld_as_unknown() -> None:
    failure = _comparison_failure()
    failure["metric"] = {"name": "private_metric", "observed": "invalid"}
    failure["excerpt"] = ["E   comparison failed"]

    report = export_failure({"failures": [failure]}, _context())

    validate_public_failure(report)
    assert report["failures"][0]["reason_code"] == "unknown"
    assert report["failures"][0]["disclosure"] == "withheld"
    assert "metric" not in report["failures"][0]
    assert "excerpt" not in report["failures"][0]


def test_unknown_reason_never_exports_an_excerpt() -> None:
    failure = _build_failure(test_id="tests/e2e/test_build.py::test_build")
    failure["reason_code"] = "private_failure_reason"
    failure["excerpt"] = ["E   safe-looking but unclassified failure detail"]

    report = export_failure({"failures": [failure]}, _context())

    validate_public_failure(report)
    assert report["failures"][0]["reason_code"] == "unknown"
    assert report["failures"][0]["disclosure"] == "withheld"
    assert "excerpt" not in report["failures"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda failure: failure.pop("metric"),
        lambda failure: failure.update(reason_code="unknown", disclosure="truncated"),
        lambda failure: failure.update(
            reason_code="unknown", disclosure="withheld", excerpt=["E   detail"]
        ),
    ],
)
def test_public_contract_rejects_invalid_reason_disclosure_invariants(mutation) -> None:
    report = _comparison_report()
    mutation(report["failures"][0])

    with pytest.raises(PublicFailureValidationError):
        validate_public_failure(report)


def test_json_schema_accepts_the_exported_report_and_is_itself_valid() -> None:
    schema_path = (
        Path(__file__).parents[2] / "tools/public_failure/assets/public-failure-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = export_failure({"failures": []}, _context(result="error"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda failure: failure.pop("metric"),
        lambda failure: failure.update(reason_code="unknown", disclosure="truncated"),
        lambda failure: failure.update(
            reason_code="unknown", disclosure="withheld", excerpt=["E   detail"]
        ),
    ],
)
def test_json_schema_enforces_reason_disclosure_invariants(mutation) -> None:
    schema_path = (
        Path(__file__).parents[2] / "tools/public_failure/assets/public-failure-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = _comparison_report()
    mutation(report["failures"][0])

    assert list(Draft202012Validator(schema).iter_errors(report))


def test_renderer_produces_deterministic_plain_text() -> None:
    report = _comparison_report()

    first = render_failure_report(report)
    second = render_failure_report(report)

    assert first == second
    document = first.decode("utf-8")
    assert "TRTMC Protected CI failure" in document
    assert "patchtsmixer" in document
    assert "max_relative_l2" in document
    assert "0.021" in document
    assert "Requirement: <= 0.01" in document
    assert "Status: FAILED" in document
    lowered = document.lower()
    assert "<!doctype" not in lowered
    assert "<html" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_renderer_includes_a_bounded_sanitized_failed_step_tail() -> None:
    failure = _comparison_failure()
    failure["excerpt"] = [
        "E   AssertionError: output mismatch",
        "FAILED tests/e2e/models/patchtsmixer/test_e2e.py::test_forecast",
    ]
    report = export_failure({"failures": [failure]}, _context())

    validate_public_failure(report)
    document = render_failure_report(report).decode("utf-8")

    assert report["failures"][0]["disclosure"] == "truncated"
    assert "Sanitized failed-step excerpt (tail):" in document
    assert "E   AssertionError: output mismatch" in document
    assert "FAILED tests/e2e/models/patchtsmixer/test_e2e.py::test_forecast" in document


@pytest.mark.parametrize(
    "excerpt",
    [
        [],
        ["x"] * 21,
        ["line with\nnewline"],
        ["x" * 241],
        ["x" * 201] * 20,
    ],
)
def test_public_contract_rejects_invalid_excerpts(excerpt: list[str]) -> None:
    report = _comparison_report()
    report["failures"][0]["excerpt"] = excerpt

    with pytest.raises(PublicFailureValidationError):
        validate_public_failure(report)


def test_safety_scan_rejects_a_sensitive_excerpt() -> None:
    failure = _comparison_failure()
    failure["excerpt"] = ["request failed at https://runner.internal/log"]
    report = export_failure({"failures": [failure]}, _context())
    document = render_failure_report(report)

    with pytest.raises(PublicFailureSafetyError, match="URL"):
        assert_public_payload_safe(report, document)


def test_safety_scan_rejects_an_unredacted_registry_image_reference() -> None:
    failure = _comparison_failure()
    failure["excerpt"] = ["pull nvcr.io/private/image:build failed"]
    report = export_failure({"failures": [failure]}, _context())
    document = render_failure_report(report)

    with pytest.raises(PublicFailureSafetyError, match="registry image reference"):
        assert_public_payload_safe(report, document)


def test_safety_scan_allows_a_dotted_relative_test_path() -> None:
    report = export_failure(
        {"failures": [_build_failure(test_id="docs/schema.json/examples")]},
        _context(),
    )
    document = render_failure_report(report)

    assert_public_payload_safe(report, document)


@pytest.mark.parametrize(
    ("internal_failure", "expected_lines"),
    [
        (
            {
                "failure_type": "unit_fail",
                "stage": "unit",
                "test_id": "tests/tools/test_public_failure.py::collection",
                "reason_code": "python_dependency_missing",
                "subject": "jsonschema",
            },
            (
                "Cause: A required Python dependency is unavailable.",
                "Subject: jsonschema",
                "Test: tests/tools/test_public_failure.py::collection",
            ),
        ),
        (
            {
                "failure_type": "legal_fail",
                "stage": "legal",
                "test_id": "bindings/nodejs/example.js",
                "reason_code": "spdx_preamble_invalid",
            },
            (
                "Cause: An SPDX directive is outside the approved file preamble.",
                "Test: bindings/nodejs/example.js",
            ),
        ),
        (
            {
                "failure_type": "source_quality_fail",
                "stage": "source-quality",
                "test_id": "cpp/src/example.cpp:42",
                "reason_code": "source_formatting_failed",
            },
            (
                "Cause: Source formatting validation failed.",
                "Test: cpp/src/example.cpp:42",
            ),
        ),
        (
            {
                "failure_type": "infrastructure_error",
                "stage": "runtime-control",
                "test_id": "runtime-catalog",
                "reason_code": "runtime_catalog_miss",
            },
            (
                "Cause: No qualified runtime exists for this exact Source revision.",
                "Test: runtime-catalog",
            ),
        ),
        (
            {
                "failure_type": "compare_fail",
                "stage": "model-proof",
                "model": "internvl3-2b",
                "test_id": "internvl3-2b::full_generation",
                "reason_code": "model_output_mismatch",
            },
            (
                "Cause: Model output did not match the required reference.",
                "Test: internvl3-2b::full_generation",
            ),
        ),
        (
            {
                "failure_type": "package_fail",
                "stage": "package",
                "test_id": "python-wheel::import",
                "reason_code": "python_package_import_failed",
                "subject": "tensorrt_model_connect",
            },
            (
                "Cause: The built Python package could not be imported.",
                "Subject: tensorrt_model_connect",
            ),
        ),
        (
            {
                "failure_type": "infrastructure_error",
                "stage": "model-cache",
                "test_id": "model-cache",
                "reason_code": "model_cache_warm_failed",
            },
            (
                "Cause: A required public model cache could not be warmed.",
                "Test: model-cache",
            ),
        ),
        (
            {
                "failure_type": "infrastructure_error",
                "stage": "model-proof",
                "test_id": "qwen3_omni::gpu-admission",
                "reason_code": "gpu_capacity_unavailable",
            },
            (
                "Cause: The protected GPU did not have enough allocatable capacity.",
                "Test: qwen3_omni::gpu-admission",
            ),
        ),
    ],
)
def test_real_internal_ci_failure_classes_render_actionable_text(
    internal_failure: dict[str, object], expected_lines: tuple[str, ...]
) -> None:
    artifacts = build_failure_artifacts({"failures": [internal_failure]}, _context())
    document = artifacts.log_bytes.decode("utf-8")

    for line in expected_lines:
        assert line in document


def test_public_failure_relay_uses_the_existing_status_context() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/internal-ci-failure-log.yml"
    ).read_text(encoding="utf-8")

    assert "repository_dispatch:" in workflow
    assert "types: [trtmc-public-failure-v1]" in workflow
    assert "Publish public-failure.log" in workflow
    assert "validate_public_failure(report)" in workflow
    assert "assert_public_payload_safe(report, document)" in workflow
    assert workflow.count("TRTMC Internal CI / Automated premerge gate") == 1
    assert "actions/runs/$GITHUB_RUN_ID" in workflow
    assert "pulls/$PR_NUMBER" in workflow
    assert workflow.index("- name: Confirm the exact open pull-request head") < workflow.index(
        "- name: Print public-failure.log"
    )


def test_internal_ci_bridge_publishes_the_private_sanitized_artifact() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/internal-ci-bridge.yml"
    ).read_text(encoding="utf-8")

    assert "--name public-failure-payload" in workflow
    assert "--log" not in workflow
    assert "validate_public_failure(report)" in workflow
    assert "assert_public_payload_safe(report, document)" in workflow
    assert "name: public-failure-log" in workflow
    assert workflow.count("TRTMC Internal CI / Automated premerge gate") == 1
    assert "actions/runs/$GITHUB_RUN_ID" in workflow
    assert workflow.index("- name: Confirm the exact open pull-request head") < workflow.index(
        "- name: Print public-failure.log"
    )


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
    assert b"TRTMC Protected CI failure" in artifacts.log_bytes
    assert b"approved structured fields" in artifacts.log_bytes


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
    output_dir.mkdir()
    (output_dir / "report.html").write_text("stale legacy output", encoding="utf-8")

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
    assert "TRTMC Protected CI failure" in (
        output_dir / "public-failure.log"
    ).read_text()
    assert not (output_dir / "report.html").exists()
