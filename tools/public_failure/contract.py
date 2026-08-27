# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and deterministic serialization for public-failure-v1."""

from __future__ import annotations

import json
import math
import re
from typing import Collection, Mapping

from .policy import (
    FAILURE_CLASS_BY_INTERNAL_TYPE,
    POLICY_VERSION,
    PUBLIC_BACKENDS,
    PUBLIC_GPU_TYPES,
    PUBLIC_METRIC_NAMES,
    PUBLIC_METRIC_OPERATORS,
    PUBLIC_MODELS,
    PUBLIC_REASON_CODES,
    PUBLIC_SUBJECTS,
    PUBLIC_STAGE_BY_INTERNAL_STAGE,
)


REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "report_id",
        "repository",
        "pr_number",
        "head_sha",
        "base_sha",
        "tested_revision",
        "tested_revision_kind",
        "dispatch_nonce",
        "run_attempt",
        "result",
        "failures",
        "omitted_failure_count",
        "generated_at",
    }
)
REPORT_REQUIRED_FIELDS = REPORT_FIELDS - {"dispatch_nonce"}
FAILURE_FIELDS = frozenset(
    {
        "public_stage",
        "model",
        "backend",
        "gpu_type",
        "test_id",
        "failure_class",
        "reason_code",
        "metric",
        "subject",
        "excerpt",
        "disclosure",
    }
)
METRIC_FIELDS = frozenset({"name", "observed", "operator", "threshold"})
FAILURE_REQUIRED_FIELDS = FAILURE_FIELDS - {"metric", "subject", "excerpt"}
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
DISPATCH_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
REPORT_ID_PATTERN = re.compile(r"trtmc-pr[1-9][0-9]*-[0-9a-f]{7}-attempt[1-9][0-9]*\Z")
TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
TEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_./:\[\],=+-]+\Z")
EXCERPT_LINE_PATTERN = re.compile(r"[\x20-\x7e]{1,240}\Z")
PUBLIC_FAILURE_CLASSES = frozenset(FAILURE_CLASS_BY_INTERNAL_TYPE.values()) | {"unknown"}
PUBLIC_STAGES = frozenset(PUBLIC_STAGE_BY_INTERNAL_STAGE.values()) | {"protected-ci"}
PUBLIC_MODEL_NAMES = PUBLIC_MODELS | {"other-model"}
PUBLIC_BACKEND_NAMES = PUBLIC_BACKENDS | {"other-backend"}
PUBLIC_GPU_NAMES = PUBLIC_GPU_TYPES | {"protected-gpu"}


class PublicFailureValidationError(ValueError):
    """Raised when an object violates public-failure-v1."""


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: Collection[str], path: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise PublicFailureValidationError(f"{path} has unknown fields: {unknown}")


def _require_fields(value: Mapping[str, object], required: Collection[str], path: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise PublicFailureValidationError(f"{path} is missing fields: {missing}")


def _require_enum(value: object, allowed: Collection[str], path: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise PublicFailureValidationError(f"{path} is not an approved value")


def _require_string(
    value: object, path: str, *, max_length: int, pattern: re.Pattern[str] | None = None
) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise PublicFailureValidationError(f"{path} must be a bounded non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise PublicFailureValidationError(f"{path} has an invalid format")
    return value


def _require_integer(value: object, path: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise PublicFailureValidationError(f"{path} must be a bounded integer")


def _validate_metric(metric: Mapping[str, object], path: str) -> None:
    _reject_unknown_fields(metric, METRIC_FIELDS, path)
    _require_fields(metric, METRIC_FIELDS, path)
    _require_enum(metric.get("name"), PUBLIC_METRIC_NAMES, f"{path}.name")
    _require_enum(metric.get("operator"), PUBLIC_METRIC_OPERATORS, f"{path}.operator")
    for key in ("observed", "threshold"):
        number = metric.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise PublicFailureValidationError(f"{path}.{key} must be finite")


def _validate_failure(failure: Mapping[str, object], index: int) -> None:
    path = f"failures[{index}]"
    _reject_unknown_fields(failure, FAILURE_FIELDS, path)
    _require_fields(failure, FAILURE_REQUIRED_FIELDS, path)
    _require_enum(failure.get("public_stage"), PUBLIC_STAGES, f"{path}.public_stage")
    _require_enum(failure.get("model"), PUBLIC_MODEL_NAMES, f"{path}.model")
    _require_enum(failure.get("backend"), PUBLIC_BACKEND_NAMES, f"{path}.backend")
    _require_enum(failure.get("gpu_type"), PUBLIC_GPU_NAMES, f"{path}.gpu_type")
    _require_enum(failure.get("failure_class"), PUBLIC_FAILURE_CLASSES, f"{path}.failure_class")
    _require_enum(failure.get("reason_code"), PUBLIC_REASON_CODES, f"{path}.reason_code")
    if "subject" in failure:
        _require_enum(failure.get("subject"), PUBLIC_SUBJECTS, f"{path}.subject")
    if "excerpt" in failure:
        excerpt = failure.get("excerpt")
        if not isinstance(excerpt, list) or not excerpt or len(excerpt) > 20:
            raise PublicFailureValidationError(
                f"{path}.excerpt must be a non-empty array of at most 20 lines"
            )
        if sum(len(line) for line in excerpt if isinstance(line, str)) > 4000:
            raise PublicFailureValidationError(
                f"{path}.excerpt must contain at most 4000 characters"
            )
        for line_index, line in enumerate(excerpt):
            _require_string(
                line,
                f"{path}.excerpt[{line_index}]",
                max_length=240,
                pattern=EXCERPT_LINE_PATTERN,
            )
    _require_enum(
        failure.get("disclosure"),
        {"full", "truncated", "withheld"},
        f"{path}.disclosure",
    )
    test_id = _require_string(
        failure.get("test_id"),
        f"{path}.test_id",
        max_length=300,
        pattern=TEST_ID_PATTERN,
    )
    if test_id.startswith("/") or ".." in test_id or "\\" in test_id:
        raise PublicFailureValidationError(f"{path}.test_id is not a safe relative test ID")
    reason_code = failure.get("reason_code")
    metric = failure.get("metric")
    if reason_code == "metric_threshold_exceeded" and metric is None:
        raise PublicFailureValidationError(f"{path} must include metric threshold evidence")
    if reason_code == "unknown":
        if failure.get("disclosure") != "withheld":
            raise PublicFailureValidationError(f"{path} must withhold an unknown reason")
        if "excerpt" in failure:
            raise PublicFailureValidationError(f"{path} cannot excerpt an unknown reason")
    if metric is None:
        return
    if not isinstance(metric, Mapping):
        raise PublicFailureValidationError(f"{path}.metric must be an object")
    _validate_metric(metric, f"{path}.metric")


def validate_public_failure(report: Mapping[str, object]) -> None:
    """Reject values that do not satisfy public-failure-v1."""
    if not isinstance(report, Mapping):
        raise PublicFailureValidationError("report must be an object")
    _reject_unknown_fields(report, REPORT_FIELDS, "report")
    _require_fields(report, REPORT_REQUIRED_FIELDS, "report")
    if report.get("schema_version") != 1 or isinstance(report.get("schema_version"), bool):
        raise PublicFailureValidationError("schema_version must be 1")
    if report.get("policy_version") != POLICY_VERSION:
        raise PublicFailureValidationError("policy_version is not supported")
    if report.get("repository") != "NVIDIA/TensorRT-Model-Connect":
        raise PublicFailureValidationError("repository is not supported")
    _require_string(
        report.get("report_id"),
        "report_id",
        max_length=100,
        pattern=REPORT_ID_PATTERN,
    )
    _require_integer(report.get("pr_number"), "pr_number", minimum=1, maximum=2**31 - 1)
    _require_integer(report.get("run_attempt"), "run_attempt", minimum=1, maximum=1000)
    _require_integer(
        report.get("omitted_failure_count"),
        "omitted_failure_count",
        minimum=0,
        maximum=2**31 - 1,
    )
    for key in ("head_sha", "base_sha", "tested_revision"):
        _require_string(report.get(key), key, max_length=40, pattern=SHA_PATTERN)
    if "dispatch_nonce" in report:
        _require_string(
            report.get("dispatch_nonce"),
            "dispatch_nonce",
            max_length=32,
            pattern=DISPATCH_NONCE_PATTERN,
        )
    _require_enum(report.get("tested_revision_kind"), {"head", "merge"}, "tested_revision_kind")
    _require_enum(report.get("result"), {"failure", "error"}, "result")
    _require_string(
        report.get("generated_at"),
        "generated_at",
        max_length=20,
        pattern=TIMESTAMP_PATTERN,
    )
    failures = report.get("failures")
    if not isinstance(failures, list) or len(failures) > 20:
        raise PublicFailureValidationError("failures must be an array")
    for index, failure in enumerate(failures):
        if not isinstance(failure, Mapping):
            raise PublicFailureValidationError(f"failures[{index}] must be an object")
        _validate_failure(failure, index)


def serialize_public_failure(report: Mapping[str, object]) -> bytes:
    """Serialize a public report deterministically for audits and hashing."""
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
