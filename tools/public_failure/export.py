# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit allowlist conversion from protected artifacts to public data."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from .policy import (
    POLICY_VERSION,
    PUBLIC_METRIC_NAMES,
    PUBLIC_METRIC_OPERATORS,
    public_backend,
    public_failure_class,
    public_gpu_type,
    public_model,
    public_reason_code,
    public_stage,
    public_subject,
)


MAX_PUBLIC_FAILURES = 20
SAFE_TEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_./:\[\],=+-]{1,300}\Z")
SAFE_EXCERPT_LINE_PATTERN = re.compile(r"[\x20-\x7e]{1,240}\Z")


@dataclass(frozen=True)
class ExportContext:
    """Trusted run identity supplied by the private CI finalizer."""

    repository: str
    pr_number: int
    head_sha: str
    base_sha: str
    tested_revision: str
    run_attempt: int
    result: str
    generated_at: str
    tested_revision_kind: str = "head"


def _export_metric(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    operator = value.get("operator")
    observed = value.get("observed")
    threshold = value.get("threshold")
    if name not in PUBLIC_METRIC_NAMES or operator not in PUBLIC_METRIC_OPERATORS:
        return None
    for number in (observed, threshold):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            return None
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }


def _export_test_id(value: object) -> str:
    if not isinstance(value, str) or SAFE_TEST_ID_PATTERN.fullmatch(value) is None:
        return "withheld"
    if value.startswith("/") or ".." in value or "\\" in value:
        return "withheld"
    return value


def _export_excerpt(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value or len(value) > 20:
        return None
    lines: list[str] = []
    for item in value:
        if not isinstance(item, str) or SAFE_EXCERPT_LINE_PATTERN.fullmatch(item) is None:
            return None
        lines.append(item)
    if sum(len(line) for line in lines) > 4000:
        return None
    return lines


def _export_one_failure(value: Mapping[str, object]) -> dict[str, Any]:
    metric = _export_metric(value.get("metric"))
    reason_code = public_reason_code(value.get("reason_code"))
    subject = public_subject(value.get("subject"))
    excerpt = _export_excerpt(value.get("excerpt"))
    public = {
        "public_stage": public_stage(value.get("stage")),
        "model": public_model(value.get("model")),
        "backend": public_backend(value.get("backend")),
        "gpu_type": public_gpu_type(value.get("gpu_type")),
        "test_id": _export_test_id(value.get("test_id")),
        "failure_class": public_failure_class(value.get("failure_type")),
        "reason_code": reason_code,
        "disclosure": (
            "truncated"
            if excerpt is not None
            else "full"
            if reason_code != "unknown"
            else "withheld"
        ),
    }
    if metric is not None:
        public["metric"] = metric
    if subject is not None:
        public["subject"] = subject
    if excerpt is not None:
        public["excerpt"] = excerpt
    return public


def export_failure(
    internal_artifacts: Mapping[str, object], context: ExportContext
) -> dict[str, Any]:
    """Return a new public object; never copy unknown internal fields."""
    raw_failures = internal_artifacts.get("failures")
    failures = raw_failures if isinstance(raw_failures, list) else []
    approved_failures = [item for item in failures if isinstance(item, Mapping)]
    visible_failures = approved_failures[:MAX_PUBLIC_FAILURES]
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "report_id": (
            f"trtmc-pr{context.pr_number}-{context.head_sha[:7]}-attempt{context.run_attempt}"
        ),
        "repository": context.repository,
        "pr_number": context.pr_number,
        "head_sha": context.head_sha,
        "base_sha": context.base_sha,
        "tested_revision": context.tested_revision,
        "tested_revision_kind": context.tested_revision_kind,
        "run_attempt": context.run_attempt,
        "result": context.result,
        "failures": [_export_one_failure(item) for item in visible_failures],
        "omitted_failure_count": len(approved_failures) - len(visible_failures),
        "generated_at": context.generated_at,
    }
