# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One safe entry point for building local public failure artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contract import serialize_public_failure, validate_public_failure
from .export import ExportContext, export_failure
from .render import render_failure_report
from .safety import assert_public_payload_safe


@dataclass(frozen=True)
class PublicFailureArtifacts:
    """Validated report data and deterministic serialized representations."""

    report: dict[str, Any]
    json_bytes: bytes
    log_bytes: bytes


def build_failure_artifacts(
    internal_artifacts: Mapping[str, object], context: ExportContext
) -> PublicFailureArtifacts:
    """Export, validate, render, and poison-scan one protected CI failure."""
    report = export_failure(internal_artifacts, context)
    validate_public_failure(report)
    json_bytes = serialize_public_failure(report)
    log_bytes = render_failure_report(report)
    assert_public_payload_safe(report, log_bytes)
    return PublicFailureArtifacts(
        report=report,
        json_bytes=json_bytes,
        log_bytes=log_bytes,
    )
