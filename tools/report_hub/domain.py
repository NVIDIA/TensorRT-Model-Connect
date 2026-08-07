# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain types and validation shared by Report Hub adapters and storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any


SAFE_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
RUN_LIFECYCLES = {"active", "trashed", "purge_scheduled", "purged"}
TRIAGE_STATUSES = {
    "new",
    "investigating",
    "linked",
    "monitoring",
    "resolved",
    "accepted_risk",
}
SEVERITIES = {"unassessed", "blocker", "high", "medium", "low"}
EXTERNAL_SYSTEMS = {"github", "devtest", "nvbug"}


class ReportHubError(Exception):
    """Base class for errors safe to translate into an API response."""

    status = 400
    code = "invalid_request"


class NotFoundError(ReportHubError):
    status = 404
    code = "not_found"


class ConflictError(ReportHubError):
    status = 409
    code = "conflict"


class PermissionDeniedError(ReportHubError):
    status = 403
    code = "permission_denied"


class UpstreamError(ReportHubError):
    status = 502
    code = "upstream_error"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: str) -> str:
    canonical = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def require_folder(value: Any) -> str:
    if not isinstance(value, str) or not SAFE_FOLDER.fullmatch(value):
        raise ReportHubError("report folder contains unsupported characters")
    return value


def require_text(value: Any, field: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ReportHubError(f"{field} must be text")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ReportHubError(f"{field} is required")
    if len(cleaned) > maximum:
        raise ReportHubError(f"{field} exceeds {maximum} characters")
    return cleaned


def require_string_list(value: Any, field: str, *, maximum_items: int = 20) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ReportHubError(f"{field} must contain at most {maximum_items} values")
    result: list[str] = []
    for item in value:
        cleaned = require_text(item, field, maximum=64)
        if cleaned not in result:
            result.append(cleaned)
    return result


@dataclass(frozen=True)
class Observation:
    finding_id: str
    model: str
    workload: str
    family: str
    operation: str
    status: str
    metric_name: str
    metric_value: float | None
    details: dict[str, Any]
