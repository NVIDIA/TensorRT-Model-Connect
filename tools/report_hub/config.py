# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed Report Hub configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    database_path: Path
    evidence_root_url: str = ""
    catalog_url: str = ""
    bind_host: str = "127.0.0.1"
    bind_port: int = 4180
    auth_mode: str = "proxy"
    dev_user: str = ""
    dev_role: str = "admin"
    auth_user_header: str = "X-Forwarded-User"
    auth_role_header: str = "X-Report-Hub-Role"
    secret: str = ""
    retention_days: int = 30
    max_report_bytes: int = 8 * 1024 * 1024
    request_timeout_seconds: float = 20.0
    allowed_external_hosts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_environment(cls, *, development: bool = False) -> "Settings":
        root = os.environ.get("REPORT_HUB_EVIDENCE_ROOT_URL", "").rstrip("/")
        catalog = os.environ.get("REPORT_HUB_CATALOG_URL", "")
        if root and not catalog:
            catalog = f"{root}/report-browser/reports-index.json"
        auth_mode = "development" if development else os.environ.get("REPORT_HUB_AUTH_MODE", "proxy")
        secret = os.environ.get("REPORT_HUB_SECRET", "")
        if auth_mode == "development" and not secret:
            secret = secrets.token_urlsafe(32)
        return cls(
            database_path=Path(os.environ.get("REPORT_HUB_DATABASE", "report-hub.sqlite3")),
            evidence_root_url=root,
            catalog_url=catalog,
            bind_host=os.environ.get("REPORT_HUB_HOST", "127.0.0.1"),
            bind_port=int(os.environ.get("REPORT_HUB_PORT", "4180")),
            auth_mode=auth_mode,
            dev_user=os.environ.get("REPORT_HUB_DEV_USER", "local-qa" if development else ""),
            dev_role=os.environ.get("REPORT_HUB_DEV_ROLE", "admin"),
            auth_user_header=os.environ.get("REPORT_HUB_USER_HEADER", "X-Forwarded-User"),
            auth_role_header=os.environ.get("REPORT_HUB_ROLE_HEADER", "X-Report-Hub-Role"),
            secret=secret,
            retention_days=int(os.environ.get("REPORT_HUB_RETENTION_DAYS", "30")),
            max_report_bytes=int(os.environ.get("REPORT_HUB_MAX_REPORT_BYTES", str(8 * 1024 * 1024))),
            request_timeout_seconds=float(os.environ.get("REPORT_HUB_REQUEST_TIMEOUT", "20")),
            allowed_external_hosts=tuple(
                item.strip().lower()
                for item in os.environ.get("REPORT_HUB_EXTERNAL_HOSTS", "github.com").split(",")
                if item.strip()
            ),
        )

    def validate(self) -> None:
        if self.auth_mode not in {"proxy", "development"}:
            raise ValueError("REPORT_HUB_AUTH_MODE must be proxy or development")
        if self.auth_mode == "proxy" and not self.secret:
            raise ValueError("REPORT_HUB_SECRET is required in proxy mode")
        if self.auth_mode == "development" and not self.dev_user:
            raise ValueError("development mode requires a development user")
        if not 1 <= self.retention_days <= 3650:
            raise ValueError("retention days must be between 1 and 3650")
        if self.max_report_bytes < 1024:
            raise ValueError("maximum report size must be at least 1024 bytes")
        for name, value in (("evidence root", self.evidence_root_url), ("catalog", self.catalog_url)):
            if not value:
                continue
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} URL must use HTTP or HTTPS")
