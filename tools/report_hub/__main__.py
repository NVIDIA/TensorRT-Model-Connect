# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run TRTMC Report Hub."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .app import ReportHubApplication
from .config import Settings
from .server import serve
from .source import HttpEvidenceSource
from .storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="use an explicit local development identity")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--evidence-root")
    parser.add_argument("--catalog-url")
    arguments = parser.parse_args()

    settings = Settings.from_environment(development=arguments.dev)
    overrides: dict[str, object] = {}
    if arguments.host:
        overrides["bind_host"] = arguments.host
    if arguments.port:
        overrides["bind_port"] = arguments.port
    if arguments.database:
        overrides["database_path"] = arguments.database
    if arguments.evidence_root:
        overrides["evidence_root_url"] = arguments.evidence_root.rstrip("/")
        if not arguments.catalog_url:
            overrides["catalog_url"] = (
                f"{arguments.evidence_root.rstrip('/')}/report-browser/reports-index.json"
            )
    if arguments.catalog_url:
        overrides["catalog_url"] = arguments.catalog_url
    settings = replace(settings, **overrides)
    settings.validate()
    if settings.auth_mode == "development" and settings.bind_host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("development auth may only bind to a loopback address")

    store = Store(settings.database_path, retention_days=settings.retention_days)
    store.initialize()
    application = ReportHubApplication(settings, store, HttpEvidenceSource(settings))
    serve(application, settings.bind_host, settings.bind_port)


if __name__ == "__main__":
    main()
