# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from tools.report_hub.app import ReportHubApplication
from tools.report_hub.auth import AuthManager, AuthenticationError, Identity
from tools.report_hub.config import Settings
from tools.report_hub.domain import ConflictError, ReportHubError
from tools.report_hub.normalize import normalize_report
from tools.report_hub.integrations import IntegrationRegistry, PublishEnvelope
from tools.report_hub.source import HttpEvidenceSource
from tools.report_hub.storage import Store


CATALOG = {
    "schema_version": "trtmc.report-browser-index/v1",
    "sources": {
        "benchmark": {
            "reports": [
                {
                    "folder": "accuracy-20260807-a1b2c3d4",
                    "date": "2026-08-07",
                    "updated_at": "2026-08-07T02:00:00Z",
                    "_signature": "abc",
                    "summary": {"total": 2, "passed": 1, "failed": 1, "other": 0},
                },
                {
                    "folder": "accuracy-20260806-d4c3b2a1",
                    "date": "2026-08-06",
                    "updated_at": "2026-08-06T02:00:00Z",
                    "_signature": "def",
                    "summary": {"total": 1, "passed": 1, "failed": 0, "other": 0},
                },
            ]
        },
        "perf": {"reports": []},
    },
}

VALIDATION_REPORT = {
    "schema_version": "trtmc.validation-report/v2",
    "results": [
        {
            "model": "qwen-test",
            "family": "qwen",
            "workload": "mmlu",
            "operation": "generate",
            "execution": {"status": "completed", "attempts": [{"error": ""}]},
            "comparison": {
                "status": "disagreement",
                "primary_metric": {"name": "agreement", "value": 0.5},
                "failures": [{"gate": "minimum", "actual": 0.5}],
            },
            "validation": {"status": "failed"},
        },
        {
            "model": "t5-test",
            "workload": "summary",
            "execution": {"status": "completed"},
            "comparison": {
                "status": "agreement",
                "primary_metric": {"name": "agreement", "value": 1.0},
            },
            "validation": {"status": "passed"},
        },
    ],
}

BENCHMARK_REPORT = {
    "schema_version": "trtmc.benchmark-report/v1",
    "runs": [
        {
            "run_id": "source-run",
            "cells": [
                {
                    "model": "gpt-test",
                    "name": "default",
                    "operation": "generate",
                    "status": "completed",
                    "metrics": {
                        "primary": {
                            "name": "runtime_e2e_wall_ms.p50",
                            "unit": "ms",
                            "value": 12.5,
                        },
                        "sample_count": 20,
                    },
                }
            ],
        }
    ],
}

PERFORMANCE_REPORT = {
    "schema_version": "trtmc.perf-matrix/v1",
    "cases": [
        {
            "id": "nemotron.generate",
            "model": "nemotron-test",
            "family": "nemotron",
            "operation": "generate",
            "status": "red",
            "workload_contract": {"testcase": "default"},
            "baseline": {"metrics": {"latency_ms": {"p50": 100.0}}},
            "candidate": {
                "metrics": {
                    "latency_ms": {"p50": 125.0},
                    "primary": {"name": "runtime_e2e_wall_ms.p50", "value": 125.0},
                }
            },
            "comparison": {
                "baseline_over_trtmc_p50": 0.8,
                "equivalence_margin_percent": 5.0,
            },
        }
    ],
}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    value = Store(tmp_path / "hub.sqlite3", retention_days=30)
    value.initialize()
    value.sync_catalog(CATALOG, actor="qa-user")
    return value


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "hub.sqlite3",
        evidence_root_url="https://reports.example.test/root",
        catalog_url="https://reports.example.test/root/catalog.json",
        auth_mode="development",
        dev_user="qa-user",
        dev_role="admin",
        secret="test-secret",
        allowed_external_hosts=("github.com", "bugs.example.test"),
    )


def _run(store: Store, index: int = 0) -> dict[str, Any]:
    return store.list_runs(source="benchmark")[index]


def test_validation_and_benchmark_reports_normalize_to_stable_findings() -> None:
    validation = normalize_report("benchmark", VALIDATION_REPORT)
    benchmark = normalize_report("perf", BENCHMARK_REPORT)

    assert [item.status for item in validation] == ["failed", "passed"]
    assert validation[0].metric_name == "agreement"
    assert validation[0].metric_value == 0.5
    assert validation[0].finding_id == normalize_report("benchmark", VALIDATION_REPORT)[0].finding_id
    assert benchmark[0].status == "passed"
    assert benchmark[0].metric_value == 12.5
    assert benchmark[0].details["sample_count"] == 20


def test_performance_matrix_preserves_comparison_contract() -> None:
    observation = normalize_report("perf", PERFORMANCE_REPORT)[0]

    assert observation.status == "failed"
    assert observation.metric_name == "baseline_over_trtmc_p50"
    assert observation.metric_value == 0.8
    assert observation.details["baseline_p50_ms"] == 100.0
    assert observation.details["candidate_p50_ms"] == 125.0


def test_unknown_report_schema_fails_closed() -> None:
    with pytest.raises(ReportHubError, match="unsupported report schema"):
        normalize_report("benchmark", {"schema_version": "unknown/v9"})


def test_finding_identity_changes_with_metric_contract() -> None:
    changed = json.loads(json.dumps(VALIDATION_REPORT))
    changed["results"][0]["comparison"]["primary_metric"]["name"] = "exact_match_rate"

    original_id = normalize_report("benchmark", VALIDATION_REPORT)[0].finding_id
    changed_id = normalize_report("benchmark", changed)[0].finding_id

    assert original_id != changed_id


def test_catalog_sync_is_idempotent_and_never_restores_trashed_runs(store: Store) -> None:
    run = _run(store)
    trashed = store.trash_run(
        run["id"],
        reason="Superseded by complete run",
        confirmation=run["folder"],
        expected_version=run["version"],
        actor="qa-user",
    )

    result = store.sync_catalog(CATALOG, actor="qa-user")

    assert result == {"inserted": 0, "updated": 2}
    assert store.get_run(run["id"])["lifecycle"] == "trashed"
    assert store.get_run(run["id"])["version"] == trashed["version"]


def test_trash_requires_exact_confirmation_and_supports_restore(store: Store) -> None:
    run = _run(store)
    with pytest.raises(ReportHubError, match="exactly match"):
        store.trash_run(
            run["id"],
            reason="Duplicate",
            confirmation="wrong-run",
            expected_version=run["version"],
            actor="qa-user",
        )

    trashed = store.trash_run(
        run["id"],
        reason="Duplicate",
        confirmation=run["folder"],
        expected_version=run["version"],
        actor="qa-user",
    )
    with pytest.raises(ConflictError, match="changed"):
        store.restore_run(run["id"], expected_version=run["version"], actor="qa-user")
    restored = store.restore_run(
        run["id"], expected_version=trashed["version"], actor="qa-user"
    )

    assert restored["lifecycle"] == "active"
    assert restored["trashed_at"] is None
    assert [event["action"] for event in store.audit_events(entity_id=run["id"])][:2] == [
        "run.restored",
        "run.trashed",
    ]


def test_restore_is_rejected_after_retention_expires(store: Store) -> None:
    run = _run(store)
    trashed = store.trash_run(
        run["id"],
        reason="Duplicate",
        confirmation=run["folder"],
        expected_version=run["version"],
        actor="qa-user",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(ConflictError, match="retention window has expired"):
        store.restore_run(
            run["id"],
            expected_version=trashed["version"],
            actor="qa-user",
            now=datetime(2026, 2, 2, tzinfo=timezone.utc),
        )


def test_purge_schedule_waits_for_retention_and_closed_findings(store: Store) -> None:
    run = _run(store)
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trashed = store.trash_run(
        run["id"],
        reason="Obsolete evidence",
        confirmation=run["folder"],
        expected_version=run["version"],
        actor="qa-user",
        now=january,
    )
    store.ingest_observations(
        run["id"], normalize_report("benchmark", VALIDATION_REPORT), actor="qa-user"
    )
    with pytest.raises(ConflictError, match="retention"):
        store.schedule_purge(
            run["id"],
            confirmation=run["folder"],
            acknowledge_irreversible=True,
            expected_version=trashed["version"],
            actor="admin-user",
            now=datetime(2026, 1, 15, tzinfo=timezone.utc),
        )
    with pytest.raises(ConflictError, match="open findings"):
        store.schedule_purge(
            run["id"],
            confirmation=run["folder"],
            acknowledge_irreversible=True,
            expected_version=trashed["version"],
            actor="admin-user",
            now=datetime(2026, 2, 2, tzinfo=timezone.utc),
        )
    for finding in store.list_findings(run_id=run["id"]):
        store.update_triage(
            finding["id"],
            {
                "status": "resolved",
                "severity": "low",
                "owner": "qa-user",
                "note": "No longer actionable",
                "tags": ["obsolete"],
                "expected_version": 0,
            },
            actor="qa-user",
        )
    scheduled = store.schedule_purge(
        run["id"],
        confirmation=run["folder"],
        acknowledge_irreversible=True,
        expected_version=trashed["version"],
        actor="admin-user",
        now=datetime(2026, 2, 2, tzinfo=timezone.utc),
    )
    assert scheduled["lifecycle"] == "purge_scheduled"


def test_triage_persists_across_multiple_run_observations(store: Store) -> None:
    first, second = store.list_runs(source="benchmark")
    observations = normalize_report("benchmark", VALIDATION_REPORT)
    store.ingest_observations(first["id"], observations, actor="qa-user")
    finding = store.list_findings(run_id=first["id"])[0]
    saved = store.update_triage(
        finding["id"],
        {
            "status": "investigating",
            "severity": "high",
            "owner": "qa-user",
            "note": "Reproducing on a clean engine",
            "tags": ["accuracy-regression"],
            "expected_version": 0,
        },
        actor="qa-user",
    )
    store.ingest_observations(second["id"], observations, actor="qa-user")

    same_finding = next(
        item for item in store.list_findings(run_id=second["id"]) if item["id"] == finding["id"]
    )
    assert same_finding["triage"] == saved


def test_auth_proxy_requires_identity_and_mutation_token(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = Settings(**{**settings.__dict__, "auth_mode": "proxy"})
    auth = AuthManager(settings)
    with pytest.raises(AuthenticationError):
        auth.authenticate({})
    identity = auth.authenticate(
        {settings.auth_user_header: "qa-user", settings.auth_role_header: "qa"}
    )
    with pytest.raises(ReportHubError, match="mutation token"):
        auth.verify_mutation(identity, {})
    auth.verify_mutation(identity, {"X-Report-Hub-CSRF": auth.csrf_token(identity)})


def test_application_sync_analyze_and_update_triage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = Store(settings.database_path)
    store.initialize()

    def loader(url: str, _maximum: int) -> dict[str, Any]:
        return CATALOG if url.endswith("catalog.json") else VALIDATION_REPORT

    application = ReportHubApplication(settings, store, HttpEvidenceSource(settings, loader=loader))
    session = _json_response(application.handle("GET", "/api/v1/session", {}))
    headers = {"X-Report-Hub-CSRF": session["csrf_token"]}
    sync = application.handle("POST", "/api/v1/catalog/sync", headers, b"{}")
    assert sync.status == 200
    run = store.list_runs(source="benchmark")[0]
    analyzed = _json_response(
        application.handle("POST", f"/api/v1/runs/{run['id']}/analyze", headers, b"{}")
    )
    finding = analyzed["findings"][0]
    triage_body = json.dumps(
        {
            "status": "investigating",
            "severity": "high",
            "owner": "qa-user",
            "note": "Reproduced",
            "tags": ["accuracy"],
            "expected_version": 0,
        }
    ).encode()
    response = application.handle(
        "PUT", f"/api/v1/findings/{finding['id']}/triage", headers, triage_body
    )
    assert response.status == 200
    assert _json_response(response)["triage"]["version"] == 1


def test_external_github_link_is_repository_scoped(tmp_path: Path, store: Store) -> None:
    settings = _settings(tmp_path)
    application = ReportHubApplication(settings, store, HttpEvidenceSource(settings))
    identity = Identity("qa-user", "qa")

    with pytest.raises(ReportHubError, match="this repository"):
        application._create_link(  # noqa: SLF001 - direct policy unit test
            {
                "system": "github",
                "record_type": "issue",
                "external_id": "42",
                "url": "https://github.com/example/other/issues/42",
            },
            identity,
            run_id=_run(store)["id"],
        )


def test_phase_zero_adapters_fail_closed_and_idempotency_is_persistent(store: Store) -> None:
    adapter = IntegrationRegistry.phase_zero().get("devtest")
    with pytest.raises(ReportHubError, match="write adapter is disabled"):
        adapter.publish(PublishEnvelope("create_plan", "plan-1", "qa-user", {"name": "Nightly"}))

    first = store.prepare_adapter_operation(
        system="devtest",
        operation="create_plan",
        idempotency_key="plan-1",
        request={"name": "Nightly"},
        actor="qa-user",
    )
    replay = store.prepare_adapter_operation(
        system="devtest",
        operation="create_plan",
        idempotency_key="plan-1",
        request={"name": "Nightly"},
        actor="qa-user",
    )
    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert first["id"] == replay["id"]
    with pytest.raises(ConflictError, match="different request"):
        store.prepare_adapter_operation(
            system="devtest",
            operation="create_plan",
            idempotency_key="plan-1",
            request={"name": "Changed"},
            actor="qa-user",
        )


def _json_response(response: Any) -> dict[str, Any]:
    return json.loads(response.body.decode("utf-8"))
