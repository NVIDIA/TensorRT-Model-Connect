# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite persistence for Report Hub state and audit history."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from .domain import (
    EXTERNAL_SYSTEMS,
    SEVERITIES,
    TRIAGE_STATUSES,
    ConflictError,
    NotFoundError,
    Observation,
    ReportHubError,
    isoformat,
    require_folder,
    require_string_list,
    require_text,
    stable_id,
    utc_now,
)


SCHEMA_VERSION = 1


class Store:
    def __init__(self, path: Path, *, retention_days: int = 30):
        self.path = Path(path)
        self.retention_days = retention_days

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current} is newer than supported schema {SCHEMA_VERSION}"
                )
            if current == 0:
                connection.executescript(_SCHEMA)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def sync_catalog(self, catalog: Mapping[str, Any], *, actor: str) -> dict[str, int]:
        sources = catalog.get("sources")
        if not isinstance(sources, Mapping):
            raise ReportHubError("report catalog sources must be an object")
        now = isoformat(utc_now())
        inserted = 0
        updated = 0
        with self._transaction() as connection:
            for source in ("benchmark", "perf"):
                group = sources.get(source, {})
                reports = group.get("reports", []) if isinstance(group, Mapping) else []
                if not isinstance(reports, list):
                    raise ReportHubError(f"catalog source {source} reports must be a list")
                for raw in reports:
                    if not isinstance(raw, Mapping):
                        continue
                    folder = require_folder(raw.get("folder"))
                    run_id = stable_id("run", source, folder)
                    existing = connection.execute(
                        "SELECT id FROM runs WHERE id = ?", (run_id,)
                    ).fetchone()
                    summary = raw.get("summary") if isinstance(raw.get("summary"), Mapping) else {}
                    connection.execute(
                        """
                        INSERT INTO runs (
                            id, source, folder, report_date, catalog_updated_at,
                            signature, summary_json, lifecycle, version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            report_date = excluded.report_date,
                            catalog_updated_at = excluded.catalog_updated_at,
                            signature = excluded.signature,
                            summary_json = excluded.summary_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            run_id,
                            source,
                            folder,
                            raw.get("date"),
                            raw.get("updated_at"),
                            raw.get("_signature"),
                            _json(summary),
                            now,
                            now,
                        ),
                    )
                    if existing:
                        updated += 1
                    else:
                        inserted += 1
            self._audit(
                connection,
                entity_type="catalog",
                entity_id="report-catalog",
                action="catalog.synced",
                actor=actor,
                before=None,
                after={"inserted": inserted, "updated": updated},
                at=now,
            )
        return {"inserted": inserted, "updated": updated}

    def list_runs(self, *, source: str | None = None, lifecycle: str = "active") -> list[dict[str, Any]]:
        clauses = ["lifecycle = ?"]
        parameters: list[Any] = [lifecycle]
        if source:
            if source not in {"benchmark", "perf"}:
                raise ReportHubError("unsupported report source")
            clauses.append("source = ?")
            parameters.append(source)
        sql = f"SELECT * FROM runs WHERE {' AND '.join(clauses)} ORDER BY report_date DESC, folder DESC"
        with self._connect() as connection:
            return [self._run_dict(row) for row in connection.execute(sql, parameters)]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("report run was not found")
        return self._run_dict(row)

    def trash_run(
        self,
        run_id: str,
        *,
        reason: str,
        confirmation: str,
        expected_version: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        reason = require_text(reason, "reason", maximum=500)
        current_time = now or utc_now()
        at = isoformat(current_time)
        purge_after = isoformat(current_time + timedelta(days=self.retention_days))
        with self._transaction() as connection:
            before_row = self._locked_run(connection, run_id)
            before = self._run_dict(before_row)
            if before["lifecycle"] != "active":
                raise ConflictError("only active reports can be moved to Trash")
            if confirmation != before["folder"]:
                raise ReportHubError("confirmation must exactly match the report folder")
            if expected_version != before["version"]:
                raise ConflictError("report changed; reload before deleting")
            connection.execute(
                """
                UPDATE runs SET lifecycle = 'trashed', trashed_at = ?, purge_after = ?,
                    purge_reason = ?, version = version + 1, updated_at = ?
                WHERE id = ?
                """,
                (at, purge_after, reason, at, run_id),
            )
            after = self._run_dict(self._locked_run(connection, run_id))
            self._audit(connection, "run", run_id, "run.trashed", actor, before, after, at)
        return after

    def restore_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = now or utc_now()
        at = isoformat(current_time)
        with self._transaction() as connection:
            before = self._run_dict(self._locked_run(connection, run_id))
            if before["lifecycle"] != "trashed":
                raise ConflictError("only reports in Trash can be restored")
            if expected_version != before["version"]:
                raise ConflictError("report changed; reload before restoring")
            deadline = _parse_datetime(before["purge_after"])
            if deadline is not None and current_time > deadline:
                raise ConflictError("retention window has expired; report can no longer be restored")
            connection.execute(
                """
                UPDATE runs SET lifecycle = 'active', trashed_at = NULL, purge_after = NULL,
                    purge_reason = NULL, version = version + 1, updated_at = ? WHERE id = ?
                """,
                (at, run_id),
            )
            after = self._run_dict(self._locked_run(connection, run_id))
            self._audit(connection, "run", run_id, "run.restored", actor, before, after, at)
        return after

    def schedule_purge(
        self,
        run_id: str,
        *,
        confirmation: str,
        acknowledge_irreversible: bool,
        expected_version: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = now or utc_now()
        at = isoformat(current_time)
        with self._transaction() as connection:
            before = self._run_dict(self._locked_run(connection, run_id))
            if before["lifecycle"] != "trashed":
                raise ConflictError("only reports in Trash can be scheduled for purge")
            if confirmation != before["folder"]:
                raise ReportHubError("confirmation must exactly match the report folder")
            if not acknowledge_irreversible:
                raise ReportHubError("irreversibility acknowledgement is required")
            if expected_version != before["version"]:
                raise ConflictError("report changed; reload before scheduling purge")
            deadline = _parse_datetime(before["purge_after"])
            if deadline is None or current_time < deadline:
                raise ConflictError("retention window has not expired")
            references = connection.execute(
                "SELECT COUNT(*) FROM external_links WHERE run_id = ? OR finding_id IN "
                "(SELECT finding_id FROM observations WHERE run_id = ?)",
                (run_id, run_id),
            ).fetchone()[0]
            open_findings = connection.execute(
                """
                SELECT COUNT(*) FROM observations o
                LEFT JOIN triage t ON t.finding_id = o.finding_id
                WHERE o.run_id = ? AND (
                    (o.status IN ('failed', 'error')
                        AND COALESCE(t.status, 'new') NOT IN ('resolved', 'accepted_risk'))
                    OR (t.status IS NOT NULL AND t.status NOT IN ('resolved', 'accepted_risk'))
                )
                """,
                (run_id,),
            ).fetchone()[0]
            if references or open_findings:
                raise ConflictError(
                    f"purge blocked by {references} external references and {open_findings} open findings"
                )
            connection.execute(
                "UPDATE runs SET lifecycle = 'purge_scheduled', version = version + 1, updated_at = ? WHERE id = ?",
                (at, run_id),
            )
            after = self._run_dict(self._locked_run(connection, run_id))
            self._audit(connection, "run", run_id, "run.purge_scheduled", actor, before, after, at)
        return after

    def ingest_observations(
        self,
        run_id: str,
        observations: Iterable[Observation],
        *,
        actor: str,
    ) -> int:
        run = self.get_run(run_id)
        if run["lifecycle"] == "purged":
            raise ConflictError("purged report evidence cannot be analyzed")
        now = isoformat(utc_now())
        count = 0
        with self._transaction() as connection:
            for observation in observations:
                connection.execute(
                    """
                    INSERT INTO findings (
                        id, source, model, workload, metric_contract, family, title,
                        created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        family = excluded.family, title = excluded.title, last_seen_at = excluded.last_seen_at
                    """,
                    (
                        observation.finding_id,
                        run["source"],
                        observation.model,
                        observation.workload,
                        observation.metric_name,
                        observation.family,
                        f"{observation.model} · {observation.workload}",
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO observations (
                        run_id, finding_id, operation, status, metric_name, metric_value,
                        details_json, observed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, finding_id) DO UPDATE SET
                        operation = excluded.operation, status = excluded.status,
                        metric_name = excluded.metric_name, metric_value = excluded.metric_value,
                        details_json = excluded.details_json, observed_at = excluded.observed_at
                    """,
                    (
                        run_id,
                        observation.finding_id,
                        observation.operation,
                        observation.status,
                        observation.metric_name,
                        observation.metric_value,
                        _json(observation.details),
                        now,
                    ),
                )
                count += 1
            self._audit(
                connection,
                "run",
                run_id,
                "run.analyzed",
                actor,
                None,
                {"observations": count},
                now,
            )
        return count

    def list_findings(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        run_join = ""
        if run_id:
            run_join = "JOIN observations selected ON selected.finding_id = f.id AND selected.run_id = ?"
            parameters.append(run_id)
        else:
            run_join = """
                LEFT JOIN observations selected ON selected.rowid = (
                    SELECT latest.rowid FROM observations latest
                    WHERE latest.finding_id = f.id
                    ORDER BY latest.observed_at DESC, latest.run_id DESC LIMIT 1
                )
            """
        sql = f"""
            SELECT f.*,
                COALESCE(t.status, 'new') AS triage_status,
                COALESCE(t.severity, 'unassessed') AS severity,
                COALESCE(t.owner, 'Unassigned') AS owner,
                COALESCE(t.note, '') AS note,
                COALESCE(t.tags_json, '[]') AS tags_json,
                COALESCE(t.version, 0) AS triage_version,
                t.updated_by AS triage_updated_by,
                t.updated_at AS triage_updated_at,
                selected.status AS observation_status,
                selected.operation AS operation,
                selected.metric_name AS metric_name,
                selected.metric_value AS metric_value,
                selected.details_json AS details_json
            FROM findings f
            {run_join}
            LEFT JOIN triage t ON t.finding_id = f.id
            ORDER BY f.last_seen_at DESC, f.title
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._finding_dict(row) for row in rows]

    def get_finding(self, finding_id: str) -> dict[str, Any]:
        matches = [item for item in self.list_findings() if item["id"] == finding_id]
        if not matches:
            raise NotFoundError("finding was not found")
        return matches[0]

    def update_triage(
        self,
        finding_id: str,
        payload: Mapping[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        status = require_text(payload.get("status"), "status", maximum=32)
        severity = require_text(payload.get("severity"), "severity", maximum=32)
        owner = require_text(payload.get("owner", "Unassigned"), "owner", maximum=160)
        note = require_text(payload.get("note", ""), "note", maximum=8000, allow_empty=True)
        tags = require_string_list(payload.get("tags", []), "tags")
        expected_version = payload.get("expected_version")
        if not isinstance(expected_version, int):
            raise ReportHubError("expected_version must be an integer")
        if status not in TRIAGE_STATUSES:
            raise ReportHubError("unsupported triage status")
        if severity not in SEVERITIES:
            raise ReportHubError("unsupported severity")
        now = isoformat(utc_now())
        with self._transaction() as connection:
            finding = connection.execute("SELECT id FROM findings WHERE id = ?", (finding_id,)).fetchone()
            if finding is None:
                raise NotFoundError("finding was not found")
            row = connection.execute("SELECT * FROM triage WHERE finding_id = ?", (finding_id,)).fetchone()
            before = self._triage_dict(row) if row else None
            current_version = int(row["version"]) if row else 0
            if expected_version != current_version:
                raise ConflictError("triage changed; reload before saving")
            new_version = current_version + 1
            connection.execute(
                """
                INSERT INTO triage (
                    finding_id, status, severity, owner, note, tags_json, version, updated_by, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    status = excluded.status, severity = excluded.severity,
                    owner = excluded.owner, note = excluded.note, tags_json = excluded.tags_json,
                    version = excluded.version, updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (finding_id, status, severity, owner, note, _json(tags), new_version, actor, now),
            )
            after = self._triage_dict(
                connection.execute("SELECT * FROM triage WHERE finding_id = ?", (finding_id,)).fetchone()
            )
            self._audit(connection, "finding", finding_id, "triage.updated", actor, before, after, now)
        return after

    def add_external_link(
        self,
        *,
        finding_id: str | None,
        run_id: str | None,
        system: str,
        record_type: str,
        external_id: str,
        url: str,
        actor: str,
    ) -> dict[str, Any]:
        if bool(finding_id) == bool(run_id):
            raise ReportHubError("link must target exactly one finding or run")
        if system not in EXTERNAL_SYSTEMS:
            raise ReportHubError("unsupported external system")
        record_type = require_text(record_type, "record_type", maximum=64)
        external_id = require_text(external_id, "external_id", maximum=160)
        url = require_text(url, "url", maximum=2000, allow_empty=True)
        link_id = stable_id("link", system, record_type, external_id, finding_id or "", run_id or "")
        now = isoformat(utc_now())
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO external_links (
                        id, finding_id, run_id, system, record_type, external_id, url,
                        sync_status, created_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', ?, ?, ?)
                    """,
                    (link_id, finding_id, run_id, system, record_type, external_id, url, actor, now, now),
                )
            except sqlite3.IntegrityError as error:
                raise ConflictError("external link already exists or target is missing") from error
            link = self._link_dict(
                connection.execute("SELECT * FROM external_links WHERE id = ?", (link_id,)).fetchone()
            )
            self._audit(
                connection,
                "finding" if finding_id else "run",
                finding_id or run_id or "",
                "external_link.added",
                actor,
                None,
                link,
                now,
            )
        return link

    def list_external_links(
        self, *, finding_id: str | None = None, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if bool(finding_id) == bool(run_id):
            raise ReportHubError("provide exactly one finding or run")
        field, value = ("finding_id", finding_id) if finding_id else ("run_id", run_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM external_links WHERE {field} = ? ORDER BY created_at DESC", (value,)
            ).fetchall()
        return [self._link_dict(row) for row in rows]

    def get_draft(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        table, field = _draft_table(kind)
        with self._connect() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE {field} = ?", (entity_id,)).fetchone()
        return self._draft_dict(row, field) if row else None

    def save_draft(
        self,
        kind: str,
        entity_id: str,
        data: Mapping[str, Any],
        *,
        expected_version: int,
        actor: str,
    ) -> dict[str, Any]:
        table, field = _draft_table(kind)
        if len(_json(data)) > 64 * 1024:
            raise ReportHubError("draft exceeds 64 KiB")
        now = isoformat(utc_now())
        with self._transaction() as connection:
            parent_table = "runs" if kind == "test_plan" else "findings"
            if connection.execute(f"SELECT id FROM {parent_table} WHERE id = ?", (entity_id,)).fetchone() is None:
                raise NotFoundError(f"{parent_table[:-1]} was not found")
            row = connection.execute(f"SELECT * FROM {table} WHERE {field} = ?", (entity_id,)).fetchone()
            before = self._draft_dict(row, field) if row else None
            current_version = int(row["version"]) if row else 0
            if expected_version != current_version:
                raise ConflictError("draft changed; reload before saving")
            new_version = current_version + 1
            connection.execute(
                f"""
                INSERT INTO {table} ({field}, data_json, version, updated_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT({field}) DO UPDATE SET data_json = excluded.data_json,
                    version = excluded.version, updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (entity_id, _json(data), new_version, actor, now),
            )
            after = self._draft_dict(
                connection.execute(f"SELECT * FROM {table} WHERE {field} = ?", (entity_id,)).fetchone(),
                field,
            )
            self._audit(
                connection,
                "run" if kind == "test_plan" else "finding",
                entity_id,
                f"{kind}.saved",
                actor,
                before,
                after,
                now,
            )
        return after

    def audit_events(
        self, *, entity_type: str | None = None, entity_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if entity_type:
            clauses.append("entity_type = ?")
            parameters.append(entity_type)
        if entity_id:
            clauses.append("entity_id = ?")
            parameters.append(entity_id)
        parameters.append(max(1, min(limit, 500)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM audit_events {where} ORDER BY sequence DESC LIMIT ?", parameters
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "action": row["action"],
                "actor": row["actor"],
                "at": row["at"],
                "before": _loads(row["before_json"]),
                "after": _loads(row["after_json"]),
            }
            for row in rows
        ]

    def prepare_adapter_operation(
        self,
        *,
        system: str,
        operation: str,
        idempotency_key: str,
        request: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        if system not in EXTERNAL_SYSTEMS:
            raise ReportHubError("unsupported external system")
        operation = require_text(operation, "operation", maximum=80)
        idempotency_key = require_text(idempotency_key, "idempotency_key", maximum=160)
        request_json = _json(request) or "{}"
        if len(request_json) > 64 * 1024:
            raise ReportHubError("adapter request exceeds 64 KiB")
        digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        operation_id = stable_id("operation", system, idempotency_key)
        now = isoformat(utc_now())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM adapter_operations WHERE system = ? AND idempotency_key = ?",
                (system, idempotency_key),
            ).fetchone()
            if row:
                if row["request_sha256"] != digest or row["operation"] != operation:
                    raise ConflictError("idempotency key was already used for a different request")
                result = self._adapter_operation_dict(row)
                result["replayed"] = True
                return result
            connection.execute(
                """
                INSERT INTO adapter_operations (
                    id, system, operation, idempotency_key, request_sha256, request_json,
                    status, actor, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?)
                """,
                (
                    operation_id,
                    system,
                    operation,
                    idempotency_key,
                    digest,
                    request_json,
                    actor,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM adapter_operations WHERE id = ?", (operation_id,)
            ).fetchone()
            result = self._adapter_operation_dict(row)
            result["replayed"] = False
            self._audit(
                connection,
                "adapter_operation",
                operation_id,
                "adapter_operation.prepared",
                actor,
                None,
                result,
                now,
            )
            return result

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _transaction(self) -> _Transaction:
        return _Transaction(self._connect())

    @staticmethod
    def _locked_run(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("report run was not found")
        return row

    @staticmethod
    def _run_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source": row["source"],
            "folder": row["folder"],
            "date": row["report_date"],
            "catalog_updated_at": row["catalog_updated_at"],
            "signature": row["signature"],
            "summary": _loads(row["summary_json"]) or {},
            "lifecycle": row["lifecycle"],
            "trashed_at": row["trashed_at"],
            "purge_after": row["purge_after"],
            "purge_reason": row["purge_reason"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _finding_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source": row["source"],
            "model": row["model"],
            "workload": row["workload"],
            "metric_contract": row["metric_contract"],
            "family": row["family"],
            "title": row["title"],
            "last_seen_at": row["last_seen_at"],
            "observation": {
                "status": row["observation_status"],
                "operation": row["operation"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "details": _loads(row["details_json"]) or {},
            },
            "triage": {
                "status": row["triage_status"],
                "severity": row["severity"],
                "owner": row["owner"],
                "note": row["note"],
                "tags": _loads(row["tags_json"]) or [],
                "version": row["triage_version"],
                "updated_by": row["triage_updated_by"],
                "updated_at": row["triage_updated_at"],
            },
        }

    @staticmethod
    def _triage_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "status": row["status"],
            "severity": row["severity"],
            "owner": row["owner"],
            "note": row["note"],
            "tags": _loads(row["tags_json"]) or [],
            "version": row["version"],
            "updated_by": row["updated_by"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _link_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "finding_id": row["finding_id"],
            "run_id": row["run_id"],
            "system": row["system"],
            "record_type": row["record_type"],
            "external_id": row["external_id"],
            "url": row["url"],
            "sync_status": row["sync_status"],
            "snapshot": _loads(row["snapshot_json"]),
            "last_synced_at": row["last_synced_at"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _draft_dict(row: sqlite3.Row, field: str) -> dict[str, Any]:
        return {
            "entity_id": row[field],
            "data": _loads(row["data_json"]) or {},
            "version": row["version"],
            "updated_by": row["updated_by"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _adapter_operation_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "system": row["system"],
            "operation": row["operation"],
            "idempotency_key": row["idempotency_key"],
            "request_sha256": row["request_sha256"],
            "status": row["status"],
            "response": _loads(row["response_json"]),
            "actor": row["actor"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
        at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (
                entity_type, entity_id, action, actor, at, before_json, after_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_type, entity_id, action, actor, at, _json(before), _json(after)),
        )


class _Transaction:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _draft_table(kind: str) -> tuple[str, str]:
    if kind == "test_plan":
        return "test_plan_drafts", "run_id"
    if kind == "defect":
        return "defect_drafts", "finding_id"
    raise ReportHubError("unsupported draft kind")


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL CHECK (source IN ('benchmark', 'perf')),
    folder TEXT NOT NULL,
    report_date TEXT,
    catalog_updated_at TEXT,
    signature TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}',
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('active', 'trashed', 'purge_scheduled', 'purged')),
    trashed_at TEXT,
    purge_after TEXT,
    purge_reason TEXT,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (source, folder)
);

CREATE TABLE findings (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    model TEXT NOT NULL,
    workload TEXT NOT NULL,
    metric_contract TEXT NOT NULL,
    family TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (source, model, workload, metric_contract)
);

CREATE TABLE observations (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
    finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE RESTRICT,
    operation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('passed', 'failed', 'error', 'other')),
    metric_name TEXT NOT NULL,
    metric_value REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id)
);

CREATE TABLE triage (
    finding_id TEXT PRIMARY KEY REFERENCES findings(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('new', 'investigating', 'linked', 'monitoring', 'resolved', 'accepted_risk')),
    severity TEXT NOT NULL CHECK (severity IN ('unassessed', 'blocker', 'high', 'medium', 'low')),
    owner TEXT NOT NULL,
    note TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_by TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE external_links (
    id TEXT PRIMARY KEY,
    finding_id TEXT REFERENCES findings(id) ON DELETE RESTRICT,
    run_id TEXT REFERENCES runs(id) ON DELETE RESTRICT,
    system TEXT NOT NULL CHECK (system IN ('github', 'devtest', 'nvbug')),
    record_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    sync_status TEXT NOT NULL,
    snapshot_json TEXT,
    last_synced_at TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK ((finding_id IS NOT NULL) != (run_id IS NOT NULL)),
    UNIQUE (system, record_type, external_id, finding_id, run_id)
);

CREATE TABLE test_plan_drafts (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE RESTRICT,
    data_json TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_by TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE defect_drafts (
    finding_id TEXT PRIMARY KEY REFERENCES findings(id) ON DELETE RESTRICT,
    data_json TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_by TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    at TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT
);

CREATE TABLE adapter_operations (
    id TEXT PRIMARY KEY,
    system TEXT NOT NULL CHECK (system IN ('github', 'devtest', 'nvbug')),
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    request_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('prepared', 'publishing', 'succeeded', 'failed')),
    response_json TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (system, idempotency_key)
);

CREATE INDEX observations_finding_idx ON observations(finding_id, observed_at DESC);
CREATE INDEX runs_lifecycle_idx ON runs(lifecycle, source, report_date DESC);
CREATE INDEX audit_entity_idx ON audit_events(entity_type, entity_id, sequence DESC);
"""
