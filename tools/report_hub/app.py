# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free HTTP application for Report Hub."""

from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from .auth import AuthManager, Identity
from .config import Settings
from .domain import ReportHubError
from .integrations import IntegrationRegistry
from .normalize import normalize_report
from .source import HttpEvidenceSource
from .storage import Store


_ENTITY_ID = re.compile(r"^(?:run|finding)_[0-9a-f]{24}$")
_GITHUB_PATH = re.compile(r"^/NVIDIA/TensorRT-Model-Connect/(?:issues|pull)/([1-9][0-9]*)/?$")


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: tuple[tuple[str, str], ...] = ()

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> "Response":
        return cls(
            status=status,
            body=(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )


class ReportHubApplication:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        evidence: HttpEvidenceSource,
        *,
        static_root: Path | None = None,
        integrations: IntegrationRegistry | None = None,
    ):
        self.settings = settings
        self.store = store
        self.evidence = evidence
        self.auth = AuthManager(settings)
        self.static_root = static_root or Path(__file__).with_name("static")
        self.integrations = integrations or IntegrationRegistry.phase_zero()

    def handle(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        body: bytes = b"",
    ) -> Response:
        parsed = urlsplit(target)
        path = parsed.path.rstrip("/") or "/"
        if not path.startswith("/api/"):
            return self._static(path)
        try:
            if path == "/api/v1/health" and method == "GET":
                return Response.json({"ok": True, "schema_version": "trtmc.report-hub-health/v1"})
            identity = self.auth.authenticate(headers)
            if method in {"POST", "PUT", "PATCH", "DELETE"}:
                self.auth.verify_mutation(identity, headers)
            payload = self._payload(body) if body else {}
            query = parse_qs(parsed.query)
            return self._api(method, path, query, payload, identity)
        except ReportHubError as error:
            return Response.json(
                {"error": {"code": error.code, "message": str(error)}}, status=error.status
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response.json(
                {"error": {"code": "invalid_json", "message": "request body must be UTF-8 JSON"}},
                status=400,
            )

    def _api(
        self,
        method: str,
        path: str,
        query: Mapping[str, list[str]],
        payload: Mapping[str, Any],
        identity: Identity,
    ) -> Response:
        if path == "/api/v1/session" and method == "GET":
            return Response.json(
                {
                    "user": identity.user,
                    "role": identity.role,
                    "csrf_token": self.auth.csrf_token(identity),
                    "auth_mode": self.settings.auth_mode,
                }
            )
        if path == "/api/v1/integrations" and method == "GET":
            return Response.json({"integrations": self.integrations.status()})
        if path == "/api/v1/catalog/sync" and method == "POST":
            self.auth.require_role(identity, "qa")
            result = self.store.sync_catalog(self.evidence.fetch_catalog(), actor=identity.user)
            return Response.json({"result": result}, status=200)
        if path == "/api/v1/runs" and method == "GET":
            source = _query_one(query, "source") or None
            lifecycle = _query_one(query, "lifecycle") or "active"
            return Response.json({"runs": self.store.list_runs(source=source, lifecycle=lifecycle)})
        if path == "/api/v1/findings" and method == "GET":
            run_id = _query_one(query, "run_id") or None
            return Response.json({"findings": self.store.list_findings(run_id=run_id)})
        if path == "/api/v1/audit" and method == "GET":
            events = self.store.audit_events(
                entity_type=_query_one(query, "entity_type") or None,
                entity_id=_query_one(query, "entity_id") or None,
                limit=_query_int(query, "limit", 100),
            )
            return Response.json({"events": events})

        run_match = re.fullmatch(r"/api/v1/runs/([^/]+)(?:/(analyze|trash|restore|purge-schedule|test-plan|links))?", path)
        if run_match:
            run_id = _entity_id(run_match.group(1), "run")
            action = run_match.group(2)
            return self._run_api(method, run_id, action, payload, identity)

        finding_match = re.fullmatch(
            r"/api/v1/findings/([^/]+)(?:/(triage|defect-draft|links))?", path
        )
        if finding_match:
            finding_id = _entity_id(finding_match.group(1), "finding")
            action = finding_match.group(2)
            return self._finding_api(method, finding_id, action, payload, identity)

        return Response.json(
            {"error": {"code": "not_found", "message": "API route was not found"}}, status=404
        )

    def _run_api(
        self,
        method: str,
        run_id: str,
        action: str | None,
        payload: Mapping[str, Any],
        identity: Identity,
    ) -> Response:
        if action is None and method == "GET":
            run = self.store.get_run(run_id)
            result = dict(run)
            if run["lifecycle"] != "purged" and self.settings.evidence_root_url:
                result["evidence"] = {
                    "html_url": self.evidence.evidence_url(run["source"], run["folder"], "report.html")
                }
            return Response.json({"run": result})
        if action == "analyze" and method == "POST":
            self.auth.require_role(identity, "qa")
            run = self.store.get_run(run_id)
            document = self.evidence.fetch_report(run["source"], run["folder"])
            count = self.store.ingest_observations(
                run_id, normalize_report(run["source"], document.payload), actor=identity.user
            )
            return Response.json(
                {
                    "observations": count,
                    "findings": self.store.list_findings(run_id=run_id),
                    "evidence": {"json_url": document.json_url, "html_url": document.html_url},
                }
            )
        if action == "trash" and method == "POST":
            self.auth.require_role(identity, "qa")
            run = self.store.trash_run(
                run_id,
                reason=payload.get("reason"),
                confirmation=payload.get("confirmation"),
                expected_version=_expected_version(payload),
                actor=identity.user,
            )
            return Response.json({"run": run})
        if action == "restore" and method == "POST":
            self.auth.require_role(identity, "qa")
            run = self.store.restore_run(
                run_id, expected_version=_expected_version(payload), actor=identity.user
            )
            return Response.json({"run": run})
        if action == "purge-schedule" and method == "POST":
            self.auth.require_role(identity, "admin")
            run = self.store.schedule_purge(
                run_id,
                confirmation=payload.get("confirmation"),
                acknowledge_irreversible=payload.get("acknowledge_irreversible") is True,
                expected_version=_expected_version(payload),
                actor=identity.user,
            )
            return Response.json({"run": run})
        if action == "test-plan":
            if method == "GET":
                return Response.json({"draft": self.store.get_draft("test_plan", run_id)})
            if method == "PUT":
                self.auth.require_role(identity, "qa")
                data = payload.get("data")
                if not isinstance(data, Mapping):
                    raise ReportHubError("draft data must be an object")
                draft = self.store.save_draft(
                    "test_plan",
                    run_id,
                    data,
                    expected_version=_expected_version(payload),
                    actor=identity.user,
                )
                return Response.json({"draft": draft})
        if action == "links":
            if method == "GET":
                return Response.json({"links": self.store.list_external_links(run_id=run_id)})
            if method == "POST":
                self.auth.require_role(identity, "qa")
                link = self._create_link(payload, identity, run_id=run_id)
                return Response.json({"link": link}, status=201)
        return Response.json(
            {"error": {"code": "method_not_allowed", "message": "method is not allowed"}},
            status=405,
        )

    def _finding_api(
        self,
        method: str,
        finding_id: str,
        action: str | None,
        payload: Mapping[str, Any],
        identity: Identity,
    ) -> Response:
        if action is None and method == "GET":
            return Response.json({"finding": self.store.get_finding(finding_id)})
        if action == "triage" and method == "PUT":
            self.auth.require_role(identity, "qa")
            return Response.json(
                {"triage": self.store.update_triage(finding_id, payload, actor=identity.user)}
            )
        if action == "defect-draft":
            if method == "GET":
                return Response.json({"draft": self.store.get_draft("defect", finding_id)})
            if method == "PUT":
                self.auth.require_role(identity, "qa")
                data = payload.get("data")
                if not isinstance(data, Mapping):
                    raise ReportHubError("draft data must be an object")
                draft = self.store.save_draft(
                    "defect",
                    finding_id,
                    data,
                    expected_version=_expected_version(payload),
                    actor=identity.user,
                )
                return Response.json({"draft": draft})
        if action == "links":
            if method == "GET":
                return Response.json({"links": self.store.list_external_links(finding_id=finding_id)})
            if method == "POST":
                self.auth.require_role(identity, "qa")
                link = self._create_link(payload, identity, finding_id=finding_id)
                return Response.json({"link": link}, status=201)
        return Response.json(
            {"error": {"code": "method_not_allowed", "message": "method is not allowed"}},
            status=405,
        )

    def _create_link(
        self,
        payload: Mapping[str, Any],
        identity: Identity,
        *,
        finding_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        system = str(payload.get("system", "")).lower()
        url = str(payload.get("url", "")).strip()
        self._validate_external_url(system, url)
        return self.store.add_external_link(
            finding_id=finding_id,
            run_id=run_id,
            system=system,
            record_type=payload.get("record_type"),
            external_id=payload.get("external_id"),
            url=url,
            actor=identity.user,
        )

    def _validate_external_url(self, system: str, url: str) -> None:
        if not url:
            return
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ReportHubError("external link must use HTTPS")
        hostname = parsed.hostname.lower()
        if hostname not in self.settings.allowed_external_hosts:
            raise ReportHubError("external link host is not allowed")
        if system == "github" and (hostname != "github.com" or not _GITHUB_PATH.fullmatch(parsed.path)):
            raise ReportHubError("GitHub link must target this repository's issue or pull request")

    def _static(self, path: str) -> Response:
        relative = unquote(path).lstrip("/") or "index.html"
        candidate = (self.static_root / relative).resolve()
        static_root = self.static_root.resolve()
        if candidate != static_root and static_root not in candidate.parents:
            return Response.json(
                {"error": {"code": "not_found", "message": "file was not found"}}, status=404
            )
        if not candidate.is_file():
            if "." in Path(relative).name:
                return Response.json(
                    {"error": {"code": "not_found", "message": "file was not found"}}, status=404
                )
            candidate = static_root / "index.html"
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        cache = "public, max-age=3600" if candidate.name != "index.html" else "no-store"
        return Response(
            status=200,
            body=candidate.read_bytes(),
            content_type=f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type,
            headers=(("Cache-Control", cache),),
        )

    @staticmethod
    def _payload(body: bytes) -> Mapping[str, Any]:
        if len(body) > 128 * 1024:
            raise ReportHubError("request body exceeds 128 KiB")
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ReportHubError("request body must be a JSON object")
        return payload


def _query_one(query: Mapping[str, list[str]], key: str) -> str:
    values = query.get(key, [])
    return values[0] if values else ""


def _query_int(query: Mapping[str, list[str]], key: str, default: int) -> int:
    raw = _query_one(query, key)
    try:
        return int(raw) if raw else default
    except ValueError as error:
        raise ReportHubError(f"{key} must be an integer") from error


def _expected_version(payload: Mapping[str, Any]) -> int:
    value = payload.get("expected_version")
    if not isinstance(value, int):
        raise ReportHubError("expected_version must be an integer")
    return value


def _entity_id(value: str, prefix: str) -> str:
    if not _ENTITY_ID.fullmatch(value) or not value.startswith(f"{prefix}_"):
        raise ReportHubError(f"invalid {prefix} ID")
    return value
