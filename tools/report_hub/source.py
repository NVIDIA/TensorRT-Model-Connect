# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP adapter for the immutable report evidence store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from .config import Settings
from .domain import ReportHubError, UpstreamError, require_folder


SOURCE_FILES = {
    "benchmark": ("report.json",),
    "perf": ("results.json", "report.json"),
}


@dataclass(frozen=True)
class ReportDocument:
    payload: Mapping[str, Any]
    json_url: str
    html_url: str


class HttpEvidenceSource:
    def __init__(
        self,
        settings: Settings,
        *,
        loader: Callable[[str, int], Mapping[str, Any]] | None = None,
    ):
        self.settings = settings
        self._loader = loader or self._load_json

    def fetch_catalog(self) -> Mapping[str, Any]:
        if not self.settings.catalog_url:
            raise UpstreamError("report catalog URL is not configured")
        payload = self._loader(self.settings.catalog_url, 2 * 1024 * 1024)
        if payload.get("schema_version") != "trtmc.report-browser-index/v1":
            raise UpstreamError("report catalog has an unsupported schema")
        return payload

    def fetch_report(self, source: str, folder: str) -> ReportDocument:
        if source not in SOURCE_FILES:
            raise ReportHubError("unsupported report source")
        require_folder(folder)
        if not self.settings.evidence_root_url:
            raise UpstreamError("evidence root URL is not configured")
        last_error: Exception | None = None
        for filename in SOURCE_FILES[source]:
            url = self.evidence_url(source, folder, filename)
            try:
                payload = self._loader(url, self.settings.max_report_bytes)
                return ReportDocument(
                    payload=payload,
                    json_url=url,
                    html_url=self.evidence_url(source, folder, "report.html"),
                )
            except UpstreamError as error:
                last_error = error
        raise UpstreamError(f"report evidence could not be read: {last_error}")

    def evidence_url(self, source: str, folder: str, filename: str) -> str:
        if source not in SOURCE_FILES:
            raise ReportHubError("unsupported report source")
        require_folder(folder)
        if filename not in {*SOURCE_FILES[source], "report.html"}:
            raise ReportHubError("unsupported evidence file")
        quoted = urllib.parse.quote(folder, safe="")
        return f"{self.settings.evidence_root_url.rstrip('/')}/{source}/{quoted}/{filename}"

    def _load_json(self, url: str, maximum: int) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "TRTMC-Report-Hub/1"})
        try:
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                final_url = response.geturl()
                if _origin(final_url) != _origin(url):
                    raise UpstreamError("evidence request redirected to a different origin")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > maximum:
                    raise UpstreamError("evidence document exceeds configured size limit")
                raw = response.read(maximum + 1)
        except urllib.error.HTTPError as error:
            raise UpstreamError(f"evidence request failed with HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise UpstreamError(f"evidence request failed: {error}") from error
        if len(raw) > maximum:
            raise UpstreamError("evidence document exceeds configured size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("evidence document is not valid UTF-8 JSON") from error
        if not isinstance(payload, Mapping):
            raise UpstreamError("evidence document must be a JSON object")
        return payload


def _origin(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower(), parsed.netloc.lower()
