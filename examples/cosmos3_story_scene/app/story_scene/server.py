# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small standard-library HTTP API and static-file server."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import AppConfig
from .jobs import JobManager, JobManagerClosed, JobNotFound
from .prompts import (
    MAX_JSON_BYTES,
    ValidationError,
    parse_json_body,
    preset_catalog,
    validate_submission,
)
from .runtime import CommandRunner, StoryScenePipeline, run_command


JOB_ROUTE = re.compile(r"^/api/jobs/([0-9a-fA-F-]{36})$")
OUTPUT_ROUTE = re.compile(
    r"^/outputs/([0-9a-fA-F-]{36})/(horizontal|social)\.mp4$"
)
RANGE_HEADER = re.compile(r"^bytes=(\d*)-(\d*)$")


@dataclass(slots=True)
class StorySceneApplication:
    config: AppConfig
    jobs: JobManager


def _safe_static_file(root: Path, request_path: str) -> Path | None:
    try:
        decoded = unquote(request_path, errors="strict")
    except UnicodeDecodeError:
        return None
    if "\x00" in decoded or "\\" in decoded:
        return None
    relative = PurePosixPath(decoded.lstrip("/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    root = root.resolve()
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        if os.path.commonpath((root, candidate)) != str(root):
            return None
    except ValueError:
        return None
    return candidate


class StorySceneHandler(BaseHTTPRequestHandler):
    """A bound handler subclass receives ``application`` from create_server."""

    application: StorySceneApplication
    protocol_version = "HTTP/1.1"
    server_version = "Cosmos3StoryScene/1"

    def log_message(self, _format: str, *_args: object) -> None:
        # Request lines and headers are intentionally never logged.
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._route_get(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._route_get(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path != "/api/jobs":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if media_type.strip().lower() != "application/json":
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Content-Type must be application/json"},
            )
            return
        if self.headers.get("Transfer-Encoding"):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Chunked request bodies are not supported"},
            )
            return
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED,
                {"error": "Content-Length is required"},
            )
            return
        try:
            length = int(length_header, 10)
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Content-Length is invalid"},
            )
            return
        if length > MAX_JSON_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Request body is too large"},
            )
            return
        try:
            payload = parse_json_body(self.rfile.read(length))
            submission = validate_submission(payload)
            snapshot = self.application.jobs.submit(submission)
        except ValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except JobManagerClosed:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "The generator is shutting down"},
            )
            return
        self._send_json(HTTPStatus.ACCEPTED, snapshot.as_dict())

    def _route_get(self, *, send_body: bool) -> None:
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "worker_ready": self.application.jobs.is_alive,
                },
                send_body=send_body,
            )
            return
        if path == "/api/presets":
            self._send_json(
                HTTPStatus.OK,
                {"presets": preset_catalog()},
                send_body=send_body,
            )
            return
        match = JOB_ROUTE.fullmatch(path)
        if match:
            try:
                snapshot = self.application.jobs.get(match.group(1).lower())
            except JobNotFound:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Job not found"},
                    send_body=send_body,
                )
                return
            self._send_json(
                HTTPStatus.OK,
                snapshot.as_dict(),
                send_body=send_body,
            )
            return
        match = OUTPUT_ROUTE.fullmatch(path)
        if match:
            job_id = match.group(1).lower()
            try:
                snapshot = self.application.jobs.get(job_id)
            except JobNotFound:
                snapshot = None
            if snapshot is None or snapshot.status != "succeeded":
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Output not found"},
                    send_body=send_body,
                )
                return
            filename = f"{match.group(2)}.mp4"
            self._send_file(
                self.application.config.output_root / job_id / filename,
                "video/mp4",
                send_body=send_body,
            )
            return
        if path.startswith("/api/") or path.startswith("/outputs/"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Not found"},
                send_body=send_body,
            )
            return

        static_path = "index.html" if path == "/" else path
        if static_path.startswith("/static/"):
            static_path = static_path.removeprefix("/static/")
        candidate = _safe_static_file(
            self.application.config.static_root,
            static_path,
        )
        if candidate is None or not candidate.is_file():
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Not found"},
                send_body=send_body,
            )
            return
        media_type = mimetypes.guess_type(candidate.name)[0]
        self._send_file(
            candidate,
            media_type or "application/octet-stream",
            send_body=send_body,
        )

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        send_body: bool = True,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_file(
        self,
        path: Path,
        media_type: str,
        *,
        send_body: bool,
    ) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Not found"},
                send_body=send_body,
            )
            return
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_value = self.headers.get("Range")
        if range_value:
            parsed = RANGE_HEADER.fullmatch(range_value.strip())
            if parsed is None or size == 0:
                self._range_not_satisfiable(size)
                return
            first, last = parsed.groups()
            if not first:
                suffix = int(last or "0")
                if suffix <= 0:
                    self._range_not_satisfiable(size)
                    return
                start = max(0, size - suffix)
            else:
                start = int(first)
                if last:
                    end = min(end, int(last))
            if start >= size or start > end:
                self._range_not_satisfiable(size)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Cache-Control",
            "public, max-age=3600" if media_type == "video/mp4" else "no-cache",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if media_type == "text/html":
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "media-src 'self' blob:; style-src 'self'; script-src 'self'; "
                "object-src 'none'; base-uri 'none'; form-action 'self'; "
                "frame-ancestors 'none'",
            )
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body or length == 0:
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                block = source.read(min(64 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def _range_not_satisfiable(self, size: int) -> None:
        body = b'{"error":"Requested range is not satisfiable"}'
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(
    config: AppConfig | None = None,
    *,
    runner: CommandRunner = run_command,
) -> tuple[ThreadingHTTPServer, JobManager]:
    config = AppConfig.from_env() if config is None else config
    pipeline = StoryScenePipeline(config, runner=runner)
    jobs = JobManager(config.output_root, pipeline.run)
    application = StorySceneApplication(config=config, jobs=jobs)

    class BoundStorySceneHandler(StorySceneHandler):
        pass

    BoundStorySceneHandler.application = application
    server = ThreadingHTTPServer((config.host, config.port), BoundStorySceneHandler)
    server.daemon_threads = True
    return server, jobs


def main() -> None:
    server, jobs = create_server()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        jobs.close()
