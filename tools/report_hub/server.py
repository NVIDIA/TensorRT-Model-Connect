# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Threaded HTTP transport for the Report Hub application."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Type

from .app import ReportHubApplication


def handler_for(application: ReportHubApplication) -> Type[BaseHTTPRequestHandler]:
    class ReportHubHandler(BaseHTTPRequestHandler):
        server_version = "TRTMCReportHub/1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            self._dispatch("POST")

        def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
            self._dispatch("PUT")

        def _dispatch(self, method: str) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
                return
            if length > 128 * 1024:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            body = self.rfile.read(length) if length else b""
            response = application.handle(method, self.path, self.headers, body)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            for name, value in response.headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[report-hub] {self.address_string()} {format % args}")

    return ReportHubHandler


def serve(application: ReportHubApplication, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), handler_for(application))
    print(f"TRTMC Report Hub listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
