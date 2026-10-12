# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal request timing records, without prompt or generated text."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse

_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class RequestRecords:
    def __init__(self, path: Path | None) -> None:
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8") if path is not None else None

    def write(self, record: dict[str, Any]) -> None:
        if self._file is not None:
            with self._lock:
                self._file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class RequestPolicy:
    """Measure until the final ASGI body, including streamed response delivery."""
    def __init__(self, app: Any, *, records: RequestRecords) -> None:
        self.app = app
        self.records = records

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ids = [value.decode("latin1") for key, value in scope.get("headers", [])
               if key.lower() == b"x-request-id"]
        valid = len(ids) <= 1 and (not ids or _REQUEST_ID.fullmatch(ids[0]) is not None)
        request_id = ids[0] if ids and valid else f"req-{uuid.uuid4().hex}"
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        context: dict[str, Any] = {"request_id": request_id, "route": scope["path"],
            "model": None, "replica": None, "input_tokens": None,
            "input_token_source": "unavailable", "completion_tokens": None,
            "completion_token_source": "unavailable"}
        state["timing"] = context
        started = time.monotonic()
        status = 500
        ended = False

        async def tracked_send(message: Any) -> None:
            nonlocal status, ended
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                message["headers"] = [*headers, (b"x-request-id", request_id.encode("ascii"))]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                ended = True
        try:
            if not valid:
                await JSONResponse({"error": {"message": "invalid X-Request-ID header",
                    "type": "invalid_request_error", "param": "X-Request-ID", "code": "invalid_request"}},
                    status_code=400)(scope, receive, tracked_send)
            else:
                await self.app(scope, receive, tracked_send)
        except asyncio.CancelledError:
            status = 499
            raise
        finally:
            status = context.pop("terminal_status", status if ended else 499 if status == 200 else status)
            context.update(status=status, handler_ms=(time.monotonic() - started) * 1000)
            logging.getLogger("uvicorn.error").info(json.dumps({"event": "http_request",
                "request_id": request_id, "route": scope["path"], "status": status,
                "duration_seconds": context["handler_ms"] / 1000}, separators=(",", ":")))
            if scope["path"] in {"/v1/completions", "/v1/chat/completions"}:
                self.records.write(context)
