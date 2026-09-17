# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from concurrent.futures import Future
from typing import Any

from fastapi.testclient import TestClient

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import (
    ModelNotFoundError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
)


class FakeSession:
    def __init__(
        self,
        requests: list[tuple[str, dict[str, Any]]],
        request_error: Exception | None,
    ) -> None:
        self.requests = requests
        self.request_error = request_error
        self.closed = False

    def submit(self, operation: str, payload: dict[str, Any]) -> Future[Any]:
        self.requests.append((operation, payload))
        result: Future[Any] = Future()
        if self.request_error is not None:
            result.set_exception(self.request_error)
        else:
            result.set_result(
                {
                    "text": "Paris",
                    "completion_tokens": 1,
                    "setup_ms": 1.0,
                    "prefill_ms": 2.0,
                    "decode_ms": 3.0,
                }
            )
        return result

    def close(self) -> None:
        self.closed = True


class FakeRegistry:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.started = False
        self.saturated = False
        self.request_error: Exception | None = None

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.started,
            "degraded": False,
            "models": {
                "test/model": {
                    "ready_replicas": 1,
                    "idle_replicas": 1,
                }
            },
        }

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "test/model",
                "object": "model",
                "created": 1,
                "owned_by": "tensorrt-model-connect",
            }
        ]

    def max_tokens(self, model: str, hard_cap: int) -> int:
        if model != "test/model":
            raise ModelNotFoundError(model)
        return min(17, hard_cap)

    def acquire(self, model: str) -> FakeSession:
        if model != "test/model":
            raise ModelNotFoundError(model)
        if self.saturated:
            raise WorkerSaturatedError("busy")
        return FakeSession(self.requests, self.request_error)


def make_client(registry: FakeRegistry) -> TestClient:
    app = create_app(
        registry,  # type: ignore[arg-type]
        ServerConfig(
            max_body_bytes=4096,
            max_prompt_bytes=1024,
            max_generation_tokens=64,
        ),
    )
    return TestClient(app)


def test_completion_and_chat_keep_model_semantics_in_worker(caplog: Any) -> None:
    registry = FakeRegistry()
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with make_client(registry) as client:
        completion = client.post(
            "/v1/completions",
            json={"model": "test/model", "prompt": "Capital?", "temperature": 0},
        )
        assert completion.status_code == 200
        assert completion.json()["choices"][0]["text"] == "Paris"
        assert completion.json()["usage"]["completion_tokens"] == 1
        completion_request_id = completion.headers["x-request-id"]

        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "test/model",
                "messages": [
                    {"role": "system", "content": "Be brief"},
                    {"role": "user", "content": "Capital?"},
                ],
                "max_completion_tokens": 8,
            },
        )
        assert chat.status_code == 200
        config = registry.requests[-1][1]["config"]
        assert config["use_chat_template"] is True
        assert config["system_prompt"] == "Be brief"
        assert config["max_new_tokens"] == 8

    request_logs = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "uvicorn.error" and '"event":"http_request"' in record.message
    ]
    completion_log = next(
        record for record in request_logs if record["request_id"] == completion_request_id
    )
    assert completion_log["route"] == "/v1/completions"
    assert completion_log["status"] == 200
    assert completion_log["duration_seconds"] >= 0
    assert "Capital?" not in json.dumps(request_logs)


def test_validation_overload_and_metrics_are_explicit() -> None:
    registry = FakeRegistry()
    with make_client(registry) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        unknown = client.post(
            "/v1/completions",
            json={"model": "test/model", "prompt": "x", "unknown": True},
        )
        assert unknown.status_code == 400
        assert unknown.json()["error"]["param"] == "unknown"

        missing = client.post(
            "/v1/completions",
            json={"model": "missing/model", "prompt": "x"},
        )
        assert missing.status_code == 404

        oversized = client.post(
            "/v1/completions",
            headers={"Content-Length": "5000"},
            content=b"{}",
        )
        assert oversized.status_code == 413

        registry.saturated = True
        busy = client.post(
            "/v1/completions",
            json={"model": "test/model", "prompt": "x"},
        )
        assert busy.status_code == 429
        assert busy.headers["retry-after"] == "1"
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert 'route="/v1/completions",status="429"' in metrics.text


def test_worker_request_too_large_returns_413_and_finishes_metrics() -> None:
    registry = FakeRegistry()
    registry.request_error = WorkerRequestTooLargeError("private serialized size")
    with make_client(registry) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "test/model", "prompt": "x"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
        assert "private serialized size" not in response.text

        metrics = client.get("/metrics")
        assert 'route="/v1/completions",status="413"' in metrics.text
        assert "trtmc_server_active_requests 0" in metrics.text
