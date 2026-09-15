# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from fastapi.testclient import TestClient

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import ModelNotFoundError, WorkerSaturatedError


class FakeSession:
    def __init__(self, requests: list[tuple[str, dict[str, Any]]]) -> None:
        self.requests = requests
        self.closed = False

    def submit(self, operation: str, payload: dict[str, Any]) -> Future[Any]:
        self.requests.append((operation, payload))
        result: Future[Any] = Future()
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
        return FakeSession(self.requests)


def make_client(
    registry: FakeRegistry, *, api_key: str | None = None
) -> TestClient:
    app = create_app(
        registry,  # type: ignore[arg-type]
        ServerConfig(
            api_key=api_key,
            max_body_bytes=4096,
            max_prompt_bytes=1024,
            max_generation_tokens=64,
        ),
    )
    return TestClient(app)


def test_completion_and_chat_keep_model_semantics_in_worker() -> None:
    registry = FakeRegistry()
    with make_client(registry) as client:
        completion = client.post(
            "/v1/completions",
            json={"model": "test/model", "prompt": "Capital?", "temperature": 0},
        )
        assert completion.status_code == 200
        assert completion.json()["choices"][0]["text"] == "Paris"
        assert completion.json()["usage"]["completion_tokens"] == 1

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


def test_validation_overload_auth_and_metrics_are_explicit() -> None:
    registry = FakeRegistry()
    with make_client(registry, api_key="secret") as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        unknown = client.post(
            "/v1/completions",
            headers=headers,
            json={"model": "test/model", "prompt": "x", "unknown": True},
        )
        assert unknown.status_code == 400
        assert unknown.json()["error"]["param"] == "unknown"

        missing = client.post(
            "/v1/completions",
            headers=headers,
            json={"model": "missing/model", "prompt": "x"},
        )
        assert missing.status_code == 404

        oversized = client.post(
            "/v1/completions",
            headers={**headers, "Content-Length": "5000"},
            content=b"{}",
        )
        assert oversized.status_code == 413

        registry.saturated = True
        busy = client.post(
            "/v1/completions",
            headers=headers,
            json={"model": "test/model", "prompt": "x"},
        )
        assert busy.status_code == 429
        assert busy.headers["retry-after"] == "1"
        metrics = client.get("/metrics", headers=headers)
        assert metrics.status_code == 200
        assert 'route="/v1/completions",status="429"' in metrics.text
