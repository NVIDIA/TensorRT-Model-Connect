# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASGI send barriers prove HTTP delivery precedes native completion."""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.registry import ModelRegistry, ModelSpec
from trtmc_server.worker import WorkerLoadOptions
from apps.server.python.tests.test_app import FakeRegistry
from apps.server.python.tests.test_stream_transport import SCRIPT


def test_ids_terminal_records_and_text_privacy(tmp_path):
    registry = FakeRegistry()
    records = tmp_path / "records.jsonl"
    app = create_app(registry, ServerConfig(max_body_bytes=4096, max_prompt_bytes=1024,
                                           max_generation_tokens=64, records=records))
    with TestClient(app) as client:
        for key, saturated, expected in (("success", False, 200), ("busy", True, 429)):
            registry.saturated = saturated
            response = client.post("/v1/completions", headers={"X-Request-ID": key},
                json={"model": "test/model", "prompt": "PRIVATE_INPUT"})
            assert response.status_code == expected and response.headers["x-request-id"] == key
        invalid = client.post("/v1/completions", headers={"X-Request-ID": "bad id"},
                              json={"model": "test/model", "prompt": "PRIVATE_INPUT"})
        assert invalid.status_code == 400 and invalid.headers["x-request-id"] != "bad id"
    saved = [json.loads(line) for line in records.read_text().split("\n") if line]
    assert [record["status"] for record in saved] == [200, 429, 400]
    assert saved[0]["request_id"] == "success" and saved[0]["completion_tokens"] == 1
    assert saved[0]["input_tokens"] is None and saved[0]["input_token_source"] == "unavailable"
    assert "PRIVATE_INPUT" not in records.read_text() and "Paris" not in records.read_text()


def test_first_sse_delta_is_sent_before_worker_completion(tmp_path):
    # Insert a release barrier after the first delta in the real subprocess.
    release = tmp_path / "release"
    script = SCRIPT.replace('release=r.get("release")', f'release={str(release)!r}')
    binary = tmp_path / "worker"
    binary.write_text(script); binary.chmod(0o755)
    registry = ModelRegistry([ModelSpec("test", tmp_path / "unused.bundle")], worker_binary=binary,
                             load_options=WorkerLoadOptions(), startup_timeout=3, request_timeout=3)
    records = tmp_path / "records.jsonl"
    app = create_app(registry, ServerConfig(max_body_bytes=4096, max_prompt_bytes=1024,
                                           max_generation_tokens=64, records=records))
    body = json.dumps({"model": "test", "prompt": "hello", "max_tokens": 1, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
             "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/completions",
             "raw_path": b"/v1/completions", "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json"), (b"x-request-id", b"aiperf-001")],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}
    messages = []
    async def run():
        received = False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()
        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                for event in message.get("body", b"").decode().split("\n\n"):
                    if not event.startswith("data: {"):
                        continue
                    choices = json.loads(event.removeprefix("data: ")).get("choices", [])
                    if choices and choices[0].get("text") == "中":
                        assert not release.exists()
                        assert registry.status()["models"]["test"]["idle_replicas"] == 0
                        release.touch()  # completion needs the first HTTP delta to arrive
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
            assert registry.status()["models"]["test"]["idle_replicas"] == 1
    asyncio.run(run())
    assert messages[0]["status"] == 200
    # json.dumps may escape Unicode; check parsed SSE content in addition to raw bytes.
    content = b"".join(m.get("body", b"") for m in messages).decode()
    assert content.endswith("data: [DONE]\n\n")
    saved = [json.loads(line) for line in records.read_text().split("\n") if line]
    assert len(saved) == 1 and saved[0]["status"] == 200 and saved[0]["request_id"] == "aiperf-001"


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_http_disconnect_closes_stream_and_retires_lane(tmp_path, spec_version):
    release = tmp_path / "release"
    binary = tmp_path / "worker"
    binary.write_text(SCRIPT.replace('release=r.get("release")', f'release={str(release)!r}'))
    binary.chmod(0o755)
    registry = ModelRegistry([ModelSpec("test", tmp_path / "unused.bundle")], worker_binary=binary,
                             load_options=WorkerLoadOptions(), startup_timeout=3, request_timeout=3)
    records = tmp_path / "records.jsonl"
    app = create_app(registry, ServerConfig(max_body_bytes=4096, max_prompt_bytes=1024,
                                           max_generation_tokens=64, records=records))
    body = json.dumps({"model": "test", "prompt": "hello", "max_tokens": 1, "stream": True}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
             "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/completions",
             "raw_path": b"/v1/completions", "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}

    async def run():
        received = False
        disconnected = asyncio.Event()

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                assert not release.exists()
                assert registry.status()["models"]["test"]["idle_replicas"] == 0
                disconnected.set()
                if spec_version == "2.4":
                    raise OSError("client disconnected during send")

        async with app.router.lifespan_context(app):
            try:
                await asyncio.wait_for(app(scope, receive, send), timeout=5)
            except ClientDisconnect:
                assert spec_version == "2.4"
            assert disconnected.is_set()
            model = registry.status()["models"]["test"]
            assert model["idle_replicas"] == 0 and model["ready_replicas"] == 0
            worker = registry._groups["test"]._workers[0]
            assert worker._process.poll() is not None
    asyncio.run(run())
    saved = [json.loads(line) for line in records.read_text().split("\n") if line]
    assert len(saved) == 1 and saved[0]["status"] == 499
