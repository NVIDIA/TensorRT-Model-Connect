# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from apps.server.python.tests.test_app import FakeRegistry, make_client
from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import ModelNotFoundError


class TextRegistry(FakeRegistry):
    def supports(self, model, capability):
        if model != "test/model":
            raise ModelNotFoundError(model)
        return capability == "text_generation"


@pytest.mark.parametrize("shape", ["string", "message", "parts"])
def test_responses_preserves_text_and_native_controls_and_returns_typed_output(tmp_path, shape):
    prompt = "  First\nsecond\u2028third  "
    value = prompt if shape == "string" else [{"type": "message", "role": "user",
        "content": prompt if shape == "message" else [{"type": "input_text", "text": prompt[:7]},
                                                       {"type": "input_text", "text": prompt[7:]}]}]
    registry, path = TextRegistry(), tmp_path / "records.jsonl"
    app = create_app(registry, ServerConfig(max_body_bytes=4096, max_prompt_bytes=1024,
                                          max_generation_tokens=64, records=path))
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={"model": "test/model", "input": value,
            "instructions": "  System\n", "max_output_tokens": 16, "temperature": 0,
            "enable_thinking": False, "store": False}, headers={"X-Request-ID": "responses-original"})
    assert response.status_code == 200 and response.headers["X-Request-ID"] == "responses-original"
    body = response.json()
    assert body["id"].startswith("resp_") and body["object"] == "response" and body["status"] == "completed"
    message, = body["output"]
    assert message["type"] == "message" and message["id"].startswith("msg_")
    assert message["role"] == "assistant" and message["status"] == "completed"
    assert message["content"] == [{"type": "output_text", "text": "Paris", "annotations": []}]
    assert body["usage"] is None and body["store"] is False and body["max_output_tokens"] == 16
    operation, payload = registry.requests[0]
    assert operation == "generate" and payload["prompt"] == prompt
    assert payload["config"]["max_new_tokens"] == 16 and payload["config"]["temperature"] == 0
    assert payload["config"]["use_chat_template"] is True and payload["config"]["system_prompt"] == "  System\n"
    row, = [json.loads(line) for line in path.read_text().split("\n") if line]
    assert row["route"] == "/v1/responses" and row["request_id"] == "responses-original"
    assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert row["completion_sha256"] == hashlib.sha256(b"Paris").hexdigest()


@pytest.mark.parametrize("extra", [
    {"store": True}, {"stream": True}, {"previous_response_id": "resp_old"},
    {"tools": []}, {"reasoning": {"effort": "low"}}, {"conversation": "conv_old"},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]},
    {"input": [{"role": "assistant", "content": "old answer"}]},
    {"input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
    {"input": ""}, {"input": [{"role": "user", "content": [{"type": "input_text", "text": ""}]}]},
    {"max_output_tokens": 15},
])
def test_unsupported_responses_features_fail_before_native_dispatch(extra):
    registry = TextRegistry()
    with make_client(registry) as client:
        response = client.post("/v1/responses", json={"model": "test/model", "input": "hello",
                                                     "max_output_tokens": 16, **extra})
    assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
    assert not registry.requests


def test_responses_reuses_admission_unknown_model_and_generation_limit_errors():
    registry = TextRegistry()
    with make_client(registry) as client:
        request = {"model": "test/model", "input": "hello", "max_output_tokens": 16}
        assert client.post("/v1/responses", json={**request, "model": "missing"}).status_code == 404
        over = client.post("/v1/responses", json={**request, "max_output_tokens": 65})
        assert over.status_code == 400 and over.json()["error"]["param"] == "max_output_tokens"
        registry.saturated = True
        busy = client.post("/v1/responses", json=request)
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
    assert not registry.requests


def test_responses_rejects_models_without_native_text_capability():
    registry = FakeRegistry()
    with make_client(registry) as client:
        response = client.post("/v1/responses", json={"model": "test/model", "input": "hello"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "unsupported_task"
    assert not registry.requests


def test_responses_input_limit_identifies_the_responses_field_before_native_dispatch():
    registry = TextRegistry()
    app = create_app(registry, ServerConfig(max_body_bytes=4096, max_prompt_bytes=4, max_generation_tokens=64))
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={"model": "test/model", "input": "hello"})
    assert response.status_code == 413
    assert response.json()["error"]["param"] == "input"
    assert not registry.requests


@pytest.mark.parametrize("declared_length", [True, False])
@pytest.mark.parametrize("route,field", [("/v1/responses", "input"), ("/v1/completions", "prompt")])
def test_text_body_limit_rejects_oversized_json_before_native_dispatch(declared_length, route, field):
    registry = TextRegistry()
    app = create_app(registry, ServerConfig(max_body_bytes=128, max_prompt_bytes=1024, max_generation_tokens=64))
    body = json.dumps({"model": "test/model", field: "x" * 256}).encode()
    content = body if declared_length else iter([body[:64], body[64:]])
    with TestClient(app) as client:
        response = client.post(route, content=content, headers={"Content-Type": "application/json"})
    assert ("content-length" in response.request.headers) is declared_length
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "content_too_large"
    assert not registry.requests


@pytest.mark.parametrize("route,field", [("/v1/responses", "input"), ("/v1/completions", "prompt")])
def test_text_body_limit_preserves_valid_chunked_requests(route, field):
    registry = TextRegistry()
    app = create_app(registry, ServerConfig(max_body_bytes=128, max_prompt_bytes=1024, max_generation_tokens=64))
    body = json.dumps({"model": "test/model", field: "hello"}).encode()
    with TestClient(app) as client:
        response = client.post(route, content=iter([body[:16], body[16:]]),
                               headers={"Content-Type": "application/json"})
    assert "content-length" not in response.request.headers
    assert response.status_code == 200
    assert registry.requests[0][1]["prompt"] == "hello"


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN"])
def test_nonfinite_responses_temperature_is_rejected_before_native_dispatch(value):
    registry = TextRegistry()
    with make_client(registry) as client:
        response = client.post("/v1/responses", content=(
            '{"model":"test/model","input":"hello","temperature":' + value + '}'),
            headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "temperature"
    assert not registry.requests
