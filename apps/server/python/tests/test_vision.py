# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from concurrent.futures import Future
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import WorkerCrashedError, WorkerSaturatedError
from trtmc_server.registry import ModelRegistry


class Registry:
    def __init__(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 4), color="red").save(data, format="PNG")
        self.image = data.getvalue()
        self.paths = []
        self.calls = 0
        self.closed = 0
        self.retired = False
        self.failure = None
        self.result = {"text": "Red. Complete answer.", "completion_tokens": 3,
                       "image_height": 4, "image_width": 8,
                       "model_call_ms": 12.5, "timing_scope": "public_task_call_wall"}

    def start(self):
        pass

    def close(self):
        pass

    def max_tokens(self, model, cap):
        return min(10, cap)

    def supports(self, model, capability):
        return model == "vision" and capability == "vision_language_generation"

    def acquire(self, model):
        if self.failure == "busy":
            raise WorkerSaturatedError("busy")
        return Session(self)


class Session:
    replica_name = "vision-1"

    def __init__(self, registry):
        self.registry = registry

    def submit(self, operation, payload):
        path = Path(payload["image_path"])
        self.registry.paths.append(path)
        assert path.read_bytes() == self.registry.image
        assert operation == "generate_vision" and payload["prompt"] == "Color?"
        assert payload["config"] == {"max_new_tokens": 10, "temperature": 1.0, "top_k": 1}
        self.registry.calls += 1
        if self.registry.failure == "crashed":
            raise WorkerCrashedError("private path")
        result = Future()
        result.set_result(self.registry.result)
        return result

    def close(self):
        self.registry.closed += 1

    def retire(self, error):
        self.registry.retired = True
        result = Future()
        result.set_result(None)
        return result


def payload(registry):
    return {"model": "vision", "max_tokens": 10, "temperature": 1.0, "top_k": 1,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Color?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                    base64.b64encode(registry.image).decode()}}]}]}


def client_for(registry, records=None, **limits):
    return TestClient(create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits)))


def test_image_question_full_answer_and_request_timing(tmp_path):
    registry = Registry()
    records = tmp_path / "requests.jsonl"
    ModelRegistry._validate_ready({"event": "ready", "protocol_version": 2,
                                  "capabilities": ["vision_language_generation"]})
    with client_for(registry, records) as client:
        response = client.post("/v1/chat/completions", json=payload(registry), headers={"X-Request-ID": "vqa-42"})
    assert response.status_code == 200 and response.headers["x-request-id"] == "vqa-42"
    body = response.json()
    assert body["choices"][0]["message"] == {"role": "assistant", "content": registry.result["text"]}
    assert body["usage"]["completion_tokens"] == 3 and body["object"] == "chat.completion"
    record = json.loads(records.read_text())
    assert record["request_id"] == "vqa-42" and record["image_width"] == 8 and record["image_height"] == 4
    assert record["image_bytes"] == len(registry.image) and record["model_call_ms"] == 12.5
    assert record["timing_scope"] == "public_task_call_wall" and record["streaming"] == "none"
    assert registry.closed == 1 and all(not path.exists() for path in registry.paths)


@pytest.mark.parametrize("kind", ["two_images", "two_turns", "system", "no_text", "stream", "stop", "thinking", "n",
                                 "external_url", "local_path", "bad_media", "other_model", "tokens", "duplicate_tokens"])
def test_unsupported_image_requests_never_invoke_native(kind):
    registry = Registry()
    body = payload(registry)
    message = body["messages"][0]
    if kind == "two_images":
        message["content"].append(message["content"][-1])
    elif kind in {"system", "two_turns"}:
        body["messages"].insert(0, {"role": "system" if kind == "system" else "user", "content": "Hi"})
    elif kind == "no_text":
        message["content"].pop(0)
    elif kind in {"external_url", "local_path", "bad_media"}:
        message["content"][-1]["image_url"]["url"] = {
            "external_url": "https://example.org/test.png", "local_path": "/etc/passwd",
            "bad_media": "data:image/png;base64,???"}[kind]
    else:
        field, value = {"stream": ("stream", True), "stop": ("stop", "red"), "n": ("n", 2),
                        "thinking": ("enable_thinking", False), "other_model": ("model", "text-only"),
                        "tokens": ("max_tokens", 129), "duplicate_tokens": ("max_completion_tokens", 10)}[kind]
        body[field] = value
    with client_for(registry) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400 and registry.calls == 0
    assert all(not path.exists() for path in registry.paths)


@pytest.mark.parametrize("text", ["", " "])
def test_empty_native_text_is_preserved_for_the_family_to_grade(text):
    registry = Registry()
    registry.result.update(text=text, completion_tokens=0)
    with client_for(registry) as client:
        response = client.post("/v1/chat/completions", json=payload(registry))
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == text
    assert response.json()["usage"]["completion_tokens"] == 0 and not registry.retired
    assert registry.closed == 1 and all(not path.exists() for path in registry.paths)


@pytest.mark.parametrize("patch", [{"image_width": 7}, {"image_height": True},
                                  {"timing_scope": "kernel"}, {"completion_tokens": True},
                                  {"model_call_ms": -1}])
def test_invalid_native_result_retires_worker_and_releases_upload(patch):
    registry = Registry()
    registry.result.update(patch)
    with client_for(registry) as client:
        response = client.post("/v1/chat/completions", json=payload(registry))
    assert response.status_code == 502 and registry.retired and registry.closed == 1
    assert all(not path.exists() for path in registry.paths)


@pytest.mark.parametrize("failure,status", [("busy", 429), ("crashed", 503), ("pixels", 413)])
def test_image_failure_records_and_cleanup(tmp_path, failure, status):
    registry = Registry()
    registry.failure = failure
    records = tmp_path / "requests.jsonl"
    with client_for(registry, records, max_image_pixels=16 if failure == "pixels" else 32) as client:
        response = client.post("/v1/chat/completions", json=payload(registry))
    assert response.status_code == status and "private" not in response.text
    record = json.loads(records.read_text())
    assert record["status"] == status and record["image_bytes"] == len(registry.image)
    assert all(not path.exists() for path in registry.paths)
