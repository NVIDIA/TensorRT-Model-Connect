# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future
import json

from fastapi.testclient import TestClient
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import ModelNotFoundError, WorkerProtocolError, WorkerSaturatedError
from trtmc_server.registry import ModelRegistry


class Session:
    replica_name = "encoder-1"

    def __init__(self, registry):
        self.registry = registry

    def submit(self, operation, payload):
        self.registry.calls.append((operation, payload))
        future = Future()
        future.set_result(self.registry.result)
        return future

    def retire(self, error):
        self.registry.retired = True
        future = Future()
        future.set_result(None)
        return future

    def close(self):
        self.registry.closed += 1


class Registry:
    def __init__(self):
        self.calls = []
        self.result = {"values": [0.25] * 383 + [-2.0], "dim": 384,
                       "model_call_ms": 1.5, "timing_scope": "public_task_call_wall"}
        self.retired = False
        self.closed = 0
        self.saturated = False

    def start(self):
        pass

    def close(self):
        pass

    def supports(self, model, capability):
        if model == "missing":
            raise ModelNotFoundError("model is not registered")
        return model == "encoder" and capability == "embeddings"

    def acquire(self, model):
        if self.saturated:
            raise WorkerSaturatedError("busy")
        return Session(self)


def client_for(registry, records=None, **limits):
    config = ServerConfig(max_body_bytes=1024, max_prompt_bytes=128,
                          max_generation_tokens=16, records=records)
    for name, value in limits.items():
        setattr(config, name, value)
    return TestClient(create_app(registry, config))


@pytest.mark.parametrize("value", ["Exact input: café", ["Exact input: café"]])
def test_full_vector_input_identity_and_timing(value, tmp_path):
    registry = Registry()
    path = tmp_path / "records.jsonl"
    with client_for(registry, path) as client:
        response = client.post("/v1/embeddings", json={"model": "encoder", "input": value},
                               headers={"X-Request-ID": "embedding-1"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "embedding-1"
    assert response.json()["data"] == [{"object": "embedding", "index": 0,
                                        "embedding": registry.result["values"]}]
    assert "usage" not in response.json()
    assert registry.calls == [("embed", {"input": "Exact input: café"})]
    assert registry.closed == 1
    record = json.loads(path.read_text())
    assert record["request_id"] == "embedding-1"
    assert record["model_call_ms"] == 1.5
    assert record["timing_scope"] == "public_task_call_wall"
    assert record["embedding_dimensions"] == 384
    assert record["completion_tokens"] is None
    assert "Exact input" not in path.read_text()


@pytest.mark.parametrize("extra", [
    {"dimensions": 384}, {"encoding_format": "base64"}, {"stream": True},
    {"input": []}, {"input": ["one", "two"]}, {"input": [1, 2]}, {"input": ""},
])
def test_unsupported_inputs_are_rejected_before_execution(extra):
    registry = Registry()
    with client_for(registry) as client:
        response = client.post("/v1/embeddings", json={"model": "encoder", "input": "one", **extra})
    assert response.status_code == 400
    assert not registry.calls


def test_capability_admission_limits_and_rejection_records(tmp_path):
    registry = Registry()
    path = tmp_path / "records.jsonl"
    with client_for(registry, path) as client:
        for model, status in [("missing", 404), ("text-model", 400)]:
            assert client.post("/v1/embeddings", json={"model": model, "input": "one"}).status_code == status
        registry.saturated = True
        busy = client.post("/v1/embeddings", json={"model": "encoder", "input": "one"})
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
        registry.saturated = False
        assert client.post("/v1/embeddings", json={"model": "encoder", "input": "x" * 129}).status_code == 413
        assert client.post("/v1/embeddings", content=b"x" * 1025).status_code == 413
    assert not registry.calls
    assert [json.loads(line)["status"] for line in path.read_text().splitlines()] == [404, 400, 429, 413, 413]


@pytest.mark.parametrize("updates", [
    {"values": {"$array_length": 384}}, {"dim": 383}, {"values": [True], "dim": 1},
    {"values": [10**400], "dim": 1}, {"model_call_ms": 10**400},
    {"model_call_ms": -1}, {"timing_scope": "unknown"},
])
def test_invalid_native_output_is_rejected_and_retires_lane(updates):
    registry = Registry()
    registry.result.update(updates)
    with client_for(registry) as client:
        response = client.post("/v1/embeddings", json={"model": "encoder", "input": "one"})
    assert response.status_code == 502
    assert registry.retired and registry.closed == 1


def test_registry_accepts_embedding_capability_and_rejects_unknown_only():
    ModelRegistry._validate_ready({"event": "ready", "protocol_version": 2,
                                   "capabilities": ["embeddings"]})
    with pytest.raises(WorkerProtocolError):
        ModelRegistry._validate_ready({"event": "ready", "protocol_version": 2,
                                       "capabilities": ["not-a-serving-task"]})
