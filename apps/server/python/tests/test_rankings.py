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
    replica_name = "ranker-1"

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
        self.result = {"scores": [-2.0, 3.0], "model_call_ms": 1.5, "timing_scope": "public_task_call_wall"}
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
        return model == "ranker" and capability == "reranking"

    def acquire(self, model):
        if self.saturated:
            raise WorkerSaturatedError("busy")
        return Session(self)


BODY = {"model": "ranker", "query": {"text": "Exact query: café"},
        "passages": [{"text": "first"}, {"text": "second"}]}


def client_for(registry, records=None):
    return TestClient(create_app(registry, ServerConfig(max_body_bytes=1024, max_prompt_bytes=128,
                                                       max_generation_tokens=16, records=records)))


def test_real_scores_sorted_with_original_indices_and_timing(tmp_path):
    registry = Registry()
    path = tmp_path / "records.jsonl"
    with client_for(registry, path) as client:
        response = client.post("/v1/ranking", json=BODY, headers={"X-Request-ID": "ranking-1"})
    assert response.status_code == 200 and response.headers["X-Request-ID"] == "ranking-1"
    assert response.json() == {"rankings": [{"index": 1, "logit": 3.0}, {"index": 0, "logit": -2.0}]}
    assert registry.calls == [("rerank", {"query": "Exact query: café", "documents": ["first", "second"]})]
    assert registry.closed == 1
    record = json.loads(path.read_text())
    assert record["request_id"] == "ranking-1" and record["model_call_ms"] == 1.5
    assert record["timing_scope"] == "public_task_call_wall" and record["passages"] == 2
    assert record["passage_bytes"] == [5, 6] and record["completion_tokens"] is None
    assert "Exact query" not in path.read_text()


def test_ties_preserve_original_passage_order():
    registry = Registry()
    registry.result["scores"] = [3.0, 3.0]
    with client_for(registry) as client:
        response = client.post("/v1/ranking", json=BODY)
    assert [item["index"] for item in response.json()["rankings"]] == [0, 1]


@pytest.mark.parametrize("extra", [
    {"truncate": "END"}, {"stream": True}, {"top_n": 1}, {"passages": []},
    {"query": {"text": ""}}, {"passages": [{"text": 1}]}, {"passages": [{"image": "x"}]},
])
def test_unsupported_inputs_are_rejected_before_execution(extra):
    registry = Registry()
    with client_for(registry) as client:
        response = client.post("/v1/ranking", json={**BODY, **extra})
    assert response.status_code == 400 and not registry.calls


def test_capability_admission_limits_and_rejection_records(tmp_path):
    registry = Registry()
    path = tmp_path / "records.jsonl"
    with client_for(registry, path) as client:
        for model, status in [("missing", 404), ("text-model", 400)]:
            assert client.post("/v1/ranking", json={**BODY, "model": model}).status_code == status
        registry.saturated = True
        busy = client.post("/v1/ranking", json=BODY)
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
        registry.saturated = False
        assert client.post("/v1/ranking", json={**BODY, "query": {"text": "x" * 129}}).status_code == 413
        assert client.post("/v1/ranking", content=b"x" * 1025).status_code == 413
    assert not registry.calls
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in records] == [404, 400, 429, 413, 413]
    assert [row["query_bytes"] for row in records[:4]] == [18, 18, 18, 129]
    for row in records[:4]:
        assert row["passages"] == row["input_items"] == 2 and row["passage_bytes"] == [5, 6]


@pytest.mark.parametrize("updates", [
    {"scores": []}, {"scores": [True, 2.0]}, {"scores": {"$array_length": 2}},
    {"scores": [10**400, 2.0]}, {"model_call_ms": 10**400},
    {"model_call_ms": -1}, {"timing_scope": "unknown"},
])
def test_invalid_native_output_is_rejected_and_retires_lane(updates, tmp_path):
    registry = Registry()
    registry.result.update(updates)
    path = tmp_path / "records.jsonl"
    with client_for(registry, path) as client:
        response = client.post("/v1/ranking", json=BODY)
    assert response.status_code == 502 and registry.retired and registry.closed == 1
    record = json.loads(path.read_text())
    assert record["passages"] == 2 and record["query_bytes"] == 18
    assert record["passage_bytes"] == [5, 6]


def test_registry_accepts_ranking_capability_and_rejects_unknown_only():
    ModelRegistry._validate_ready({"event": "ready", "protocol_version": 2, "capabilities": ["reranking"]})
    with pytest.raises(WorkerProtocolError):
        ModelRegistry._validate_ready({"event": "ready", "protocol_version": 2, "capabilities": ["unknown"]})
