# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # the serving extra (apps/perf_serving/requirements.txt)

from fastapi.testclient import TestClient  # noqa: E402

from trtmc_perf_serving.app import Lane, Saturated, ServingConfig, create_app  # noqa: E402
from trtmc_perf_serving.backends.base import BackendError, Invocation  # noqa: E402


class FakeBackend:
    def __init__(self, operation="classify", fail=None, gate=None):
        self.operation = operation
        self.requests = []
        self.fail = fail
        self.gate = gate

    def describe(self):
        return {"backend": "fake", "operation": self.operation}

    def invoke(self, request, artifact_base):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise self.fail
        files = {key: Path(value).read_bytes() for key, value in request.items() if key.endswith("_path")}
        self.requests.append({"request": dict(request), "files": files})
        # Like the worker, write an output artifact next to artifact_base.
        artifact_base.with_suffix(".audio.1.wav").write_bytes(b"wav")
        if self.operation == "generate":
            return Invocation({"text": getattr(self, "text", "out"), "output_tokens": 3}, 12.5, {"prompt_tokens": 5})
        if self.operation == "segment":
            return Invocation({"mask": list(range(100)), "classes": [1, 2]}, 4.0)
        return Invocation({"shape": [1, 1000]}, 4.0)

    def close(self):
        pass


def client_for(tmp_path, backend, **config):
    config = ServingConfig(base_request={"image_path": "/base.png", "top_k": 5}, records=tmp_path / "records.jsonl",
                           scratch=tmp_path / "scratch", **config)
    return TestClient(create_app(backend, config)), config


def test_task_route_merges_base_request_materializes_files_and_records_request_id(tmp_path):
    backend = FakeBackend()
    client, config = client_for(tmp_path, backend)
    body = {"request": {"image_path": {"$file": {"suffix": ".png", "b64": base64.b64encode(b"img").decode()}}}}

    response = client.post("/v1/tasks/classify", json=body, headers={"X-Request-ID": "req-1"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-1"
    assert response.headers["server-timing"] == "model;dur=4.000"
    payload = response.json()
    assert payload["trtmc_timing"]["model_call_ms"] == 4.0
    assert payload["trtmc_observation"] == {"shape": [1, 1000]}
    assert payload["text"] == "classify"
    assert backend.requests[0]["request"]["top_k"] == 5
    assert backend.requests[0]["files"]["image_path"] == b"img"
    record = json.loads(config.records.read_text())
    assert record["request_id"] == "req-1" and record["timing"]["model_call_ms"] == 4.0
    assert not (config.scratch / "req-1").exists()  # artifacts are not kept by default


@pytest.mark.parametrize("request_id", ["..", ".", ".hidden"])
def test_request_id_cannot_name_a_directory_outside_scratch(tmp_path, request_id):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    client, config = client_for(tmp_path, FakeBackend())
    body = {"request": {"image_path": {"$file": {"suffix": ".png", "b64": base64.b64encode(b"img").decode()}}}}

    response = client.post("/v1/tasks/classify", json=body, headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    assert response.headers["x-request-id"] != request_id
    assert sentinel.read_text() == "keep" and not (tmp_path / "inputs").exists()


def test_routes_for_other_operations_are_not_served(tmp_path):
    client, _ = client_for(tmp_path, FakeBackend("classify"))
    assert client.post("/v1/completions", json={"prompt": "x"}).status_code == 404
    assert client.post("/v1/tasks/detect", json={"request": {}}).status_code == 400


def test_backend_rejection_and_bad_body_map_to_client_errors(tmp_path):
    client, _ = client_for(tmp_path, FakeBackend(fail=BackendError("bad image")))
    rejected = client.post("/v1/tasks/classify", json={"request": {}})
    assert rejected.status_code == 422 and "bad image" in rejected.json()["error"]["message"]
    assert client.post("/v1/tasks/classify", json={"unexpected": 1}).status_code == 400


def test_long_arrays_are_compacted_unless_full_observations(tmp_path):
    client, config = client_for(tmp_path, FakeBackend("segment"))
    compacted = client.post("/v1/tasks/segment", json={"request": {"image_path": "/dev/null"}}).json()
    assert compacted["trtmc_observation"] == {"mask": {"$array_length": 100}, "classes": [1, 2]}
    assert json.loads(config.records.read_text())["observation"]["mask"] == {"$array_length": 100}

    full_client, _ = client_for(tmp_path / "full", FakeBackend("segment"), full_observations=True)
    full = full_client.post("/v1/tasks/segment", json={"request": {"image_path": "/dev/null"}}).json()
    assert full["trtmc_observation"]["mask"] == list(range(100))


def test_unexpected_backend_failure_is_500_with_request_id(tmp_path):
    client, _ = client_for(tmp_path, FakeBackend(fail=KeyError("prompt")))
    response = client.post("/v1/tasks/classify", json={"request": {}}, headers={"X-Request-ID": "req-9"})
    assert response.status_code == 500
    assert response.headers["x-request-id"] == "req-9"
    assert response.json()["error"]["code"] == "backend_failed"


def test_completions_stream_is_buffered_sse_with_usage_and_timing(tmp_path):
    client, _ = client_for(tmp_path, FakeBackend("generate"))
    response = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 3, "stream": True,
                                                    "stream_options": {"include_usage": True}})
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first, usage = json.loads(events[0]), json.loads(events[1])
    assert first["choices"][0]["text"] == "out"
    assert usage["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert usage["trtmc_timing"]["model_call_ms"] == 12.5


def test_chat_stop_truncates_returned_text(tmp_path):
    backend = FakeBackend("generate")
    backend.text = "B\nQuestion"
    client, _ = client_for(tmp_path, backend, chat_renderer=lambda messages, thinking: "rendered")
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"},
                         {"role": "user", "content": "q2"}], "stop": ["\n"], "max_completion_tokens": 5}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "B"
    assert backend.requests[0]["request"]["prompt"] == "rendered"
    assert "stop" not in backend.requests[0]["request"]


def test_lane_rejects_when_busy_and_queue_is_full():
    import asyncio

    async def scenario():
        lane = Lane(max_queue=0)
        gate = threading.Event()
        def blocked_call():
            gate.wait(5)
            return Invocation({}, 1.0)

        first = asyncio.create_task(lane.run(blocked_call))
        await asyncio.sleep(0.05)
        with pytest.raises(Saturated):
            await lane.run(lambda: Invocation({}, 1.0))
        gate.set()
        result, queue_ms, _ = await first
        assert result.model_call_ms == 1.0 and queue_ms >= 0

    asyncio.run(scenario())
