# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import Future
import io
import json
from pathlib import Path
import struct
import wave

from fastapi.testclient import TestClient
import pytest

from trtmc_server.app import ServerConfig, _worker_request, create_app
from trtmc_server.errors import ModelNotFoundError, WorkerCrashedError, WorkerSaturatedError
from trtmc_server.uploads import UploadedInput


def wav(*, channels=1, width=2, frames=4):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(8000)
        audio.writeframes((struct.pack("<h", 16384) if width == 2 else b"\0" * width) * frames * channels)
    return data.getvalue()


class Session:
    replica_name = "asr-1"

    def __init__(self, registry):
        self.registry = registry

    def submit(self, operation, payload):
        path = Path(payload["audio_path"])
        self.registry.paths.append(path)
        assert path.name == "input.wav" and path.read_bytes() == self.registry.audio
        self.registry.calls.append((operation, payload["config"]))
        if self.registry.submit_error:
            raise WorkerCrashedError("private worker path")
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
        self.audio = wav()
        self.result = {"text": "Exact transcript: café.", "audio_sample_count": 4,
                       "audio_sample_rate": 8000, "model_call_ms": 1.5,
                       "timing_scope": "public_task_call_wall"}
        self.calls, self.paths = [], []
        self.retired, self.saturated, self.draining, self.submit_error = False, False, False, False
        self.closed = 0

    def start(self):
        pass

    def close(self):
        pass

    def supports(self, model, capability):
        if model == "missing":
            raise ModelNotFoundError("model is not registered")
        return model == "asr" and capability == "audio_transcription"

    def acquire(self, model):
        if self.saturated:
            raise WorkerSaturatedError("busy")
        if self.draining:
            raise RuntimeError("draining")
        return Session(self)


def client_for(registry, records=None, **limits):
    return TestClient(create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits)))


def post(client, data=None, fields=None, **kwargs):
    return client.post("/v1/audio/transcriptions", data={"model": "asr", **(fields or {})},
                       files={"file": ("../../ignored.wav", data if data is not None else wav(), "audio/wav")},
                       **kwargs)


@pytest.mark.parametrize("response_format", ["json", "text"])
def test_complete_transcript_audio_identity_and_records(tmp_path, response_format):
    registry = Registry()
    records = tmp_path / "records.jsonl"
    with client_for(registry, records) as client:
        response = post(client, fields={"response_format": response_format, "max_output_tokens": "120"},
                        headers={"X-Request-ID": "clip-1"})
    assert response.status_code == 200 and response.headers["X-Request-ID"] == "clip-1"
    assert (response.json()["text"] if response_format == "json" else response.text) == registry.result["text"]
    assert registry.calls == [("transcribe", {"max_output_tokens": 120})]
    assert registry.closed == 1 and not registry.paths[0].exists()
    row = json.loads(records.read_text())
    assert row["input_items"] == 1 and row["audio_sample_count"] == 4 and row["audio_sample_rate"] == 8000
    assert row["audio_duration_seconds"] == 4 / 8000
    assert row["transcript_bytes"] == len(registry.result["text"].encode("utf-8"))
    assert row["model_call_ms"] == 1.5 and row["timing_scope"] == "public_task_call_wall"
    assert row["completion_tokens"] is None and "café" not in records.read_text()


@pytest.mark.parametrize("fields", [{"stream": "true"}, {"language": "en"}, {"prompt": "words"},
                         {"temperature": "0"}, {"response_format": "verbose_json"},
                         {"max_output_tokens": "129"}, {"max_output_tokens": "0"}, {"audio_path": "/etc/passwd"}])
def test_unsupported_controls_are_rejected(fields):
    registry = Registry()
    with client_for(registry) as client:
        assert post(client, fields=fields).status_code == 400
    assert not registry.calls


@pytest.mark.parametrize("data", [b"", b"not a WAV", wav(channels=2), wav(width=1), wav(frames=0), wav()[:-2]])
def test_invalid_audio_is_rejected_before_native_work(data):
    registry = Registry()
    with client_for(registry) as client:
        assert post(client, data).status_code == 400
    assert not registry.calls


def test_admission_duration_and_body_failures_remain_recorded(tmp_path, monkeypatch):
    registry = Registry()
    paths = []
    original = UploadedInput.__init__

    def track(self, *args):
        original(self, *args)
        paths.append(self.path)

    monkeypatch.setattr(UploadedInput, "__init__", track)
    records = tmp_path / "records.jsonl"
    with client_for(registry, records, max_audio_seconds=0.1) as client:
        assert post(client, fields={"model": "missing"}).status_code == 404
        assert post(client, fields={"model": "text"}).status_code == 400
        registry.saturated = True
        response = post(client)
        assert response.status_code == 429 and response.headers["Retry-After"] == "1"
        registry.saturated, registry.draining = False, True
        assert post(client).status_code == 503
        assert post(client, wav(frames=2000)).status_code == 413
        assert client.post("/v1/audio/transcriptions", content=b"x" * 16001).status_code == 413
    assert paths and not any(path.exists() for path in paths) and not registry.calls
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    assert [row["status"] for row in rows] == [404, 400, 429, 503, 413, 413]
    assert all(row["audio_sample_rate"] == 8000 and row["input_items"] == 1 for row in rows[:5])
    assert rows[4]["audio_duration_seconds"] == 0.25


@pytest.mark.parametrize("updates", [{"text": None}, {"audio_sample_count": True},
                         {"audio_sample_count": 3}, {"audio_sample_rate": 16000},
                         {"model_call_ms": 10**400}, {"timing_scope": "decode"}])
def test_invalid_native_output_retires_worker_and_releases_upload(updates):
    registry = Registry()
    registry.result.update(updates)
    with client_for(registry) as client:
        response = post(client)
    assert response.status_code == 502 and registry.retired and registry.closed == 1
    assert not registry.paths[0].exists() and "private" not in response.text


def test_submit_failure_releases_upload():
    registry = Registry()
    registry.submit_error = True
    with client_for(registry) as client:
        assert post(client).status_code == 503
    assert registry.closed == 1 and not registry.paths[0].exists()


def test_malformed_multipart_and_duplicate_fields_use_the_error_contract():
    registry = Registry()
    with client_for(registry) as client:
        malformed = client.post("/v1/audio/transcriptions", content=b"broken", headers={
            "Content-Type": "multipart/form-data; boundary=missing"})
        assert malformed.status_code == 400 and "error" in malformed.json()
        duplicate = client.post("/v1/audio/transcriptions", files=[
            ("model", (None, "asr")), ("model", (None, "text")),
            ("file", ("audio.wav", wav(), "audio/wav"))])
        assert duplicate.status_code == 400 and "error" in duplicate.json()
    assert not registry.calls


def test_cancellation_retains_input_until_native_completion():
    async def run():
        pending = Future()
        media = UploadedInput(wav(), ".wav")

        class PendingSession:
            closed = 0

            def submit(self, *args):
                return pending

            def close(self):
                self.closed += 1

        session = PendingSession()
        task = asyncio.create_task(_worker_request(session, {}, operation="transcribe", validate=lambda _: None,
                                                  release_input=media.close))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert media.path.exists() and session.closed == 0
        pending.set_result({})
        for _ in range(10):
            await asyncio.sleep(0)
            if session.closed:
                break
        assert session.closed == 1 and not media.path.exists()

    asyncio.run(run())
