# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from starlette.websockets import WebSocketDisconnect

from tensorrt_model_connect.serve import realtime as realtime_module
from tensorrt_model_connect.serve.app import ServerConfig, _worker_request, create_app
from tensorrt_model_connect.serve.errors import WorkerProtocolError, WorkerRemoteError
from tensorrt_model_connect.serve.protocol import (
    extract_text,
    invalid_request_message,
    prepare_chat_prompt,
)
from tensorrt_model_connect.serve.registry import ModelRegistry, ModelSpec
from tensorrt_model_connect.serve.realtime import RealtimeTranscriptionConnection
from tensorrt_model_connect.serve.worker import WorkerProcess, WorkerSession


FAKE_TRTMC = Path(__file__).with_name("fake_serve_worker.py")


def test_private_v2_text_and_invalid_request_shapes_are_strict() -> None:
    assert extract_text({"text": "canonical"}, operation="generate") == "canonical"
    for result in ("legacy", {"transcript": "legacy"}, {"output_text": "legacy"}):
        with pytest.raises(WorkerProtocolError, match="string text field"):
            extract_text(result, operation="generate")

    def remote(error_type: str) -> WorkerRemoteError:
        return WorkerRemoteError("bad", details={"type": error_type, "message": "detail"})

    assert invalid_request_message(remote("invalid_request_error")) == "detail"
    assert invalid_request_message(remote("invalid_request")) is None
    missing_message = WorkerRemoteError(
        "private provider diagnostic",
        details={"type": "invalid_request_error", "provider_path": "/private/model"},
    )
    assert invalid_request_message(missing_message) == "The model worker rejected the request"


def test_registry_start_is_one_transaction(tmp_path: Path) -> None:
    bundle = tmp_path / "chat.bundle"
    bundle.write_bytes(b"chat")
    created = 0
    counter_lock = threading.Lock()

    def make_worker(spec: ModelSpec) -> WorkerProcess:
        nonlocal created
        time.sleep(0.05)
        with counter_lock:
            created += 1
            replica = created
        return WorkerProcess(
            name=f"{spec.name}-{replica}",
            bundle=spec.bundle,
            trtmc_binary=FAKE_TRTMC,
            startup_timeout=1,
            request_timeout=1,
        )

    registry = ModelRegistry(
        [ModelSpec("chat", bundle, "chat")],
        trtmc_binary=FAKE_TRTMC,
        model_replicas={"chat": 2},
        startup_timeout=1,
        request_timeout=1,
        worker_factory=make_worker,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [executor.submit(registry.start) for _ in range(2)]
            for start in starts:
                start.result()
        assert created == 2
        assert registry.status()["models"]["chat"]["ready_replicas"] == 2
    finally:
        registry.close()


def test_registry_rejects_mismatched_replicas_and_rolls_back(tmp_path: Path) -> None:
    canonical = tmp_path / "asr.bundle"
    incompatible = tmp_path / "protocol-v1-asr.bundle"
    canonical.write_bytes(b"asr")
    incompatible.write_bytes(b"asr")
    workers: list[WorkerProcess] = []

    def make_worker(spec: ModelSpec) -> WorkerProcess:
        bundle = canonical if not workers else incompatible
        worker = WorkerProcess(
            name=f"{spec.name}-{len(workers) + 1}",
            bundle=bundle,
            trtmc_binary=FAKE_TRTMC,
            startup_timeout=1,
            request_timeout=1,
        )
        workers.append(worker)
        return worker

    registry = ModelRegistry(
        [ModelSpec("asr", canonical, "transcription")],
        trtmc_binary=FAKE_TRTMC,
        model_replicas={"asr": 2},
        startup_timeout=1,
        request_timeout=1,
        worker_factory=make_worker,
    )
    with pytest.raises(WorkerProtocolError, match="unsupported protocol version"):
        registry.start()
    assert not registry.ready
    assert len(workers) == 2
    assert all(worker.state == "closed" for worker in workers)


def wav_fixture(samples: bytes = b"\x01\x00\x02\x00", *, sample_width: int = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(sample_width)
        wav.setframerate(16000)
        wav.writeframes(samples)
    return output.getvalue()


def make_registry(
    tmp_path: Path,
    *,
    required_streaming: tuple[str, ...] = (),
    model_replicas: dict[str, int] | None = None,
) -> ModelRegistry:
    chat = tmp_path / "chat.bundle"
    asr = tmp_path / "asr.bundle"
    chat.write_bytes(b"chat")
    asr.write_bytes(b"asr")
    return ModelRegistry(
        [
            ModelSpec("chat", chat, "chat"),
            ModelSpec("asr", asr, "transcription"),
        ],
        trtmc_binary=FAKE_TRTMC,
        startup_timeout=1,
        request_timeout=1,
        model_replicas=model_replicas,
        required_streaming_transcription=required_streaming,
    )


def make_single_chat_registry(
    tmp_path: Path,
    bundle_stem: str,
    *,
    request_timeout: float,
    replicas: int = 1,
) -> ModelRegistry:
    bundle = tmp_path / f"{bundle_stem}.bundle"
    bundle.write_bytes(b"chat")
    return ModelRegistry(
        [ModelSpec("chat", bundle, "chat")],
        trtmc_binary=FAKE_TRTMC,
        startup_timeout=1,
        request_timeout=request_timeout,
        model_replicas={"chat": replicas},
    )


def make_single_asr_registry(
    tmp_path: Path,
    bundle_stem: str,
    *,
    require_streaming: bool = False,
    replicas: int = 1,
) -> ModelRegistry:
    bundle = tmp_path / f"{bundle_stem}.bundle"
    bundle.write_bytes(b"asr")
    return ModelRegistry(
        [ModelSpec("asr", bundle, "transcription")],
        trtmc_binary=FAKE_TRTMC,
        startup_timeout=1,
        request_timeout=1,
        model_replicas={"asr": replicas},
        required_streaming_transcription=("asr",) if require_streaming else (),
    )


def authorization() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def chat_request(prompt: str, **options: object) -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        **options,
    }


def chat_text(response: Response) -> str:
    return str(response.json()["choices"][0]["message"]["content"])


def test_worker_request_releases_lane_when_submit_fails() -> None:
    class FailingSession:
        closed = False

        def submit(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("executor is unavailable")

        def close(self) -> None:
            self.closed = True

    session = FailingSession()
    with pytest.raises(RuntimeError, match="executor is unavailable"):
        asyncio.run(_worker_request(session, "generate", {}))
    assert session.closed


def test_health_readiness_models_and_bearer_auth(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path), config=ServerConfig(api_key="test-token"))
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 401

        readiness = client.get("/readyz", headers=authorization())
        assert readiness.status_code == 200
        readiness_payload = readiness.json()
        assert readiness_payload["ready"] is True
        assert set(readiness_payload["models"]) == {"chat", "asr"}
        for model_status in readiness_payload["models"].values():
            assert set(model_status) == {
                "ready",
                "replicas",
                "ready_replicas",
                "idle_replicas",
                "busy",
            }
            assert {"pid", "pids", "returncode", "error"}.isdisjoint(model_status)

        denied = client.get("/v1/models")
        assert denied.status_code == 401
        assert denied.headers["www-authenticate"] == "Bearer"

        non_ascii = client.get(
            "/v1/models",
            headers=[(b"authorization", b"Bearer \xff")],
        )
        assert non_ascii.status_code == 401

        response = client.get("/v1/models", headers=authorization())
        assert response.status_code == 200
        models = response.json()["data"]
        assert [model["id"] for model in models] == ["chat", "asr"]
        assert models[0]["capabilities"] == ["chat"]
        assert models[0]["metadata"] == {
            "default_max_new_tokens": 64,
            "max_cache_length": 2048,
            "streaming_transcription": False,
        }
        assert models[1]["metadata"] == {
            "default_max_new_tokens": 64,
            "max_cache_length": 2048,
            "streaming_transcription": False,
        }

        preflight = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "null"
        assert "Authorization" in preflight.headers["access-control-allow-headers"]

        local_origin = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://127.0.0.1:4173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert local_origin.status_code == 200
        assert local_origin.headers["access-control-allow-origin"] == ("http://127.0.0.1:4173")

        denied_origin = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert denied_origin.status_code == 400
        assert "access-control-allow-origin" not in denied_origin.headers


def test_chat_generation_and_explicit_stream_rejection(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        assert client.get("/readyz").json() == {"status": "ready", "ready": True}
        stopped = client.post(
            "/v1/chat/completions",
            json=chat_request("hello STOP tail", stop="STOP", max_tokens=8),
        )
        assert stopped.status_code == 200
        assert stopped.json()["model"] == "chat"
        assert chat_text(stopped) == "generated:hello "
        assert stopped.json()["usage"]["total_tokens"] > 0

        thinking_disabled = client.post(
            "/v1/chat/completions",
            json=chat_request("summary", enable_thinking=False),
        )
        assert thinking_disabled.status_code == 200
        assert chat_text(thinking_disabled) == ("enable_thinking=false:generated:summary")

        chat = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Be brief"},
                    {"role": "user", "content": "Hi"},
                ]
            },
        )
        assert chat.status_code == 200
        assert (
            "--- trtmc-role:system ---\nBe brief\n"
            "--- trtmc-role:user ---\nHi\n"
            "--- trtmc-role:assistant ---" in (chat.json()["choices"][0]["message"]["content"])
        )
        assert chat.json()["trtmc"]["chat_prompt_mode"] == "role_annotated_flattened"

        thinking_enabled = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "reason"}],
                "enable_thinking": True,
            },
        )
        assert thinking_enabled.status_code == 200
        assert chat_text(thinking_enabled) == ("enable_thinking=true:generated:reason")

        streamed = client.post(
            "/v1/chat/completions",
            json=chat_request("stream", stream=True),
        )
        assert streamed.status_code == 400
        assert streamed.json()["error"]["code"] == "streaming_not_supported"

    flattened = prepare_chat_prompt(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
        ]
    )
    assert flattened[1:] == (False, "role_annotated_flattened")
    single = prepare_chat_prompt([{"role": "user", "content": "question"}])
    assert single == ("question", True, "single_user_template")


def test_chat_rejects_empty_messages(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "messages"


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n", 2),
        ("best_of", 2),
        ("logprobs", True),
        ("top_logprobs", 2),
        ("frequency_penalty", 0.5),
        ("presence_penalty", 0.5),
        ("logit_bias", {"42": 1}),
        ("ignore_eos", True),
        ("tools", [{"type": "function"}]),
        ("tool_choice", "auto"),
        ("parallel_tool_calls", True),
        ("response_format", {"type": "json_object"}),
        ("stream_options", {"include_usage": True}),
        ("functions", [{"name": "lookup"}]),
        ("function_call", "auto"),
        ("modalities", ["text", "audio"]),
        ("audio", {"format": "wav"}),
        ("prediction", {"type": "content", "content": "expected"}),
        ("reasoning_effort", "high"),
        ("future_semantic_option", True),
    ],
)
def test_chat_rejects_meaningful_unsupported_parameters(
    tmp_path: Path,
    parameter: str,
    value: object,
) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat_request("hello", **{parameter: value}),
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": f"{parameter} is not supported",
        "type": "invalid_request_error",
        "param": parameter,
        "code": "unsupported_parameter",
    }


def test_chat_accepts_no_op_and_metadata_fields(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat_request(
                "hello",
                n=1,
                best_of=1,
                logprobs=False,
                top_logprobs=0,
                frequency_penalty=0,
                presence_penalty=0,
                logit_bias={},
                ignore_eos=False,
                tools=[],
                tool_choice="none",
                parallel_tool_calls=False,
                response_format={"type": "text"},
                stream_options={},
                user="client-metadata",
                metadata={"request_class": "interactive"},
            ),
        )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "tool output"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "unsupported": True}],
        },
        {"role": "user", "content": "hello", "name": "named-participant"},
    ],
)
def test_chat_rejects_unsupported_message_semantics(
    tmp_path: Path,
    message: dict[str, object],
) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [message]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "messages"


@pytest.mark.parametrize("part_type", [[], {}, None, 7])
def test_chat_rejects_non_string_content_part_types(
    tmp_path: Path,
    part_type: object,
) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": part_type, "text": "hello"}],
                    }
                ]
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"


def test_model_capability_and_unknown_model_errors_are_openai_shaped(
    tmp_path: Path,
) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        wrong_kind = client.post("/v1/chat/completions", json=chat_request("hello", model="asr"))
        assert wrong_kind.status_code == 400
        assert wrong_kind.json()["error"]["code"] == "model_capability_mismatch"

        missing = client.post("/v1/chat/completions", json=chat_request("hello", model="missing"))
        assert missing.status_code == 404
        assert missing.json()["error"]["param"] == "model"

        invalid = client.post("/v1/chat/completions", json={"max_tokens": 1})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_request"
        assert invalid.json()["error"]["param"] == "messages"


def test_oversized_prompt_is_rejected_before_worker_protocol(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    app = create_app(registry, config=ServerConfig(max_prompt_bytes=5))
    with TestClient(app) as client:
        oversized = client.post("/v1/chat/completions", json=chat_request("123456"))
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "prompt_too_large"

        healthy = client.post("/v1/chat/completions", json=chat_request("ok"))
        assert healthy.status_code == 200
        assert registry.ready


def test_generation_token_hard_cap_applies_to_explicit_and_default_requests(
    tmp_path: Path,
) -> None:
    app = create_app(make_registry(tmp_path), config=ServerConfig(max_generation_tokens=4))
    with TestClient(app) as client:
        rejected = client.post("/v1/chat/completions", json=chat_request("hello", max_tokens=5))
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "max_tokens_exceeded"

        bounded_default = client.post("/v1/chat/completions", json=chat_request("hello"))
        assert bounded_default.status_code == 200
        assert bounded_default.json()["trtmc"]["effective_max_tokens"] == 4


def test_streaming_probe_controls_exposed_capability_and_startup(tmp_path: Path) -> None:
    registry = make_registry(tmp_path, required_streaming=("asr",))
    app = create_app(registry)
    with TestClient(app) as client:
        models = client.get("/v1/models").json()["data"]
        asr = next(model for model in models if model["id"] == "asr")
        assert asr["metadata"]["streaming_transcription"] is True

    unsupported = make_single_asr_registry(tmp_path, "no-stream-asr", require_streaming=True)
    with pytest.raises(WorkerRemoteError, match="streaming transcription is unavailable"):
        unsupported.start()
    assert not unsupported.ready

    old_protocol = make_single_asr_registry(tmp_path, "protocol-v1-asr")
    with pytest.raises(WorkerProtocolError, match="unsupported protocol version"):
        old_protocol.start()
    assert not old_protocol.ready


def test_saturated_worker_returns_429_without_waiting_for_active_request(
    tmp_path: Path,
) -> None:
    registry = make_single_chat_registry(
        tmp_path,
        "saturation-chat",
        request_timeout=1,
    )
    app = create_app(registry)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(client.post, "/v1/chat/completions", json=chat_request("first"))
        deadline = time.monotonic() + 1
        while not registry.status()["models"]["chat"]["busy"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert registry.status()["models"]["chat"]["busy"]
        assert client.get("/healthz").status_code == 200
        saturated = client.post("/v1/chat/completions", json=chat_request("second"))
        assert saturated.status_code == 429
        assert not active.done()
        assert saturated.json()["error"]["code"] == "server_busy"
        assert saturated.headers["retry-after"] == "1"
        assert active.result().status_code == 200


def test_model_replicas_execute_requests_in_parallel_and_bound_overload(
    tmp_path: Path,
) -> None:
    registry = make_single_chat_registry(
        tmp_path,
        "saturation-chat",
        request_timeout=1,
        replicas=2,
    )
    app = create_app(registry)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        active = [
            executor.submit(client.post, "/v1/chat/completions", json=chat_request(prompt))
            for prompt in ("first", "second")
        ]
        deadline = time.monotonic() + 1
        while registry.status()["models"]["chat"]["idle_replicas"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert registry.status()["models"]["chat"]["idle_replicas"] == 0

        saturated = client.post("/v1/chat/completions", json=chat_request("third"))
        assert saturated.status_code == 429
        assert saturated.json()["error"]["code"] == "server_busy"
        assert [response.result().status_code for response in active] == [200, 200]


def test_health_remains_ok_while_one_replica_can_serve(tmp_path: Path) -> None:
    registry = make_single_chat_registry(
        tmp_path,
        "crash-request-chat",
        request_timeout=1,
        replicas=2,
    )
    app = create_app(registry)
    with TestClient(app) as client:
        failed = client.post("/v1/chat/completions", json=chat_request("trigger"))
        assert failed.status_code == 503
        assert registry.status()["models"]["chat"]["ready_replicas"] == 1
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200


def test_health_and_registry_readiness_have_distinct_failure_scopes(tmp_path: Path) -> None:
    chat = tmp_path / "crash-request-chat.bundle"
    asr = tmp_path / "healthy-asr.bundle"
    chat.write_bytes(b"chat")
    asr.write_bytes(b"asr")
    registry = ModelRegistry(
        [
            ModelSpec("chat", chat, "chat"),
            ModelSpec("asr", asr, "transcription"),
        ],
        trtmc_binary=FAKE_TRTMC,
        request_timeout=1,
    )
    app = create_app(registry)
    with TestClient(app) as client:
        failed = client.post("/v1/chat/completions", json=chat_request("trigger"))
        assert failed.status_code == 503
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize(
    (
        "bundle_stem",
        "expected_status",
        "expected_code",
        "expected_message",
        "private_detail",
    ),
    [
        (
            "slow-request-chat",
            504,
            "worker_timeout",
            "The model worker timed out",
            "slow-request-chat",
        ),
        (
            "crash-request-chat",
            503,
            "worker_crashed",
            "The model worker is unavailable",
            "intentional fake crash",
        ),
    ],
)
def test_worker_failures_are_http_errors_and_clear_readiness(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    bundle_stem: str,
    expected_status: int,
    expected_code: str,
    expected_message: str,
    private_detail: str,
) -> None:
    registry = make_single_chat_registry(tmp_path, bundle_stem, request_timeout=0.05)
    app = create_app(registry, config=ServerConfig(api_key="test-token"))
    with TestClient(app) as client:
        failed = client.post(
            "/v1/chat/completions",
            json=chat_request("trigger"),
            headers=authorization(),
        )
        assert failed.status_code == expected_status
        public_error = failed.json()["error"]
        assert public_error["code"] == expected_code
        assert public_error["message"] == expected_message
        assert private_detail not in failed.text
        assert any(record.message == "Model worker request failed" for record in caplog.records)
        assert private_detail not in caplog.text
        assert "worker-secret" not in caplog.text

        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json() == {"status": "unavailable"}
        assert private_detail not in health.text

        readiness = client.get("/readyz", headers=authorization())
        assert readiness.status_code == 503
        assert private_detail not in readiness.text
        for model_status in readiness.json()["models"].values():
            assert {"pid", "pids", "returncode", "error"}.isdisjoint(model_status)


def test_native_invalid_request_maps_to_400_without_crashing_worker(
    tmp_path: Path,
) -> None:
    registry = make_single_chat_registry(tmp_path, "invalid-request-chat", request_timeout=1)
    app = create_app(registry)
    with TestClient(app) as client:
        failed = client.post("/v1/chat/completions", json=chat_request("trigger"))
        assert failed.status_code == 400
        assert failed.json()["error"]["code"] == "invalid_request"
        assert "invalid fake generation request" in failed.json()["error"]["message"]
        assert client.get("/readyz").status_code == 200


def test_multipart_audio_transcription_json_text_and_empty_rejection(
    tmp_path: Path,
) -> None:
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        audio = wav_fixture()
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", audio, "audio/wav")},
            data={"model": "asr"},
        )
        assert response.status_code == 200
        assert response.json() == {"text": f"transcribed {len(audio)} bytes"}

        shorter_audio = wav_fixture(b"\x01\x00")
        text = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", shorter_audio, "audio/wav")},
            data={"response_format": "text"},
        )
        assert text.status_code == 200
        assert text.text == f"transcribed {len(shorter_audio)} bytes"

        empty = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"", "audio/wav")},
        )
        assert empty.status_code == 400
        assert empty.json()["error"]["code"] == "empty_audio"

        unsupported = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.mp3", b"not an mp3 either", "audio/mpeg")},
        )
        assert unsupported.status_code == 415
        assert unsupported.json()["error"]["code"] == "unsupported_media_type"
        assert unsupported.json()["error"]["param"] == "file"

        unsupported_wav = client.post(
            "/v1/audio/transcriptions",
            files={
                "file": (
                    "sample.wav",
                    wav_fixture(b"\x01\x02", sample_width=1),
                    "audio/wav",
                )
            },
        )
        assert unsupported_wav.status_code == 415
        assert "PCM16 or IEEE float32" in unsupported_wav.json()["error"]["message"]


def test_verbose_transcription_response_has_a_fixed_public_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def native_result(
        session: WorkerSession,
        _operation: str,
        _payload: object,
    ) -> object:
        session.close()
        return {
            "text": "public transcript",
            "model_path": "/private/models/asr.bundle",
            "token_ids": [11, 12],
            "setup_ms": 1.25,
            "prefill_ms": 2.5,
            "decode_ms": 3.75,
            "segments": [
                {
                    "start_seconds": 0,
                    "end_seconds": 1.5,
                    "text": "public transcript",
                    "token_ids": [11, 12],
                    "provider_debug": "private provider state",
                }
            ],
        }

    monkeypatch.setattr("tensorrt_model_connect.serve.app._worker_request", native_result)
    app = create_app(make_registry(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", wav_fixture(), "audio/wav")},
            data={"model": "asr", "response_format": "verbose_json"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "public transcript",
        "model": "asr",
        "segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.5,
                "text": "public transcript",
            }
        ],
    }
    for private_field in (
        "model_path",
        "token_ids",
        "setup_ms",
        "prefill_ms",
        "decode_ms",
        "provider_debug",
    ):
        assert private_field not in response.text


def test_realtime_transcription_events_use_cumulative_transcript(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path), config=ServerConfig(api_key="test-token"))
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime?intent=transcription&access_token=test-token"
        ) as websocket:
            created = websocket.receive_json()
            assert created["type"] == "session.created"
            assert created["session"]["model"] == "asr"

            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "model": "asr",
                        "input_audio_format": "pcm16",
                        "trtmc": {"sample_rate_hz": 16000},
                    },
                }
            )
            updated = websocket.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["session"]["trtmc"]["sample_rate_hz"] == 16000

            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x01\x00\x02\x00").decode(),
                }
            )
            delta = websocket.receive_json()
            assert delta["type"] == ("conversation.item.input_audio_transcription.delta")
            assert delta["transcript"] == "2 samples"

            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x03\x00").decode(),
                }
            )
            second_delta = websocket.receive_json()
            assert second_delta["transcript"] == "3 samples"

            websocket.send_json({"type": "input_audio_buffer.commit"})
            completed = websocket.receive_json()
            assert completed["type"] == ("conversation.item.input_audio_transcription.completed")
            assert completed["transcript"] == "3 samples"

            websocket.send_json({"type": "input_audio_buffer.clear"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.cleared"


def test_realtime_rejects_bad_token_and_invalid_audio(tmp_path: Path) -> None:
    app = create_app(make_registry(tmp_path), config=ServerConfig(api_key="test-token"))
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect("/v1/realtime?intent=transcription"):
                pass
        assert denied.value.code == 4401

        with pytest.raises(WebSocketDisconnect) as non_ascii_token:
            with client.websocket_connect("/v1/realtime?intent=transcription&access_token=%C3%BF"):
                pass
        assert non_ascii_token.value.code == 4401

        with pytest.raises(WebSocketDisconnect) as denied_origin:
            with client.websocket_connect(
                "/v1/realtime?intent=transcription&access_token=test-token",
                headers={"Origin": "https://example.com"},
            ):
                pass
        assert denied_origin.value.code == 4403

        with client.websocket_connect(
            "/v1/realtime?intent=transcription&access_token=test-token",
            headers={"Origin": "http://localhost:4173"},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": "not base64"})
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "invalid_audio"


def test_realtime_handles_distinct_asyncio_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyAsyncioTimeoutError(Exception):
        pass

    class TimeoutWebSocket:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

        async def receive_json(self) -> dict[str, object]:
            raise LegacyAsyncioTimeoutError

    class Registry:
        default_transcription_model = "asr"

    websocket = TimeoutWebSocket()
    monkeypatch.setattr(
        realtime_module,
        "_TIMEOUT_ERRORS",
        (TimeoutError, LegacyAsyncioTimeoutError),
    )
    connection = RealtimeTranscriptionConnection(
        websocket,  # type: ignore[arg-type]
        Registry(),  # type: ignore[arg-type]
        idle_timeout_seconds=1,
    )

    asyncio.run(connection.run())

    assert websocket.events[0]["type"] == "session.created"
    assert websocket.events[1]["error"]["code"] == "session_idle_timeout"  # type: ignore[index]


def test_realtime_worker_diagnostics_are_not_returned_to_clients(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(make_single_asr_registry(tmp_path, "crash-asr"))
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x01\x00").decode(),
                }
            )
            failure = websocket.receive_json()

    assert failure["type"] == "conversation.item.input_audio_transcription.failed"
    assert failure["error"]["code"] == "worker_crashed"
    assert failure["error"]["message"] == "The model worker is unavailable"
    assert "intentional fake crash" not in str(failure)
    assert any(
        record.message == "Realtime model worker request failed" for record in caplog.records
    )
    assert "intentional fake crash" not in caplog.text
    assert "worker-secret" not in caplog.text


def test_realtime_chunk_error_resets_native_session_for_next_turn(tmp_path: Path) -> None:
    registry = make_single_asr_registry(tmp_path, "chunk-error-asr")
    app = create_app(registry)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription") as websocket:
            websocket.receive_json()
            audio = base64.b64encode(b"\x01\x00").decode()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "invalid_request"
            assert error["error"]["message"] == "invalid fake audio chunk"

            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            recovered = websocket.receive_json()
            assert recovered["type"] == ("conversation.item.input_audio_transcription.delta")
            assert recovered["transcript"] == "1 samples"
            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == (
                "conversation.item.input_audio_transcription.completed"
            )


def test_realtime_sessions_lease_distinct_replicas_and_bound_overload(
    tmp_path: Path,
) -> None:
    registry = make_single_asr_registry(tmp_path, "asr-replicas", replicas=2)
    app = create_app(registry)
    audio = base64.b64encode(b"\x01\x00").decode()
    with TestClient(app) as client:
        with (
            client.websocket_connect("/v1/realtime?intent=transcription") as first,
            client.websocket_connect("/v1/realtime?intent=transcription") as second,
            client.websocket_connect("/v1/realtime?intent=transcription") as third,
        ):
            for websocket in (first, second, third):
                assert websocket.receive_json()["type"] == "session.created"
            for websocket in (first, second):
                websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
                assert websocket.receive_json()["type"].endswith(".delta")

            third.send_json({"type": "input_audio_buffer.append", "audio": audio})
            rejected = third.receive_json()
            assert rejected["type"] == "error"
            assert rejected["error"]["code"] == "server_busy"

            for websocket in (first, second):
                websocket.send_json({"type": "input_audio_buffer.commit"})
                assert websocket.receive_json()["type"].endswith(".completed")


def test_realtime_disconnect_during_stream_start_resets_lane(tmp_path: Path) -> None:
    registry = make_single_asr_registry(tmp_path, "asr-delayed-start")
    app = create_app(registry)
    audio = base64.b64encode(b"\x01\x00").decode()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            deadline = time.monotonic() + 1
            while not registry.status()["models"]["asr"]["busy"] and time.monotonic() < deadline:
                time.sleep(0.005)
            assert registry.status()["models"]["asr"]["busy"]

        with client.websocket_connect("/v1/realtime?intent=transcription") as recovered:
            recovered.receive_json()
            recovered.send_json({"type": "input_audio_buffer.append", "audio": audio})
            assert recovered.receive_json()["type"].endswith(".delta")
            recovered.send_json({"type": "input_audio_buffer.commit"})
            assert recovered.receive_json()["type"].endswith(".completed")


@pytest.mark.parametrize(
    ("config", "expected_code"),
    [
        ({"realtime_idle_timeout_seconds": 0.05}, "session_idle_timeout"),
        (
            {
                "realtime_idle_timeout_seconds": 1,
                "realtime_max_session_seconds": 0.05,
            },
            "session_duration_exceeded",
        ),
    ],
)
def test_realtime_time_limits_emit_structured_failure_and_cleanup(
    tmp_path: Path,
    config: dict[str, float],
    expected_code: str,
) -> None:
    # Delaying reset makes client-close cancellation overlap cleanup
    # deterministically. The worker lock must still be released before the
    # websocket context exits and the next operation starts.
    registry = make_single_asr_registry(tmp_path, f"asr-delayed-reset-{expected_code}")
    app = create_app(registry, config=ServerConfig(**config))
    audio = base64.b64encode(b"\x01\x00").decode()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            assert websocket.receive_json()["type"].endswith(".delta")
            failure = websocket.receive_json()
            assert failure["type"].endswith(".failed")
            assert failure["error"]["code"] == expected_code

        _spec, session = registry.acquire_session("transcription", "asr")
        with session:
            probe = session.request(
                "probe_transcription_stream",
                {
                    "config": {
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "audio_format": "pcm16le",
                    }
                },
            )
        assert probe["supported"] is True


def test_realtime_audio_limit_is_cumulative_across_clear(tmp_path: Path) -> None:
    registry = make_single_asr_registry(tmp_path, "asr-byte-limit")
    app = create_app(registry, config=ServerConfig(max_realtime_session_bytes=2))
    audio = base64.b64encode(b"\x01\x00").decode()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            assert websocket.receive_json()["type"].endswith(".delta")
            websocket.send_json({"type": "input_audio_buffer.clear"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.cleared"
            websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
            failure = websocket.receive_json()
            assert failure["type"].endswith(".failed")
            assert failure["error"]["code"] == "audio_session_too_large"
