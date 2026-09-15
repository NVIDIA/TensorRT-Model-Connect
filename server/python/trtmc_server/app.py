# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI control plane for persistent TensorRT-Model-Connect workers."""

from __future__ import annotations

import asyncio
import hmac
import logging
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import (
    ModelCapabilityError,
    ModelNotFoundError,
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerRemoteError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
    WorkerTimeoutError,
)
from .protocol import (
    extract_transcription_segments,
    extract_text,
    extract_usage,
    invalid_request_message,
    is_text_only_content,
    prepare_chat_prompt,
    public_worker_error_message,
)
from .realtime import RealtimeTranscriptionConnection
from .registry import ModelRegistry
from .schemas import ChatCompletionRequest, model_to_dict
from .worker import WorkerSession


_LOGGER = logging.getLogger(__name__)
_HTTP_ENVELOPE_OVERHEAD_BYTES = 64 * 1024


class _RequestBodyTooLarge(Exception):
    pass


class _BodyLimitMiddleware:
    """Reject oversized HTTP bodies before framework parsing or spooling."""

    def __init__(self, app: ASGIApp, *, limits: Mapping[str, int]) -> None:
        self.app = app
        self.limits = dict(limits)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (limit := self.limits.get(scope["path"])) is None:
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > limit:
            await _request_body_too_large()(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await _request_body_too_large()(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, raw_value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        try:
            value = int(raw_value)
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


def _request_body_too_large() -> JSONResponse:
    return _error_response(
        413,
        "request_body_too_large",
        "Request body exceeds the server ingress limit",
        headers={"Connection": "close"},
    )


@dataclass(frozen=True)
class ServerConfig:
    """HTTP-layer policy independent of model process configuration."""

    api_key: str | None = None
    max_generation_tokens: int = 4096
    max_prompt_bytes: int = 8 * 1024 * 1024
    max_upload_bytes: int = 512 * 1024 * 1024
    max_realtime_chunk_bytes: int = 1024 * 1024
    max_realtime_session_bytes: int = 512 * 1024 * 1024
    realtime_idle_timeout_seconds: float = 30.0
    realtime_max_session_seconds: float = 4 * 60 * 60

    def __post_init__(self) -> None:
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key cannot be empty")
        if (
            self.max_prompt_bytes <= 0
            or self.max_generation_tokens <= 0
            or self.max_upload_bytes <= 0
            or self.max_realtime_chunk_bytes <= 0
            or self.max_realtime_session_bytes <= 0
            or self.realtime_idle_timeout_seconds <= 0
            or self.realtime_max_session_seconds <= 0
        ):
            raise ValueError("server limits must be positive")


def create_app(
    registry: ModelRegistry,
    *,
    config: ServerConfig | None = None,
) -> FastAPI:
    """Create an application whose lifespan owns the supplied registry."""

    server_config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(registry.start)
        try:
            yield
        finally:
            await asyncio.to_thread(registry.close)

    app = FastAPI(
        title="TensorRT-Model-Connect Serve",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        _BodyLimitMiddleware,
        limits={
            "/v1/chat/completions": (
                server_config.max_prompt_bytes + _HTTP_ENVELOPE_OVERHEAD_BYTES
            ),
            "/v1/audio/transcriptions": (
                server_config.max_upload_bytes + _HTTP_ENVELOPE_OVERHEAD_BYTES
            ),
        },
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null", "http://localhost", "http://127.0.0.1"],
        allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1)(?::[0-9]{1,5})?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if (
            server_config.api_key is not None
            and request.url.path != "/healthz"
            and request.method != "OPTIONS"
            and not _valid_authorization(
                request.headers.get("authorization"), server_config.api_key
            )
        ):
            return _error_response(
                401,
                "invalid_api_key",
                "Missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _error_response(404, exc.code, str(exc), param="model")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = first.get("loc", ())
        param = str(location[-1]) if location else None
        message = str(first.get("msg") or "request validation failed")
        return _error_response(422, "invalid_request", message, param=param)

    @app.exception_handler(ModelCapabilityError)
    async def model_capability(_request: Request, exc: ModelCapabilityError) -> JSONResponse:
        return _error_response(400, exc.code, str(exc), param="model")

    @app.exception_handler(WorkerTimeoutError)
    async def worker_timeout(_request: Request, exc: WorkerTimeoutError) -> JSONResponse:
        _log_worker_failure()
        return _error_response(
            504,
            exc.code,
            public_worker_error_message(exc),
            error_type="server_error",
        )

    @app.exception_handler(WorkerCrashedError)
    async def worker_crashed(_request: Request, exc: WorkerCrashedError) -> JSONResponse:
        _log_worker_failure()
        return _error_response(
            503,
            exc.code,
            public_worker_error_message(exc),
            error_type="server_error",
        )

    @app.exception_handler(WorkerRemoteError)
    async def worker_remote(_request: Request, exc: WorkerRemoteError) -> JSONResponse:
        message = invalid_request_message(exc)
        if message is not None:
            details = exc.details
            if (
                isinstance(details, Mapping)
                and details.get("code") == "unsupported_media_type"
                and details.get("param") == "file"
            ):
                return _error_response(
                    415,
                    "unsupported_media_type",
                    message,
                    param="file",
                )
            return _error_response(
                400,
                "invalid_request",
                message,
            )
        _log_worker_failure()
        return _error_response(
            502,
            exc.code,
            public_worker_error_message(exc),
            error_type="server_error",
        )

    @app.exception_handler(WorkerRequestTooLargeError)
    async def worker_request_too_large(
        _request: Request, exc: WorkerRequestTooLargeError
    ) -> JSONResponse:
        return _error_response(413, exc.code, public_worker_error_message(exc))

    @app.exception_handler(WorkerSaturatedError)
    async def worker_saturated(_request: Request, exc: WorkerSaturatedError) -> JSONResponse:
        return _error_response(
            429,
            exc.code,
            public_worker_error_message(exc),
            error_type="rate_limit_error",
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(WorkerProtocolError)
    async def worker_protocol(_request: Request, exc: WorkerProtocolError) -> JSONResponse:
        _log_worker_failure()
        return _error_response(
            502,
            exc.code,
            public_worker_error_message(exc),
            error_type="server_error",
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        available = registry.has_healthy_worker
        degraded = registry.public_status()["degraded"]
        return JSONResponse(
            status_code=200 if available else 503,
            content={
                "status": "ok" if available else "unavailable",
                "degraded": degraded,
            },
        )

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        status = registry.public_status()
        code = 200 if status["ready"] else 503
        content: dict[str, Any] = {
            "status": "ready" if code == 200 else "not_ready",
            "ready": status["ready"],
            "degraded": status["degraded"],
        }
        if server_config.api_key is not None:
            content.update(status)
        return JSONResponse(
            status_code=code,
            content=content,
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": registry.list_models()}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        if request.stream:
            return _error_response(
                400,
                "streaming_not_supported",
                "text token streaming is not available",
                param="stream",
            )
        raw_messages = [model_to_dict(message) for message in request.messages]
        unsupported = _unsupported_chat_parameter(request, raw_messages)
        if unsupported is not None:
            return _error_response(
                400,
                "unsupported_parameter",
                f"{unsupported} is not supported",
                param=unsupported,
            )
        prompt = prepare_chat_prompt(raw_messages)
        if len(prompt.encode("utf-8")) > server_config.max_prompt_bytes:
            return _error_response(
                413,
                "prompt_too_large",
                f"rendered messages exceed {server_config.max_prompt_bytes} UTF-8 bytes",
                param="messages",
            )
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens is not None and max_tokens > server_config.max_generation_tokens:
            return _error_response(
                400,
                "max_tokens_exceeded",
                "requested generation exceeds the server hard cap of "
                f"{server_config.max_generation_tokens} tokens",
                param="max_tokens",
            )
        spec, session = registry.acquire_session("chat", request.model)
        max_tokens = _effective_generation_tokens(
            registry,
            spec.name,
            max_tokens,
            server_config.max_generation_tokens,
        )
        result = await _worker_request(
            session,
            "generate",
            {
                "prompt": prompt,
                "config": _generation_config(
                    request,
                    max_tokens=max_tokens,
                ),
            },
        )
        text = extract_text(result, operation="generate")
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": spec.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": extract_usage(result),
            "trtmc": {
                "effective_max_tokens": max_tokens,
            },
        }

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        response_format: str = Form(default="json"),
    ) -> Any:
        spec = registry.resolve_model("transcription", model)
        if response_format not in {"json", "text", "verbose_json"}:
            return _error_response(
                400,
                "unsupported_response_format",
                "response_format must be json, verbose_json, or text",
                param="response_format",
            )

        suffix = Path(file.filename or "audio.bin").suffix[:16]
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="trtmc-serve-audio-", suffix=suffix, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                total = 0
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > server_config.max_upload_bytes:
                        return _error_response(
                            413,
                            "audio_file_too_large",
                            f"audio upload exceeds {server_config.max_upload_bytes} bytes",
                            param="file",
                        )
                    temporary.write(chunk)
            if total == 0:
                return _error_response(
                    400, "empty_audio", "uploaded audio file is empty", param="file"
                )
            transcription_config: dict[str, Any] = {}
            if language:
                transcription_config["language"] = language
            _spec, session = registry.acquire_session("transcription", spec.name)
            result = await _worker_request(
                session,
                "transcribe",
                {
                    "audio_path": str(temporary_path),
                    "config": transcription_config,
                },
            )
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        text = extract_text(result, operation="transcribe")
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json" and isinstance(result, Mapping):
            return {
                "text": text,
                "model": spec.name,
                "segments": extract_transcription_segments(result),
            }
        return {"text": text}

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if not _valid_websocket_origin(websocket.headers.get("origin")):
            await websocket.close(code=4403, reason="websocket origin is not allowed")
            return
        if websocket.query_params.get("intent") != "transcription":
            await websocket.close(code=4400, reason="intent must be transcription")
            return
        if server_config.api_key is not None and not _valid_websocket_token(
            websocket, server_config.api_key
        ):
            await websocket.close(code=4401, reason="missing or invalid access token")
            return
        await websocket.accept()
        connection = RealtimeTranscriptionConnection(
            websocket,
            registry,
            max_audio_chunk_bytes=server_config.max_realtime_chunk_bytes,
            max_session_audio_bytes=server_config.max_realtime_session_bytes,
            idle_timeout_seconds=server_config.realtime_idle_timeout_seconds,
            max_session_seconds=server_config.realtime_max_session_seconds,
        )
        await connection.run()

    return app


async def _worker_request(
    session: WorkerSession,
    op: str,
    payload: Mapping[str, Any],
) -> Any:
    try:
        task = asyncio.wrap_future(session.submit(op, payload))
    except BaseException:
        session.close()
        raise
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:

        def release_when_done(completed: asyncio.Future[Any]) -> None:
            session.close()
            try:
                completed.exception()
            except BaseException:
                pass

        task.add_done_callback(release_when_done)
        raise
    finally:
        if task.done():
            session.close()


def _generation_config(
    request: ChatCompletionRequest,
    *,
    max_tokens: int | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"use_chat_template": True}
    if max_tokens is not None:
        config["max_new_tokens"] = max_tokens
    for name in (
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "seed",
        "enable_thinking",
    ):
        value = getattr(request, name, None)
        if value is not None:
            config[name] = value
    return config


def _effective_generation_tokens(
    registry: ModelRegistry,
    model_name: str,
    requested: int | None,
    hard_cap: int,
) -> int:
    if requested is not None:
        return requested
    default = registry.metadata_for(model_name).get("default_max_new_tokens", 128)
    if isinstance(default, bool) or not isinstance(default, int) or default <= 0:
        default = 128
    return min(default, hard_cap)


def _unsupported_chat_parameter(
    request: ChatCompletionRequest,
    messages: list[Mapping[str, Any]],
) -> str | None:
    """Reject meaningful OpenAI options that this server cannot honor."""

    payload = model_to_dict(request, exclude_none=True)
    supported = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "seed",
        "enable_thinking",
        "stream",
    }
    no_op = {
        "n",
        "best_of",
        "logprobs",
        "top_logprobs",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "ignore_eos",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "stream_options",
        "stop",
    }
    metadata = {"user", "metadata"}
    unknown = sorted(set(payload) - supported - no_op - metadata)
    if unknown:
        return unknown[0]
    for name in sorted(no_op):
        if name in payload and not _is_no_op_chat_parameter(name, payload[name]):
            return name
    if len(messages) != 1:
        return "messages"
    message = messages[0]
    if (
        set(message) != {"role", "content"}
        or message["role"] != "user"
        or not is_text_only_content(message["content"])
    ):
        return "messages"
    return None


def _is_no_op_chat_parameter(name: str, value: Any) -> bool:
    if name in {"n", "best_of"}:
        return isinstance(value, int) and not isinstance(value, bool) and value == 1
    if name == "top_logprobs":
        return isinstance(value, int) and not isinstance(value, bool) and value == 0
    if name in {"frequency_penalty", "presence_penalty"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
    if name in {"logprobs", "ignore_eos", "parallel_tool_calls"}:
        return value is False
    if name == "logit_bias":
        return isinstance(value, Mapping) and not value
    if name == "tools":
        return isinstance(value, list) and not value
    if name == "tool_choice":
        return value == "none"
    if name == "response_format":
        return value == {"type": "text"}
    if name == "stream_options":
        return isinstance(value, Mapping) and not value
    return False


def _valid_authorization(header: str | None, expected: str) -> bool:
    if header is None:
        return False
    scheme, separator, token = header.partition(" ")
    return bool(separator) and scheme.lower() == "bearer" and _tokens_match(token, expected)


def _tokens_match(candidate: str, expected: str) -> bool:
    """Compare arbitrary text tokens without leaking timing or raising on Unicode."""

    try:
        return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def _valid_websocket_token(websocket: WebSocket, expected: str) -> bool:
    query_token = websocket.query_params.get("access_token")
    if query_token is not None and _tokens_match(query_token, expected):
        return True
    return _valid_authorization(websocket.headers.get("authorization"), expected)


def _valid_websocket_origin(origin: str | None) -> bool:
    """Apply the HTTP CORS boundary to browser WebSocket handshakes."""

    if origin is None or origin == "null":
        return True
    try:
        parsed = urlsplit(origin)
        return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
    except ValueError:
        return False


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=dict(headers or {}),
    )


def _log_worker_failure() -> None:
    _LOGGER.error("Model worker request failed")
