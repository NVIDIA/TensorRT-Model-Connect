# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI control plane for persistent text-generation workers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import (
    ModelNotFoundError,
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerRequestTooLargeError,
    WorkerRemoteError,
    WorkerSaturatedError,
    WorkerTimeoutError,
)
from .metrics import Metrics
from .protocol import chat_prompt, extract_result, generation_config, public_worker_error
from .registry import ModelRegistry
from .schemas import ChatCompletionRequest, CompletionRequest, GenerationRequest
from .worker import WorkerSession


_REQUEST_LOGGER = logging.getLogger("uvicorn.error")


class ServerConfig:
    def __init__(
        self,
        *,
        max_body_bytes: int,
        max_prompt_bytes: int,
        max_generation_tokens: int,
    ) -> None:
        self.max_body_bytes = max_body_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_generation_tokens = max_generation_tokens


class _BodyLimitMiddleware:
    """Reject oversized generation bodies before FastAPI parses them."""

    def __init__(self, app: ASGIApp, *, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/completions",
            "/v1/chat/completions",
        }:
            await self.app(scope, receive, send)
            return
        for name, raw_value in scope.get("headers", ()):
            if name.lower() == b"content-length":
                try:
                    if int(raw_value) > self.limit:
                        await error_response(
                            413, "content_too_large", "request body exceeds limit"
                        )(scope, receive, send)
                        return
                except ValueError:
                    pass
                break
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await error_response(
                413, "content_too_large", "request body exceeds limit"
            )(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    param: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    headers = {"X-Request-ID": request_id} if request_id else None
    if status >= 500:
        error_type = "server_error"
    elif status == 429:
        error_type = "rate_limit_error"
    else:
        error_type = "invalid_request_error"
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers,
    )


def create_app(registry: ModelRegistry, config: ServerConfig) -> FastAPI:
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(registry.start)
        try:
            yield
        finally:
            await asyncio.to_thread(registry.close)

    app = FastAPI(title="TensorRT-Model-Connect Text Server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(_BodyLimitMiddleware, limit=config.max_body_bytes)

    @app.middleware("http")
    async def policy(request: Request, call_next: Any) -> Any:
        request_id = f"req-{uuid.uuid4().hex}"
        request.state.request_id = request_id
        started = time.monotonic()
        response = None
        try:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > config.max_body_bytes:
                        response = error_response(
                            413, "content_too_large", "request body exceeds limit"
                        )
                        return response
                except ValueError:
                    response = error_response(
                        400, "invalid_request", "invalid Content-Length header"
                    )
                    return response
            response = await call_next(request)
            return response
        finally:
            status = response.status_code if response is not None else 500
            if response is not None:
                response.headers.setdefault("X-Request-ID", request_id)
            _REQUEST_LOGGER.info(
                json.dumps(
                    {
                        "duration_seconds": round(time.monotonic() - started, 6),
                        "event": "http_request",
                        "request_id": request_id,
                        "route": request.url.path,
                        "status": status,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        location = first.get("loc", ())
        param = str(location[-1]) if location else None
        return error_response(
            400,
            "invalid_request",
            str(first.get("msg", "request validation failed")),
            param=param,
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        status = registry.status()
        return JSONResponse(
            status_code=200 if status["ready"] else 503,
            content={
                "status": "ready" if status["ready"] else "not_ready",
                "degraded": status["degraded"],
            },
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": registry.models()}

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        status = registry.status()
        busy = sum(
            model["ready_replicas"] - model["idle_replicas"]
            for model in status["models"].values()
        )
        return PlainTextResponse(
            metrics.render(ready=status["ready"], busy=busy),
            media_type="text/plain; version=0.0.4",
        )

    async def execute(
        route: str,
        request: GenerationRequest,
        prompt: str,
        *,
        request_id: str,
        system_prompt: str = "",
        chat: bool,
        chat_max_tokens: int | None = None,
    ) -> Any:
        if request.n != 1:
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter", "n must be 1", param="n")
        if request.stream:
            metrics.reject(route, 400)
            return error_response(
                400, "streaming_not_supported", "streaming is not available", param="stream"
            )
        if request.stop is not None:
            metrics.reject(route, 400)
            return error_response(
                400, "unsupported_parameter", "stop is not supported", param="stop"
            )
        if len(prompt.encode("utf-8")) + len(system_prompt.encode("utf-8")) > config.max_prompt_bytes:
            metrics.reject(route, 413)
            return error_response(
                413, "content_too_large", "prompt exceeds the server limit", param="prompt"
            )
        requested_tokens = chat_max_tokens if chat_max_tokens is not None else request.max_tokens
        try:
            default_tokens = registry.max_tokens(request.model, config.max_generation_tokens)
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        max_tokens = requested_tokens or default_tokens
        if max_tokens > config.max_generation_tokens:
            metrics.reject(route, 400)
            return error_response(
                400,
                "max_tokens_exceeded",
                f"requested generation exceeds {config.max_generation_tokens} tokens",
                param="max_tokens",
            )

        queued_at = time.monotonic()
        try:
            session = registry.acquire(request.model)
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        except WorkerSaturatedError:
            metrics.reject(route, 429)
            response = error_response(
                429, "server_busy", "all model worker replicas are busy", request_id=request_id
            )
            response.headers["Retry-After"] = "1"
            return response
        except RuntimeError:
            metrics.reject(route, 503)
            return error_response(
                503, "server_draining", "server is not accepting requests", request_id=request_id
            )

        queue_seconds = time.monotonic() - queued_at
        worker_config = generation_config(request, max_tokens)
        worker_config["use_chat_template"] = chat
        if system_prompt:
            worker_config["system_prompt"] = system_prompt
        metrics.begin()
        inference_started = time.monotonic()
        try:
            result = await _worker_request(
                session, {"prompt": prompt, "config": worker_config}
            )
            text, completion_tokens, timings = extract_result(result)
        except asyncio.CancelledError:
            metrics.finish(
                route,
                499,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            raise
        except WorkerTimeoutError as error:
            metrics.finish(
                route,
                504,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(504, error.code, public_worker_error(error), request_id=request_id)
        except WorkerCrashedError as error:
            metrics.finish(
                route,
                503,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(503, error.code, public_worker_error(error), request_id=request_id)
        except WorkerRemoteError as error:
            details = error.details
            invalid = isinstance(details, Mapping) and details.get("type") == "invalid_request_error"
            status = 400 if invalid else 502
            metrics.finish(
                route,
                status,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(
                status, error.code, public_worker_error(error), request_id=request_id
            )
        except WorkerRequestTooLargeError as error:
            metrics.finish(
                route,
                413,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(
                413,
                error.code,
                public_worker_error(error),
                request_id=request_id,
            )
        except WorkerProtocolError as error:
            metrics.finish(
                route,
                502,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(502, error.code, public_worker_error(error), request_id=request_id)

        inference_seconds = time.monotonic() - inference_started
        metrics.finish(
            route,
            200,
            queue_seconds=queue_seconds,
            inference_seconds=inference_seconds,
            timings=timings,
        )
        choice: dict[str, Any]
        if chat:
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": None,
            }
            object_name = "chat.completion"
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
        else:
            choice = {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": None,
            }
            object_name = "text_completion"
            response_id = f"cmpl-{uuid.uuid4().hex}"
        return JSONResponse(
            content={
                "id": response_id,
                "object": object_name,
                "created": int(time.time()),
                "model": request.model,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
            },
            headers={"X-Request-ID": request_id},
        )

    @app.post("/v1/completions")
    async def completions(http_request: Request, request: CompletionRequest) -> Any:
        return await execute(
            "/v1/completions",
            request,
            request.prompt,
            request_id=http_request.state.request_id,
            chat=False,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(http_request: Request, request: ChatCompletionRequest) -> Any:
        try:
            prompt, system_prompt = chat_prompt(request)
        except ValueError as error:
            metrics.reject("/v1/chat/completions", 400)
            return error_response(400, "invalid_request", str(error), param="messages")
        if request.max_tokens is not None and request.max_completion_tokens is not None:
            metrics.reject("/v1/chat/completions", 400)
            return error_response(
                400,
                "invalid_request",
                "max_tokens and max_completion_tokens are mutually exclusive",
            )
        return await execute(
            "/v1/chat/completions",
            request,
            prompt,
            request_id=http_request.state.request_id,
            system_prompt=system_prompt,
            chat=True,
            chat_max_tokens=request.max_completion_tokens,
        )

    return app


async def _worker_request(session: WorkerSession, payload: dict[str, Any]) -> Any:
    try:
        task = asyncio.wrap_future(session.submit("generate", payload))
    except BaseException:
        session.close()
        raise
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        def release(completed: asyncio.Future[Any]) -> None:
            session.close()
            try:
                completed.exception()
            except BaseException:
                pass

        task.add_done_callback(release)
        raise
    finally:
        if task.done():
            session.close()
