# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI control plane for persistent text-generation workers."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
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
from .records import RequestPolicy, RequestRecords
from .registry import ModelRegistry
from .schemas import ChatCompletionRequest, CompletionRequest, GenerationRequest
from .worker import WorkerSession



class ServerConfig:
    def __init__(
        self,
        *,
        max_body_bytes: int,
        max_prompt_bytes: int,
        max_generation_tokens: int,
        records: Path | None = None,
    ) -> None:
        self.max_body_bytes = max_body_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_generation_tokens = max_generation_tokens
        self.records = records


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
                    await error_response(400, "invalid_request", "invalid Content-Length header")(scope, receive, send)
                    return
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


def _sse_data(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def _streaming_completion_response(
    *,
    response_id: str,
    created: int,
    model: str,
    text: str,
    completion_tokens: int,
    chat: bool,
    include_usage: bool,
    request_id: str,
) -> StreamingResponse:
    if chat:
        object_name = "chat.completion.chunk"
        content_choice: dict[str, Any] = {
            "index": 0,
            "delta": {"role": "assistant", "content": text},
            "logprobs": None,
            "finish_reason": None,
        }
        terminal_choice: dict[str, Any] = {
            "index": 0,
            "delta": {},
            "logprobs": None,
            "finish_reason": None,
        }
    else:
        object_name = "text_completion"
        content_choice = {
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": None,
        }
        terminal_choice = {
            "index": 0,
            "text": "",
            "logprobs": None,
            "finish_reason": None,
        }

    def chunk(choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": object_name,
            "created": created,
            "model": model,
            "choices": choices,
        }

    async def events() -> AsyncIterator[bytes]:
        yield _sse_data(chunk([content_choice]))
        yield _sse_data(chunk([terminal_choice]))
        if include_usage:
            usage_chunk = chunk([])
            usage_chunk["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            }
            yield _sse_data(usage_chunk)
        yield _sse_data("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
            "X-TRTMC-Streaming": "buffered",
        },
    )


def create_app(registry: ModelRegistry, config: ServerConfig) -> FastAPI:
    metrics = Metrics()
    records = RequestRecords(config.records)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await asyncio.to_thread(registry.start)
            yield
        finally:
            await asyncio.to_thread(registry.close)
            records.close()

    app = FastAPI(title="TensorRT-Model-Connect Text Server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(_BodyLimitMiddleware, limit=config.max_body_bytes)

    app.add_middleware(RequestPolicy, records=records)

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
        context: dict[str, Any],
        system_prompt: str = "",
        chat: bool,
        chat_max_tokens: int | None = None,
    ) -> Any:
        context["model"] = request.model
        if request.n != 1:
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter", "n must be 1", param="n")
        if request.stream_options is not None and not request.stream:
            metrics.reject(route, 400)
            return error_response(
                400,
                "invalid_request",
                "stream_options requires stream=true",
                param="stream_options",
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
        context["replica"] = getattr(session, "replica_name", None)
        context["admission_ms"] = queue_seconds * 1000
        worker_config = generation_config(request, max_tokens)
        worker_config["use_chat_template"] = chat
        if system_prompt:
            worker_config["system_prompt"] = system_prompt
        context["generation_config"] = {key: value for key, value in worker_config.items() if key != "system_prompt"}
        context["has_system_prompt"] = bool(system_prompt)
        metrics.begin()
        inference_started = time.monotonic()
        if request.stream and registry.supports_streaming(request.model):
            return await incremental_response(session, {"prompt": prompt, "config": worker_config},
                model=request.model, chat=chat, include_usage=(request.stream_options is not None
                and request.stream_options.include_usage), request_id=request_id, context=context,
                route=route, queue_seconds=queue_seconds, started=inference_started, metrics=metrics)
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
        context.update(timings, worker_roundtrip_ms=inference_seconds * 1000,
                       completion_tokens=completion_tokens, completion_token_source="native_task",
                       timing_scope=result.get("timing_scope", "unavailable"),
                       streaming="buffered" if request.stream else "none")
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
        created = int(time.time())
        if request.stream:
            return _streaming_completion_response(
                response_id=response_id,
                created=created,
                model=request.model,
                text=text,
                completion_tokens=completion_tokens,
                chat=chat,
                include_usage=(
                    request.stream_options is not None and request.stream_options.include_usage
                ),
                request_id=request_id,
            )
        return JSONResponse(
            content={
                "id": response_id,
                "object": object_name,
                "created": created,
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
            context=http_request.state.timing,
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
            context=http_request.state.timing,
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
    ownership = task
    try:
        result = await asyncio.shield(task)
        try:
            extract_result(result)
        except WorkerProtocolError as error:
            ownership = asyncio.wrap_future(session.retire(error))
            await asyncio.shield(ownership)
            raise
        return result
    except asyncio.CancelledError:
        def release(completed: asyncio.Future[Any]) -> None:
            session.close()
            try:
                completed.exception()
            except BaseException:
                pass

        ownership.add_done_callback(release)
        raise
    finally:
        if ownership.done():
            session.close()


async def incremental_response(session: WorkerSession, payload: dict[str, Any], *, model: str,
                         chat: bool, include_usage: bool, request_id: str, context: dict[str, Any],
                         route: str, queue_seconds: float, started: float, metrics: Metrics) -> StreamingResponse:
    response_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(text: str, terminal: bool = False) -> dict[str, Any]:
        choice = {"index": 0, "logprobs": None, "finish_reason": None}
        if chat:
            choice["delta"] = {} if terminal else {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {"id": response_id, "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created, "model": model, "choices": [choice]}

    stream = None
    try:
        stream = session.stream(payload)
        first = await asyncio.to_thread(stream.next)
    except asyncio.CancelledError:
        if stream is not None:
            stream.abort()
            stream.future.add_done_callback(lambda _future: session.close())
            with CancelScope(shield=True):
                await asyncio.shield(asyncio.to_thread(stream.cancel))
        else:
            session.close()
        metrics.finish(route, 499, queue_seconds=queue_seconds,
                       inference_seconds=time.monotonic() - started)
        raise
    except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError, WorkerRemoteError,
            WorkerRequestTooLargeError) as error:
        session.close()
        invalid = (isinstance(error, WorkerRemoteError) and isinstance(error.details, Mapping)
                   and error.details.get("type") == "invalid_request_error")
        status = (400 if invalid else 504 if isinstance(error, WorkerTimeoutError)
                  else 503 if isinstance(error, WorkerCrashedError) else 502)
        metrics.finish(route, status, queue_seconds=queue_seconds,
                       inference_seconds=time.monotonic() - started)
        return error_response(status, error.code, public_worker_error(error), request_id=request_id)

    state = {"started": False}

    async def events() -> AsyncIterator[bytes]:
        state["started"] = True
        status = 499
        timings = {}
        retire = False
        try:
            fragments = []
            token_count = 0
            delta = first
            while delta is not None:
                fragments.append(delta["text_delta"])
                token_count += delta["token_count"]
                if delta["text_delta"]:
                    yield _sse_data(chunk(delta["text_delta"]))
                delta = await asyncio.to_thread(stream.next)
            result = stream.future.result()
            text, completion_tokens, timings = extract_result(result)
            if text != "".join(fragments) or completion_tokens != token_count:
                raise WorkerProtocolError("stream deltas disagree with the final result")
            context.update(timings, completion_tokens=completion_tokens,
                completion_token_source="native_task", timing_scope=result.get("timing_scope", "unavailable"),
                streaming="incremental", worker_roundtrip_ms=(time.monotonic() - started) * 1000)
            yield _sse_data(chunk("", terminal=True))
            if include_usage:
                usage = chunk("", terminal=True)
                usage["choices"] = []
                usage["usage"] = {"prompt_tokens": 0, "completion_tokens": completion_tokens,
                                  "total_tokens": completion_tokens}
                yield _sse_data(usage)
            yield _sse_data("[DONE]")
            status = 200
        except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError, WorkerRemoteError,
                WorkerRequestTooLargeError) as error:
            status = (504 if isinstance(error, WorkerTimeoutError)
                      else 503 if isinstance(error, WorkerCrashedError) else 502)
            retire = isinstance(error, WorkerProtocolError)
            context["terminal_status"] = status
            yield _sse_data({"error": {"message": public_worker_error(error), "type": "server_error",
                                      "code": error.code}})
            yield _sse_data("[DONE]")
        finally:
            if not stream.future.done() or retire:
                stream.abort()
                # The lease stays held until confirmed worker exit / native completion.
                stream.future.add_done_callback(lambda _future: session.close())
                # Starlette's disconnect listener cancels an AnyIO scope.
                # Shield that scope too so cleanup confirms worker exit before returning.
                with CancelScope(shield=True):
                    await asyncio.shield(asyncio.to_thread(stream.cancel))
            else:
                session.close()
            context.setdefault("terminal_status", status)
            metrics.finish(route, status, queue_seconds=queue_seconds,
                           inference_seconds=time.monotonic() - started, timings=timings)

    body = events()

    class LeasedResponse(StreamingResponse):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                if state["started"]:
                    # A failed ASGI send can leave the iterator suspended at
                    # its yield. Close it explicitly to finish lane retirement.
                    await body.aclose()
                else:
                    if not stream.future.done():
                        stream.abort()
                        stream.future.add_done_callback(lambda _future: session.close())
                        with CancelScope(shield=True):
                            await asyncio.shield(asyncio.to_thread(stream.cancel))
                    else:
                        session.close()
                    context["terminal_status"] = 499
                    metrics.finish(route, 499, queue_seconds=queue_seconds,
                                   inference_seconds=time.monotonic() - started)

    return LeasedResponse(body, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id,
        "X-TRTMC-Streaming": "incremental"})
